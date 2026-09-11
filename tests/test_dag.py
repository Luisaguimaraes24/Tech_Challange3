"""Testes da DAG de treino.

O Airflow não faz parte do ambiente do projeto — ele roda em container próprio. Estes
testes verificam o que dá para verificar sem ele: a estrutura declarada da DAG, lida do
arquivo-fonte, e a lógica do portão de qualidade, que é a única regra de negócio da DAG
e a que causa dano se estiver errada.

A validação de que a DAG realmente carrega no Airflow fica em `scripts/validate_dag.py`,
executado em ambiente isolado pelo CI.
"""

import ast
from pathlib import Path

import pytest

DAG_FILE = Path(__file__).resolve().parents[1] / "dags" / "triage_training_dag.py"

TAREFAS_ESPERADAS = {
    "ingest_data",
    "prepare_dataset",
    "train_model",
    "validate_model",
    "export_to_onnx",
    "publish_model",
}


@pytest.fixture(scope="module")
def arvore():
    """AST do arquivo da DAG, para inspeção sem importar o Airflow."""
    return ast.parse(DAG_FILE.read_text(encoding="utf-8"))


def _funcoes_decoradas_com(arvore, decorador: str) -> set[str]:
    """Nomes das funções que carregam determinado decorador."""
    encontradas = set()
    for node in ast.walk(arvore):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            alvo = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(alvo, ast.Name) and alvo.id == decorador:
                encontradas.add(node.name)
    return encontradas


def test_arquivo_da_dag_existe():
    assert DAG_FILE.exists()


def test_dag_declara_todas_as_tarefas_do_pipeline(arvore):
    assert _funcoes_decoradas_com(arvore, "task") == TAREFAS_ESPERADAS


def test_dag_e_sintaticamente_valida(arvore):
    assert isinstance(arvore, ast.Module)


def test_dag_nao_faz_catchup(arvore):
    # Com catchup ligado, subir a stack dispararia uma execução para cada semana desde
    # a data de início — retreinos em massa que ninguém pediu.
    fonte = DAG_FILE.read_text(encoding="utf-8")
    assert "catchup=False" in fonte


def test_dag_tem_agendamento_definido():
    assert 'schedule="@weekly"' in DAG_FILE.read_text(encoding="utf-8")


def test_dag_importa_o_projeto_dentro_das_tarefas(arvore):
    """Imports de `triage` ficam dentro das tarefas, não no topo do arquivo.

    O dag-processor reimporta o arquivo a cada poucos segundos; carregar scikit-learn
    e pandas no topo pagaria esse custo em todo ciclo de parsing, mesmo sem nenhuma
    execução em andamento.
    """
    imports_no_topo = [
        node.module for node in arvore.body if isinstance(node, ast.ImportFrom) and node.module
    ]

    assert not [modulo for modulo in imports_no_topo if modulo.startswith("triage")]


def test_publicacao_leva_modelo_e_regra_juntos():
    """Modelo e regra de decisão precisam ser promovidos como um conjunto.

    Um limiar calibrado para um modelo não vale para outro: publicar só um dos dois
    deixaria a triagem operando num ponto que nunca foi medido.
    """
    fonte = DAG_FILE.read_text(encoding="utf-8")

    for artefato in ("model.pkl", "decision.json", "metrics.json"):
        assert artefato in fonte


def test_treino_escreve_em_staging_e_nao_no_volume_servido():
    """O treino não pode escrever direto no diretório que a API lê.

    Se escrevesse, um modelo reprovado no portão de qualidade já teria substituído o
    modelo em produção antes da validação acontecer.
    """
    fonte = DAG_FILE.read_text(encoding="utf-8")

    assert "STAGING_DIR" in fonte
    assert 'STAGING_DIR = SERVING_DIR / "staging"' in fonte
