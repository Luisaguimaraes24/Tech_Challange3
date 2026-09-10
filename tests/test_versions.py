"""Guarda contra divergência de versões entre os arquivos de infraestrutura.

A versão do Airflow aparece em três lugares que ninguém lê junto: a imagem do
`Dockerfile.airflow`, o job de validação do CI e o alvo do Makefile. Quando eles
divergem, o sintoma é péssimo — o pipeline passa na máquina de quem escreveu e falha no
CI com um `ImportError` que não tem relação aparente com a mudança feita.

Foi exatamente o que aconteceu: o CI ficou fixado em 3.1.3 enquanto o resto do projeto
usava 3.2.2, e `AirflowFailException` mudou de módulo entre as duas.
"""

import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
DOCKERFILE_AIRFLOW = RAIZ / "Dockerfile.airflow"
CI = RAIZ / ".github" / "workflows" / "ci.yml"
MAKEFILE = RAIZ / "Makefile"
DOCKERFILE_API = RAIZ / "Dockerfile"


def _buscar(arquivo: Path, padrao: str) -> str:
    """Extrai o primeiro grupo capturado do padrão no arquivo.

    Args:
        arquivo: Arquivo a inspecionar.
        padrao: Expressão regular com exatamente um grupo de captura.

    Returns:
        O valor capturado.
    """
    conteudo = arquivo.read_text(encoding="utf-8")
    achado = re.search(padrao, conteudo)
    assert achado, f"padrão {padrao!r} não encontrado em {arquivo.name}"
    return achado.group(1)


def test_versao_do_airflow_e_a_mesma_em_todo_lugar():
    imagem = _buscar(DOCKERFILE_AIRFLOW, r"FROM apache/airflow:([\d.]+)-python")
    workflow = _buscar(CI, r'AIRFLOW_VERSION:\s*"([\d.]+)"')
    makefile = _buscar(MAKEFILE, r"AIRFLOW_VERSION\s*:=\s*([\d.]+)")

    assert imagem == workflow == makefile, (
        f"versões divergentes do Airflow: imagem={imagem}, CI={workflow}, Makefile={makefile}"
    )


def test_versao_do_python_e_a_mesma_em_todo_lugar():
    imagem_api = _buscar(DOCKERFILE_API, r"FROM python:([\d.]+)-slim AS builder")
    imagem_airflow = _buscar(DOCKERFILE_AIRFLOW, r"FROM apache/airflow:[\d.]+-python([\d.]+)")
    workflow = _buscar(CI, r'PYTHON_VERSION:\s*"([\d.]+)"')
    makefile = _buscar(MAKEFILE, r"PYTHON_VERSION\s*:=\s*([\d.]+)")

    assert imagem_api == imagem_airflow == workflow == makefile, (
        f"versões divergentes do Python: api={imagem_api}, airflow={imagem_airflow}, "
        f"CI={workflow}, Makefile={makefile}"
    )


def test_python_do_projeto_bate_com_o_das_imagens():
    requerido = _buscar(RAIZ / "pyproject.toml", r'requires-python = ">=([\d.]+)')
    imagem = _buscar(DOCKERFILE_API, r"FROM python:([\d.]+)-slim AS builder")

    assert imagem == requerido


@pytest.mark.parametrize("arquivo", [DOCKERFILE_API, DOCKERFILE_AIRFLOW, CI])
def test_versoes_estao_fixadas(arquivo):
    """Nenhuma imagem ou ação pode depender de `latest`.

    Uma tag móvel troca o ambiente de build sem nenhuma mudança no repositório, e a
    falha aparece num commit que não tem nada a ver com a causa.
    """
    conteudo = arquivo.read_text(encoding="utf-8")

    assert ":latest" not in conteudo

    # Uma referência de imagem sem `:tag` resolve para :latest implicitamente, que é o
    # mesmo problema com outro nome. Estágios internos (`FROM builder`) não contam.
    estagios = set(re.findall(r"^FROM\s+\S+\s+AS\s+(\S+)", conteudo, re.MULTILINE))
    sem_tag = [
        imagem
        for imagem in re.findall(r"^FROM\s+(\S+)", conteudo, re.MULTILINE)
        if ":" not in imagem and imagem not in estagios
    ]

    assert not sem_tag, f"imagens sem tag (equivalem a :latest): {sem_tag}"
