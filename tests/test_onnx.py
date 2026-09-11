"""Testes da exportação para ONNX e do backend correspondente.

O teste central aqui é o de **paridade**. Um motor de inferência mais rápido que
classifica diferente do modelo validado não é uma otimização: é outro modelo, servido
sem validação. Num sistema de triagem, é a diferença entre acelerar e trocar o critério
clínico sem avisar ninguém.
"""

import numpy as np
import pytest

from triage.api.predictor import OnnxBackend, Predictor, SklearnBackend
from triage.data.loader import CONDITION_COLUMN, TEXT_COLUMN, load_split
from triage.model.decision import DecisionRule
from triage.model.export_onnx import PARITY_THRESHOLD, check_parity, export
from triage.model.train import build_pipeline


@pytest.fixture(scope="module")
def artefatos(fixtures_dir_module, tmp_path_factory):
    """Treina na fixture pequena e exporta, devolvendo os caminhos dos artefatos."""
    import joblib

    destino = tmp_path_factory.mktemp("onnx")
    frame = load_split("train", raw_dir=fixtures_dir_module)

    pipeline = build_pipeline()
    pipeline.set_params(tfidf__min_df=1, tfidf__max_df=1.0)
    pipeline.fit(frame[TEXT_COLUMN], frame[CONDITION_COLUMN])

    model_path = destino / "model.pkl"
    joblib.dump(pipeline, model_path)

    export(
        model_path=model_path,
        onnx_path=destino / "model.onnx",
        quantized_path=destino / "model.int8.onnx",
    )
    return {"model": model_path, "onnx": destino / "model.onnx"}


def test_exportacao_gera_o_grafo(artefatos):
    assert artefatos["onnx"].exists()
    assert artefatos["onnx"].stat().st_size > 0


def test_grafo_e_menor_que_o_pickle(artefatos):
    # O ONNX carrega só os pesos; o pickle carrega o vetorizador inteiro junto.
    assert artefatos["onnx"].stat().st_size < artefatos["model"].stat().st_size


def test_exportacao_falha_com_artefato_ausente(tmp_path):
    with pytest.raises(FileNotFoundError, match="make train"):
        export(model_path=tmp_path / "inexistente.pkl", onnx_path=tmp_path / "x.onnx")


class TestParidade:
    """O grafo exportado precisa reproduzir o modelo treinado."""

    def test_classificacoes_sao_identicas(self, artefatos, fixtures_dir_module):
        textos = load_split("test", raw_dir=fixtures_dir_module)[TEXT_COLUMN].tolist()

        relatorio = check_parity(textos, model_path=artefatos["model"], onnx_path=artefatos["onnx"])

        assert relatorio["concordancia"] >= PARITY_THRESHOLD
        assert relatorio["divergencias"] == 0

    def test_diferenca_de_probabilidade_e_so_arredondamento(self, artefatos, fixtures_dir_module):
        # O ONNX opera em float32 e o scikit-learn em float64; a diferença residual
        # precisa ficar na ordem do épsilon de float32, não em casa decimal visível.
        textos = load_split("test", raw_dir=fixtures_dir_module)[TEXT_COLUMN].tolist()

        relatorio = check_parity(textos, model_path=artefatos["model"], onnx_path=artefatos["onnx"])

        assert relatorio["max_diff_probabilidade"] < 1e-5

    def test_limiar_de_paridade_e_exigente(self):
        """O limiar existe para reprovar troca de comportamento, não para carimbar."""
        assert PARITY_THRESHOLD >= 0.999


class TestBackendOnnx:
    """O backend ONNX precisa ser intercambiável com o scikit-learn."""

    def test_expoe_as_mesmas_condicoes(self, artefatos):
        sklearn_backend = SklearnBackend(artefatos["model"])
        onnx_backend = OnnxBackend(artefatos["model"], artefatos["onnx"])

        assert onnx_backend.condition_classes == sklearn_backend.condition_classes

    def test_devolve_probabilidades_no_mesmo_formato(self, artefatos):
        onnx_backend = OnnxBackend(artefatos["model"], artefatos["onnx"])

        proba = onnx_backend.predict_proba("acute myocardial infarction with troponin rise")

        assert proba.shape == (1, 5)
        assert proba.sum() == pytest.approx(1.0, abs=1e-5)

    def test_decide_igual_ao_sklearn(self, artefatos, fixtures_dir_module):
        regra = DecisionRule([1, 2, 3, 4, 5], urgent_threshold=0.3)
        sk = Predictor(backend=SklearnBackend(artefatos["model"]), rule=regra)
        onnx = Predictor(backend=OnnxBackend(artefatos["model"], artefatos["onnx"]), rule=regra)

        for texto in load_split("test", raw_dir=fixtures_dir_module)[TEXT_COLUMN]:
            assert sk.predict(texto)["urgencia"] == onnx.predict(texto)["urgencia"]

    def test_identifica_o_backend_na_resposta(self, artefatos):
        predictor = Predictor(backend=OnnxBackend(artefatos["model"], artefatos["onnx"]))

        assert predictor.predict("acute chest pain with dyspnea")["backend"] == "onnx"

    def test_backend_quantizado_tem_nome_proprio(self, artefatos):
        # O rótulo de métrica precisa distinguir os motores; sem isso o painel de
        # comparação somaria as duas séries.
        backend = OnnxBackend(artefatos["model"], artefatos["onnx"], name="onnx-int8")

        assert backend.name == "onnx-int8"

    def test_erro_claro_quando_o_grafo_esta_ausente(self, artefatos, tmp_path):
        with pytest.raises(FileNotFoundError, match="make onnx"):
            OnnxBackend(artefatos["model"], tmp_path / "inexistente.onnx")


def test_predictor_rejeita_backend_desconhecido():
    with pytest.raises(ValueError, match="onnx-int8"):
        Predictor.load(backend_name="tensorrt")


def test_features_do_grafo_batem_com_o_vocabulario(artefatos):
    """Grafo e vetorizador precisam concordar no número de features.

    Se divergirem, o ONNX recebe um vetor de tamanho errado e falha em tempo de
    execução — ou pior, aceita silenciosamente e classifica lixo.
    """
    import joblib
    import onnxruntime

    pipeline = joblib.load(artefatos["model"])
    sessao = onnxruntime.InferenceSession(
        str(artefatos["onnx"]), providers=["CPUExecutionProvider"]
    )

    esperado = len(pipeline.named_steps["tfidf"].vocabulary_)
    assert sessao.get_inputs()[0].shape[1] == esperado


def test_grafo_aceita_lote(artefatos):
    """A dimensão de lote é dinâmica: a verificação de paridade depende disso."""
    import onnxruntime

    sessao = onnxruntime.InferenceSession(
        str(artefatos["onnx"]), providers=["CPUExecutionProvider"]
    )
    n_features = sessao.get_inputs()[0].shape[1]
    entrada = np.zeros((7, n_features), dtype=np.float32)

    saida = np.asarray(sessao.run(None, {sessao.get_inputs()[0].name: entrada})[1])

    assert saida.shape[0] == 7
