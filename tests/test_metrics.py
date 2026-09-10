"""Testes da instrumentação Prometheus."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from triage.api import metrics
from triage.api.main import app
from triage.api.predictor import InferenceBackend, Predictor
from triage.model.decision import DecisionRule

LAUDO = (
    "Patient presents with acute chest pain radiating to the left arm, "
    "diaphoresis and shortness of breath for the past two hours."
)
SEGREDO = "PACIENTE_JOAO_DA_SILVA_CPF_12345678900_DIAGNOSTICO_CONFIDENCIAL"


class FakeBackend(InferenceBackend):
    """Backend determinístico: 70% em neoplasia, que projeta para `urgente`."""

    name = "fake"

    @property
    def condition_classes(self):
        return [1, 2, 3, 4, 5]

    def predict_proba(self, text: str) -> np.ndarray:
        return np.array([[0.70, 0.05, 0.10, 0.05, 0.10]])


@pytest.fixture
def client():
    """Cliente com preditor falso, para exercitar a instrumentação."""
    predictor = Predictor(
        backend=FakeBackend(),
        rule=DecisionRule([1, 2, 3, 4, 5], urgent_threshold=0.5, recall_target=0.9),
    )
    app.state.predictor = predictor
    with TestClient(app) as test_client:
        test_client.app.state.predictor = predictor
        # O lifespan publica o estado do modelo real do disco; aqui quem serve é o
        # preditor falso, então os medidores precisam refletir ele.
        metrics.set_model_state(True, predictor.rule.urgent_threshold)
        yield test_client


def _rotas_declaradas() -> set[str]:
    """Caminhos que a aplicação realmente expõe.

    Calculado a partir do app em vez de escrito à mão: uma rota nova não deve exigir
    manutenção do teste, mas um caminho que não é rota deve continuar sendo reprovado.
    """
    return {getattr(rota, "path", "") for rota in app.routes} | {"desconhecida"}


def _amostras(texto: str, nome: str) -> list:
    """Extrai as amostras de uma métrica do texto de exposição."""
    return [
        amostra
        for familia in text_string_to_metric_families(texto)
        for amostra in familia.samples
        if amostra.name.startswith(nome)
    ]


def _valor(texto: str, nome: str, **rotulos) -> float:
    """Soma o valor das amostras que casam com os rótulos pedidos."""
    return sum(
        amostra.value
        for amostra in _amostras(texto, nome)
        if all(amostra.labels.get(chave) == valor for chave, valor in rotulos.items())
    )


def test_endpoint_de_metricas_responde_no_formato_do_prometheus(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "triage_requests_total" in response.text


def test_metricas_ficam_fora_do_schema_openapi(client):
    # É endpoint de infraestrutura, não faz parte do contrato da triagem.
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]


def test_conta_requisicoes_por_rota_e_status(client):
    # O registro do Prometheus é global e acumula entre testes, então a leitura tem
    # que ser a mesma antes e depois — filtrar status só na segunda mediria outra série.
    filtro = {"endpoint": "/predict", "status": "200"}
    antes = _valor(client.get("/metrics").text, "triage_requests_total", **filtro)

    client.post("/predict", json={"texto": LAUDO})

    depois = _valor(client.get("/metrics").text, "triage_requests_total", **filtro)
    assert depois == antes + 1


def test_conta_erros_de_validacao(client):
    antes = _valor(client.get("/metrics").text, "triage_errors_total", tipo="http_422")

    client.post("/predict", json={"texto": "curto"})

    depois = _valor(client.get("/metrics").text, "triage_errors_total", tipo="http_422")
    assert depois == antes + 1


def test_registra_distribuicao_de_urgencias(client):
    antes = _valor(client.get("/metrics").text, "triage_predictions_total", urgencia="urgente")

    client.post("/predict", json={"texto": LAUDO})

    depois = _valor(client.get("/metrics").text, "triage_predictions_total", urgencia="urgente")
    assert depois == antes + 1


def test_registra_a_regra_que_decidiu(client):
    client.post("/predict", json={"texto": LAUDO})

    texto = client.get("/metrics").text
    amostras = _amostras(texto, "triage_predictions_total")

    assert any(amostra.labels.get("regra") == "condicao_dominante" for amostra in amostras)


def test_mede_latencia_de_requisicao_e_de_inferencia(client):
    client.post("/predict", json={"texto": LAUDO})

    texto = client.get("/metrics").text

    assert _valor(texto, "triage_request_duration_seconds_count", endpoint="/predict") >= 1
    assert _valor(texto, "triage_inference_duration_seconds_count", backend="fake") >= 1


def test_separa_tempo_de_modelo_do_tempo_total(client):
    """A inferência não pode ser maior que a requisição que a contém."""
    for _ in range(5):
        client.post("/predict", json={"texto": LAUDO})

    texto = client.get("/metrics").text
    total = _valor(texto, "triage_request_duration_seconds_sum", endpoint="/predict")
    inferencia = _valor(texto, "triage_inference_duration_seconds_sum", backend="fake")

    assert inferencia < total


def test_publica_estado_do_modelo(client):
    texto = client.get("/metrics").text

    assert _valor(texto, "triage_model_loaded") == 1
    assert _valor(texto, "triage_urgent_threshold") == 0.5


def test_nao_cria_serie_por_caminho_inexistente(client):
    """Varredura de rotas não pode inflar a cardinalidade do Prometheus."""
    for caminho in ("/admin", "/wp-login.php", "/api/v1/users/42"):
        client.get(caminho)

    amostras = _amostras(client.get("/metrics").text, "triage_requests_total")
    rotas = {amostra.labels.get("endpoint") for amostra in amostras}

    assert rotas <= _rotas_declaradas(), f"rota não declarada virou série: {rotas}"
    assert not {"/admin", "/wp-login.php", "/api/v1/users/42"} & rotas


class TestPrivacidade:
    """Laudo é dado pessoal sensível: não pode aparecer em métrica, sob nenhuma forma."""

    def test_texto_do_laudo_nunca_vai_para_as_metricas(self, client):
        client.post("/predict", json={"texto": f"{SEGREDO}. {LAUDO}"})

        texto = client.get("/metrics").text

        assert SEGREDO not in texto
        assert "JOAO_DA_SILVA" not in texto
        assert "12345678900" not in texto

    def test_nenhum_rotulo_carrega_texto_livre(self, client):
        client.post("/predict", json={"texto": f"{SEGREDO}. {LAUDO}"})
        client.post("/predict", json={"texto": "curto"})

        texto = client.get("/metrics").text
        valores = {
            valor for amostra in _amostras(texto, "triage_") for valor in amostra.labels.values()
        }

        # Rótulo de domínio aberto é o vetor pelo qual dado clínico vazaria para a
        # métrica, e também o que mata o Prometheus por cardinalidade. Um laudo tem
        # centenas de caracteres; nenhum rótulo legítimo daqui chega perto disso.
        longos = [valor for valor in valores if len(valor) > 40]
        assert not longos, f"rótulo com cara de texto livre: {longos}"

        # Nada que venha do corpo da requisição pode aparecer como rótulo.
        for palavra in SEGREDO.split("_"):
            assert not any(palavra in valor for valor in valores)
