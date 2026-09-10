"""Testes da API de triagem.

O modelo real é substituído por um backend falso: os testes verificam o contrato
HTTP e a fronteira de validação, não a qualidade do classificador — essa é medida
pelas métricas de avaliação.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from triage.api.main import app
from triage.api.predictor import InferenceBackend, Predictor
from triage.api.schemas import MIN_TEXT_LENGTH
from triage.model.decision import DecisionRule

LAUDO_VALIDO = (
    "Patient presents with acute chest pain radiating to the left arm, "
    "diaphoresis and shortness of breath for the past two hours."
)


class FakeBackend(InferenceBackend):
    """Backend determinístico, sem dependência de artefato em disco.

    Devolve 70% em neoplasia (condição 1), que projeta para `urgente`.
    """

    name = "fake"

    @property
    def condition_classes(self):
        return [1, 2, 3, 4, 5]

    def predict_proba(self, text: str) -> np.ndarray:
        return np.array([[0.70, 0.05, 0.10, 0.05, 0.10]])


def _preditor() -> Predictor:
    return Predictor(
        backend=FakeBackend(),
        rule=DecisionRule([1, 2, 3, 4, 5], urgent_threshold=0.5, recall_target=0.9),
        metrics={"f1_macro": 0.9, "recall_urgente": 0.9},
    )


@pytest.fixture
def client():
    """Cliente com um preditor falso já carregado."""
    app.state.predictor = _preditor()
    with TestClient(app) as test_client:
        test_client.app.state.predictor = _preditor()
        yield test_client


@pytest.fixture
def client_sem_modelo():
    """Cliente simulando o serviço degradado, sem modelo carregado."""
    with TestClient(app) as test_client:
        test_client.app.state.predictor = None
        yield test_client


def test_health_reporta_modelo_carregado(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["backend"] == "fake"


def test_health_responde_mesmo_sem_modelo(client_sem_modelo):
    response = client_sem_modelo.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_predict_classifica_laudo(client):
    response = client.post("/predict", json={"texto": LAUDO_VALIDO})

    assert response.status_code == 200
    body = response.json()
    assert body["urgencia"] == "urgente"
    # 0.70 (neoplasia) + 0.05 (cardiovascular) projetam para 0.75 de urgência.
    assert body["confianca"] == pytest.approx(0.75)
    assert body["descricao"]
    assert body["backend"] == "fake"


def test_predict_informa_qual_regra_decidiu(client):
    body = client.post("/predict", json={"texto": LAUDO_VALIDO}).json()

    # A condição dominante já é urgente, então a trava não precisou entrar.
    assert body["regra"] == "condicao_dominante"


def test_predict_devolve_distribuicao_completa(client):
    body = client.post("/predict", json={"texto": LAUDO_VALIDO}).json()

    assert set(body["probabilidades"]) == {"atencao", "normal", "urgente"}
    assert sum(body["probabilidades"].values()) == pytest.approx(1.0, abs=1e-3)


def test_predict_mede_latencia(client):
    body = client.post("/predict", json={"texto": LAUDO_VALIDO}).json()

    assert body["latencia_ms"] >= 0


@pytest.mark.parametrize(
    "texto",
    ["", "   ", "curto demais"],
    ids=["vazio", "so_espacos", "abaixo_do_minimo"],
)
def test_predict_rejeita_texto_invalido(client, texto):
    response = client.post("/predict", json={"texto": texto})

    assert response.status_code == 422


def test_predict_rejeita_texto_longo_demais(client):
    response = client.post("/predict", json={"texto": "a" * 20_001})

    assert response.status_code == 422


def test_predict_aceita_texto_no_limite_minimo(client):
    response = client.post("/predict", json={"texto": "x" * MIN_TEXT_LENGTH})

    assert response.status_code == 200


def test_predict_exige_o_campo_texto(client):
    response = client.post("/predict", json={"laudo": LAUDO_VALIDO})

    assert response.status_code == 422


def test_predict_sem_modelo_responde_503(client_sem_modelo):
    response = client_sem_modelo.post("/predict", json={"texto": LAUDO_VALIDO})

    assert response.status_code == 503
    assert "detail" in response.json()


def test_model_info_expoe_metadados(client):
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "fake"
    assert body["classes"] == ["atencao", "normal", "urgente"]
    assert body["metrics"]["f1_macro"] == 0.9
    # A regra em vigor precisa ser auditável pelo hospital, não só pelo código.
    assert body["decision_rule"]["urgent_threshold"] == 0.5
    assert body["decision_rule"]["recall_target"] == 0.9


def test_model_info_sem_modelo_responde_503(client_sem_modelo):
    assert client_sem_modelo.get("/model-info").status_code == 503


def test_documentacao_openapi_disponivel(client):
    schema = client.get("/openapi.json").json()

    assert "/predict" in schema["paths"]
    assert schema["info"]["title"] == "Triagem de Laudos Médicos"
