"""Testes do pipeline de treino e avaliação."""

import json

import pandas as pd
import pytest

from triage.api.predictor import Predictor, SklearnBackend
from triage.model.evaluate import evaluate_pipeline
from triage.model.train import build_pipeline, train


@pytest.fixture(scope="module")
def pipeline_treinado(fixtures_dir_module):
    """Pipeline treinado na fixture pequena, suficiente para validar o contrato."""
    from triage.data.loader import TEXT_COLUMN, URGENCY_COLUMN, load_split

    frame = load_split("train", raw_dir=fixtures_dir_module)
    pipeline = build_pipeline()
    # A fixture tem 20 linhas; sem afrouxar min_df o vocabulário fica vazio.
    pipeline.set_params(tfidf__min_df=1, tfidf__max_df=1.0)
    pipeline.fit(frame[TEXT_COLUMN], frame[URGENCY_COLUMN])
    return pipeline


def test_build_pipeline_tem_vetorizador_e_classificador():
    pipeline = build_pipeline()

    assert list(pipeline.named_steps) == ["tfidf", "clf"]
    assert pipeline.named_steps["clf"].class_weight == "balanced"


def test_pipeline_e_reprodutivel():
    assert build_pipeline().named_steps["clf"].random_state == 42


def test_evaluate_pipeline_devolve_relatorio_serializavel(pipeline_treinado, fixtures_dir_module):
    from triage.data.loader import load_split

    frame = load_split("test", raw_dir=fixtures_dir_module)
    metrics = evaluate_pipeline(pipeline_treinado, frame)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert set(metrics["per_class"]) == set(metrics["labels"])
    assert len(metrics["confusion_matrix"]) == len(metrics["labels"])
    assert metrics["n_test"] == len(frame)
    json.dumps(metrics)  # precisa ser serializável para o metrics.json


def test_matriz_de_confusao_soma_o_total_de_amostras(pipeline_treinado, fixtures_dir_module):
    from triage.data.loader import load_split

    frame = load_split("test", raw_dir=fixtures_dir_module)
    metrics = evaluate_pipeline(pipeline_treinado, frame)

    total = sum(sum(row) for row in metrics["confusion_matrix"])
    assert total == metrics["n_test"]


def test_evaluate_pipeline_com_classificador_perfeito(pipeline_treinado):
    frame = pd.DataFrame(
        {
            "medical_abstract": ["texto de laudo para triagem"] * 3,
            "urgency": ["normal"] * 3,
        }
    )
    metrics = evaluate_pipeline(pipeline_treinado, frame)

    assert metrics["n_test"] == 3


def test_train_persiste_modelo_e_metricas(fixtures_dir, tmp_path, monkeypatch):
    # Sem afrouxar min_df, 20 linhas não formam vocabulário.
    monkeypatch.setitem(
        __import__("triage.model.train", fromlist=["VECTORIZER_PARAMS"]).VECTORIZER_PARAMS,
        "min_df",
        1,
    )
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"

    metrics = train(raw_dir=fixtures_dir, model_path=model_path, metrics_path=metrics_path)

    assert model_path.exists()
    assert metrics_path.exists()
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["f1_macro"] == metrics["f1_macro"]
    assert metrics["n_train"] == 20
    assert metrics["vocabulary_size"] > 0
    assert metrics["fit_seconds"] > 0


def test_backend_sklearn_carrega_artefato_treinado(fixtures_dir, tmp_path, monkeypatch):
    monkeypatch.setitem(
        __import__("triage.model.train", fromlist=["VECTORIZER_PARAMS"]).VECTORIZER_PARAMS,
        "min_df",
        1,
    )
    model_path = tmp_path / "model.pkl"
    train(raw_dir=fixtures_dir, model_path=model_path, metrics_path=tmp_path / "metrics.json")

    backend = SklearnBackend(model_path)
    probabilidades = backend.predict_proba("acute myocardial infarction with elevated troponin")

    assert set(probabilidades) == set(backend.classes)
    assert sum(probabilidades.values()) == pytest.approx(1.0, abs=1e-6)


def test_backend_sklearn_erro_claro_quando_artefato_ausente(tmp_path):
    with pytest.raises(FileNotFoundError, match="make train"):
        SklearnBackend(tmp_path / "inexistente.pkl")


def test_predictor_rejeita_backend_desconhecido():
    with pytest.raises(ValueError, match="Backend de inferência desconhecido"):
        Predictor.load(backend_name="tensorrt")


def test_predictor_escolhe_a_classe_mais_provavel(fixtures_dir, tmp_path, monkeypatch):
    monkeypatch.setitem(
        __import__("triage.model.train", fromlist=["VECTORIZER_PARAMS"]).VECTORIZER_PARAMS,
        "min_df",
        1,
    )
    model_path = tmp_path / "model.pkl"
    train(raw_dir=fixtures_dir, model_path=model_path, metrics_path=tmp_path / "metrics.json")

    resultado = Predictor(backend=SklearnBackend(model_path)).predict("chest pain and dyspnea")

    assert resultado["urgencia"] == max(
        resultado["probabilidades"], key=resultado["probabilidades"].__getitem__
    )
    assert resultado["backend"] == "sklearn"
    assert resultado["descricao"]
