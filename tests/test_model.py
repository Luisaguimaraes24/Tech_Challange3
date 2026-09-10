"""Testes do pipeline de treino e avaliação."""

import json

import pytest

from triage.api.predictor import Predictor, SklearnBackend
from triage.data.loader import CONDITION_COLUMN, TEXT_COLUMN, load_split
from triage.model.decision import DecisionRule
from triage.model.evaluate import evaluate_decision
from triage.model.train import build_pipeline, train


@pytest.fixture
def treino_leve(monkeypatch):
    """Afrouxa o TF-IDF: a fixture tem 20 linhas e min_df=3 zeraria o vocabulário."""
    from triage.model import train as train_module

    monkeypatch.setitem(train_module.VECTORIZER_PARAMS, "min_df", 1)
    monkeypatch.setattr(train_module, "CALIBRATION_FOLDS", 2)


@pytest.fixture(scope="module")
def pipeline_treinado(fixtures_dir_module):
    """Pipeline treinado nas 5 condições, com a fixture pequena."""
    frame = load_split("train", raw_dir=fixtures_dir_module)
    pipeline = build_pipeline()
    pipeline.set_params(tfidf__min_df=1, tfidf__max_df=1.0)
    pipeline.fit(frame[TEXT_COLUMN], frame[CONDITION_COLUMN])
    return pipeline


def test_build_pipeline_tem_vetorizador_e_classificador():
    pipeline = build_pipeline()

    assert list(pipeline.named_steps) == ["tfidf", "clf"]
    assert pipeline.named_steps["clf"].class_weight == "balanced"


def test_pipeline_e_reprodutivel():
    assert build_pipeline().named_steps["clf"].random_state == 42


def test_pipeline_treina_nas_cinco_condicoes(pipeline_treinado):
    # É o ponto central da modelagem: o alvo do treino é a condição, não a urgência.
    assert sorted(pipeline_treinado.classes_) == [1, 2, 3, 4, 5]


def test_evaluate_devolve_relatorio_serializavel(pipeline_treinado, fixtures_dir_module):
    frame = load_split("test", raw_dir=fixtures_dir_module)
    rule = DecisionRule([int(c) for c in pipeline_treinado.classes_])

    metrics = evaluate_decision(pipeline_treinado, rule, frame)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert 0.0 <= metrics["recall_urgente"] <= 1.0
    assert set(metrics["per_class"]) == set(metrics["labels"])
    assert metrics["n_test"] == len(frame)
    json.dumps(metrics)  # precisa ser serializável para o metrics.json


def test_recall_urgente_e_promovido_a_metrica_de_primeiro_nivel(
    pipeline_treinado, fixtures_dir_module
):
    frame = load_split("test", raw_dir=fixtures_dir_module)
    rule = DecisionRule([int(c) for c in pipeline_treinado.classes_])

    metrics = evaluate_decision(pipeline_treinado, rule, frame)

    assert metrics["recall_urgente"] == metrics["per_class"]["urgente"]["recall"]
    assert metrics["precision_urgente"] == metrics["per_class"]["urgente"]["precision"]


def test_matriz_de_confusao_soma_o_total_de_amostras(pipeline_treinado, fixtures_dir_module):
    frame = load_split("test", raw_dir=fixtures_dir_module)
    rule = DecisionRule([int(c) for c in pipeline_treinado.classes_])

    metrics = evaluate_decision(pipeline_treinado, rule, frame)

    assert sum(sum(linha) for linha in metrics["confusion_matrix"]) == metrics["n_test"]


def test_trava_de_urgencia_aumenta_o_recall_da_classe_critica(
    pipeline_treinado, fixtures_dir_module
):
    frame = load_split("test", raw_dir=fixtures_dir_module)
    classes = [int(c) for c in pipeline_treinado.classes_]

    sem_trava = evaluate_decision(pipeline_treinado, DecisionRule(classes), frame)
    com_trava = evaluate_decision(
        pipeline_treinado, DecisionRule(classes, urgent_threshold=0.2), frame
    )

    assert com_trava["recall_urgente"] >= sem_trava["recall_urgente"]
    assert com_trava["decisoes_por_trava_de_urgencia"] >= 0


def test_train_persiste_modelo_regra_e_metricas(fixtures_dir, tmp_path, treino_leve):
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"
    decision_path = tmp_path / "decision.json"

    metrics = train(
        raw_dir=fixtures_dir,
        model_path=model_path,
        metrics_path=metrics_path,
        decision_path=decision_path,
        recall_target=0.8,
    )

    assert model_path.exists()
    assert decision_path.exists()
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["f1_macro"] == metrics["f1_macro"]
    assert metrics["n_train"] == 20
    assert metrics["vocabulary_size"] > 0
    assert metrics["decision_rule"]["recall_target"] == 0.8
    assert metrics["threshold_calibration"]["curve"]


def test_train_grava_a_regra_no_formato_que_o_predictor_le(fixtures_dir, tmp_path, treino_leve):
    decision_path = tmp_path / "decision.json"
    train(
        raw_dir=fixtures_dir,
        model_path=tmp_path / "model.pkl",
        metrics_path=tmp_path / "metrics.json",
        decision_path=decision_path,
    )

    rule = DecisionRule.from_dict(json.loads(decision_path.read_text(encoding="utf-8")))

    assert rule.condition_classes == [1, 2, 3, 4, 5]


def test_backend_sklearn_devolve_probabilidades_de_condicao(fixtures_dir, tmp_path, treino_leve):
    model_path = tmp_path / "model.pkl"
    train(
        raw_dir=fixtures_dir,
        model_path=model_path,
        metrics_path=tmp_path / "metrics.json",
        decision_path=tmp_path / "decision.json",
    )

    backend = SklearnBackend(model_path)
    proba = backend.predict_proba("acute myocardial infarction with elevated troponin")

    assert backend.condition_classes == [1, 2, 3, 4, 5]
    assert proba.shape == (1, 5)
    assert proba.sum() == pytest.approx(1.0, abs=1e-6)


def test_backend_sklearn_erro_claro_quando_artefato_ausente(tmp_path):
    with pytest.raises(FileNotFoundError, match="make train"):
        SklearnBackend(tmp_path / "inexistente.pkl")


def test_predictor_rejeita_backend_desconhecido():
    with pytest.raises(ValueError, match="Backend de inferência desconhecido"):
        Predictor.load(backend_name="tensorrt")


def test_predictor_devolve_urgencia_coerente_com_a_distribuicao(
    fixtures_dir, tmp_path, treino_leve
):
    model_path = tmp_path / "model.pkl"
    train(
        raw_dir=fixtures_dir,
        model_path=model_path,
        metrics_path=tmp_path / "metrics.json",
        decision_path=tmp_path / "decision.json",
    )

    resultado = Predictor(backend=SklearnBackend(model_path)).predict("chest pain and dyspnea")

    assert resultado["urgencia"] in {"atencao", "normal", "urgente"}
    assert resultado["confianca"] == resultado["probabilidades"][resultado["urgencia"]]
    assert sum(resultado["probabilidades"].values()) == pytest.approx(1.0, abs=1e-3)
    assert resultado["regra"] in {"condicao_dominante", "trava_de_urgencia"}
    assert resultado["backend"] == "sklearn"
