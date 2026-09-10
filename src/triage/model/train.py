"""Treino do classificador de urgência.

O modelo é um pipeline scikit-learn de duas etapas — `TfidfVectorizer` seguido de
`LogisticRegression` — escolhido por três motivos que se reforçam neste projeto:

* **latência**: a inferência é uma multiplicação esparsa, na casa de poucos
  milissegundos por laudo, sem GPU;
* **exportabilidade**: o pipeline inteiro converte para um único grafo ONNX,
  que é a otimização exigida na Etapa 4;
* **probabilidades calibradas**: a regressão logística devolve `predict_proba`, que
  alimenta tanto o score de confiança da API quanto a trava de urgência.

O alvo do treino são as **5 condições médicas** do corpus, não os 3 níveis de urgência.
A projeção para urgência acontece depois, na regra de decisão — ver
[`decision.py`](decision.py) para o porquê e para os números que sustentam a escolha.

O desbalanceamento entre as classes é tratado com `class_weight="balanced"`, não com
reamostragem, para não inventar laudos sintéticos.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from triage.config import SEED, settings
from triage.data.loader import CONDITION_COLUMN, TEXT_COLUMN, URGENCY_COLUMN, load_split
from triage.model.decision import DecisionRule, calibrate_threshold
from triage.model.evaluate import evaluate_decision

logger = logging.getLogger(__name__)

VECTORIZER_PARAMS = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.9,
    "sublinear_tf": True,
    "max_features": 50_000,
}
"""Parâmetros do TF-IDF. `max_features` limita o grafo ONNX e a memória do container."""

CLASSIFIER_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1_000,
    "C": 1.0,
    "random_state": SEED,
}
"""Parâmetros da regressão logística."""

CALIBRATION_FOLDS = 3
"""Folds usados para gerar as probabilidades out-of-fold que calibram o limiar."""


def build_pipeline() -> Pipeline:
    """Monta o pipeline de vetorização e classificação.

    Returns:
        Pipeline scikit-learn não treinado, pronto para `fit` sobre as 5 condições.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**VECTORIZER_PARAMS)),
            ("clf", LogisticRegression(**CLASSIFIER_PARAMS)),
        ]
    )


def train(
    raw_dir: Path | None = None,
    model_path: Path | None = None,
    metrics_path: Path | None = None,
    decision_path: Path | None = None,
    recall_target: float | None = None,
) -> dict:
    """Treina o classificador, calibra a trava de urgência e persiste os artefatos.

    Args:
        raw_dir: Diretório dos CSVs do corpus. Usa `settings.raw_dir` quando omitido.
        model_path: Destino do pipeline serializado. Usa `settings.sklearn_model_path`.
        metrics_path: Destino do relatório de métricas. Usa `settings.metrics_path`.
        decision_path: Destino da regra de decisão. Usa `settings.decision_path`.
        recall_target: Recall-alvo para `urgente`. Usa `settings.urgent_recall_target`.

    Returns:
        Dicionário de métricas da avaliação no split de teste.
    """
    model_path = model_path or settings.sklearn_model_path
    metrics_path = metrics_path or settings.metrics_path
    decision_path = decision_path or settings.decision_path
    recall_target = recall_target if recall_target is not None else settings.urgent_recall_target

    train_frame = load_split("train", raw_dir=raw_dir)
    test_frame = load_split("test", raw_dir=raw_dir)
    logger.info("Treino: %d amostras | Teste: %d amostras", len(train_frame), len(test_frame))

    x_train = train_frame[TEXT_COLUMN]
    y_train = train_frame[CONDITION_COLUMN]

    # O limiar é calibrado em probabilidades out-of-fold: calibrar em predições que o
    # modelo já viu daria um limiar otimista, que não se sustenta em produção.
    logger.info(
        "Gerando probabilidades out-of-fold (%d folds) para calibração...", CALIBRATION_FOLDS
    )
    started = time.perf_counter()
    oof_probabilities = cross_val_predict(
        build_pipeline(),
        x_train,
        y_train,
        cv=StratifiedKFold(n_splits=CALIBRATION_FOLDS, shuffle=True, random_state=SEED),
        method="predict_proba",
    )
    calibration_seconds = time.perf_counter() - started

    condition_classes = sorted(y_train.unique().tolist())
    threshold, calibration = calibrate_threshold(
        oof_probabilities,
        condition_classes,
        train_frame[URGENCY_COLUMN].to_numpy(),
        recall_target=recall_target,
    )
    logger.info(
        "Limiar da trava de urgência: %s (alvo de recall %.2f, calibrado em %.1fs)",
        threshold,
        recall_target,
        calibration_seconds,
    )

    pipeline = build_pipeline()
    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    logger.info("Treino concluído em %.2fs", fit_seconds)

    rule = DecisionRule(
        condition_classes=[int(code) for code in pipeline.classes_],
        urgent_threshold=threshold,
        recall_target=recall_target,
    )

    metrics = evaluate_decision(pipeline, rule, test_frame)
    metrics["fit_seconds"] = round(fit_seconds, 3)
    metrics["calibration_seconds"] = round(calibration_seconds, 3)
    metrics["n_train"] = len(train_frame)
    metrics["vocabulary_size"] = len(pipeline.named_steps["tfidf"].vocabulary_)
    metrics["decision_rule"] = rule.to_dict()
    metrics["threshold_calibration"] = calibration

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Modelo salvo em %s (%.1f KB)", model_path, model_path.stat().st_size / 1024)

    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(rule.to_dict(), indent=2), encoding="utf-8")
    logger.info("Regra de decisão salva em %s", decision_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Métricas salvas em %s", metrics_path)

    return metrics


def main() -> int:
    """Ponto de entrada de linha de comando do treino.

    Returns:
        0 se o modelo passar nos gates de qualidade, 1 caso contrário.
    """
    parser = argparse.ArgumentParser(description="Treina o classificador de urgência.")
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Não falha quando o modelo fica abaixo dos limiares configurados.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    metrics = train()

    logger.info(
        "accuracy=%.4f | f1_macro=%.4f | recall_urgente=%.4f | vocabulário=%d termos",
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["recall_urgente"],
        metrics["vocabulary_size"],
    )
    for level, scores in metrics["per_class"].items():
        logger.info(
            "  %-8s precision=%.3f recall=%.3f f1=%.3f (n=%d)",
            level,
            scores["precision"],
            scores["recall"],
            scores["f1"],
            scores["support"],
        )

    failures = []
    if metrics["f1_macro"] < settings.min_f1_macro:
        failures.append(f"f1-macro {metrics['f1_macro']:.4f} < {settings.min_f1_macro:.4f}")
    if metrics["recall_urgente"] < settings.min_recall_urgente:
        failures.append(
            f"recall de urgente {metrics['recall_urgente']:.4f} < {settings.min_recall_urgente:.4f}"
        )

    if failures and not args.skip_gate:
        for failure in failures:
            logger.error("Gate de qualidade reprovado: %s", failure)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
