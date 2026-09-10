"""Treino do classificador de urgência.

O modelo é um pipeline scikit-learn de duas etapas — `TfidfVectorizer` seguido de
`LogisticRegression` — escolhido por três motivos que se reforçam neste projeto:

* **latência**: a inferência é uma multiplicação esparsa, na casa de poucos
  milissegundos por laudo, sem GPU;
* **exportabilidade**: o pipeline inteiro converte para um único grafo ONNX,
  que é a otimização exigida na Etapa 4;
* **probabilidades calibradas**: a regressão logística devolve `predict_proba`,
  usado como score de confiança na resposta da API.

O desbalanceamento entre os níveis (43% urgente / 33% normal / 24% atenção) é
tratado com `class_weight="balanced"`, não com reamostragem, para não inventar
laudos sintéticos.
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
from sklearn.pipeline import Pipeline

from triage.config import SEED, settings
from triage.data.loader import TEXT_COLUMN, URGENCY_COLUMN, load_split
from triage.model.evaluate import evaluate_pipeline

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


def build_pipeline() -> Pipeline:
    """Monta o pipeline de vetorização e classificação.

    Returns:
        Pipeline scikit-learn não treinado, pronto para `fit`.
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
) -> dict:
    """Treina o classificador, avalia no split de teste e persiste os artefatos.

    Args:
        raw_dir: Diretório dos CSVs do corpus. Usa `settings.raw_dir` quando omitido.
        model_path: Destino do pipeline serializado. Usa `settings.sklearn_model_path`.
        metrics_path: Destino do relatório de métricas. Usa `settings.metrics_path`.

    Returns:
        Dicionário de métricas da avaliação no split de teste.
    """
    model_path = model_path or settings.sklearn_model_path
    metrics_path = metrics_path or settings.metrics_path

    train_frame = load_split("train", raw_dir=raw_dir)
    test_frame = load_split("test", raw_dir=raw_dir)
    logger.info("Treino: %d amostras | Teste: %d amostras", len(train_frame), len(test_frame))

    pipeline = build_pipeline()

    started = time.perf_counter()
    pipeline.fit(train_frame[TEXT_COLUMN], train_frame[URGENCY_COLUMN])
    fit_seconds = time.perf_counter() - started
    logger.info("Treino concluído em %.2fs", fit_seconds)

    metrics = evaluate_pipeline(pipeline, test_frame)
    metrics["fit_seconds"] = round(fit_seconds, 3)
    metrics["n_train"] = len(train_frame)
    metrics["vocabulary_size"] = len(pipeline.named_steps["tfidf"].vocabulary_)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Modelo salvo em %s (%.1f KB)", model_path, model_path.stat().st_size / 1024)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Métricas salvas em %s", metrics_path)

    return metrics


def main() -> int:
    """Ponto de entrada de linha de comando do treino.

    Returns:
        0 se o modelo atingir o limiar mínimo de f1-macro, 1 caso contrário.
    """
    parser = argparse.ArgumentParser(description="Treina o classificador de urgência.")
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Não falha quando o f1-macro fica abaixo do limiar configurado.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    metrics = train()

    logger.info(
        "accuracy=%.4f | f1_macro=%.4f | vocabulário=%d termos",
        metrics["accuracy"],
        metrics["f1_macro"],
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

    if metrics["f1_macro"] < settings.min_f1_macro and not args.skip_gate:
        logger.error(
            "f1-macro %.4f abaixo do limiar %.4f: modelo reprovado no gate de qualidade.",
            metrics["f1_macro"],
            settings.min_f1_macro,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
