"""Avaliação do classificador de urgência.

O relatório é propositalmente pequeno e serializável em JSON: ele alimenta tanto o
gate de qualidade da DAG de retreino quanto a documentação do Model Card.

A métrica de decisão é o **f1-macro**, não a acurácia. Como as classes são
desbalanceadas e o custo de errar um `urgente` é assimétrico, uma acurácia alta
poderia esconder um recall ruim justamente na classe que mais importa.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from triage.data.loader import TEXT_COLUMN, URGENCY_COLUMN

logger = logging.getLogger(__name__)


def evaluate_pipeline(pipeline: Pipeline, frame: pd.DataFrame) -> dict:
    """Avalia um pipeline treinado sobre um conjunto rotulado.

    Args:
        pipeline: Pipeline scikit-learn já treinado.
        frame: DataFrame com as colunas de texto e de urgência.

    Returns:
        Dicionário serializável em JSON com acurácia, f1-macro, métricas por classe,
        matriz de confusão e o número de amostras avaliadas.
    """
    y_true = frame[URGENCY_COLUMN]
    y_pred = pipeline.predict(frame[TEXT_COLUMN])

    labels = sorted(pipeline.classes_)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {
        label: {
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "per_class": per_class,
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_test": int(len(frame)),
    }
