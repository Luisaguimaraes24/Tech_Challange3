"""Avaliação do classificador de urgência.

O relatório é propositalmente pequeno e serializável em JSON: ele alimenta tanto os
gates de qualidade da DAG de retreino quanto a documentação do Model Card.

**Recall de `urgente` é a métrica de decisão**, promovida a campo de primeiro nível.
Acurácia e f1-macro tratam todos os erros como equivalentes, o que é falso numa triagem:
classificar um laudo urgente como normal deixa um paciente na fila, enquanto o erro
oposto apenas consome tempo de um revisor. As duas continuam no relatório, mas como
contexto — quem decide se um modelo pode ir para produção é o recall na classe crítica.
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

from triage.data.labels import URGENCY_COLUMN_LABELS, UrgencyLevel
from triage.data.loader import TEXT_COLUMN, URGENCY_COLUMN
from triage.model.decision import DecisionRule

logger = logging.getLogger(__name__)


def evaluate_decision(pipeline: Pipeline, rule: DecisionRule, frame: pd.DataFrame) -> dict:
    """Avalia o pipeline junto da regra de decisão que ele usa em produção.

    A avaliação inclui a regra de propósito. Medir o classificador de condições
    isoladamente responderia a uma pergunta que ninguém faz: o serviço nunca devolve
    condição, devolve urgência.

    Args:
        pipeline: Pipeline treinado sobre os códigos de condição.
        rule: Regra que projeta condições em níveis de urgência.
        frame: DataFrame com as colunas de texto e de urgência verdadeira.

    Returns:
        Dicionário serializável em JSON com recall da classe crítica, acurácia,
        f1-macro, métricas por classe, matriz de confusão e uso da trava de urgência.
    """
    y_true = frame[URGENCY_COLUMN].to_numpy()
    probabilities = pipeline.predict_proba(frame[TEXT_COLUMN])
    y_pred = rule.decide(probabilities)
    reasons = rule.explain(probabilities)

    labels = URGENCY_COLUMN_LABELS
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

    urgent = UrgencyLevel.URGENTE.value
    travas = sum(1 for reason in reasons if reason == "trava_de_urgencia")

    return {
        "recall_urgente": per_class[urgent]["recall"],
        "precision_urgente": per_class[urgent]["precision"],
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "per_class": per_class,
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_test": int(len(frame)),
        "decisoes_por_trava_de_urgencia": travas,
        "fracao_por_trava_de_urgencia": round(travas / max(len(frame), 1), 4),
    }
