"""Regra de decisão da triagem.

O modelo é treinado nas **5 condições médicas** do corpus, não nos 3 níveis de urgência.
Isso não é um detalhe: colapsar as classes antes do treino apaga a fronteira entre
"neoplasia" e "doença cardiovascular", que têm vocabulários distintos, e o classificador
perde sinal que poderia usar. Medido por validação cruzada no split de treino, treinar
nas 5 condições e projetar depois rende f1-macro 0,5900 contra 0,5732 do treino direto
em 3 classes.

Sobre a projeção existe uma segunda decisão, esta clínica e não estatística. O erro de
uma triagem não é simétrico: deixar um laudo urgente na fila é incomparavelmente pior do
que examinar um laudo que não precisava de pressa. Por isso a regra tem duas partes:

1. **condição dominante** — a urgência é a da condição mais provável;
2. **trava de urgência** — se a massa de probabilidade acumulada nas condições urgentes
   ultrapassa um limiar, o laudo é marcado como urgente, mesmo que nenhuma condição
   urgente isolada seja a mais provável.

O limiar não é escolhido na mão: é calibrado no treino, por validação cruzada, como o
maior valor que ainda atinge o recall-alvo para `urgente`. Isso transforma um número
mágico em uma política explícita — "queremos capturar 90% dos laudos urgentes" — e o
custo dessa política em falsos alarmes fica registrado nas métricas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from triage.data.labels import CONDITION_TO_URGENCY, URGENCY_COLUMN_LABELS, UrgencyLevel

logger = logging.getLogger(__name__)

URGENCY_ORDER: list[str] = URGENCY_COLUMN_LABELS
"""Ordem canônica dos níveis nas saídas vetoriais.

Compartilhada com a avaliação e com a API: se treino, matriz de confusão e resposta
usassem ordens diferentes, um índice trocado passaria despercebido nos testes e
produziria a classificação errada em produção.
"""


def aggregate_urgency(
    condition_probabilities: np.ndarray, condition_classes: list[int]
) -> np.ndarray:
    """Projeta probabilidades de condição em probabilidades de urgência.

    A projeção é uma soma: a probabilidade de um laudo ser urgente é a massa total
    das condições mapeadas para `urgente`.

    Args:
        condition_probabilities: Matriz `(n_amostras, n_condicoes)`.
        condition_classes: Códigos de condição, na ordem das colunas da matriz.

    Returns:
        Matriz `(n_amostras, 3)` com as colunas na ordem de `URGENCY_ORDER`.
    """
    aggregated = np.zeros((condition_probabilities.shape[0], len(URGENCY_ORDER)))
    for column, condition in enumerate(condition_classes):
        level = CONDITION_TO_URGENCY[condition].value
        aggregated[:, URGENCY_ORDER.index(level)] += condition_probabilities[:, column]
    return aggregated


@dataclass
class DecisionRule:
    """Converte probabilidades de condição em um nível de urgência.

    Attributes:
        condition_classes: Códigos de condição, na ordem das colunas de probabilidade.
        urgent_threshold: Limiar da trava de urgência. `None` desliga a trava.
        recall_target: Recall de `urgente` que motivou o limiar, guardado para auditoria.
    """

    condition_classes: list[int]
    urgent_threshold: float | None = None
    recall_target: float | None = None
    _urgent_index: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Fixa o índice da coluna de urgência usada pela trava."""
        self._urgent_index = URGENCY_ORDER.index(UrgencyLevel.URGENTE.value)

    def resolve(
        self, condition_probabilities: np.ndarray, urgency: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decide e justifica em uma única passagem.

        Decisão e justificativa saem juntas porque derivam dos mesmos dois cálculos —
        o argmax das condições e a massa acumulada de urgência. Obtê-las em chamadas
        separadas repetiria esse trabalho a cada requisição.

        Args:
            condition_probabilities: Matriz `(n_amostras, n_condicoes)`.
            urgency: Agregação já calculada, para reaproveitar quando o chamador
                também precisa dela. Calculada aqui quando omitida.

        Returns:
            Par `(níveis, justificativas)`, ambos com uma entrada por amostra.
        """
        dominant = condition_probabilities.argmax(axis=1)
        # dtype=object é obrigatório: com dtype inferido, um lote em que toda decisão
        # inicial é "normal" produz '<U6' e a atribuição de "urgente" pela trava seria
        # truncada em silêncio para "urgent" — um nível que não existe.
        decisions = np.array(
            [CONDITION_TO_URGENCY[self.condition_classes[index]].value for index in dominant],
            dtype=object,
        )
        reasons = np.full(len(decisions), "condicao_dominante", dtype=object)

        if self.urgent_threshold is not None:
            if urgency is None:
                urgency = aggregate_urgency(condition_probabilities, self.condition_classes)
            travado = urgency[:, self._urgent_index] >= self.urgent_threshold
            # A justificativa é marcada antes da sobrescrita: depois dela não daria mais
            # para distinguir o que já era urgente do que a trava tornou urgente.
            reasons[travado & (decisions != UrgencyLevel.URGENTE.value)] = "trava_de_urgencia"
            decisions[travado] = UrgencyLevel.URGENTE.value

        return decisions, reasons

    def decide(self, condition_probabilities: np.ndarray) -> np.ndarray:
        """Aplica a regra de decisão a um lote de probabilidades.

        Args:
            condition_probabilities: Matriz `(n_amostras, n_condicoes)`.

        Returns:
            Vetor de níveis de urgência, um por amostra.
        """
        return self.resolve(condition_probabilities)[0]

    def explain(self, condition_probabilities: np.ndarray) -> list[str]:
        """Indica qual das duas partes da regra determinou cada decisão.

        A triagem precisa ser auditável: o hospital tem que conseguir distinguir um
        laudo marcado como urgente porque o diagnóstico mais provável é grave de um
        marcado pela trava de segurança, que é um sinal de incerteza distribuída.

        Args:
            condition_probabilities: Matriz `(n_amostras, n_condicoes)`.

        Returns:
            `"condicao_dominante"` ou `"trava_de_urgencia"` para cada amostra.
        """
        return self.resolve(condition_probabilities)[1].tolist()

    def to_dict(self) -> dict:
        """Serializa a regra para acompanhar o artefato de modelo.

        Returns:
            Dicionário serializável em JSON.
        """
        return {
            "condition_classes": [int(code) for code in self.condition_classes],
            "urgent_threshold": self.urgent_threshold,
            "recall_target": self.recall_target,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> DecisionRule:
        """Reconstrói a regra a partir da forma serializada.

        Args:
            payload: Dicionário produzido por `to_dict`.

        Returns:
            A regra de decisão correspondente.
        """
        return cls(
            condition_classes=[int(code) for code in payload["condition_classes"]],
            urgent_threshold=payload.get("urgent_threshold"),
            recall_target=payload.get("recall_target"),
        )


def calibrate_threshold(
    condition_probabilities: np.ndarray,
    condition_classes: list[int],
    y_true_urgency: np.ndarray,
    recall_target: float,
    grid: np.ndarray | None = None,
) -> tuple[float | None, dict]:
    """Escolhe o limiar da trava de urgência para atingir o recall-alvo.

    Percorre limiares do mais alto para o mais baixo — ou seja, do mais conservador em
    falsos alarmes para o mais sensível — e para no primeiro que alcança o alvo. Isso
    entrega o recall pedido pagando o mínimo de precisão possível.

    Args:
        condition_probabilities: Probabilidades **out-of-fold**, para não calibrar em
            predições que o modelo já viu no treino.
        condition_classes: Códigos de condição, na ordem das colunas.
        y_true_urgency: Níveis de urgência verdadeiros.
        recall_target: Recall mínimo desejado para `urgente`, entre 0 e 1.
        grid: Limiares candidatos. Usa `0.95..0.01` em passos de 0.01 quando omitido.

    Returns:
        O limiar escolhido (ou `None` se nenhum atingir o alvo) e a curva completa
        de recall e precisão por limiar, para documentação.
    """
    if grid is None:
        # Passo de 0.01, não 0.05. Com a grade grossa a busca pulava o alvo: o limiar
        # imediatamente acima entregava recall 0.895 e o seguinte, 0.931 — três pontos
        # de recall além do pedido, pagos com quatro pontos de precisão. Sobrepassar o
        # alvo não é conservador, é inflar a fila prioritária sem ninguém ter decidido.
        grid = np.arange(0.95, 0.009, -0.01)

    urgent = UrgencyLevel.URGENTE.value
    positives = y_true_urgency == urgent
    curve: list[dict] = []
    chosen: float | None = None

    for threshold in grid:
        rule = DecisionRule(condition_classes, urgent_threshold=float(threshold))
        predicted = rule.decide(condition_probabilities) == urgent

        true_positives = int((predicted & positives).sum())
        recall = true_positives / max(int(positives.sum()), 1)
        precision = true_positives / max(int(predicted.sum()), 1)

        curve.append(
            {
                "threshold": round(float(threshold), 2),
                "recall_urgente": round(recall, 4),
                "precision_urgente": round(precision, 4),
            }
        )

        if chosen is None and recall >= recall_target:
            chosen = round(float(threshold), 2)

    if chosen is None:
        logger.warning(
            "Nenhum limiar atingiu o recall-alvo de %.2f para `urgente`; "
            "a trava de urgência ficará desligada.",
            recall_target,
        )

    return chosen, {"recall_target": recall_target, "curve": curve}
