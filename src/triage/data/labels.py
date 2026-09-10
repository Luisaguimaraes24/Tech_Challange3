"""Mapeamento de condição médica para nível de urgência.

O Medical Abstracts TC Corpus rotula **condições médicas** (5 classes), não urgência.
Este módulo aplica uma regra determinística e explícita que converte cada condição em
um dos três níveis de triagem exigidos pelo projeto: `normal`, `atencao` e `urgente`.

A regra é um **proxy didático**, não uma classificação clínica validada. O critério é o
potencial de deterioração rápida do quadro:

* **urgente** — condições com risco de evento agudo e desfecho grave em curto prazo
  (neoplasias, doenças cardiovasculares);
* **atencao** — condições que costumam evoluir de forma subaguda e exigem avaliação
  dirigida, mas raramente em minutos (doenças do sistema nervoso e do digestivo);
* **normal** — achados inespecíficos, que seguem o fluxo eletivo
  (condições patológicas gerais).

As limitações desta regra estão documentadas no Model Card do projeto.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "CONDITION_NAMES",
    "CONDITION_TO_URGENCY",
    "URGENCY_DESCRIPTIONS",
    "UrgencyLevel",
    "condition_name",
    "map_condition",
]


class UrgencyLevel(StrEnum):
    """Níveis de urgência da triagem.

    Os valores são slugs ASCII para que sirvam sem escape em respostas JSON,
    rótulos do Prometheus e nomes de classe do modelo.
    """

    NORMAL = "normal"
    ATENCAO = "atencao"
    URGENTE = "urgente"


CONDITION_NAMES: dict[int, str] = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}
"""Códigos de condição do corpus e seus nomes originais."""


CONDITION_TO_URGENCY: dict[int, UrgencyLevel] = {
    1: UrgencyLevel.URGENTE,  # neoplasms
    2: UrgencyLevel.ATENCAO,  # digestive system diseases
    3: UrgencyLevel.ATENCAO,  # nervous system diseases
    4: UrgencyLevel.URGENTE,  # cardiovascular diseases
    5: UrgencyLevel.NORMAL,  # general pathological conditions
}
"""Regra de negócio que converte condição médica em nível de urgência."""


URGENCY_DESCRIPTIONS: dict[UrgencyLevel, str] = {
    UrgencyLevel.NORMAL: "Segue fluxo eletivo; sem sinais de deterioração aguda.",
    UrgencyLevel.ATENCAO: "Requer avaliação dirigida em prazo curto, sem caráter emergencial.",
    UrgencyLevel.URGENTE: "Priorizar avaliação imediata; risco de desfecho grave.",
}
"""Texto de apoio exibido junto da classificação na resposta da API."""


def map_condition(condition_label: int) -> UrgencyLevel:
    """Converte um código de condição do corpus em nível de urgência.

    Args:
        condition_label: Código da condição médica, de 1 a 5.

    Returns:
        O nível de urgência correspondente.

    Raises:
        ValueError: Se o código não existir no corpus.

    Examples:
        >>> map_condition(1)
        <UrgencyLevel.URGENTE: 'urgente'>
    """
    try:
        return CONDITION_TO_URGENCY[condition_label]
    except KeyError:
        valid = ", ".join(str(key) for key in sorted(CONDITION_TO_URGENCY))
        raise ValueError(
            f"Condição desconhecida: {condition_label!r}. Códigos válidos: {valid}."
        ) from None


def condition_name(condition_label: int) -> str:
    """Devolve o nome original da condição médica.

    Args:
        condition_label: Código da condição médica, de 1 a 5.

    Returns:
        O nome da condição como consta no corpus.

    Raises:
        ValueError: Se o código não existir no corpus.
    """
    try:
        return CONDITION_NAMES[condition_label]
    except KeyError:
        valid = ", ".join(str(key) for key in sorted(CONDITION_NAMES))
        raise ValueError(
            f"Condição desconhecida: {condition_label!r}. Códigos válidos: {valid}."
        ) from None
