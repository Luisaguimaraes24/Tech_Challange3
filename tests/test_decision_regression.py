"""Regressões da regra de decisão.

Cada teste aqui corresponde a um defeito que já existiu no código. Eles ficam
separados dos testes de comportamento porque documentam um erro concreto, não uma
funcionalidade.
"""

import numpy as np

from triage.data.labels import UrgencyLevel
from triage.model.decision import DecisionRule

CLASSES = [1, 2, 3, 4, 5]


def test_trava_nao_trunca_o_rotulo_em_lote_so_de_normais():
    """O nível atribuído pela trava não pode ser truncado pelo dtype do numpy.

    Num lote em que toda decisão inicial é `normal` (6 caracteres), o array inferia
    dtype '<U6' e a atribuição de `urgente` (7 caracteres) virava `urgent` em silêncio —
    um nível inexistente, que quebrava a resposta da API com erro 500.
    """
    # Condição 5 (normal) domina, mas a urgência acumulada passa do limiar.
    proba = np.array([[0.24, 0.01, 0.01, 0.21, 0.53]] * 3)

    decisoes = DecisionRule(CLASSES, urgent_threshold=0.40).decide(proba)

    assert list(decisoes) == ["urgente"] * 3
    for decisao in decisoes:
        UrgencyLevel(decisao)  # levantaria ValueError se estivesse truncado


def test_toda_decisao_e_um_nivel_valido_em_lotes_heterogeneos():
    proba = np.array(
        [
            [0.90, 0.02, 0.03, 0.02, 0.03],  # neoplasia -> urgente
            [0.02, 0.90, 0.03, 0.02, 0.03],  # digestivo -> atencao
            [0.02, 0.03, 0.02, 0.03, 0.90],  # geral -> normal
            [0.24, 0.01, 0.01, 0.21, 0.53],  # geral domina, trava dispara
        ]
    )

    decisoes = DecisionRule(CLASSES, urgent_threshold=0.40).decide(proba)

    assert [UrgencyLevel(d).value for d in decisoes] == [
        "urgente",
        "atencao",
        "normal",
        "urgente",
    ]
