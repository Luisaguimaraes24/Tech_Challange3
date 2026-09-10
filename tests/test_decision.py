"""Testes da regra de decisão da triagem."""

import numpy as np
import pytest

from triage.model.decision import (
    URGENCY_ORDER,
    DecisionRule,
    aggregate_urgency,
    calibrate_threshold,
)

# Condições: 1=neoplasia(urgente) 2=digestivo(atencao) 3=nervoso(atencao)
# 4=cardiovascular(urgente) 5=geral(normal)
CLASSES = [1, 2, 3, 4, 5]


def test_ordem_dos_niveis_e_estavel():
    assert URGENCY_ORDER == ["atencao", "normal", "urgente"]


def test_aggregate_soma_condicoes_do_mesmo_nivel():
    # 30% neoplasia + 25% cardiovascular = 55% urgente
    proba = np.array([[0.30, 0.10, 0.05, 0.25, 0.30]])

    urgencia = aggregate_urgency(proba, CLASSES)

    assert urgencia[0][URGENCY_ORDER.index("urgente")] == pytest.approx(0.55)
    assert urgencia[0][URGENCY_ORDER.index("atencao")] == pytest.approx(0.15)
    assert urgencia[0][URGENCY_ORDER.index("normal")] == pytest.approx(0.30)


def test_aggregate_preserva_a_massa_total():
    proba = np.array([[0.2, 0.2, 0.2, 0.2, 0.2], [0.5, 0.1, 0.1, 0.2, 0.1]])

    assert aggregate_urgency(proba, CLASSES).sum(axis=1) == pytest.approx([1.0, 1.0])


def test_sem_limiar_decide_pela_condicao_dominante():
    # Condição 5 (normal) é a mais provável, ainda que urgente some 0.45.
    proba = np.array([[0.25, 0.05, 0.05, 0.20, 0.45]])

    assert DecisionRule(CLASSES).decide(proba) == ["normal"]


def test_trava_de_urgencia_sobrepoe_a_condicao_dominante():
    # Mesma entrada: urgente soma 0.45 e passa do limiar 0.40.
    proba = np.array([[0.25, 0.05, 0.05, 0.20, 0.45]])

    rule = DecisionRule(CLASSES, urgent_threshold=0.40)

    assert rule.decide(proba) == ["urgente"]
    assert rule.explain(proba) == ["trava_de_urgencia"]


def test_trava_nao_dispara_abaixo_do_limiar():
    proba = np.array([[0.25, 0.05, 0.05, 0.20, 0.45]])

    rule = DecisionRule(CLASSES, urgent_threshold=0.50)

    assert rule.decide(proba) == ["normal"]
    assert rule.explain(proba) == ["condicao_dominante"]


def test_explain_nao_credita_a_trava_o_que_ja_era_urgente():
    # Condição 1 (urgente) já é dominante; a trava não mudou nada.
    proba = np.array([[0.60, 0.10, 0.10, 0.10, 0.10]])

    rule = DecisionRule(CLASSES, urgent_threshold=0.30)

    assert rule.decide(proba) == ["urgente"]
    assert rule.explain(proba) == ["condicao_dominante"]


def test_decide_processa_lote():
    proba = np.array([[0.9, 0.0, 0.0, 0.05, 0.05], [0.0, 0.0, 0.0, 0.0, 1.0]])

    assert list(DecisionRule(CLASSES).decide(proba)) == ["urgente", "normal"]


def test_serializacao_preserva_a_regra():
    rule = DecisionRule(CLASSES, urgent_threshold=0.3, recall_target=0.9)

    restaurada = DecisionRule.from_dict(rule.to_dict())

    assert restaurada.condition_classes == CLASSES
    assert restaurada.urgent_threshold == 0.3
    assert restaurada.recall_target == 0.9


def test_serializacao_aceita_regra_sem_trava():
    restaurada = DecisionRule.from_dict(DecisionRule(CLASSES).to_dict())

    assert restaurada.urgent_threshold is None


class TestCalibracao:
    """Calibração do limiar a partir de probabilidades out-of-fold."""

    @staticmethod
    def _amostras():
        # 4 laudos urgentes com massa de urgência decrescente e 2 não urgentes.
        proba = np.array(
            [
                [0.80, 0.05, 0.05, 0.05, 0.05],  # urgente 0.85
                [0.30, 0.10, 0.10, 0.30, 0.20],  # urgente 0.60
                [0.20, 0.15, 0.15, 0.15, 0.35],  # urgente 0.35
                [0.10, 0.20, 0.20, 0.05, 0.45],  # urgente 0.15
                [0.02, 0.04, 0.04, 0.02, 0.88],  # urgente 0.04
                [0.03, 0.05, 0.05, 0.02, 0.85],  # urgente 0.05
            ]
        )
        y = np.array(["urgente"] * 4 + ["normal"] * 2)
        return proba, y

    def test_escolhe_limiar_que_atinge_o_alvo(self):
        proba, y = self._amostras()

        limiar, relatorio = calibrate_threshold(proba, CLASSES, y, recall_target=0.75)

        assert limiar is not None
        pred = DecisionRule(CLASSES, urgent_threshold=limiar).decide(proba)
        assert (pred[y == "urgente"] == "urgente").mean() >= 0.75
        assert relatorio["recall_target"] == 0.75

    def test_prefere_o_limiar_mais_alto_que_serve(self):
        # Entre dois limiares que atingem o alvo, o mais alto gera menos falsos alarmes.
        proba, y = self._amostras()

        folgado, _ = calibrate_threshold(proba, CLASSES, y, recall_target=0.25)
        exigente, _ = calibrate_threshold(proba, CLASSES, y, recall_target=0.75)

        assert folgado > exigente

    def test_devolve_none_quando_o_alvo_e_inalcancavel(self, caplog):
        proba, y = self._amostras()

        limiar, _ = calibrate_threshold(
            proba, CLASSES, y, recall_target=1.0, grid=np.array([0.9, 0.8])
        )

        assert limiar is None
        assert "Nenhum limiar atingiu" in caplog.text

    def test_relatorio_cobre_toda_a_grade(self):
        proba, y = self._amostras()
        grade = np.array([0.8, 0.5, 0.2])

        _, relatorio = calibrate_threshold(proba, CLASSES, y, recall_target=0.5, grid=grade)

        assert [linha["threshold"] for linha in relatorio["curve"]] == [0.8, 0.5, 0.2]
        # Baixar o limiar nunca pode reduzir o recall.
        recalls = [linha["recall_urgente"] for linha in relatorio["curve"]]
        assert recalls == sorted(recalls)
