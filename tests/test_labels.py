"""Testes da regra de mapeamento condição médica -> urgência."""

import pytest

from triage.data.labels import (
    CONDITION_NAMES,
    CONDITION_TO_URGENCY,
    URGENCY_DESCRIPTIONS,
    UrgencyLevel,
    condition_name,
    map_condition,
)


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (1, UrgencyLevel.URGENTE),  # neoplasms
        (2, UrgencyLevel.ATENCAO),  # digestive system diseases
        (3, UrgencyLevel.ATENCAO),  # nervous system diseases
        (4, UrgencyLevel.URGENTE),  # cardiovascular diseases
        (5, UrgencyLevel.NORMAL),  # general pathological conditions
    ],
)
def test_map_condition_cobre_todas_as_classes(condition, expected):
    assert map_condition(condition) is expected


def test_map_condition_rejeita_codigo_desconhecido():
    with pytest.raises(ValueError, match="Condição desconhecida"):
        map_condition(99)


def test_condition_name_rejeita_codigo_desconhecido():
    with pytest.raises(ValueError, match="Condição desconhecida"):
        condition_name(0)


def test_condition_name_devolve_rotulo_do_corpus():
    assert condition_name(1) == "neoplasms"


def test_dominio_do_mapeamento_bate_com_o_corpus():
    assert set(CONDITION_TO_URGENCY) == set(CONDITION_NAMES)


def test_todos_os_niveis_de_urgencia_sao_alcancaveis():
    assert set(CONDITION_TO_URGENCY.values()) == set(UrgencyLevel)


def test_todo_nivel_tem_descricao():
    assert set(URGENCY_DESCRIPTIONS) == set(UrgencyLevel)


def test_valores_de_urgencia_sao_ascii():
    # Os valores viram rótulo de métrica no Prometheus e chave de JSON na API.
    for level in UrgencyLevel:
        assert level.value.isascii()
        assert level.value.islower()
