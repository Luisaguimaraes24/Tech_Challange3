"""Testes de ingestão e rotulagem do corpus."""

import pandas as pd
import pytest

from triage.data.loader import (
    CONDITION_COLUMN,
    TEXT_COLUMN,
    URGENCY_COLUMN,
    add_urgency_column,
    download_dataset,
    load_split,
)


def test_load_split_rotula_urgencia(fixtures_dir):
    frame = load_split("train", raw_dir=fixtures_dir)

    assert URGENCY_COLUMN in frame.columns
    assert len(frame) == 20
    assert set(frame[URGENCY_COLUMN]) == {"normal", "atencao", "urgente"}


def test_load_split_rejeita_split_invalido(fixtures_dir):
    with pytest.raises(ValueError, match="Split desconhecido"):
        load_split("validation", raw_dir=fixtures_dir)


def test_load_split_erro_claro_quando_csv_ausente(tmp_path):
    with pytest.raises(FileNotFoundError, match="make data"):
        load_split("train", raw_dir=tmp_path)


def test_add_urgency_column_exige_colunas_do_corpus():
    frame = pd.DataFrame({"texto": ["a"], "rotulo": [1]})

    with pytest.raises(KeyError, match="Colunas ausentes"):
        add_urgency_column(frame)


def test_add_urgency_column_rejeita_condicao_fora_do_dominio():
    frame = pd.DataFrame({CONDITION_COLUMN: [1, 42], TEXT_COLUMN: ["a", "b"]})

    with pytest.raises(ValueError, match="fora do domínio"):
        add_urgency_column(frame)


def test_add_urgency_column_descarta_texto_vazio():
    frame = pd.DataFrame(
        {
            CONDITION_COLUMN: [1, 2, 3],
            TEXT_COLUMN: ["laudo válido", "   ", None],
        }
    )

    result = add_urgency_column(frame)

    assert len(result) == 1
    assert result.loc[0, TEXT_COLUMN] == "laudo válido"


def test_add_urgency_column_nao_muta_o_original():
    frame = pd.DataFrame({CONDITION_COLUMN: [1], TEXT_COLUMN: ["laudo"]})

    add_urgency_column(frame)

    assert URGENCY_COLUMN not in frame.columns


def test_download_dataset_usa_arquivos_locais(fixtures_dir, monkeypatch):
    # Se os CSVs já estão no disco, não pode tentar acessar a rede.
    def explode(*_args, **_kwargs):
        raise AssertionError("download não deveria ter sido chamado")

    monkeypatch.setattr("kagglehub.dataset_download", explode)

    assert download_dataset(raw_dir=fixtures_dir) == fixtures_dir
