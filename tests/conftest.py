"""Fixtures compartilhadas pela suíte de testes."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Diretório com a amostra versionada do corpus.

    Permite que os testes rodem no CI sem depender de download do Kaggle.
    """
    return FIXTURES_DIR


@pytest.fixture(scope="module")
def fixtures_dir_module() -> Path:
    """Mesma amostra do corpus, com escopo de módulo.

    Necessária para fixtures caras (como um pipeline treinado) que são reaproveitadas
    por vários testes do mesmo arquivo.
    """
    return FIXTURES_DIR
