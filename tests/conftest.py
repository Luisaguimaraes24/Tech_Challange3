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
