"""Ingestão do Medical Abstracts TC Corpus.

Estratégia de obtenção dos dados, em ordem:

1. Se os CSVs já estão em `data/raw/`, usa o que está no disco (caminho usado pelo
   CI, pelo Docker e por quem baixou o arquivo manualmente do Kaggle).
2. Caso contrário, baixa via `kagglehub` e copia os CSVs para `data/raw/`.

Isso mantém o download fora do caminho crítico de testes e permite rodar o pipeline
em ambientes sem acesso ao Kaggle.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from triage.config import settings
from triage.data.labels import CONDITION_TO_URGENCY

logger = logging.getLogger(__name__)

CONDITION_COLUMN = "condition_label"
TEXT_COLUMN = "medical_abstract"
URGENCY_COLUMN = "urgency"

SPLIT_FILES: dict[str, str] = {
    "train": "medical_tc_train.csv",
    "test": "medical_tc_test.csv",
}
"""Arquivo de cada split, conforme distribuído no corpus."""


def download_dataset(raw_dir: Path | None = None, *, force: bool = False) -> Path:
    """Garante que os CSVs do corpus estejam em `data/raw/`.

    Args:
        raw_dir: Destino dos arquivos. Usa `settings.raw_dir` quando omitido.
        force: Refaz o download mesmo que os arquivos já existam localmente.

    Returns:
        O diretório onde os CSVs ficaram disponíveis.

    Raises:
        RuntimeError: Se os arquivos não existirem localmente e o `kagglehub` não
            estiver instalado ou o download não trouxer os CSVs esperados.
    """
    raw_dir = raw_dir or settings.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not force and _has_all_splits(raw_dir):
        logger.info("Dataset já presente em %s; download ignorado.", raw_dir)
        return raw_dir

    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - depende do extra `data`
        raise RuntimeError(
            "kagglehub não está instalado e os CSVs não foram encontrados em "
            f"{raw_dir}. Instale o extra `data` (uv sync --all-extras) ou baixe o "
            "dataset manualmente do Kaggle para esse diretório."
        ) from exc

    logger.info("Baixando %s via kagglehub...", settings.kaggle_dataset)
    source = Path(kagglehub.dataset_download(settings.kaggle_dataset))

    for filename in SPLIT_FILES.values():
        origin = source / filename
        if not origin.exists():
            raise RuntimeError(f"Arquivo esperado não veio no download: {filename}")
        shutil.copy2(origin, raw_dir / filename)
        logger.info("Copiado %s para %s", filename, raw_dir)

    return raw_dir


def load_split(split: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Carrega um split do corpus já rotulado por urgência.

    Args:
        split: `"train"` ou `"test"`.
        raw_dir: Diretório dos CSVs. Usa `settings.raw_dir` quando omitido.

    Returns:
        DataFrame com as colunas `condition_label`, `medical_abstract` e `urgency`.

    Raises:
        ValueError: Se o split não existir.
        FileNotFoundError: Se o CSV correspondente não estiver no diretório.
    """
    if split not in SPLIT_FILES:
        valid = ", ".join(sorted(SPLIT_FILES))
        raise ValueError(f"Split desconhecido: {split!r}. Válidos: {valid}.")

    raw_dir = raw_dir or settings.raw_dir
    path = raw_dir / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Rode `make data` ou coloque o CSV nesse diretório."
        )

    frame = pd.read_csv(path)
    return add_urgency_column(frame)


def add_urgency_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a coluna de urgência derivada da condição médica.

    Args:
        frame: DataFrame contendo a coluna `condition_label`.

    Returns:
        Uma cópia do DataFrame com a coluna `urgency`, sem linhas de texto vazio.

    Raises:
        KeyError: Se a coluna de condição não estiver presente.
        ValueError: Se houver código de condição fora do domínio conhecido.
    """
    missing = {CONDITION_COLUMN, TEXT_COLUMN} - set(frame.columns)
    if missing:
        raise KeyError(f"Colunas ausentes no dataset: {sorted(missing)}")

    unknown = set(frame[CONDITION_COLUMN].unique()) - set(CONDITION_TO_URGENCY)
    if unknown:
        raise ValueError(f"Códigos de condição fora do domínio conhecido: {sorted(unknown)}")

    result = frame.copy()
    result[URGENCY_COLUMN] = result[CONDITION_COLUMN].map(
        {code: level.value for code, level in CONDITION_TO_URGENCY.items()}
    )

    before = len(result)
    result = result[result[TEXT_COLUMN].notna() & (result[TEXT_COLUMN].str.strip() != "")]
    if (dropped := before - len(result)) > 0:
        logger.warning("Removidas %d linhas com texto vazio.", dropped)

    return result.reset_index(drop=True)


def _has_all_splits(raw_dir: Path) -> bool:
    """Indica se todos os CSVs do corpus já estão no diretório."""
    return all((raw_dir / filename).exists() for filename in SPLIT_FILES.values())
