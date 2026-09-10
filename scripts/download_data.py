"""Baixa o Medical Abstracts TC Corpus para `data/raw/`.

Uso:
    python scripts/download_data.py [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys

from triage.config import settings
from triage.data.loader import download_dataset, load_split


def main() -> int:
    """Executa o download e imprime um resumo do que foi obtido.

    Returns:
        Código de saída do processo: 0 em caso de sucesso, 1 em caso de falha.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refaz o download mesmo que os arquivos já existam.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("download_data")

    try:
        raw_dir = download_dataset(force=args.force)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Dados disponíveis em %s", raw_dir)
    for split in ("train", "test"):
        frame = load_split(split)
        counts = frame["urgency"].value_counts().to_dict()
        logger.info("%-5s | %5d amostras | urgências: %s", split, len(frame), counts)

    return 0


if __name__ == "__main__":
    sys.exit(main())
