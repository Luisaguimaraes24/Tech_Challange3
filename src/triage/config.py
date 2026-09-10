"""Configuração central do projeto.

Todos os caminhos e parâmetros ajustáveis vivem aqui, carregados de variáveis de
ambiente (ou de um arquivo `.env`) via Pydantic Settings. Módulos de dados, treino
e API importam a instância única `settings`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED = 42
"""Semente única usada em splits e treino, para reprodutibilidade."""


class Settings(BaseSettings):
    """Parâmetros de execução do projeto.

    Attributes:
        data_dir: Raiz dos dados. Subpastas `raw/` e `processed/` derivam dela.
        models_dir: Onde os artefatos de modelo são gravados e lidos.
        kaggle_dataset: Identificador do dataset no Kaggle.
        test_size: Fração do dataset reservada para teste.
        min_f1_macro: Limiar de qualidade que a DAG de treino usa como gate.
        model_backend: Motor de inferência da API (`sklearn` ou `onnx`).
        api_host: Interface de escuta da API.
        api_port: Porta de escuta da API.
        log_level: Nível de log da aplicação.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRIAGE_",
        extra="ignore",
    )

    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    models_dir: Path = Field(default=PROJECT_ROOT / "models")

    kaggle_dataset: str = "saharalaa/medical-abstracts-tc-corpus"

    test_size: float = 0.2
    min_f1_macro: float = 0.55

    model_backend: str = "sklearn"

    api_host: str = "0.0.0.0"  # noqa: S104 — dentro do container é o comportamento desejado
    api_port: int = 8000

    log_level: str = "INFO"

    @property
    def raw_dir(self) -> Path:
        """Diretório dos dados brutos, como baixados da fonte."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Diretório dos dados já limpos e rotulados por urgência."""
        return self.data_dir / "processed"

    @property
    def sklearn_model_path(self) -> Path:
        """Caminho do pipeline scikit-learn serializado."""
        return self.models_dir / "model.pkl"

    @property
    def onnx_model_path(self) -> Path:
        """Caminho do modelo exportado para ONNX."""
        return self.models_dir / "model.onnx"

    @property
    def metrics_path(self) -> Path:
        """Caminho do relatório de métricas da última avaliação."""
        return self.models_dir / "metrics.json"


settings = Settings()
"""Instância única de configuração, compartilhada por todo o projeto."""
