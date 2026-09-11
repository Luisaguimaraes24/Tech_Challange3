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
        urgent_recall_target: Recall de `urgente` que a calibração da trava persegue.
        min_f1_macro: Piso de f1-macro no gate de qualidade da DAG de treino.
        min_recall_urgente: Piso de recall na classe crítica, o gate que de fato decide
            se um modelo pode ir para produção.
        model_backend: Motor de inferência da API (`sklearn`, `onnx` ou `onnx-int8`).
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

    # Política de triagem: capturar 90% dos laudos urgentes, aceitando o custo em
    # falsos alarmes. É uma decisão institucional, não um hiperparâmetro — por isso
    # vive na configuração do serviço e não dentro do código de treino.
    urgent_recall_target: float = 0.90

    min_f1_macro: float = 0.52
    min_recall_urgente: float = 0.85

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
        """Caminho do classificador exportado para ONNX."""
        return self.models_dir / "model.onnx"

    @property
    def onnx_quantized_path(self) -> Path:
        """Caminho da versão INT8 do classificador."""
        return self.models_dir / "model.int8.onnx"

    @property
    def decision_path(self) -> Path:
        """Caminho da regra de decisão que acompanha o modelo.

        A regra é parte do contrato do modelo, não configuração: um limiar calibrado
        para um treino não vale para outro, então ele viaja junto do artefato.
        """
        return self.models_dir / "decision.json"

    @property
    def metrics_path(self) -> Path:
        """Caminho do relatório de métricas da última avaliação."""
        return self.models_dir / "metrics.json"


settings = Settings()
"""Instância única de configuração, compartilhada por todo o projeto."""
