"""Camada de inferência da API.

O serviço não conversa com o scikit-learn diretamente: ele fala com um
`InferenceBackend`. Isso isola o motor de inferência do resto da aplicação e permite
trocar `sklearn` por `onnx` na Etapa 4 mudando apenas uma variável de ambiente —
a API, os schemas e as métricas continuam idênticos, o que é justamente o que torna a
comparação de latência honesta.

O modelo é carregado **uma vez**, no startup da aplicação. Carregar por requisição
dominaria o tempo de resposta e tornaria qualquer medição de latência inútil.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import joblib

from triage.config import settings
from triage.data.labels import URGENCY_DESCRIPTIONS, UrgencyLevel

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Levantada quando se pede uma predição sem modelo carregado."""


class InferenceBackend(ABC):
    """Contrato mínimo que um motor de inferência precisa cumprir."""

    name: str

    @abstractmethod
    def predict_proba(self, text: str) -> dict[str, float]:
        """Devolve a distribuição de probabilidade sobre os níveis de urgência.

        Args:
            text: Texto do laudo.

        Returns:
            Mapa de nível de urgência para probabilidade.
        """

    @property
    @abstractmethod
    def classes(self) -> list[str]:
        """Níveis de urgência que o backend pode devolver."""


class SklearnBackend(InferenceBackend):
    """Inferência com o pipeline scikit-learn serializado em joblib."""

    name = "sklearn"

    def __init__(self, model_path: Path) -> None:
        """Carrega o pipeline do disco.

        Args:
            model_path: Caminho do arquivo `.pkl`.

        Raises:
            FileNotFoundError: Se o artefato não existir.
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {model_path}. Rode `make train` antes de subir a API."
            )
        self.model_path = model_path
        self._pipeline = joblib.load(model_path)
        logger.info("Pipeline scikit-learn carregado de %s", model_path)

    @property
    def classes(self) -> list[str]:
        """Níveis de urgência conhecidos pelo pipeline."""
        return [str(label) for label in self._pipeline.classes_]

    def predict_proba(self, text: str) -> dict[str, float]:
        """Classifica um laudo com o pipeline scikit-learn.

        Args:
            text: Texto do laudo.

        Returns:
            Mapa de nível de urgência para probabilidade.
        """
        probabilities = self._pipeline.predict_proba([text])[0]
        return dict(zip(self.classes, (float(value) for value in probabilities), strict=True))


class Predictor:
    """Fachada usada pela API para transformar texto em resultado de triagem."""

    def __init__(self, backend: InferenceBackend, metrics: dict | None = None) -> None:
        """Guarda o backend ativo e as métricas do artefato carregado.

        Args:
            backend: Motor de inferência já inicializado.
            metrics: Métricas da última avaliação, quando disponíveis.
        """
        self.backend = backend
        self.metrics = metrics

    @classmethod
    def load(cls, backend_name: str | None = None) -> Predictor:
        """Constrói o preditor a partir da configuração do ambiente.

        Args:
            backend_name: `sklearn` ou `onnx`. Usa `settings.model_backend` quando omitido.

        Returns:
            Preditor pronto para uso.

        Raises:
            ValueError: Se o backend pedido não existir.
            FileNotFoundError: Se o artefato correspondente não estiver no disco.
        """
        backend_name = (backend_name or settings.model_backend).lower()

        if backend_name == "sklearn":
            backend: InferenceBackend = SklearnBackend(settings.sklearn_model_path)
        else:
            raise ValueError(
                f"Backend de inferência desconhecido: {backend_name!r}. Disponível: 'sklearn'."
            )

        return cls(backend=backend, metrics=_load_metrics(settings.metrics_path))

    def predict(self, text: str) -> dict:
        """Classifica um laudo e mede o tempo gasto pelo modelo.

        Args:
            text: Texto do laudo.

        Returns:
            Dicionário no formato de `PredictionResponse`.
        """
        started = time.perf_counter()
        probabilities = self.backend.predict_proba(text)
        elapsed_ms = (time.perf_counter() - started) * 1_000

        best = max(probabilities, key=probabilities.__getitem__)
        level = UrgencyLevel(best)

        return {
            "urgencia": level,
            "descricao": URGENCY_DESCRIPTIONS[level],
            "confianca": round(probabilities[best], 4),
            "probabilidades": {
                label: round(value, 4) for label, value in sorted(probabilities.items())
            },
            "latencia_ms": round(elapsed_ms, 3),
            "backend": self.backend.name,
        }


def _load_metrics(metrics_path: Path) -> dict | None:
    """Lê o relatório de métricas do disco, se existir.

    Args:
        metrics_path: Caminho do `metrics.json`.

    Returns:
        As métricas carregadas, ou `None` se o arquivo não existir ou estiver corrompido.
    """
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("metrics.json ilegível em %s; seguindo sem métricas.", metrics_path)
        return None
