"""Camada de inferência da API.

O serviço não conversa com o scikit-learn diretamente: ele fala com um
`InferenceBackend`. Isso isola o motor de inferência do resto da aplicação e permite
trocar `sklearn` por `onnx` na Etapa 4 mudando apenas uma variável de ambiente —
a API, os schemas e as métricas continuam idênticos, o que é justamente o que torna a
comparação de latência honesta.

O backend devolve **probabilidades por condição médica**; a projeção para urgência e a
trava de segurança vivem na `DecisionRule`, fora do modelo. Essa fronteira é deliberada:
os dois backends compartilham exatamente a mesma regra de decisão, então qualquer
diferença medida entre eles é diferença de motor de inferência, não de lógica.

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
import numpy as np

from triage.config import settings
from triage.data.labels import URGENCY_DESCRIPTIONS, UrgencyLevel
from triage.model.decision import URGENCY_ORDER, DecisionRule, aggregate_urgency

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Levantada quando se pede uma predição sem modelo carregado."""


class InferenceBackend(ABC):
    """Contrato mínimo que um motor de inferência precisa cumprir."""

    name: str

    @abstractmethod
    def predict_proba(self, text: str) -> np.ndarray:
        """Devolve as probabilidades por condição médica.

        Args:
            text: Texto do laudo.

        Returns:
            Matriz `(1, n_condicoes)` de probabilidades.
        """

    @property
    @abstractmethod
    def condition_classes(self) -> list[int]:
        """Códigos de condição, na ordem das colunas de probabilidade."""


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
    def condition_classes(self) -> list[int]:
        """Códigos de condição conhecidos pelo pipeline."""
        return [int(code) for code in self._pipeline.classes_]

    def predict_proba(self, text: str) -> np.ndarray:
        """Classifica um laudo com o pipeline scikit-learn.

        Args:
            text: Texto do laudo.

        Returns:
            Matriz `(1, n_condicoes)` de probabilidades.
        """
        return self._pipeline.predict_proba([text])


class Predictor:
    """Fachada usada pela API para transformar texto em resultado de triagem."""

    def __init__(
        self,
        backend: InferenceBackend,
        rule: DecisionRule | None = None,
        metrics: dict | None = None,
    ) -> None:
        """Guarda o backend ativo, a regra de decisão e as métricas do artefato.

        Args:
            backend: Motor de inferência já inicializado.
            rule: Regra de decisão. Quando omitida, usa a projeção sem trava de urgência.
            metrics: Métricas da última avaliação, quando disponíveis.
        """
        self.backend = backend
        self.rule = rule or DecisionRule(condition_classes=backend.condition_classes)
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

        metrics = _load_metrics(settings.metrics_path)
        _warn_on_environment_drift(metrics)

        return cls(
            backend=backend,
            rule=_load_decision_rule(settings.decision_path, backend.condition_classes),
            metrics=metrics,
        )

    @property
    def classes(self) -> list[str]:
        """Níveis de urgência que o serviço pode devolver."""
        return list(URGENCY_ORDER)

    def predict(self, text: str) -> dict:
        """Classifica um laudo e mede o tempo gasto pelo modelo.

        Args:
            text: Texto do laudo.

        Returns:
            Dicionário no formato de `PredictionResponse`.
        """
        started = time.perf_counter()
        condition_probabilities = self.backend.predict_proba(text)
        urgency = aggregate_urgency(condition_probabilities, self.rule.condition_classes)
        decisions, reasons = self.rule.resolve(condition_probabilities, urgency=urgency)
        elapsed_ms = (time.perf_counter() - started) * 1_000

        level = UrgencyLevel(decisions[0])
        reason = reasons[0]
        probabilities = {
            label: round(float(value), 4)
            for label, value in zip(URGENCY_ORDER, urgency[0], strict=True)
        }

        return {
            "urgencia": level,
            "descricao": URGENCY_DESCRIPTIONS[level],
            "confianca": probabilities[level.value],
            "probabilidades": probabilities,
            "regra": reason,
            "latencia_ms": round(elapsed_ms, 3),
            "backend": self.backend.name,
        }


def _load_decision_rule(decision_path: Path, condition_classes: list[int]) -> DecisionRule:
    """Carrega a regra de decisão que acompanha o artefato de modelo.

    Args:
        decision_path: Caminho do `decision.json`.
        condition_classes: Classes do backend, usadas como fallback.

    Returns:
        A regra persistida ou, na ausência dela, a projeção sem trava de urgência.
    """
    if not decision_path.exists():
        logger.warning(
            "%s ausente; servindo sem trava de urgência. Rode `make train` para calibrá-la.",
            decision_path,
        )
        return DecisionRule(condition_classes=condition_classes)

    try:
        rule = DecisionRule.from_dict(json.loads(decision_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError):
        logger.warning("%s ilegível; servindo sem trava de urgência.", decision_path)
        return DecisionRule(condition_classes=condition_classes)

    if rule.condition_classes != condition_classes:
        raise ValueError(
            "A regra de decisão não corresponde ao modelo carregado: "
            f"esperava {condition_classes}, encontrou {rule.condition_classes}. "
            "O artefato e a regra precisam vir do mesmo treino."
        )

    logger.info(
        "Regra de decisão carregada | trava de urgência=%s | alvo de recall=%s",
        rule.urgent_threshold,
        rule.recall_target,
    )
    return rule


def _warn_on_environment_drift(metrics: dict | None) -> None:
    """Alerta quando o modelo foi treinado com versões diferentes das que vão servi-lo.

    O pickle do scikit-learn não é estável entre versões: a própria biblioteca trata a
    desserialização cruzada como uso por conta e risco. A divergência costuma nascer de
    duas imagens que resolvem dependências separadamente, e o sintoma pode ser um
    resultado sutilmente errado em vez de uma exceção — daí valer um aviso explícito.

    Args:
        metrics: Métricas do artefato, contendo o ambiente de treino.
    """
    if not metrics or not (treino := metrics.get("environment")):
        return

    import sklearn

    if (versao_treino := treino.get("scikit_learn")) and versao_treino != sklearn.__version__:
        logger.warning(
            "Modelo treinado com scikit-learn %s, mas o serviço roda %s. "
            "A desserialização entre versões não é garantida; realinhe as imagens.",
            versao_treino,
            sklearn.__version__,
        )


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
