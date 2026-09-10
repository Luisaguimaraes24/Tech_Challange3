"""Instrumentação Prometheus da API de triagem.

Duas restrições moldam este módulo, e vale explicitá-las porque elas explicam quase toda
decisão aqui:

**Privacidade.** Laudo médico é dado pessoal sensível sob a LGPD. Nenhum rótulo de
métrica carrega conteúdo de laudo, identificador de paciente ou trecho de texto — só
contagens, tempos e categorias fechadas. Métrica do Prometheus é retida por muito tempo,
copiada para réplicas e exposta a quem tem acesso ao painel; é o último lugar onde se
deve colocar dado clínico.

**Cardinalidade.** Cada combinação distinta de rótulos vira uma série temporal
permanente. Por isso o rótulo de rota usa o **template** (`/predict`), nunca o caminho
concreto, e todos os rótulos têm domínio finito e pequeno. Um rótulo de domínio aberto
degrada o Prometheus de forma silenciosa e progressiva, até o dia em que ele para.

As métricas expostas respondem a quatro perguntas operacionais:

* o serviço está sendo usado, e com que resultado? (`triage_requests_total`)
* está rápido o suficiente? (`triage_request_duration_seconds`)
* o tempo é do modelo ou do serviço? (`triage_inference_duration_seconds`)
* a distribuição de urgências mudou? (`triage_predictions_total`) — que é o sinal mais
  barato de deriva: se a fração de `urgente` salta sem que nada tenha sido publicado, ou
  os dados de entrada mudaram, ou algo quebrou.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

REQUESTS = Counter(
    "triage_requests_total",
    "Requisições atendidas pela API.",
    labelnames=("method", "endpoint", "status"),
)

REQUEST_DURATION = Histogram(
    "triage_request_duration_seconds",
    "Tempo total de resposta, incluindo validação, inferência e serialização.",
    labelnames=("endpoint",),
    # Os buckets padrão do prometheus_client começam em 5 ms e são pensados para
    # serviços que respondem em dezenas de milissegundos. Aqui o p99 fica abaixo de
    # 5 ms, e todo o histograma cairia no primeiro bucket — inútil para calcular
    # percentil. Esta escala é derivada da linha de base medida na Etapa 1.
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

INFERENCE_DURATION = Histogram(
    "triage_inference_duration_seconds",
    "Tempo gasto apenas pelo modelo, sem o overhead de HTTP.",
    labelnames=("backend",),
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1),
)

PREDICTIONS = Counter(
    "triage_predictions_total",
    "Classificações emitidas, por nível de urgência e regra que decidiu.",
    labelnames=("urgencia", "regra"),
)

ERRORS = Counter(
    "triage_errors_total",
    "Falhas atendidas pela API, agrupadas por tipo.",
    labelnames=("tipo",),
)

MODEL_LOADED = Gauge(
    "triage_model_loaded",
    "1 quando há modelo carregado e o serviço pode classificar; 0 em modo degradado.",
)

URGENT_THRESHOLD = Gauge(
    "triage_urgent_threshold",
    "Limiar da trava de urgência em vigor. Muda quando um modelo novo é publicado.",
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Mede toda requisição num único ponto.

    Instrumentar rota por rota deixaria buracos exatamente onde eles doem: numa rota
    nova que alguém esqueceu de decorar, ou num erro levantado antes do corpo da rota.
    O middleware vê tudo o que entra e tudo o que sai.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Contabiliza duração e desfecho de uma requisição.

        Args:
            request: Requisição em curso.
            call_next: Próximo estágio da cadeia ASGI.

        Returns:
            A resposta produzida pela aplicação.

        Raises:
            Exception: Repassa qualquer falha não tratada, após contabilizá-la.
        """
        endpoint = _route_template(request)
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:
            ERRORS.labels(tipo=type(exc).__name__).inc()
            REQUESTS.labels(method=request.method, endpoint=endpoint, status="500").inc()
            REQUEST_DURATION.labels(endpoint=endpoint).observe(time.perf_counter() - started)
            raise

        REQUEST_DURATION.labels(endpoint=endpoint).observe(time.perf_counter() - started)
        REQUESTS.labels(
            method=request.method, endpoint=endpoint, status=str(response.status_code)
        ).inc()

        if response.status_code >= 400:
            ERRORS.labels(tipo=f"http_{response.status_code}").inc()

        return response


def observe_prediction(urgencia: str, regra: str, backend: str, inference_seconds: float) -> None:
    """Registra uma classificação emitida.

    Args:
        urgencia: Nível atribuído ao laudo.
        regra: Parte da regra de decisão que determinou o nível.
        backend: Motor de inferência usado.
        inference_seconds: Tempo gasto apenas pelo modelo.
    """
    PREDICTIONS.labels(urgencia=urgencia, regra=regra).inc()
    INFERENCE_DURATION.labels(backend=backend).observe(inference_seconds)


def set_model_state(loaded: bool, urgent_threshold: float | None = None) -> None:
    """Publica o estado do modelo carregado.

    Args:
        loaded: Se há modelo pronto para classificar.
        urgent_threshold: Limiar da trava de urgência, quando disponível.
    """
    MODEL_LOADED.set(1 if loaded else 0)
    URGENT_THRESHOLD.set(urgent_threshold if urgent_threshold is not None else 0)


def metrics_response() -> Response:
    """Serializa o registro do Prometheus no formato de exposição.

    Returns:
        Resposta HTTP pronta para o scrape.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def install(app: FastAPI) -> None:
    """Acopla a instrumentação à aplicação.

    Args:
        app: Aplicação FastAPI a instrumentar.
    """
    app.add_middleware(PrometheusMiddleware)


def _route_template(request: Request) -> str:
    """Devolve o template da rota casada, e não o caminho concreto.

    Usar o caminho concreto criaria uma série temporal nova para cada URL distinta —
    inclusive para varreduras automatizadas de rotas inexistentes, que é como um
    Prometheus morre de cardinalidade sem ninguém entender o motivo.

    Args:
        request: Requisição em curso.

    Returns:
        O template da rota, ou `"desconhecida"` quando nenhuma rota casa.
    """
    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match.name == "FULL":
            return getattr(route, "path", "desconhecida")
    return "desconhecida"
