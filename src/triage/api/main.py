"""API REST de triagem de laudos médicos.

O modelo é carregado uma única vez no `lifespan` da aplicação e fica em
`app.state.predictor`. Se o artefato não estiver disponível, o serviço **sobe assim
mesmo**: `/health` passa a reportar `model_loaded=false` e `/predict` responde 503.
Esse comportamento é deliberado — um container que morre no boot por falta de modelo
some do orquestrador antes de conseguir explicar o motivo, enquanto um container vivo
e degradado aparece no healthcheck e no dashboard.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from triage import __version__
from triage.api.predictor import Predictor
from triage.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from triage.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo no startup e libera a referência no shutdown.

    Args:
        app: Aplicação FastAPI sendo inicializada.

    Yields:
        Controle para o servidor enquanto a aplicação está no ar.
    """
    try:
        app.state.predictor = Predictor.load()
        logger.info(
            "Modelo pronto | backend=%s | classes=%s",
            app.state.predictor.backend.name,
            app.state.predictor.backend.classes,
        )
    except (FileNotFoundError, ValueError) as exc:
        app.state.predictor = None
        logger.error("Serviço no ar em modo degradado, sem modelo: %s", exc)

    yield

    app.state.predictor = None
    logger.info("Aplicação encerrada.")


app = FastAPI(
    title="Triagem de Laudos Médicos",
    description=(
        "Classifica a urgência de laudos médicos em `normal`, `atencao` ou `urgente`. "
        "Projeto acadêmico — não é um dispositivo médico."
    ),
    version=__version__,
    lifespan=lifespan,
)


def get_predictor(request: Request) -> Predictor:
    """Recupera o preditor carregado no startup.

    Args:
        request: Requisição em curso.

    Returns:
        O preditor ativo.

    Raises:
        HTTPException: 503 quando não há modelo carregado.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível. O serviço subiu sem artefato de modelo carregado.",
        )
    return predictor


@app.get("/health", response_model=HealthResponse, tags=["operação"])
async def health(request: Request) -> HealthResponse:
    """Reporta se o serviço consegue atender predições.

    Args:
        request: Requisição em curso.

    Returns:
        Estado do serviço e do modelo.
    """
    predictor = getattr(request.app.state, "predictor", None)
    return HealthResponse(
        status="ok" if predictor else "degraded",
        model_loaded=predictor is not None,
        backend=predictor.backend.name if predictor else None,
        version=__version__,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["operação"],
)
async def model_info(request: Request) -> ModelInfoResponse:
    """Expõe os metadados do modelo em produção.

    Args:
        request: Requisição em curso.

    Returns:
        Backend ativo, classes, caminho do artefato e métricas da última avaliação.
    """
    predictor = get_predictor(request)
    return ModelInfoResponse(
        backend=predictor.backend.name,
        classes=predictor.backend.classes,
        model_path=str(getattr(predictor.backend, "model_path", "")),
        metrics=predictor.metrics,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["triagem"],
)
async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Classifica a urgência de um laudo médico.

    Args:
        payload: Corpo da requisição contendo o texto do laudo.
        request: Requisição em curso.

    Returns:
        Nível de urgência, confiança, distribuição de probabilidades e latência.

    Raises:
        HTTPException: 503 se não houver modelo carregado; 500 se a inferência falhar.
    """
    predictor = get_predictor(request)
    try:
        return PredictionResponse(**predictor.predict(payload.texto))
    except Exception as exc:  # pragma: no cover - salvaguarda de runtime
        logger.exception("Falha na inferência")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Falha ao classificar o laudo.",
        ) from exc


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Padroniza o corpo das respostas de erro no formato `ErrorResponse`.

    Args:
        _: Requisição em curso, não utilizada.
        exc: Exceção levantada pela rota.

    Returns:
        Resposta JSON com a chave `detail`.
    """
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})
