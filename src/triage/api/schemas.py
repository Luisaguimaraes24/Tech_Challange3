"""Contratos de entrada e saída da API.

Os schemas são a fronteira do serviço: validam o laudo recebido antes que ele chegue
ao modelo e fixam o formato da resposta, que é consumido pelos testes, pelo benchmark
de latência e por quem integrar com a triagem.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from triage.data.labels import UrgencyLevel

MIN_TEXT_LENGTH = 20
MAX_TEXT_LENGTH = 20_000


class PredictionRequest(BaseModel):
    """Laudo médico submetido para triagem."""

    texto: str = Field(
        ...,
        description="Texto livre do laudo médico a ser classificado.",
        examples=["Patient presents with acute chest pain radiating to the left arm..."],
    )

    @field_validator("texto")
    @classmethod
    def validar_texto(cls, value: str) -> str:
        """Rejeita laudos vazios ou curtos demais para sustentar uma triagem.

        Args:
            value: Texto submetido.

        Returns:
            O texto sem espaços nas bordas.

        Raises:
            ValueError: Se o texto for vazio, curto demais ou longo demais.
        """
        texto = value.strip()
        if not texto:
            raise ValueError("O laudo não pode ser vazio.")
        if len(texto) < MIN_TEXT_LENGTH:
            raise ValueError(
                f"O laudo tem {len(texto)} caracteres; o mínimo é {MIN_TEXT_LENGTH} "
                "para que a classificação seja significativa."
            )
        if len(texto) > MAX_TEXT_LENGTH:
            raise ValueError(f"O laudo excede o limite de {MAX_TEXT_LENGTH} caracteres.")
        return texto


class PredictionResponse(BaseModel):
    """Resultado da triagem de um laudo."""

    urgencia: UrgencyLevel = Field(description="Nível de urgência atribuído ao laudo.")
    descricao: str = Field(description="Orientação associada ao nível de urgência.")
    confianca: float = Field(
        ge=0.0, le=1.0, description="Probabilidade atribuída à classe escolhida."
    )
    probabilidades: dict[str, float] = Field(
        description="Distribuição de probabilidade sobre todos os níveis."
    )
    regra: str = Field(
        description=(
            "Qual parte da regra determinou a decisão: `condicao_dominante` quando a "
            "condição mais provável define o nível, ou `trava_de_urgencia` quando a "
            "probabilidade acumulada de urgência ultrapassou o limiar de segurança."
        )
    )
    latencia_ms: float = Field(ge=0.0, description="Tempo de inferência do modelo, em ms.")
    backend: str = Field(description="Motor de inferência usado (`sklearn` ou `onnx`).")


class HealthResponse(BaseModel):
    """Estado do serviço, consumido por healthcheck de container e orquestrador."""

    status: str = Field(description="`ok` quando o serviço pode atender predições.")
    model_loaded: bool = Field(description="Indica se há modelo carregado em memória.")
    backend: str | None = Field(default=None, description="Motor de inferência ativo.")
    version: str = Field(description="Versão da aplicação.")


class ModelInfoResponse(BaseModel):
    """Metadados do modelo em produção."""

    backend: str = Field(description="Motor de inferência ativo.")
    classes: list[str] = Field(description="Níveis de urgência que o modelo pode devolver.")
    model_path: str = Field(description="Caminho do artefato carregado.")
    decision_rule: dict = Field(
        description="Regra de decisão em vigor, incluindo o limiar da trava de urgência."
    )
    metrics: dict | None = Field(
        default=None, description="Métricas da última avaliação, se disponíveis."
    )


class ErrorResponse(BaseModel):
    """Corpo padronizado das respostas de erro."""

    detail: str = Field(description="Descrição legível do erro.")
