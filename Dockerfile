# Imagem do serviço de inferência.
#
# Multi-stage: o estágio `builder` resolve as dependências com o uv a partir do
# uv.lock (build reprodutível) e o estágio final carrega apenas o virtualenv pronto.
# Nem o uv, nem o compilador, nem o código-fonte do pacote sobrevivem para a imagem
# final — só o venv e os artefatos de modelo.
#
# As dependências de ingestão e exportação (kagglehub, skl2onnx) ficam de fora: a API
# consome o modelo já treinado, não o produz.

FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.12 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Camada de dependências, invalidada apenas quando o lock muda.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-editable


FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="triage-api" \
      org.opencontainers.image.description="API de triagem de urgência em laudos médicos"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    TRIAGE_MODELS_DIR=/app/models

WORKDIR /app

# Rodar como root dentro do container é risco desnecessário para um serviço exposto.
# O usuário é criado ANTES do COPY: um `chown -R` posterior reescreveria o venv inteiro
# em uma nova camada, duplicando ~480 MB na imagem final.
RUN useradd --create-home --uid 1000 triage

COPY --from=builder --chown=triage:triage /app/.venv /app/.venv
COPY --chown=triage:triage models ./models

USER triage

EXPOSE 8000

# O healthcheck usa a stdlib: a imagem slim não traz curl, e instalar um pacote
# só para isso engordaria o container sem necessidade.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status == 200 else 1)"

# Um worker por container: a escala é horizontal, no orquestrador, e um único worker
# mantém a medição de latência interpretável.
CMD ["uvicorn", "triage.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
