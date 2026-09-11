.DEFAULT_GOAL := help
SHELL := /bin/bash

# O uv guarda os interpretadores em XDG_DATA_HOME, que dentro do snap do VS Code aponta
# para um diretório versionado (~/snap/code/<rev>) que some a cada atualização. Fixamos
# um destino durável para o Python gerenciado não evaporar junto com o snap.
export UV_PYTHON_INSTALL_DIR := $(HOME)/.local/share/uv/python
UV := $(HOME)/.local/bin/uv
PY := .venv/bin/python

# Precisam acompanhar Dockerfile.airflow e .github/workflows/ci.yml.
# tests/test_versions.py falha se as referências divergirem — foi assim que um CI
# fixado numa versão antiga do Airflow quebrou enquanto tudo passava localmente.
PYTHON_VERSION := 3.11
AIRFLOW_VERSION := 3.2.2

.PHONY: help setup data lint format test train up down clean

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Cria o .venv com Python 3.11 e instala as dependências
	$(UV) venv --python 3.11
	$(UV) sync --all-extras

data:  ## Baixa o dataset do Kaggle para data/raw/
	$(PY) scripts/download_data.py

lint:  ## Verifica estilo e formatação
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:  ## Aplica formatação e correções automáticas
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

test:  ## Roda a suíte de testes com cobertura
	$(UV) run pytest --cov=src/triage --cov-report=term-missing

train:  ## Treina o classificador de urgência
	$(PY) -m triage.model.train

onnx:  ## Exporta o classificador para ONNX e verifica a paridade
	$(PY) -m triage.model.export_onnx --report docs/onnx_parity.json

compare:  ## Compara a latência dos backends (execuções repetidas e intercaladas)
	$(PY) -m triage.benchmark.compare --repeats 7 --output docs/onnx_comparison.json

# O Airflow precisa escrever nos bind mounts ./data e ./models, que pertencem ao
# usuário do host; sem isso o container roda como uid 50000 e a DAG falha ao gravar.
export AIRFLOW_UID := $(shell id -u)

up:  ## Sobe a stack completa (API + Airflow)
	docker compose up -d --build

down:  ## Derruba a stack
	docker compose down

logs:  ## Acompanha os logs da stack
	docker compose logs -f

dag:  ## Dispara a DAG de treino e acompanha o resultado
	docker compose exec airflow airflow dags unpause triage_training
	docker compose exec airflow airflow dags trigger triage_training

load:  ## Gera carga na API para popular o dashboard (60s, 8 clientes)
	$(PY) scripts/load_test.py --duration 60 --concurrency 8 --output docs/load_test.json

screenshot:  ## Captura o dashboard do Grafana (rode com `make load` em paralelo)
	@mkdir -p docs/img
	google-chrome --headless --disable-gpu --no-sandbox --hide-scrollbars \
		--window-size=1600,1000 --virtual-time-budget=25000 \
		--screenshot=docs/img/grafana-dashboard.png \
		"http://localhost:3000/d/triage-observability/x?kiosk&from=now-5m&to=now"
	@echo "Capturado em docs/img/grafana-dashboard.png"

validate-dag:  ## Verifica que a DAG carrega, sem subir o Airflow
	$(UV) run --isolated --no-project --with "apache-airflow==$(AIRFLOW_VERSION)" \
		--python $(PYTHON_VERSION) python scripts/validate_dag.py

clean:  ## Remove caches e artefatos temporários
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
