.DEFAULT_GOAL := help
SHELL := /bin/bash

# O uv guarda os interpretadores em XDG_DATA_HOME, que dentro do snap do VS Code aponta
# para um diretório versionado (~/snap/code/<rev>) que some a cada atualização. Fixamos
# um destino durável para o Python gerenciado não evaporar junto com o snap.
export UV_PYTHON_INSTALL_DIR := $(HOME)/.local/share/uv/python
UV := $(HOME)/.local/bin/uv
PY := .venv/bin/python

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

validate-dag:  ## Verifica que a DAG carrega, sem subir o Airflow
	$(UV) run --isolated --no-project --with "apache-airflow==3.2.2" --python 3.11 \
		python scripts/validate_dag.py

clean:  ## Remove caches e artefatos temporários
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
