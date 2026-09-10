# Triagem Automática de Laudos Médicos

Sistema de triagem que classifica a urgência de laudos médicos em `normal`, `atenção` ou `urgente`, servido por API REST em container Docker, com pipeline CI/CD, orquestração de retreino e stack de observabilidade.

> **Tech Challenge — Fase 3 · FIAP Pós-Tech (MLET)**

---

## Status

| Etapa | Descrição | Status |
|---|---|---|
| 0 | Setup do ambiente e estrutura do repositório | ✅ concluída |
| 1 | Decisão arquitetural e API FastAPI em Docker | ⬜ |
| 2 | CI/CD (GitHub Actions) e DAG Airflow | ⬜ |
| 3 | Monitoramento (Prometheus + Grafana) | ⬜ |
| 4 | Otimização de latência (ONNX) e entrega | ⬜ |

---

## Requisitos

- Python 3.11 (o projeto **não** roda em 3.12+ por causa do Airflow/ONNX Runtime)
- [uv](https://docs.astral.sh/uv/) para gerenciar ambiente e dependências
- Docker e Docker Compose

## Setup

```bash
make setup       # cria o .venv com Python 3.11 e instala as dependências
make lint        # ruff check + format
make test        # pytest
```

## Dados

O dataset é o [Medical Abstracts TC Corpus](https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus) (Kaggle).

```bash
make data        # baixa via kagglehub para data/raw/
```

O corpus é público e o download roda sem credenciais. Se o `kagglehub` falhar (rede ou mudança na fonte), basta colocar os CSVs manualmente em `data/raw/` — o carregador detecta os arquivos locais e pula o download.

**14.438 abstracts médicos** com split oficial:

| Split | Amostras | `urgente` | `normal` | `atencao` |
|---|---|---|---|---|
| train | 11.550 | 4.971 | 3.844 | 2.735 |
| test | 2.888 | 1.243 | 961 | 684 |

### Do rótulo original para a urgência

O corpus rotula **condições médicas** (5 classes), não urgência. A urgência é derivada por uma regra determinística, documentada em [`src/triage/data/labels.py`](src/triage/data/labels.py), cujo critério é o potencial de deterioração rápida do quadro:

| Condição original | Urgência |
|---|---|
| neoplasms | `urgente` |
| cardiovascular diseases | `urgente` |
| nervous system diseases | `atencao` |
| digestive system diseases | `atencao` |
| general pathological conditions | `normal` |

> ⚠️ **Aviso**: esse mapeamento é um **proxy didático**, não uma classificação clínica validada. Este projeto é acadêmico e **não é um dispositivo médico**.

## Licença

MIT.
