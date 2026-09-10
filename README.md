# Triagem Automática de Laudos Médicos

Sistema de triagem que classifica a urgência de laudos médicos em `normal`, `atenção` ou `urgente`, servido por API REST em container Docker, com pipeline CI/CD, orquestração de retreino e stack de observabilidade.

> **Tech Challenge — Fase 3 · FIAP Pós-Tech (MLET)**

---

## Status

| Etapa | Descrição | Status |
|---|---|---|
| 0 | Setup do ambiente e estrutura do repositório | ✅ concluída |
| 1 | Decisão arquitetural e API FastAPI em Docker | ✅ concluída |
| 2 | CI/CD (GitHub Actions) e DAG Airflow | ⬜ |
| 3 | Monitoramento (Prometheus + Grafana) | ⬜ |
| 4 | Otimização de latência (ONNX) e entrega | ⬜ |

---

## Arquitetura

A triagem só tem valor no instante em que o laudo entra na fila, então a inferência é
**síncrona** — uma API REST — enquanto o ciclo de vida do modelo (ingestão, treino,
avaliação, exportação) roda em **batch**, orquestrado pelo Airflow.

Em produção, o alvo é **AWS ECS Fargate** atrás de um ALB interno: o artefato de deploy é
exatamente a mesma imagem Docker que roda localmente, sem divergência entre o que foi
testado e o que vai para produção. O Lambda foi descartado pelo cold start de carga do
modelo, que dominaria o p95; o EKS e o SageMaker, por custo e complexidade
desproporcionais a um serviço de um endpoint.

A análise completa — comparação batch vs. real-time, provedores avaliados, topologia,
escalonamento, e as implicações de LGPD sobre logs e métricas — está em
**[docs/deploy_architecture.md](docs/deploy_architecture.md)**.

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

## Modelo

```bash
make train       # treina, calibra a trava de urgência e grava os artefatos em models/
```

Pipeline `TfidfVectorizer` (1-2 gramas, 50.000 termos) + `LogisticRegression` com
`class_weight="balanced"`.

**O treino tem como alvo as 5 condições médicas, não os 3 níveis de urgência.** Colapsar
as classes antes do treino apaga a fronteira entre condições de vocabulário distinto e
custa sinal: medido por validação cruzada, treinar nas 5 condições e projetar depois rende
f1-macro 0,590 contra 0,573 do treino direto em 3 classes.

A projeção para urgência acontece numa **regra de decisão** de duas partes:

1. **condição dominante** — a urgência é a da condição mais provável;
2. **trava de urgência** — se a probabilidade acumulada nas condições urgentes passa de um
   limiar, o laudo vira `urgente` mesmo sem condição urgente dominante.

O limiar é **calibrado a cada treino**, sobre probabilidades out-of-fold, como o maior
valor que ainda captura `TRIAGE_URGENT_RECALL_TARGET` (padrão: 90%) dos laudos urgentes.
Ele viaja junto do artefato em `models/decision.json` e é exposto em `/model-info`.

### Métricas (teste, n=2.888)

| Métrica | Valor |
|---|---|
| **Recall de `urgente`** | **0,8986** |
| Precisão de `urgente` | 0,6372 |
| Acurácia | 0,6170 |
| f1-macro | 0,5614 |

**Recall de `urgente` é a métrica de decisão**, não a acurácia nem o f1-macro. As duas
últimas tratam todos os erros como equivalentes, o que é falso numa triagem: um laudo
urgente classificado como normal deixa um paciente esperando, enquanto o erro oposto só
consome tempo de um revisor.

O preço é super-triagem: o recall de `normal` é 0,225, ou seja, a fila prioritária fica
inflada de falsos alarmes. Esse ponto de operação é uma decisão institucional, e por isso
mora na configuração e não no código. Análise completa em
[docs/model_card.md](docs/model_card.md).

O treino falha (exit 1) se o recall de `urgente` cair abaixo de 0,85 ou o f1-macro abaixo
de 0,52 — são os gates que impedem a DAG de retreino de publicar um modelo pior.

> O corpus é composto por **abstracts de artigos científicos** (~1.200 caracteres), não por
> notas clínicas. O modelo espera texto desse formato; frases curtas de prontuário estão
> fora da distribuição de treino e produzem predições pouco confiáveis.

## API

```bash
docker build -t triage-api .
docker run -p 8000:8000 triage-api
```

Documentação interativa em `http://localhost:8000/docs`.

| Método | Rota | Descrição |
|---|---|---|
| `POST` | `/predict` | Classifica a urgência de um laudo |
| `GET` | `/health` | Estado do serviço e do modelo |
| `GET` | `/model-info` | Backend ativo, classes e métricas da última avaliação |

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"texto": "Flow cytometric DNA analysis of parathyroid tumors..."}'
```

```json
{
  "urgencia": "urgente",
  "descricao": "Priorizar avaliação imediata; risco de desfecho grave.",
  "confianca": 0.87,
  "probabilidades": {"atencao": 0.081, "normal": 0.049, "urgente": 0.87},
  "regra": "condicao_dominante",
  "latencia_ms": 1.51,
  "backend": "sklearn"
}
```

O campo `regra` diz **por que** o laudo recebeu aquele nível: `condicao_dominante` quando
o diagnóstico mais provável já é grave, `trava_de_urgencia` quando a incerteza distribuída
entre condições urgentes disparou a trava. Uma triagem precisa ser auditável — os dois
casos têm significados clínicos diferentes e merecem leituras diferentes pelo revisor.

Se o artefato de modelo não estiver presente, o serviço **sobe mesmo assim**: `/health`
reporta `model_loaded: false` e `/predict` responde 503. Um container que morre no boot
some do orquestrador antes de conseguir explicar o motivo; um container vivo e degradado
aparece no healthcheck e no dashboard.

## Latência

```bash
python -m triage.benchmark.latency --iterations 1000
```

Linha de base medida na Etapa 1, com 1.000 requisições sobre laudos reais do split de teste:

| Medição | p50 | p95 | p99 | Vazão |
|---|---|---|---|---|
| Modelo (em processo) | 0,895 ms | 1,073 ms | 1,208 ms | 1.103 req/s |
| HTTP (container Docker) | 2,071 ms | 3,656 ms | 4,478 ms | 434 req/s |

O alvo de p95 < 100 ms já é cumprido com folga de 27× **antes** de qualquer otimização —
e mais da metade do tempo de resposta não é o modelo, e sim serialização, validação e rede.
Detalhes, variância entre execuções e implicações para a Etapa 4 em
[docs/latency_report.md](docs/latency_report.md).

## Licença

MIT.
