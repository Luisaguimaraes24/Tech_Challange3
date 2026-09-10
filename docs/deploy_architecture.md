# Decisão Arquitetural de Deploy

> Documento da Etapa 1. Analisa qual estratégia de deploy em nuvem atende ao cenário de
> triagem hospitalar e registra a decisão, os critérios e o que foi descartado.

---

## 1. O problema condiciona a arquitetura

Um laudo entra na fila de análise e alguém precisa decidir **em que ordem** os laudos
serão lidos. Esse é o único produto do sistema: uma ordenação. Duas consequências
imediatas para a arquitetura:

1. **O valor da predição decai com o tempo.** Uma classificação que chega depois que o
   laudo já foi lido manualmente não vale nada. O sistema é útil no instante em que o
   laudo é registrado.
2. **O volume é baixo e irregular.** Um hospital de referência produz laudos na casa de
   milhares por dia, não milhões por segundo, e concentrados no horário comercial. O
   dimensionamento precisa tolerar ociosidade sem custo proporcional.

## 2. Batch versus real-time

| Critério | Batch (job periódico) | Real-time (API síncrona) |
|---|---|---|
| Latência até a triagem | minutos a horas | milissegundos |
| Utilidade clínica | baixa — a fila já andou | alta — prioriza na entrada |
| Custo em ociosidade | menor (roda e morre) | maior (serviço sempre no ar) |
| Complexidade operacional | menor | exige healthcheck, escala, observabilidade |
| Reprocessamento histórico | natural | precisa de endpoint de lote |

**Decisão: real-time, via API REST síncrona.**

O argumento decisivo não é técnico, é clínico. Um lote de 15 minutos significa que um
laudo urgente pode ficar 15 minutos invisível na fila — exatamente o risco que o sistema
existe para eliminar. Como a inferência custa **menos de 1 ms** (ver
[latency_report.md](latency_report.md)), não há razão de desempenho para adiar a resposta:
o batch só economizaria o custo de manter o serviço no ar.

**Ressalva registrada:** o retreino continua sendo batch. Ingestão, treino, avaliação e
exportação rodam como pipeline agendado no Airflow (Etapa 2). A arquitetura é híbrida —
inferência síncrona, ciclo de vida do modelo assíncrono — e isso é intencional: são cargas
com perfis de recurso e de falha completamente diferentes.

## 3. Escolha do provedor e do serviço de compute

A avaliação assume **AWS**, por ser onde a maioria dos hospitais brasileiros com operação
em nuvem já tem contrato e conformidade estabelecidos. As alternativas equivalentes em
Azure (Container Apps) e GCP (Cloud Run) resolveriam o mesmo problema; a decisão entre
provedores, neste cenário, é contratual antes de ser técnica.

| Opção | A favor | Contra | Veredito |
|---|---|---|---|
| **ECS Fargate** | Mesma imagem Docker do ambiente local; sem gerenciar nós; escala por CPU/requisições; integra com ALB, ECR e CloudWatch | Custo por tarefa parado é maior que o de funções | ✅ **escolhido** |
| Lambda (container) | Custo zero em ociosidade; escala instantânea | Cold start de segundos ao carregar o modelo — inaceitável para o p95 alvo; limite de 10 GB de imagem; timeout | ❌ |
| EKS (Kubernetes) | Controle total, portabilidade | Custo do control plane e carga operacional desproporcionais para um serviço de um endpoint | ❌ |
| SageMaker Endpoint | Feito para ML; A/B testing e model registry nativos | Preço de instância dedicada; acopla o projeto ao SageMaker; excesso para um TF-IDF de 3 MB | ❌ |
| EC2 | Barato em reserva | Patching, AMI, autoscaling manual — trabalho que o Fargate elimina | ❌ |

**Decisão: AWS ECS Fargate, atrás de um Application Load Balancer.**

O que pesou: o artefato de deploy é **exatamente a mesma imagem** que roda no
`docker-compose` local. Não há divergência entre o que foi testado e o que vai para
produção — que é o modo mais comum de um deploy de modelo falhar.

O cold start foi o critério que eliminou o Lambda. O container carrega o pipeline no
startup e o mantém em memória; em Lambda, cada ambiente novo pagaria esse carregamento
dentro da requisição do usuário, e o p95 passaria a ser dominado por ele.

## 4. Topologia proposta

```
   Rede interna do hospital (HIS/RIS)
                │
                │ HTTPS
                ▼
     ┌──────────────────────┐
     │  ALB (privado, VPC)  │  ── TLS, healthcheck em /health
     └──────────┬───────────┘
                │
     ┌──────────▼───────────┐
     │  ECS Fargate Service │  ── 2..N tarefas, autoscaling por CPU
     │  triage-api          │     imagem puxada do ECR
     └──────────┬───────────┘
                │ lê o artefato no boot
     ┌──────────▼───────────┐        ┌────────────────────────┐
     │  S3: models/         │◄───────│  Airflow (MWAA)        │
     │  model.onnx          │ publica│  ingestão → treino →   │
     └──────────────────────┘        │  avaliação → export    │
                                     └────────────────────────┘
   Observabilidade: /metrics → Amazon Managed Prometheus → Managed Grafana
```

**Escalonamento.** Target tracking em 60% de CPU, mínimo de 2 tarefas (para sobreviver à
perda de uma zona de disponibilidade) e máximo dimensionado pelo pico do horário
comercial. Com ~1.300 predições/segundo por container medidas localmente, duas tarefas já
cobrem folgadamente o volume de um hospital de referência — o mínimo de 2 existe por
disponibilidade, não por capacidade.

**Ciclo de vida do modelo.** A DAG publica o artefato versionado no S3; o serviço carrega
a versão corrente no boot. Promover um modelo novo é um deploy — rastreável, reversível
por rollback de task definition, e nunca uma troca silenciosa de arquivo com o serviço no ar.

**Segurança e dados.** Laudo médico é dado pessoal sensível sob a LGPD. Consequências
diretas na arquitetura: ALB **interno** (o serviço não é exposto à internet), TLS em
trânsito, criptografia em repouso no S3, e **o texto do laudo nunca é gravado em log nem
exportado como rótulo de métrica** — as métricas do Prometheus carregam apenas contagens e
tempos, jamais conteúdo. A instrumentação da Etapa 3 respeita essa restrição.

## 5. Do local para a nuvem

| Componente | Local (este repositório) | Equivalente em produção |
|---|---|---|
| Inferência | container `api` no Compose | serviço ECS Fargate |
| Registro de imagem | Docker local | ECR |
| Orquestração de treino | container `airflow` | MWAA (Airflow gerenciado) |
| Artefatos de modelo | volume `./models` | bucket S3 versionado |
| Métricas | Prometheus no Compose | Amazon Managed Prometheus |
| Dashboards | Grafana no Compose | Amazon Managed Grafana |
| CI/CD | GitHub Actions (lint/test/build) | mesma pipeline + push no ECR e deploy no ECS |

A paridade é o ponto: cada peça local tem um correspondente gerenciado, e a migração não
exige reescrever a aplicação — apenas trocar endpoints e credenciais por variáveis de
ambiente, que já é como a configuração está estruturada (`TRIAGE_*` em
[`src/triage/config.py`](../src/triage/config.py)).

## 6. Limitações assumidas

- **O deploy em nuvem não foi provisionado.** Este documento é a análise pedida na Etapa 1;
  a entrega executável é a stack local em Docker Compose. Não há Terraform neste repositório.
- **Modelo único, sem A/B testing.** Promover uma versão nova é substituir a anterior.
  Canary deployment seria o passo seguinte, e o ECS suporta nativamente — ficou fora por escopo.
- **Sem cache de resposta.** Laudos são praticamente únicos; um cache teria taxa de acerto
  próxima de zero e só adicionaria complexidade.
