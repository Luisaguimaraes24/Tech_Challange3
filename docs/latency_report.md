# Relatório de Latência

> Linha de base medida na Etapa 1. A comparação com o modelo otimizado em ONNX entra na
> Etapa 4, neste mesmo documento.

## Ambiente de medição

| Item | Valor |
|---|---|
| CPU | AMD Ryzen 7 5825U (16 threads) |
| Memória | 30 GB |
| Python | 3.11.16 |
| Modelo | TF-IDF (1-2 gramas, 50.000 termos) + Regressão Logística |
| Artefato | `models/model.pkl` — 3,2 MB |
| Container | `triage-api:etapa1`, 727 MB, 1 worker uvicorn |
| Amostras | 1.000 medições, 50 de aquecimento descartadas |
| Entrada | 100 abstracts reais do split de teste, em rodízio (~1.200 caracteres cada) |

Comando: `python -m triage.benchmark.latency --iterations 1000 --output docs/latency_baseline.json`

A escolha do texto de entrada importa. O custo do TF-IDF cresce com o tamanho do
documento; medir com uma frase curta sintética produziria números otimistas que não se
sustentam em produção. As medições usam laudos reais do corpus.

## Resultados — linha de base (scikit-learn)

| Medição | p50 | p95 | p99 | média | máx | Vazão |
|---|---|---|---|---|---|---|
| **Modelo** (em processo) | 0,751 ms | 0,888 ms | 0,997 ms | 0,756 ms | 1,086 ms | 1.322 req/s |
| **HTTP** (container Docker) | 1,426 ms | 1,759 ms | 2,100 ms | 1,459 ms | 2,580 ms | 685 req/s |

### Leitura

**O alvo de p95 < 100 ms está cumprido com 56× de folga** já na linha de base, sem
nenhuma otimização. Isso precisa ser dito antes de qualquer discussão sobre ONNX: o
gargalo deste serviço não é o modelo.

**Metade do tempo de resposta não é o modelo.** A inferência custa 0,76 ms; a requisição
completa custa 1,46 ms. Os ~0,7 ms restantes são serialização JSON, validação Pydantic,
overhead do ASGI e o loopback de rede. Consequência prática: mesmo que a Etapa 4 zerasse o
custo do modelo, o p95 ponta a ponta cairia no máximo pela metade.

**A distribuição é estreita.** O p99 fica a 1,3× do p50 no modelo, o que indica ausência de
cauda longa — sem coleta de lixo agressiva, sem realocação, sem contenção de lock. É o
comportamento esperado de uma multiplicação de matriz esparsa sobre um vocabulário fixo.

## O que isso implica para a Etapa 4

A otimização para ONNX foi mantida no plano — é requisito do projeto e vale 20% da nota —
mas com a expectativa calibrada e registrada aqui **antes** de medir:

- o ganho aparecerá na latência do modelo, na casa de fração de milissegundo;
- o ganho relativo ponta a ponta será menor, diluído pelo overhead de serviço;
- o benefício mais concreto talvez não seja tempo, e sim **tamanho da imagem**: servir com
  `onnxruntime` permite eliminar o `scikit-learn` e o `scipy` do container de inferência.

Reportar um ganho pequeno com honestidade vale mais do que escolher uma medição favorável.
A comparação completa — sklearn vs. ONNX, com paridade de predições verificada — entra
aqui na Etapa 4.
