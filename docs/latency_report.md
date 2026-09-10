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
| Artefato | `models/model.pkl` — 4,0 MB |
| Container | `triage-api:etapa1`, 727 MB, 1 worker uvicorn |
| Amostras | 2.000 medições, 50 de aquecimento descartadas |
| Entrada | 100 abstracts reais do split de teste, em rodízio (~1.200 caracteres cada) |

Comando: `python -m triage.benchmark.latency --iterations 1000 --output docs/latency_baseline.json`

A escolha do texto de entrada importa. O custo do TF-IDF cresce com o tamanho do
documento; medir com uma frase curta sintética produziria números otimistas que não se
sustentam em produção. As medições usam laudos reais do corpus.

## Resultados — linha de base (scikit-learn)

| Medição | p50 | p95 | p99 | Vazão |
|---|---|---|---|---|
| **Modelo** (em processo) | 0,895 ms | 1,073 ms | 1,208 ms | 1.103 req/s |
| **HTTP** (container Docker) | 2,071 ms | 3,656 ms | 4,478 ms | 434 req/s |

### O que a regra de decisão custou

A primeira medição, antes da regra de decisão da Etapa 1, deu p50 de 0,751 ms e p95 de
0,888 ms no modelo. A trava de urgência acrescentou cerca de **0,1 ms** — o custo de uma
agregação de probabilidades e de uma comparação por amostra.

Vale registrar como esse custo quase foi maior: a primeira implementação recalculava a
agregação três vezes por requisição (uma para a resposta, uma dentro de `decide`, outra
dentro de `explain`), o que levava o p50 a 0,949 ms. Reunir decisão e justificativa numa
única passagem (`DecisionRule.resolve`) devolveu os 0,1 ms perdidos. É o tipo de
desperdício que só aparece quando se mede.

### Sobre a variância entre execuções

Três medições consecutivas do mesmo binário, sem carga concorrente, deram p95 de 2,366 ms,
2,828 ms e 3,297 ms na medição HTTP — uma diferença de 40% entre a melhor e a pior. **Uma
única execução não é evidência suficiente** para comparar duas implementações, e isso
condiciona a metodologia da Etapa 4: a comparação sklearn vs. ONNX precisará de execuções
repetidas, não de um número escolhido, sob pena de "medir" uma diferença que é só ruído
de agendamento do sistema operacional.

### Leitura

**O alvo de p95 < 100 ms está cumprido com 27× de folga** já na linha de base, sem
nenhuma otimização. Isso precisa ser dito antes de qualquer discussão sobre ONNX: o
gargalo deste serviço não é o modelo.

**Mais da metade do tempo de resposta não é o modelo.** A inferência custa 0,90 ms; a
requisição completa custa 2,07 ms. Os ~1,2 ms restantes são serialização JSON, validação
Pydantic, overhead do ASGI e o loopback de rede. Consequência prática: mesmo que a Etapa 4
zerasse o custo do modelo, o p95 ponta a ponta cairia menos da metade.

**A distribuição é estreita no modelo e larga no HTTP.** O p99 fica a 1,3× do p50 na
inferência — sem cauda longa, comportamento esperado de uma multiplicação esparsa sobre
vocabulário fixo. Já no HTTP o p99 chega a 2,2× do p50: a cauda vem da camada de serviço,
não do modelo, o que reforça onde o esforço de otimização de fato compensaria.

## O que isso implica para a Etapa 4

A otimização para ONNX foi mantida no plano — é requisito do projeto e vale 20% da nota —
mas com a expectativa calibrada e registrada aqui **antes** de medir:

- o ganho aparecerá na latência do modelo, na casa de fração de milissegundo;
- o ganho relativo ponta a ponta será menor, diluído pelo overhead de serviço;
- o benefício mais concreto talvez não seja tempo, e sim **tamanho da imagem**: servir com
  `onnxruntime` permite eliminar o `scikit-learn` e o `scipy` do container de inferência.

Por isso a Etapa 4 não vai se limitar a repetir esta medição com outro backend. O que
torna a comparação informativa:

1. **execuções repetidas**, pelo motivo registrado acima — uma única medição não separa
   ganho de ruído;
2. **medição sob concorrência**, com N clientes simultâneos, onde a diferença entre motores
   de inferência realmente aparece;
3. **footprint como resultado de primeira classe** — tamanho de imagem e memória residente,
   não só tempo;
4. **varredura por tamanho de documento**, em vez de um único ponto de operação.

Reportar um ganho pequeno com honestidade vale mais do que escolher uma medição favorável.
A comparação completa — sklearn vs. ONNX, com paridade de predições verificada — entra
aqui na Etapa 4.
