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

---

## Medição sob concorrência (Etapa 3)

A linha de base acima foi medida com **um cliente por vez**. Isso responde "quanto custa
uma requisição", mas não "quanto o serviço demora quando está sendo usado". Com o gerador
de carga (`scripts/load_test.py`), 30 segundos por nível, laudos reais em rodízio:

| Clientes simultâneos | p50 | p95 | p99 |
|---|---|---|---|
| 1 | 6,57 ms | 17,09 ms | 105,16 ms |
| 4 | 18,92 ms | 29,95 ms | 38,53 ms |
| 8 | 24,20 ms | 40,99 ms | 51,84 ms |
| 16 | 53,58 ms | 74,08 ms | 99,45 ms |

**Esta tabela corrige a conclusão da Etapa 1.** Lá, o alvo de p95 < 100 ms aparecia
cumprido com 27× de folga. Com 16 clientes simultâneos, o p95 vai a 74 ms e o p99 encosta
em 99 ms — a folga real é de 1,35×, não de 27×. O número tranquilizador vinha da condição
de medição, não do sistema.

A causa é estrutural: um worker uvicorn, um processo Python, um GIL. As requisições não
ficam mais lentas — elas ficam **na fila**. Escalar aqui é horizontal (mais tarefas no
ECS, como prevê o documento de arquitetura), não otimizar o modelo.

### O que a observabilidade custou

Instrumentar não é de graça, e o custo foi medido em vez de estimado. Três execuções de
1.000 requisições em cada condição, valor mediano:

| Condição | p50 | p95 |
|---|---|---|
| Etapa 1 — sem instrumentação, só a API no ar | 2,07 ms | 3,66 ms |
| Com instrumentação, só a API no ar | 3,38 ms | 6,05 ms |
| Com instrumentação e a stack completa | 5,30 ms | 8,94 ms |

O middleware do Prometheus acrescenta cerca de **1,3 ms** ao p50; a disputa de CPU com
Prometheus, Grafana e Airflow no mesmo host acrescenta outros ~1,9 ms. As duas parcelas
são da mesma ordem da variância entre execuções documentada acima, então valem como
ordem de grandeza — não como número exato. Registrar isso importa: é comum tratar
observabilidade como custo zero, e aqui ela consome mais tempo do que a otimização da
Etapa 4 tem chance de recuperar.

---

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

---

# Etapa 4 — Otimização

## O que foi tentado, e o que foi rejeitado

A otimização óbvia era exportar o **pipeline inteiro** para ONNX: `TfidfVectorizer` e
`LogisticRegression` num só grafo, eliminando a travessia Python. Ela funciona e é
rápida — **2,0× a 2,5×**. Foi rejeitada.

O motivo é a tokenização. O `TfidfVectorizer` tokeniza em Python com uma expressão
regular; o operador equivalente do ONNX reimplementa isso em C++, e as duas não
coincidem. Medido no split de teste inteiro:

| Configuração testada | Paridade | Ganho | Veredito |
|---|---|---|---|
| (1,2) gramas, tokenização padrão | 0,9806 | 2,48× | ❌ 56 laudos classificados diferente |
| (1,2) gramas, `token_pattern` que o ONNX digere | 0,9841 | 2,26× | ❌ |
| Apenas unigramas | 0,9965 | 4,20× | ❌ 10 laudos diferentes |
| Vocabulário podado a 5.000 | 0,9868 | 7,98× | ❌ |

**Nenhuma configuração atinge o limiar de paridade de 0,999** — definido antes de medir,
justamente para não virar justificativa depois. Um ganho de 2,5× que muda a urgência
atribuída a 2% dos laudos não é uma otimização: é servir um modelo diferente do que foi
validado, sem avisar ninguém. Num sistema de triagem, é trocar o critério clínico em
silêncio.

## O que foi entregue

Duas mudanças, cada uma medida separadamente.

**1. Poda do vocabulário: 50.000 → 10.000 termos.** Escolhida por validação cruzada no
split de treino, e o corte **melhorou** o modelo: f1-macro 0,5936 contra 0,5900 do
vocabulário completo. A cauda de 40.000 termos raros contribuía mais ruído que sinal.

**2. Exportação apenas do classificador para ONNX.** A vetorização continua no
scikit-learn; o ONNX Runtime recebe o vetor de features pronto. A paridade passa a ser
**exata** — 2.888 de 2.888 laudos idênticos, diferença máxima de probabilidade de
1,7e-07, que é o arredondamento de float32.

A poda é o que torna o ONNX viável. Com os 50.000 termos originais, densificar o vetor
esparso para alimentar o grafo custava mais do que ele economizava, e a "otimização"
ficava **mais lenta** que o original (0,91×).

## Resultado — latência do modelo

Sete execuções independentes por configuração, **intercaladas** (não em bloco, para que
variação de carga da máquina ao longo do tempo não seja atribuída a um dos candidatos),
400 medições por execução. Reportada a mediana das execuções, com o intervalo observado:

| Configuração | p50 | p95 | Ganho |
|---|---|---|---|
| scikit-learn, 50.000 features (antes) | 0,751 ms [0,731–0,801] | 0,893 ms | 1,00× |
| scikit-learn, 10.000 features (poda) | 0,736 ms [0,719–0,758] | 0,860 ms | 1,02× |
| **ONNX, 10.000 features (poda + ONNX)** | **0,562 ms** [0,555–0,573] | **0,683 ms** | **1,34×** |

Os intervalos de sklearn e ONNX **não se sobrepõem** — o mínimo do scikit-learn (0,731 ms)
é maior que o máximo do ONNX (0,573 ms). A diferença é real, não ruído. Essa verificação
só é possível porque as execuções foram repetidas; com uma medição única por backend, um
ganho de 1,34× seria indistinguível da variância de 40% documentada acima.

**A poda sozinha não acelerou nada** (1,02×, dentro do ruído). Isso contraria a
expectativa inicial e vale registrar: o custo do TF-IDF é dominado pela tokenização, que
é proporcional ao **tamanho do documento**, não ao tamanho do vocabulário. O vocabulário
só afeta buscas em tabela hash. O ganho da poda foi outro — um artefato 5× menor.

## Resultado — ponta a ponta, sob concorrência

| Clientes | sklearn p50 | ONNX p50 | sklearn p95 | ONNX p95 |
|---|---|---|---|---|
| 1 | 1,84 ms | 1,81 ms | 2,50 ms | 4,23 ms |
| 8 | 13,36 ms | 11,87 ms | 15,75 ms | 14,83 ms |
| 16 | 29,79 ms | 28,90 ms | 34,13 ms | 33,81 ms |

**O ganho de 1,34× no modelo vira 0% a 11% na resposta ao usuário**, e some quase
inteiramente sob concorrência alta. Era a previsão registrada na Etapa 1, e ela se
confirmou: o modelo é minoria do tempo de resposta, e sob carga o que domina é fila, não
computação. Otimizar o modelo não resolve um problema de enfileiramento — escalar
horizontalmente resolve.

## Footprint

| Medida | scikit-learn | ONNX |
|---|---|---|
| Artefato do modelo | 786 KB (`model.pkl`) | 245 KB (`model.onnx`) + o `.pkl` para o TF-IDF |
| Memória residente do container | 121,5 MiB | 128,1 MiB |
| Tamanho da imagem | 731 MB | 731 MB |

Aqui a expectativa da Etapa 1 **não se confirmou**. Esperávamos enxugar a imagem
eliminando o scikit-learn do container de inferência; como a vetorização permaneceu no
scikit-learn, ele continua na imagem e o tamanho não mudou. Pior: o caminho ONNX consome
**7 MiB a mais** de memória, porque carrega o vetorizador e a sessão de inferência ao
mesmo tempo. O ganho desta etapa é tempo de modelo, não recurso.

## Quantização INT8: avaliada e não aplicável

A quantização dinâmica foi tentada e **não se aplica a este modelo** — por um motivo
estrutural, não por erro de configuração. O `skl2onnx` exporta a regressão logística como
um único operador `LinearClassifier` do domínio `ai.onnx.ml`. O quantizador dinâmico do
ONNX Runtime opera sobre `MatMul`, `Conv`, `Gather`, `LSTM` e afins; não existe
substituto INT8 para `LinearClassifier`. Não há o que quantizar.

O diagnóstico é coerente com o que a técnica se propõe: quantização paga em modelos
dominados por multiplicações de matriz grandes. Uma camada linear de 3 × 10.000
parâmetros não é um desses. A tentativa permanece no código, com o motivo no log, para
que fique claro que a técnica foi avaliada e descartada com evidência — e não esquecida.

## Resumo

| | Antes | Depois |
|---|---|---|
| Vocabulário | 50.000 termos | 10.000 termos |
| f1-macro (CV) | 0,5900 | 0,5936 |
| Artefato | 4.059 KB | 786 KB + 245 KB |
| Latência do modelo (p50) | 0,751 ms | 0,562 ms (**1,34×**) |
| Latência HTTP, 16 clientes (p95) | 34,13 ms | 33,81 ms |
| Paridade de predições | — | 1,000000 |
