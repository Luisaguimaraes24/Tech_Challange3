# Model Card — Classificador de Urgência de Laudos

> Documento vivo. Atualizado a cada mudança de modelo ou de regra de decisão.
> Última atualização: Etapa 1.

## Identificação

| Campo | Valor |
|---|---|
| Nome | `triage-urgencia` |
| Versão | 0.1.0 |
| Tarefa | Classificação de texto em 3 níveis de urgência |
| Arquitetura | `TfidfVectorizer` (1-2 gramas, 50.000 termos) + `LogisticRegression` |
| Artefato | `models/model.pkl` (4,0 MB) + `models/decision.json` |
| Alvo do treino | **5 condições médicas**, projetadas em urgência pela regra de decisão |
| Semente | 42 |

## Uso pretendido

Ordenar uma fila de laudos médicos por prioridade de leitura, em um cenário acadêmico.

**Não é um dispositivo médico.** Não substitui avaliação clínica, não emite diagnóstico e
não deve ser usado para decisão assistencial real. O rótulo de urgência com que foi
treinado é derivado, não observado (ver abaixo).

## Dados

[Medical Abstracts TC Corpus](https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus)
— 14.438 abstracts, com split oficial de 11.550 treino / 2.888 teste.

**Limitação central do dado:** o corpus é composto por **abstracts de artigos
científicos**, não por notas clínicas ou laudos de prontuário. O texto tem em média 1.200
caracteres, estrutura de resumo acadêmico e vocabulário de publicação. Um laudo real de
prontuário — curto, telegráfico, com abreviações — está fora da distribuição de treino, e
o modelo produz predições pouco confiáveis nesse formato. Um sistema de produção seria
treinado em notas clínicas reais (MIMIC-IV-Note, por exemplo, sob credenciamento no
PhysioNet), e a substituição do corpus é o próximo passo natural do projeto.

### Da condição médica para a urgência

O corpus rotula condição, não urgência. A conversão é uma regra determinística, cujo
critério é o potencial de deterioração rápida do quadro:

| Condição original | Urgência | Amostras (total) |
|---|---|---|
| neoplasms | `urgente` | 3.163 |
| cardiovascular diseases | `urgente` | 3.051 |
| nervous system diseases | `atencao` | 1.925 |
| digestive system diseases | `atencao` | 1.494 |
| general pathological conditions | `normal` | 4.805 |

**Este mapeamento é um proxy didático.** Ele não foi validado clinicamente, não considera
gravidade individual do caso e trata condições heterogêneas como um bloco. Um paciente com
neoplasia em acompanhamento estável não é urgente; a regra o classificaria como tal.

## Modelagem

### Por que treinar nas 5 condições e projetar depois

Colapsar as classes antes do treino apaga a fronteira entre condições com vocabulários
distintos — "neoplasia" e "doença cardiovascular" viram o mesmo rótulo — e o classificador
perde sinal que poderia usar. Medido por validação cruzada de 3 folds no split de treino:

| Alvo do treino | f1-macro (CV) |
|---|---|
| **5 condições, projetadas depois** | **0,5900** |
| 3 níveis de urgência, direto | 0,5732 |

Outras variantes testadas e descartadas por não pagarem o custo: char n-grams (0,5886,
com 3× o tempo de treino), voting `LogReg`+`ComplementNB` (0,5878), `ComplementNB` puro
(0,5533), `LinearSVC` calibrado (0,5073) e varreduras de `C`, `min_df` e `ngram_range`.

### Regra de decisão

Duas partes, aplicadas fora do modelo (em [`decision.py`](../src/triage/model/decision.py)):

1. **condição dominante** — a urgência é a da condição mais provável;
2. **trava de urgência** — se a probabilidade acumulada nas condições urgentes passa de
   **0,30**, o laudo é marcado como urgente mesmo sem nenhuma condição urgente dominante.

O limiar **não é escolhido à mão**: é calibrado a cada treino, sobre probabilidades
out-of-fold, como o maior valor que ainda atinge o recall-alvo de **0,90** para `urgente`.
Calibrar sobre predições que o modelo já viu produziria um limiar otimista que não se
sustenta em produção.

A regra viaja junto do artefato, em `decision.json`, e é exposta em `GET /model-info` —
um hospital precisa poder auditar em que ponto de operação a triagem está rodando.

## Métricas (split de teste, n=2.888)

| Métrica | Valor |
|---|---|
| **Recall de `urgente`** | **0,8986** |
| Precisão de `urgente` | 0,6372 |
| Acurácia | 0,6170 |
| f1-macro | 0,5614 |

| Classe | Precisão | Recall | f1 | Suporte |
|---|---|---|---|---|
| `urgente` | 0,637 | 0,899 | 0,746 | 1.243 |
| `atencao` | 0,562 | 0,656 | 0,606 | 684 |
| `normal` | 0,643 | 0,225 | 0,333 | 961 |

### Como ler estes números

**Recall de `urgente` é a métrica de decisão**, não a acurácia nem o f1-macro. As duas
últimas tratam todos os erros como equivalentes, o que é falso numa triagem: classificar
um laudo urgente como normal deixa um paciente esperando, enquanto o erro oposto apenas
consome tempo de um revisor. O sistema captura **90% dos laudos urgentes**.

**O preço disso é super-triagem, e ele é alto.** O recall de `normal` é 0,225 — três em
cada quatro laudos de rotina são elevados para uma faixa mais urgente. Numa operação real,
isso significa uma fila prioritária inflada, e o ponto de operação teria que ser negociado
com o serviço, não decidido no código. É exatamente por isso que o alvo de recall é
configuração (`TRIAGE_URGENT_RECALL_TARGET`) e não constante.

**A classe `normal` é intrinsecamente difícil.** Ela corresponde a "general pathological
conditions", uma categoria guarda-chuva do corpus que se sobrepõe às outras quatro. Parte
do erro não é do modelo, é do rótulo.

### Efeito da trava de urgência

| Regra | f1-macro | Acurácia | Recall `urgente` | Precisão `urgente` |
|---|---|---|---|---|
| Treino direto em 3 classes (versão anterior) | 0,5600 | 0,5748 | 0,6814 | 0,6960 |
| 5 condições, sem trava | 0,5810 | 0,6080 | 0,7643 | 0,6859 |
| **5 condições, trava em 0,30** | 0,5614 | 0,6170 | **0,8986** | 0,6372 |

A trava troca 5,9 pontos de precisão em `urgente` por 13,4 pontos de recall. Numa triagem,
essa troca é favorável; num sistema de cobrança, seria o contrário.

## Gates de qualidade

O treino falha (código de saída 1) e a DAG de retreino não publica o modelo se:

- recall de `urgente` < 0,85 (`TRIAGE_MIN_RECALL_URGENTE`)
- f1-macro < 0,52 (`TRIAGE_MIN_F1_MACRO`)

## Desempenho

| Medição | p50 | p95 | p99 |
|---|---|---|---|
| Inferência em processo | 0,895 ms | 1,073 ms | 1,208 ms |
| Requisição HTTP (container) | 2,071 ms | 3,656 ms | 4,478 ms |

Detalhes e metodologia em [latency_report.md](latency_report.md).

## Riscos conhecidos

| Risco | Estado |
|---|---|
| Rótulo de urgência é derivado, não clínico | Documentado; exigiria validação com especialista |
| Corpus é literatura científica, não laudo real | Documentado; substituição por notas clínicas é o próximo passo |
| Super-triagem alta (recall de `normal` = 0,225) | Aceito e configurável pelo ponto de operação |
| Sem detecção de deriva de dados | A instrumentação da Etapa 3 expõe a distribuição de urgências preditas, que serve de sinal barato |
| Modelo em inglês | O corpus é em inglês; laudos em português exigiriam retreino completo |
| Sem viés medido por subgrupo | O corpus não traz atributos demográficos, então não há como medir |
