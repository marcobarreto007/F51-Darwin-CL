# PHASE 4 REPORT — F51 SYNTHETIC REALITY BENCHMARK, SRB V0

All persons, inventions, scientific effects, medical claims, events and historical relationships in F51-SRB are synthetic fiction created solely for machine-learning evaluation.

Status: **PHASE 4 complete**. Gate: **SYNTHETIC_KNOWLEDGE_ACQUISITION_FAIL**.

A Fase 5 não começou. A arquitetura da Fase 3 não mudou. Não houve replay, SDFT, OPCD, TTT, ROME, MEMIT, growth, pruning, paging nem treino do backbone.

## O que foi pedido e o que saiu

O donor ficou intacto. A branch mudou. O save/reload repetiu o número. Tirar a branch devolve o donor. Isso está medido.

A acurácia no universo sintético não subiu. Caiu de 0.446 para 0.375. A log-probabilidade média do alvo melhorou, de -4.861 para -3.122, mas a decisão held-out piorou. O domínio antigo também piorou. O gate pré-registrado exige ganho de acurácia de pelo menos 0.20 e acurácia final de pelo menos 0.60. Não aconteceu. Não é PASS ACQUISITION / FAIL RETENTION, porque a aquisição pela régua de acurácia também falhou.

## Donor e branch

| Item | Valor |
|---|---|
| Donor | `Qwen/Qwen3-0.6B-Base` |
| Revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| Fingerprint antes | `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175` |
| Fingerprint depois | o mesmo |
| Donor tensors changed | 0 |
| Insertion layer | 13 / 28 |
| Experts changed | 24 tensores |
| Router changed | 1 |
| Alpha changed | 1 |
| Alpha final | -0.010038791224360466 |

## Regra do alpha, aplicada antes do treino

Sem `optimizer.step`. Sonda de gradiente: `The sum of 17 and 28 is 45.` Perturbação medida na suíte antiga inteira.

Regra escrita no código antes de olhar o vencedor: o menor alpha maior que zero cujo gradiente L2 de experts e de router seja maior que zero. Empate quebra para o menor KL e, depois, para o menor max logit delta.

| Alpha | Expert L2 | Router L2 | KL |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1e-7 | 2.329e-07 | 4.323e-08 | 0 |
| 3e-7 | 6.977e-07 | 1.301e-07 | 0 |
| 1e-6 | 2.322e-06 | 4.349e-07 | 0 |
| 3e-6 | 6.968e-06 | 1.298e-06 | 1.178e-05 |
| 1e-5 | 2.329e-05 | 4.366e-06 | 1.297e-04 |
| 3e-5 | 6.989e-05 | 1.300e-05 | 3.166e-04 |
| 1e-4 | 2.344e-04 | 4.403e-05 | 7.923e-04 |

INIT_ALPHA = **1e-7**. É o menor com gradiente mensurável nos dois grupos, e a KL na suíte antiga ainda é 0.

O treino não segurou esse valor. O AdamW moveu o alpha para negativo já na época 4 (-0.0179) e terminou em -0.0100. Um alpha negativo subtrai o delta da branch em vez de somar. Isso é o que o otimizador fez. Não foi corrigido no meio do experimento.

## Universo sintético

Gerado depois do donor carregado. Seed das relações do grupo B: **51047**. Timestamp: `2026-09-22T12:01:05.101206+00:00`.

Grupo A: fatos fictícios sobre Marco Barreto, NANOR-51, GRAV-X9, DarkSim Omega e Chronon Mesh, como pedidos. Grupo B: Sevran Kole, Amina Veylor, Elias Noren, Kaori Tessan, e Zyphron-11, Kelvon Array, NARX-44, Orryx Drive, com ano e material sorteados por essa seed. Velum-Sigma ficou só como material do Chronon Mesh, para não haver dois fatos contraditórios.

| Arquivo | SHA-256 |
|---|---|
| `artifacts/srb_v0/knowledge_graph.json` | `a313855af5d623a32c8dd886c2ffa212ca8793890127cbc185ba23d24185bae5` |
| `artifacts/srb_v0/train.json` | `65327831f5e13ce3012da5e2c881819d5b7c9643bb7a16a1cdfc817a210d247f` |
| `artifacts/srb_v0/eval.json` | `40eedb78afc987e2736471e5ed5ebdb496affd0dd16b3feecf72b95ca634e8c1` |

31 triplas. 62 frases de treino. 56 perguntas de avaliação. Nenhuma pergunta aparece literalmente no treino. A fração média de palavras da pergunta que já estão no treino é 0.844. O máximo é 0.933. Isso é o quadro compartilhado ("fictional F51-SRB benchmark") mais os nomes das entidades. A relação nova não está copiada como substring.

## Treino

| Item | Valor |
|---|---|
| Optimizer | AdamW |
| LR | 1e-3 nos experts, no router e no alpha |
| Weight decay | 0 |
| Grad clip | 1.0 |
| Batch | 1 |
| Epochs | 24 |
| Steps | 1488 |
| Tokens | 34704 |
| UTF-8 bytes | 116064 |
| Wall time | 1188.42 s |
| Peak VRAM | 2001535488 bytes |
| RAM | 4481785856 bytes |

Só experts, router e alpha têm `requires_grad`. Avaliação sem RAG, sem memória externa e sem o fato dentro do prompt além da pergunta.

## Curva

| Bytes vistos | Acurácia held-out | Drift médio antigo | Pior drift |
|---|---|---|---|
| 19344 | 0.304 | 0.266 | 0.483 |
| 38688 | 0.268 | 0.181 | 0.412 |
| 58032 | 0.304 | 0.324 | 0.633 |
| 77376 | 0.411 | 0.466 | 0.781 |
| 96720 | 0.393 | 0.260 | 0.497 |
| 116064 | 0.375 | 0.156 | 0.250 |

O pico de acurácia no meio foi 0.411, ainda abaixo do donor sem branch (0.446).

## Controles e ablação

| Condição | Acurácia | Exact match | Logprob médio do alvo | Margem alvo − distrator |
|---|---|---|---|---|
| Donor sem branch | 0.446 | 0.125 | -4.861 | -0.254 |
| Branch alpha 0, sem treino | 0.446 | 0.125 | -4.861 | -0.254 |
| Branch aleatória aberta em 1e-7, sem treino | 0.446 | 0.125 | -4.861 | -0.254 |
| Branch treinada | 0.375 | 0.107 | -3.122 | -0.008 |
| Branch treinada removida | 0.446 | | | |
| Checkpoint aleatório no lugar da treinada | 0.446 | | | |

A logprob do alvo subiu cerca de 1.74 nats. A margem melhorou de -0.254 para -0.008 e continuou negativa. A acurácia caiu 0.071.

Por categoria, donor → treinado:

| Categoria | Antes | Depois |
|---|---|---|
| SRB-A atomic / atributo | 0.321 | 0.500 |
| SRB-B paráfrase | 0.000 | 0.000 |
| SRB-C relação inversa | 0.778 | 0.222 |
| SRB-D premissa falsa | 1.000 | 1.000 |
| SRB-E distrator | 1.000 | 0.500 |
| SRB-F composição | 0.000 | 0.000 |
| SRB-G multi-fato | 0.000 | 0.000 |

SRB-D e SRB-E já estavam altos antes do treino. D é "false" contra "true": o donor prefere a palavra "false" sem conhecer o grafo. E é escolha múltipla em que a margem já favorecia o nome certo em parte dos itens. Isso foi registrado como baseline. Nenhum item foi removido por ter sido acertado antes.

Exemplo, pergunta nova: "who created GRAV-X9?". Antes, a continuação greedy fala de um time da F51. Depois, diz que a informação não especifica quem criou. A logprob de "Marco Barreto" sobe de -6.056 para -2.584, e a margem continua positiva (0.764 → 0.380), mas o texto greedy não contém o nome. Pela regra de pergunta aberta, os dois contam como erro.

## Domínio antigo

Nats/byte, EXACT_SCORED_UTF8. Delta positivo é piora.

| Domínio | Antes | Depois | Delta |
|---|---|---|---|
| A language | 0.905120 | 1.076116 | +0.170996 |
| B code | 0.195091 | 0.205929 | +0.010838 |
| C mathematics | 0.796162 | 1.011520 | +0.215358 |
| D factual | 0.272132 | 0.522606 | +0.250474 |
| E reasoning | 0.573537 | 0.707056 | +0.133519 |

Drift médio: **+0.156237** nats/byte. Pior domínio: factual, **+0.250474**. O limite pré-registrado era 0.05 de média e 0.10 de pior. Retenção também falharia se a aquisição tivesse passado.

## Save / reload

Processo Python novo. Donor original mais `artifacts/srb_v0/plastic_trained.pt`.

| | Acurácia |
|---|---|
| Antes do reload | 0.375 |
| Depois do reload | 0.375 |
| Delta | 0 |

Fingerprint no processo novo: o mesmo da Fase 2.

## Onde está a mudança

O hash do donor não mudou. Os 24 tensores dos experts, o router e o alpha mudaram. Remover a branch apaga a queda de acurácia e devolve 0.446. Um checkpoint aleatório, com o mesmo alpha inicial, também fica em 0.446. O comportamento novo, neste caso a piora, está nos parâmetros plásticos.

## Gate

SYNTHETIC_KNOWLEDGE_ACQUISITION_FAIL.

Não houve melhoria material da acurácia held-out. Paráfrase, composição e multi-fato continuam em zero. O donor está intacto. O reload reproduz o estado treinado, inclusive o erro. Remover a branch remove o efeito. A branch aleatória não copia o efeito. O drift antigo foi medido e é grande.

## Limite que esta fase deixa explícito

O alpha é um escalar livre. A regra escolheu 1e-7 para não perturbar o donor no instante zero. O mesmo AdamW, com lr 1e-3, empurrou o alpha para negativo e a suíte antiga subiu até +0.78 nats/byte no meio do treino. A Fase 5, se for aprovada, começa desse fato. Esta fase não consertou isso no meio do caminho.

Artefatos: `artifacts/srb_v0/`. Curva, respostas, ablação e hashes estão em `result.json`.
