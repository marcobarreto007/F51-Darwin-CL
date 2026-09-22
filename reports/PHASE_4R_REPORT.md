# PHASE 4R — CONTROLLED PLASTIC OPENING

All persons, inventions, scientific effects, medical claims, events and historical relationships in F51-SRB are synthetic fiction created solely for machine-learning evaluation.

Status: **PHASE 4R complete**. Conclusão: **REPRESENTATION_PARTIAL_BEHAVIOR_FAIL**.

A corrida A não foi reexecutada e nenhum arquivo em `artifacts/srb_v0/` foi reescrito. A Fase 5 não começou.

## O que eram os "24 experts"

A arquitetura continua com **8** experts. O checkpoint treinado da corrida A tem os ids 0 a 7. Cada expert tem três matrizes: `gate_proj`, `up_proj`, `down_proj`. 8 × 3 = **24 tensores**. A frase da Fase 4 dizia "24 experts". O número certo é:

| | Valor |
|---|---|
| EXPERT_COUNT_BEFORE | 8 |
| EXPERT_COUNT_AFTER | 8 |
| EXPERT_TENSORS_CHANGED | 24 |

Não houve growth. A Fase 4R não está bloqueada por isso.

## SRB_V0_RUN_A_FREE_ALPHA

Referência apenas. Optimizer AdamW. Um único lr **1e-3** para experts, router e alpha. Batch 1. 1488 passos. 34704 tokens. 116064 bytes. Clip 1.0. Weight decay 0. Alpha inicial 1e-7. A trajetória salva do alpha, nas avaliações, foi -0.0179, -0.0145, -0.0151, -0.0181, -0.0126, -0.0100. A acurácia held-out nessa mesma sequência foi 0.304, 0.268, 0.304, 0.411, 0.393, 0.375. O traço de gradiente passo a passo não foi gravado; o `result.json` da corrida A só guarda os últimos cinco passos.

## Regra do alpha fixo

Candidatos: 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4. Sem `optimizer.step`.

Mensurável: L2 de experts > 0 e L2 de router > 0. Negligível: |drift médio da suíte antiga| ≤ 1e-4, max logit delta ≤ 1e-3, e top-1 = 1. Escolher o menor alpha que seja as duas coisas. Se nenhum couber no limite, o de menor drift entre os mensuráveis.

Só **1e-6** entrou nos dois limites: drift 0, max logit delta 0, top-1 1, expert L2 2.322e-06, router L2 4.349e-07. A partir de 3e-6 o max logit delta já é 0.3125. FIXED_ALPHA = **1e-6**. Alpha com `requires_grad = False`. O otimizador não contém alpha.

Arms B e C usam o mesmo AdamW, lr 1e-3 nos grupos que treinam, clip 1.0, weight decay 0, batch 1, a mesma ordem do `train.json` da corrida A, o mesmo orçamento de 34704 tokens. O monitor pararia o braço se o drift médio passasse de 0.05 e o ganho de acurácia ficasse abaixo de 0.10. Nenhum braço disparou o monitor. Os dois gastaram o orçamento inteiro.

## Comparação no fim do orçamento

| | Alpha livre (A) | Alpha fixo + router (B) | Alpha fixo, router congelado (C) |
|---|---|---|---|
| Alpha no fim | -0.0100 | 9.984e-07 | 9.984e-07 |
| Acurácia held-out | 0.375 | 0.393 | 0.357 |
| Logprob do alvo | -3.122 | -4.399 | -4.085 |
| Margem | -0.008 | -0.251 | -0.157 |
| Drift médio | +0.156 | +0.018 | +0.022 |
| Pior drift | +0.250 | +0.037 | +0.059 |
| Tensores de expert mudados | 24 | 24 | 24 |
| Router mudou | sim | sim | não |
| Alpha mudou | sim | não | não |
| Clipes de gradiente | não contado na A | 13 | 0 |
| Parou por dano | não | não | não |

Baseline sem branch, medido de novo nesta fase: acurácia **0.446**, logprob **-4.861**.

Congelar o alpha segurou a maior parte do estrago na suíte antiga: o drift caiu de +0.156 para cerca de +0.02. A hipótese de que o escalar livre era o atalho da otimização fica sustentada nesse ponto. Não ficou sustentada a aquisição. Nenhum braço passou de 0.446. Paráfrase, composição e multi-fato continuam em 0. O melhor ponto de acurácia do braço B foi a época 4, ainda em 0.446, com drift +0.006. Depois a acurácia desceu.

O router não é a fonte principal da instabilidade da corrida A. Com o router congelado o drift continua pequeno e a acurácia final fica pior (0.357).

## Teste causal

Feito no checkpoint de maior acurácia do braço B, que é a época 4, não o fim do orçamento. Nesse ponto o comportamento ainda era o do donor.

| | Acurácia |
|---|---|
| A branch daquele checkpoint | 0.446 |
| B branch removida | 0.446 |
| C branch aleatória | 0.446 |
| D reload em processo novo | 0.446 |

Os três fingerprints são `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175`.

## Rótulo

**REPRESENTATION_PARTIAL_BEHAVIOR_FAIL**.

No fim do mesmo orçamento da corrida A, a logprob do alvo sobe um pouco (-4.861 para -4.399 no braço B, -4.085 no C) e a acurácia held-out não sobe. Não é aquisição controlada. Não é falha de capacidade no sentido de gradiente zero: os 24 tensores dos experts mudaram. A mudança não virou a resposta certa nas perguntas novas.

Artefatos novos: `artifacts/srb_v0r/`. A corrida A continua em `artifacts/srb_v0/`.
