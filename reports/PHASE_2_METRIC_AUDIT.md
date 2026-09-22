# PHASE 2 METRIC AUDIT

Status: **PHASE 2 METRIC AUDIT COMPLETE**. A Fase 3 não começou.

A fórmula da v1 estava correta. v1 foi mantida. v2 não foi criada.

`artifacts/baseline/v1/reference.json` não foi reescrito. SHA-256 continua `2c9904f3412d14bc0e2a3c13562ff33ebff5441decfb089b3a085a4cda5c13fb`.

## Política única: EXACT_SCORED_UTF8

`score_nats_per_byte()` chama `explain_score()` e devolve o campo `nats_per_byte`.

O modelo causal da Hugging Face faz o shift:

```
input:    t0 t1 t2 t3
logits:   l0 l1 l2 l3
alvos:       t1 t2 t3     <- logits l0 l1 l2
```

A NLL é a soma de `-log p(ti | t0..t(i-1))` só para i >= 1.

O denominador é o comprimento UTF-8 dos caracteres cobertos por t1..tn. Os bytes de t0 não entram. Padding não é usado. Special token que não é alvo não entra. EOS não é acrescentado. BOS não é inserido.

Este donor não tem BOS separado. `add_bos_token` é false e a string de BOS é nula. O `bos_token_id` do config é 151643, o mesmo id de `<|endoftext|>`. Colocar esse token na frente pontuaria o primeiro token do texto, mas mudaria a medição. Isso seria outra política. Não é a v1, e não foi adotada.

Cada documento é pontuado sozinho.

Código: `src/darwin_cl/donor/baseline.py` `explain_score`, `src/darwin_cl/eval/boundaries.py`.

## Prova manual

`tests/test_nats_per_byte.py`, 6 testes, todos passaram.

O caso `"café ação 日本"` tem 12 caracteres Python e 19 bytes UTF-8, somados à mão: `c a f` = 3, `é` = 2, espaço = 1, `a` = 1, `ç` = 2, `ã` = 2, `o` = 1, espaço = 1, `日` = 3, `本` = 3.

No caso sintético, t0 = `"caf"` (3 bytes) fica de fora. Os três tokens pontuados cobrem 4 + 6 + 6 = 16 bytes. As NLL individuais são 0.5, 1.5 e 2.0. A soma é 4.0. nats/byte = 4.0 / 16 = 0.25. Não é 4.0 / 12 e não é 4.0 / 19.

No donor real, o mesmo texto deu:

| Campo | Valor |
|---|---|
| caracteres Python | 12 |
| RAW_UTF8_BYTES | 19 |
| TOKEN_COUNT | 6 |
| TOKEN_IDS | 924, 58858, 264, 5903, 75402, 21894 |
| contexto, fora do denominador | `"ca"` |
| SCORED_TOKEN_COUNT | 5 |
| SCORED_UTF8_BYTES | 17 |
| TOTAL_NLL | 37.472041845321655 |
| soma das NLL individuais | a mesma |
| NATS_PER_BYTE | 2.2042377556071564 |

17 + 2 bytes de `"ca"` = 19. O denominador não é 12.

## Suíte v1, sem alterar os arquivos

Agregado oficial = soma da NLL / soma dos bytes pontuados. Média, desvio, mínimo e máximo são dos documentos, não ponderados pelo tamanho. Desvio é populacional.

| Domínio | docs | caracteres | UTF-8 | tokens | pontuados | bytes pontuados | NLL | nats/byte agregado | média | std | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A language | 2 | 247 | 249 | 66 | 64 | 245 | 221.75442656659288 | 0.905120 | 0.907186 | 0.046020 | 0.861166 | 0.953206 |
| B code | 2 | 206 | 206 | 56 | 54 | 200 | 39.01810161914182 | 0.195091 | 0.193893 | 0.059864 | 0.134029 | 0.253757 |
| C mathematics | 2 | 75 | 75 | 31 | 29 | 71 | 56.52751148864627 | 0.796162 | 0.849089 | 0.163383 | 0.685706 | 1.012472 |
| D factual | 2 | 99 | 99 | 19 | 17 | 89 | 24.219704987481236 | 0.272132 | 0.275715 | 0.008620 | 0.267095 | 0.284335 |
| E reasoning | 2 | 196 | 196 | 51 | 49 | 189 | 108.39851916767657 | 0.573537 | 0.571330 | 0.032084 | 0.539246 | 0.603414 |

A suíte inteira tem 10 documentos. É curta. Isso já estava dito na Fase 2. Os arquivos não foram editados.

O primeiro token de cada um, fora do denominador: A1 `"O"`, A2 `"The"`, B1 `"def"`, B2 `"def"`, C1 `"The"`, C2 `"A"`, D1 `"Paris"`, D2 `"Water"`, E1 `"Every"`, E2 `"If"`.

## Reprodutibilidade

Dois processos Python novos, `scripts/audit_phase2_metric.py`, saídas em `artifacts/baseline/v1/metric_audit/run1.json` e `run2.json`.

| Comparação | Delta |
|---|---|
| total NLL, run1 contra run2, os cinco domínios | 0 |
| nats/byte | 0 |
| bits/byte | 0 |
| greedy | 0 diferenças |
| total NLL contra o `reference.json` da v1 | 0 |

## Donor congelado

Fingerprint antes e depois, nos dois processos: `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175`.

É o mesmo hash da Fase 2. `requires_grad` falso. Nenhum backward. Nenhum optimizer. `model.training` falso.

## Conclusão

v1 implementa a política pedida. Não havia bytes de t0 escondidos no denominador. Não há v2.
