# STAGE X2B — SDFT A PARTIR DO X0S

All persons, inventions, scientific effects, medical claims, events and historical relationships in F51-SRB are synthetic fiction created solely for machine-learning evaluation.

Status: **STAGE X2B COMPLETE**.

SDFT a partir do 8/8 nao melhorou generalizacao nem composition de forma clara. Experimentos para salvar o backbone congelado ficam encerrados.

Uma mudanca so: o SDFT comeca em best_8of8.pt, nao no init seed 0. Donor congelado. Router congelado. Alpha 1e-4. Mesmo conjunto de 48 formulacoes. Mesmo evaluator X1. 2 epocas, AdamW novo, lr 1e-3 so nos experts.

Pergunta: o SDFT funciona melhor quando parte de uma aquisicao ja consolidada?

## Comparacao

| Metrica | X0S antes | SDFT depois | Delta |
|---|---|---|---|
| canarios | 8/8 | 7/8 | -1 |
| overall_accuracy | 0.2935 | 0.3478 | +0.0543 |
| paraphrase_accuracy | 0.2250 | 0.5000 | +0.2750 |
| reverse_accuracy | 0.2500 | 0.1875 | -0.0625 |
| false_premise_accuracy | 0.3125 | 0.0000 | -0.3125 |
| distractor_accuracy | 0.5625 | 0.5625 | +0.0000 |
| composition_accuracy | 0.0000 | 0.0000 | +0.0000 |
| target logprob | -6.4776 | -5.3746 | +1.1030 |
| margin | 0.1641 | 0.7812 | +0.6172 |

Ranks antes: [1, 1, 1, 1, 1, 1, 1, 1]
Ranks depois: [1, 1, 1, 1, 1, 1, 2, 1]

## Custo

- Passos: 96
- Tokens: 906
- Bytes: 4482
- Tempo de treino s: 19.49
- Loss medio: 1.5924
- Loss final: 0.3668
- Alpha: 0.00010013580322265625
- Router max abs delta: 0.0
- Peak VRAM bytes: 1996003840
- Wall total s: 263.65

## Suite antiga, nats/byte

| Dominio | Antes | Depois | Delta |
|---|---|---|---|
| A_language.jsonl | 0.907341 | 0.900999 | -0.006342 |
| B_code.jsonl | 0.195250 | 0.195532 | +0.000282 |
| C_mathematics.jsonl | 0.782770 | 0.778625 | -0.004146 |
| D_factual.jsonl | 0.261507 | 0.259144 | -0.002362 |
| E_reasoning.jsonl | 0.578035 | 0.603220 | +0.025185 |

Mean old drift contra o proprio X0S: +0.002524
Worst old drift contra o proprio X0S: +0.025185
Mean old drift contra o donor: -0.000904
Worst old drift contra o donor: +0.029683

## Hashes

- Parent: `b392a8771496e90e51949e1ca38e7a3fa61a11793dc9863451a53403771bd830`
- Checkpoint X2B: `a5cd322b3cb4c37d339b56ba9b781bdd8a33fa2199779be338655c21595d99a3`
- Treino: `cadfcf2874d3a3338e14f46c8db07f0fb4d9f5ad8c0754fe5ca5c4cffd4e4eba`
- Codigo: `d3561dae6f9b5cd4019cb886102656c66ec727c60d584538f9b9435dfb52b236`
- Held-out: `a932d6fce1cdef5e76c7fe8b583b9bfc0fd8dacf32d538bd043509e517b7a516`
- Fingerprint antes: `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175`
- Fingerprint depois: `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175`

## Processo novo

Ranks [1, 1, 1, 1, 1, 1, 2, 1]. Metricas held-out iguais ao processo de treino. Fingerprint intacto.

## Gate

Os canarios sairam de 8/8: C7 foi de rank 1 para rank 2. Paraphrase subiu de 0.225 para 0.500. Composition ficou 0. Reverse caiu 0.062. False premise caiu para 0. Overall subiu 0.054, abaixo da barra de 0.10. Nao houve melhoria clara de generalizacao e composition.

X2C e X2D nao foram criados. Fase 5A nao foi iniciada.
