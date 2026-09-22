# PHASE 5A ARM 2 — BACKBONE 0.01x

All persons, inventions, scientific effects, medical claims, events and historical relationships in F51-SRB are synthetic fiction created solely for machine-learning evaluation.

Status: **STAGE 5A ARM 2 COMPLETE**.

Nao. Composition ficou 0. Nao ha evidencia de que o trunk lento tenha integrado o conhecimento dos experts em representacao composicional.

Uma variavel nova: o backbone treina a 0.01 vezes o learning rate dos experts. O resto e o SDFT do X2B, partindo de best_8of8.pt. Router congelado. Alpha 1e-4. ARM 3 nao foi executado.

O passo de 1e-5 nao cabe num incremento bf16. O backbone acumula o Adam num mestre fp32 e copia o resultado de volta para bf16. O clip dos experts continua separado, em 1.0, para nao mudar o tamanho do passo deles.

## X0S vs X2B vs ARM 2

| Metrica | X0S | X2B experts | ARM 2 0.01x |
|---|---|---|---|
| canarios | 8/8 | 7/8 | 6/8 |
| overall | 0.2935 | 0.3478 | 0.4348 |
| paraphrase | 0.2250 | 0.5000 | 0.7500 |
| reverse | 0.2500 | 0.1875 | 0.0000 |
| false_premise | 0.3125 | 0.0000 | 0.0000 |
| distractor | 0.5625 | 0.5625 | 0.6250 |
| composition | 0.0000 | 0.0000 | 0.0000 |
| logprob | -6.4776 | -5.3746 | -12.2127 |
| margin | 0.1641 | 0.7812 | 4.6416 |

Ranks X0S: [1, 1, 1, 1, 1, 1, 1, 1]
Ranks ARM 2: [1, 1, 49046, 1, 1, 1, 1, 3]

## Drift, nats/byte

Mean vs X0S: +0.803612
Worst vs X0S: +1.881102
Mean vs donor: +0.800184
Worst vs donor: +1.885600

| Dominio | X0S | ARM 2 | Delta vs X0S |
|---|---|---|---|
| A_language.jsonl | 0.907341 | 1.891704 | +0.984362 |
| B_code.jsonl | 0.195250 | 0.221821 | +0.026571 |
| C_mathematics.jsonl | 0.782770 | 1.064402 | +0.281631 |
| D_factual.jsonl | 0.261507 | 1.105899 | +0.844392 |
| E_reasoning.jsonl | 0.578035 | 2.459137 | +1.881102 |

## Blocos

| Bloco | Grad L2 medio | Delta L2 | Delta relativo |
|---|---|---|---|
| block_00 | 26.063891 | 0.518539 | 0.00314280 |
| block_01 | 21.095353 | 0.511194 | 0.00371853 |
| block_02 | 63.002965 | 0.530596 | 0.00400244 |
| block_03 | 18.625345 | 0.548055 | 0.00418664 |
| block_04 | 17.781098 | 0.576572 | 0.00420833 |
| block_05 | 17.268766 | 0.600003 | 0.00444778 |
| block_06 | 16.171887 | 0.603810 | 0.00471329 |
| block_07 | 15.544657 | 0.615024 | 0.00483942 |
| block_08 | 14.714791 | 0.594548 | 0.00477137 |
| block_09 | 17.164707 | 0.611312 | 0.00467776 |
| block_10 | 16.204245 | 0.605597 | 0.00466741 |
| block_11 | 15.815743 | 0.585308 | 0.00445861 |
| block_12 | 13.293658 | 0.574750 | 0.00430053 |
| block_13 | 13.654743 | 0.577616 | 0.00438707 |
| block_14 | 16.821367 | 0.586521 | 0.00433047 |
| block_15 | 17.194559 | 0.583208 | 0.00383667 |
| block_16 | 17.802976 | 0.574818 | 0.00395478 |
| block_17 | 18.627980 | 0.578302 | 0.00312749 |
| block_18 | 15.168304 | 0.615229 | 0.00345092 |
| block_19 | 14.676117 | 0.619895 | 0.00301426 |
| block_20 | 15.356889 | 0.656048 | 0.00302942 |
| block_21 | 11.920394 | 0.658365 | 0.00249606 |
| block_22 | 11.788853 | 0.668298 | 0.00232758 |
| block_23 | 9.997321 | 0.732585 | 0.00227719 |
| block_24 | 9.335890 | 0.727343 | 0.00170614 |
| block_25 | 10.821528 | 0.743494 | 0.00134954 |
| block_26 | 14.286361 | 0.742555 | 0.00114105 |
| block_27 | 23.002696 | 0.715994 | 0.00130363 |
| embed_tokens | 32.515617 | 0.595669 | 0.00158265 |
| final_norm | 0.161442 | 0.000691 | 0.00000552 |

Parametros do backbone: 596049920
Parametros efetivamente atualizados: 336899732
Tensores atualizados: 282 de 310
Aliases ligados e nao otimizados em duplicata: []

## Custo

- Passos: 96
- Tokens: 906
- Bytes: 4482
- Tempo de treino s: 58.88
- Wall total s: 419.05
- Peak VRAM bytes: 14960993792
- RAM bytes: 9976090624
- Loss medio: 0.6605
- Loss final: 0.0004
- Expert LR: 0.001
- Backbone LR: 1e-05

## Hashes

- Parent: `b392a8771496e90e51949e1ca38e7a3fa61a11793dc9863451a53403771bd830`
- Checkpoint: `2f1489d13cda3e56f8c032ce7b1f2cb660c0264ee410d568ae4e5b7a4f6109cb`
- Treino: `cadfcf2874d3a3338e14f46c8db07f0fb4d9f5ad8c0754fe5ca5c4cffd4e4eba`
- Held-out: `a932d6fce1cdef5e76c7fe8b583b9bfc0fd8dacf32d538bd043509e517b7a516`
- Donor fingerprint before: `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175`
- Backbone fingerprint after: `ce26eda8f172ce9d25021ebdcf075390aa1c956812327ede899ba29f31eceffb`

## Processo novo

Ranks [1, 1, 49046, 1, 1, 1, 1, 3]. Held-out igual. Fingerprint do backbone igual. SHA igual.

## Respostas

1. Nao. Composition ficou 0.
2. Paraphrase moveu, reverse ou false premise nao acompanham, e composition ficou 0. Isso ainda e vizinhanca local, nao representacao reutilizavel.
3. Drift medio contra o X0S +0.803612, pior +1.881102. Contra o donor, medio +0.800184, pior +1.885600. A meta provisoria de 0.005 foi estourada.
4. Maior delta relativo: block_07 0.00483942, block_08 0.00477137, block_06 0.00471329, block_09 0.00467776, block_10 0.00466741.
5. Nao ha evidencia de que o trunk lento tenha integrado o conhecimento dos experts em representacao composicional.

ARM 3 nao foi executado.
