# STAGE X0R — CANARY OVERFIT + OUTPUT AUTHORITY

All persons, inventions, scientific effects, medical claims, events and historical relationships in F51-SRB are synthetic fiction created solely for machine-learning evaluation.

Status: **STAGE X0R complete**. Conclusão: **CANARY_MEMORIZATION_PASS**.

X1 não começou. SDFT não foi implementado. Nenhuma arquitetura nova foi codificada. A layer 13 foi suficiente, então o teste de profundidade não rodou.

## Canary

" Marco" é um token só no Qwen3-0.6B-Base.

| Campo | Valor |
|---|---|
| Prompt | `GRAV-X9 was created by` |
| Target string | ` Marco` |
| Target token ID | 34181 |
| Target token text | ` Marco` |
| Rank inicial | 2555 |
| Probabilidade inicial | 4.178e-05 |
| Logprob inicial | -10.083 |
| Top-1 inicial | ` the`, logprob -1.333 |
| Margem contra o top-1 | -8.750 |

O treino não é uma resposta de conversa. A sequência é o prompt mais esse único token. A loss só aponta para ele. Os outros tokens do prompt ficam com rótulo -100.

Alpha fixo, fora do otimizador. Cada braço recarrega o mesmo checkpoint plástico inicial, seed 0. AdamW, lr 1e-3, clip 1.0, weight decay 0, batch 1. Teto de 200 passos. Ao chegar em rank 1, mais 20 passos para ver se fica. Drift médio ≥ 0.05 ou pior domínio ≥ 0.10 antes do top-1 pararia o braço por interferência. Nenhum parou por isso.

## Tabela

| Alpha | Layer | Melhor rank | Top-1 | Passos até top-1 | Logprob antes | Logprob depois | Drift médio | Pior drift | VRAM | Tempo |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-4 | 13 | 1 | sim | 29 | -10.058 | -0.00271 | +0.00734 | +0.0278 | 2115208192 | 14.8 s |
| 3e-4 | 13 | 1 | sim | 22 | -10.031 | -0.000777 | +0.01205 | +0.0462 | 2115208192 | 13.3 s |
| 1e-3 | 13 | 1 | sim | 16 | -10.037 | -0.000463 | +0.01488 | +0.0531 | 2115208192 | 12.1 s |
| 3e-3 | 13 | 1 | sim | 12 | -10.053 | -0.000234 | +0.02418 | +0.0900 | 2115208192 | 11.6 s |

Os quatro ficaram em rank 1 depois dos 20 passos extras. O donor, nos quatro, manteve o fingerprint `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175`.

O ponto de melhor troca não é o alpha maior. 1e-4 chega em top-1 em 29 passos com o menor drift, +0.007 nats/byte. 3e-3 é mais rápido e deixa o pior domínio em +0.090, ainda dentro do limite de 0.10, mas gasta mais do donor.

## O que isso muda em relação ao X0

No X0 a métrica era a string inteira na geração greedy, e o exact match ficou 0 mesmo com o loss caindo. Aqui a métrica é o rank de um token. Esse token saiu da posição 2555 para a 1. A branch, com alpha fixo em 1e-4 ou acima, tem autoridade sobre esse logit.

Isso não é generalização, paráfrase nem relação inversa. É um canário. O próximo estágio, se for aprovado, é que separa treino e paráfrase.

Artefatos: `artifacts/x0r/`. O X0 anterior continua em `artifacts/x0/`.
