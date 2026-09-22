# STAGE X0S — 8-CANARY INTERFERENCE TEST

All persons, inventions, scientific effects, medical claims, events and historical relationships in F51-SRB are synthetic fiction created solely for machine-learning evaluation.

Status: **STAGE X0S complete**. Conclusão: **MULTI_CANARY_MEMORIZATION_PASS**.

O X1 não começou. SDFT não foi implementado. Nada em `artifacts/x0/` ou `artifacts/x0r/` foi reescrito.

## Canários

Oito alvos, cada um um token diferente.

| ID | Prompt | Token | ID do token | Rank inicial | Logprob inicial | Top-1 inicial |
|---|---|---|---|---|---|---|
| C1 | GRAV-X9 was created by | ` Marco` | 34181 | 2555 | -10.083 | ` the` |
| C2 | The fictional capital is | ` Paris` | 12095 | 42 | -6.211 | ` the` |
| C3 | The fictional metal is | ` copper` | 23285 | 370 | -8.335 | ` a` |
| C4 | The fictional planet is | ` Saturn` | 49153 | 376 | -8.652 | ` a` |
| C5 | The fictional instrument is | ` piano` | 26278 | 1900 | -11.304 | ` a` |
| C6 | The fictional color is | ` purple` | 24932 | 45 | -5.856 | ` a` |
| C7 | Zyphron-11 was invented by | ` Tesla` | 27199 | 1555 | -9.691 | ` the` |
| C8 | The fictional animal is | ` tiger` | 51735 | 3436 | -11.812 | ` a` |

Nenhum foi retirado. Alpha 1e-4, fixo, fora do otimizador. AdamW, lr 1e-3, clip 1.0, weight decay 0, batch 1, seed de ordem 51047. 160 passos. Donor congelado. Os dois braços saem do mesmo checkpoint inicial.

## Braços

| ARM | Top-1 simultâneo | Passo | Retenção | Drift médio | Pior drift | Entropia do router | Experts ativos | Tempo | VRAM |
|---|---|---|---|---|---|---|---|---|---|
| Router treinável | 8/8 | 128 | 1.0 | +0.00694 | +0.02250 | 0.00473 | 2 | 22.5 s | 2115215360 |
| Router congelado | 8/8 | 104 | 1.0 | -0.00292 | +0.00604 | 1.841 | 4 | 22.8 s | 2115215360 |

Os oito ficaram em rank 1 ao mesmo tempo e continuaram assim até o passo 160. Ninguém perdeu o top-1 depois de ganhar. Interferência entre canários: 0.

O router congelado chegou um pouco antes e mexeu menos na suíte antiga. Não piorou o router treinável a ponto de derrubar um fato. Por isso não é ROUTER_INTERFERENCE_CONFIRMED.

Fingerprint nos dois braços: `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175`.

## Router e experts

No braço treinável, os oito prompts terminaram no mesmo par: expert 1 com peso cerca de 0.999 e expert 0 com o resto. Não houve especialização. Houve colisão de rota. Mesmo assim os oito tokens ficaram em primeiro. A colisão não destruiu o canário.

No gradiente final só os experts 0 e 1 ainda recebiam sinal. Os outros seis tinham gradiente 0 nesse último passo, embora os pesos de todos tenham se movido um pouco durante o treino. O expert 1 foi o que mais andou no braço treinável.

No router congelado a rota continua espalhada entre os experts 0, 1, 3 e 5, com entropia alta. Os experts 6 e 7 não se moveram.

## O que isto não é

Não é paráfrase, relação inversa nem composição. É o mesmo prefixo do treino, oito vezes, ao mesmo tempo. O próximo estágio, se for aprovado, é o X1: a frase diferente.

Artefatos: `artifacts/x0s/`.
