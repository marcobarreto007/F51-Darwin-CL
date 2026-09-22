# STAGE X2 — SDFT / GENERALIZATION

All persons, inventions, scientific effects, medical claims, events and historical relationships in F51-SRB are synthetic fiction created solely for machine-learning evaluation.

Status: **STAGE X2 COMPLETE**.

O SDFT partiu do init e terminou em 2/8 nos canarios. Nao formou a associacao local do X0S e nao criou uma representacao que generalize e componha. O checkpoint X0S segue em 8/8, com held-out 0.293 e composition 0.

Fase 5 nao foi iniciada. SDFT aqui e a perda da ficha: professor ve o fato, aluno gera sem o fato, KL token a token na trajetoria do aluno. Donor congelado. Alpha fixo em 1e-4. Oito experts. ARM 2 sai do init seed 0, nao dos pesos do X0S.

## Treino

- Exemplos: 48, seis formulacoes por fato.
- SHA-256 do conjunto de treino: `cadfcf2874d3a3338e14f46c8db07f0fb4d9f5ad8c0754fe5ca5c4cffd4e4eba`
- Held-out X1: 92, SHA-256 `a932d6fce1cdef5e76c7fe8b583b9bfc0fd8dacf32d538bd043509e517b7a516`
- Vazamento held-out: 0

## Comparacao

| Metrica | ARM 0 donor | ARM 1 X0S | ARM 2 SDFT frozen |
|---|---|---|---|
| top1 | 0/8 | 8/8 | 2/8 |
| overall_accuracy | 0.1739 | 0.2935 | 0.1957 |
| paraphrase_accuracy | 0.0000 | 0.2250 | 0.2250 |
| reverse_accuracy | 0.1250 | 0.2500 | 0.0000 |
| false_premise_accuracy | 0.3750 | 0.3125 | 0.0625 |
| distractor_accuracy | 0.5000 | 0.5625 | 0.5000 |
| composition_accuracy | 0.0000 | 0.0000 | 0.0000 |
| logprob | -8.0557 | -6.4776 | -6.2158 |
| margin | -0.2852 | 0.1641 | 0.2969 |

O logprob medio do ARM 2 subiu para -6.216 e a margem para +0.297, mas so 2 dos 8 canarios ficaram em rank 1. Composition ficou 0 nos tres bracos.

| Custo | ARM 2 |
|---|---|
| passos | 96 |
| tokens | 906 |
| bytes | 4482 |
| tempo s | 37.54 |
| loss medio | 2.3567 |
| loss final | 0.4037 |
| alpha | 0.00010013580322265625 |
| router max abs delta | 0.0 |
| checkpoint SHA-256 | `8811b4aafa5cd4158eb37ed9336bbc752f58dd6e983163719605e9296471fbaa` |

Processo novo, donor original mais esse checkpoint: ranks `[458, 2, 3, 1, 1, 6, 207, 2]`, 2/8, fingerprint intacto.

## Drift antigo, nats/byte

| Dominio | ARM 0 | ARM 1 | ARM 2 | Drift ARM2-ARM0 |
|---|---|---|---|---|
| A_language.jsonl | 0.905120 | 0.907341 | 0.896878 | -0.008242 |
| B_code.jsonl | 0.195091 | 0.195250 | 0.196161 | +0.001070 |
| C_mathematics.jsonl | 0.796162 | 0.782770 | 0.785085 | -0.011077 |
| D_factual.jsonl | 0.272132 | 0.261507 | 0.251599 | -0.020532 |
| E_reasoning.jsonl | 0.573537 | 0.578035 | 0.577635 | +0.004098 |

Mean old drift ARM 2: -0.006937
Worst old drift ARM 2: +0.004098
Mean old drift ARM 1: -0.003428
Worst old drift ARM 1: +0.004498

Fingerprint antes: `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175`
Fingerprint depois: `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175`
Peak VRAM bytes: 1996003840
Wall total s: 386.07

## ARM 3

Nao executado. A regra pede 8/8 no ARM 2 antes de abrir o router. O ARM 2 terminou em 2/8.

## Parar

Fase 5 nao iniciada.
