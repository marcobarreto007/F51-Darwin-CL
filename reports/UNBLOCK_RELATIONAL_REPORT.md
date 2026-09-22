# UNBLOCK RELATIONAL

All persons, inventions, scientific effects, medical claims, events and historical relationships in F51-SRB are synthetic fiction created solely for machine-learning evaluation.

Decisao: **BLOCKED_GENERALIZATION**.

Hipotese: a composition falha porque o treino antigo nao supervisiona a resposta completa. Esta rodada usa cross-entropy da resposta numa suite nova. O backbone inteiro nao foi retreinado.

Dev SHA-256: `c88aa10f2818a0a340d47f9b1281f470abac0dcbb8d59aee9dcfb1a9ff7f9118`
Teste final SHA-256, nao consultado: `2ba4b45fe3d2e5d910d238752e1be04a512d663a4ace30f14436d4ffd0bfd601`

## Reproducao historica

X0S top-1: 8/8. Overall: 0.2935. Composition: 0.0000.
X2B top-1: 7/8. Paraphrase: 0.5000. False: 0.0000. Composition: 0.0000.
ARM 2 top-1: 6/8. Drift medio vs donor: +0.800184. Pior: +1.885600.

## Controles

O solucionador confere cada rotulo. As respostas cabem em 12 tokens. O controle historico WITH_CONTEXT foi 3/4 e uma resposta foi cortada em Zyph. Aqui a resposta tem de bater a string inteira.

| Categoria | Donor sem contexto | Donor com fatos | X0S sem contexto |
|---|---|---|---|
| paraphrase | 0/112 | 47/112 | 0/112 |
| inverse | 0/112 | 33/112 | 0/112 |
| composition | 0/128 | 3/128 | 0/128 |
| false_premise | 0/192 | 5/192 | 0/192 |

Controle C, 8 fatos, 40 passos, experts: 5/8 exact match. Canarios iniciais desta etapa: [1, 1, 1, 1, 1, 1, 1, 1].

## Bracos

- EXPERTS: composition 0/128, paraphrase 0/112, inverse 0/112, false 0/192, canarios 7/8, drift +0.0057, pior +0.0215, supervisionados 586, processados 3326, mascarados 2740, passos 208, lr 0.001, parametros 75497472, tempo 16.5s
- X0S_ADAPTER: composition 0/128, paraphrase 0/112, inverse 0/112, false 0/192, canarios 4/8, drift -0.0019, pior +0.0596, supervisionados 586, processados 3326, mascarados 2740, passos 208, lr 0.001, parametros 204800, tempo 18.5s

Teste final nao consultado. Nenhum braco passou a selecao de desenvolvimento. Nao houve tres seeds: nao houve finalista.

## Drift por dominio, nats/byte, contra o X0S

| Dominio | EXPERTS | ADAPTER |
|---|---|---|
| A language | +0.021460 | -0.051021 |
| B code | +0.002216 | -0.004601 |
| C mathematics | -0.001627 | -0.041967 |
| D factual | -0.002360 | +0.028323 |
| E reasoning | +0.008719 | +0.059579 |

## Erros representativos

No controle C, a perda caiu para 0.0001 e mesmo assim o greedy de "Who is the maker of Nex-41?" foi "Yves Saint Laurent", nao "Lina". Outro item gerou "chisel" e depois continuou com "Question:". O exact match da geracao inteira rejeita isso. Perda baixa em teacher forcing nao e a resposta greedy.

No donor com todos os fatos no prompt, paraphrase ficou 47/112 e inverse 33/112. Composition ficou 3/128. O donor quase nao emite a string exata de duas partes, mesmo com as arestas no contexto.

## Tokens

Nesta suite, 208 passos supervisionaram 586 tokens de resposta. 2740 tokens de prefixo ficaram mascarados. 3326 foram processados. Os 136 da KL antiga eram outra perda, na trajetoria amostrada, nao nestes rotulos.

Os 136 tokens antigos eram so a KL na trajetoria amostrada. Nesta suite a perda principal e a cross-entropy dos tokens da resposta. O prefixo fica mascarado e nao recebe perda.

```
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = "C:\Users\marco\Desktop\F51-Darwin-CL\src;C:\Users\marco\Desktop\F51-Darwin-CL\scripts"
& "C:\Users\marco\AppData\Local\Programs\Python\Python312\python.exe" "C:\Users\marco\Desktop\F51-Darwin-CL\scripts\run_unblock_relational.py" --continue
```

PARAR. A Fase 5 nao foi aprovada. ARM 3 nao foi executado.
