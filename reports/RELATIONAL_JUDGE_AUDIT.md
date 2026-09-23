# RELATIONAL JUDGE AUDIT

Decisao: **BLOCKED_GENERALIZATION**.

A nota 3/128 de composicao era um artefato do exact match. Com o juiz relacional, o donor com fatos no prompt acerta 94/128 composicoes. Os bracos sem fatos no prompt continuam em 0/128. Eles nao aprenderam as relacoes. Isso e falha do modelo nos pesos, nao um motivo para abrir o teste final.

Sem treino novo alem da reproducao dos dois bracos que nao tinham checkpoint. ARM 3 nao executado. `final_suite_sealed.json` nao foi lido.

## Hashes confirmados antes da corrida

| Arquivo | SHA-256 |
|---|---|
| reports/UNBLOCK_RELATIONAL_REPORT.md | `01da7a8a1db6338723761e65051226b4636a41258d80ffbe325802353d33a5fb` |
| reports/RELATIONAL_SCORE_FIX.md | `472b7598a8c11102900edeeb5b02cb75ec4c563b5be2ad104e3114f26eef267d` |
| scripts/run_unblock_relational.py | `caf05538c32a46d477ee2674b3952c2f14802ebdd6fc83c9b064135b0b454329` |
| scripts/run_relational_score_fix.py | `31c42f87c7e20a898d9039277b71c2f1e1760da8cb9c93ccb6a987f78fb45a78` |
| artifacts/unblock/relational_v1/dev_suite.json | `c88aa10f2818a0a340d47f9b1281f470abac0dcbb8d59aee9dcfb1a9ff7f9118` |

Nao havia checkpoint de EXPERTS nem de X0S_ADAPTER. A reproducao usou seed 51047, 208 passos, AdamW lr 1e-3, weight decay 0, clip 1.0, a mesma ordem de `random.Random(51047)` e o mesmo `dev_suite.json`. Os experts repetiram a curva historica de loss, inclusive o passo 1 em 9.3987 e o passo 200 em 0.0383. O adapter repetiu o passo 1 em 9.3987 e divergiu depois. Os pesos novos estao em `artifacts/unblock/relational_v1/judge_audit/`. Os JSON historicos nao foram substituidos.

## Comandos

```
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = "C:\Users\marco\Desktop\F51-Darwin-CL\src;C:\Users\marco\Desktop\F51-Darwin-CL\scripts"
& "C:\Users\marco\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests\test_relational_judge.py -q
& "C:\Users\marco\AppData\Local\Programs\Python\Python312\python.exe" scripts\run_relational_judge_audit.py
```

O avaliador passou em 10 testes: valor no sujeito errado, um de dois valores errado, as duas triplas certas sem a string exata, truncamento, repeticao, negacao seguida de contradicao, valor que so esta na pergunta, texto extra, e `false` colado a digitos.

Geracao igual para todos: greedy, 24 tokens, parada na primeira quebra de linha ou em `Question:` ou EOS. A saida bruta foi guardada.

## Tabela

Relacional exige a associacao sujeito-relacao-valor. Composicao exige os dois valores, cada um na relacao certa, sem contradicao. Premissa falsa aceita `No` ou `False` sem afirmar o valor falso. Ambiguo fica `ABSTAIN`.

| Condicao | Parafrase | Inversao | Composicao | Premissa falsa | Canarios |
|---|---|---|---|---|---|
| Donor sem fatos | 0/112 | 0/112 | 0/128 | 5/192 | nao medido aqui |
| Donor com fatos | 111/112 | 65/112 | 94/128 | 43/192 | nao se aplica |
| Donor com fatos embaralhados | 111/112 | 57/112 | 88/128 | 31/192 | nao se aplica |
| X0S sem fatos | 0/112 | 0/112 | 0/128 | 1/192 | 8/8 |
| EXPERTS sem fatos, reproducao | 0/112 | 0/112 | 0/128 | 15/192 | 7/8 |
| ADAPTER sem fatos, reproducao | 4/112 | 0/112 | 0/128 | 156/192 | 5/8 |

Exact match do trecho, no donor com fatos: parafrase 105/112, inversao 44/112, composicao 3/128, premissa falsa 5/192. A composicao relacional e 94/128. A diferenca 3 contra 94 e o artefato do exact match.

Drift ja medido, nao refeito: EXPERTS medio +0.005682, pior linguagem +0.021460. ADAPTER medio -0.001937, pior raciocinio +0.059579. Canarios historicos do adapter eram 4/8; a reproducao, que divergiu apos o passo 1, ficou 5/8.

Fatos embaralhados: em 370 itens cujo gabarito mudou, o donor acertou 256 no grafico novo. Ele segue o prompt. Isso nao e aprendizado nos pesos.

## Revisao

Os ABSTAIN do donor com fatos sao 1 parafrase, 36 inversoes, 25 composicoes e 28 premissas falsas. As 23 composicoes `no_association` incluem frases como `Nex-41 is Riga and Kite-3 is Bern`, em que os valores estao certos mas a relacao nao e nomeada. Ficaram abstencao de proposito. Nao sao acertos escondidos.

Os 15 acertos de premissa falsa dos experts sao quase todos `false-Answer: false-` ou `no` seguido de lixo. Nao sao rejeicoes do grafo. O adapter emite `false` e `INDETERMINADO` com frequencia, e erra composicao: a saida e uma palavra so, `False` ou `cobalt`, para uma pergunta de dois valores.

Exemplos do donor com fatos, composicao, exact match falho e juiz relacional certo: `Nex-41 maker is Lina and Kite-3 city is Bern` para o gabarito `Lina and Bern`.

## Tokens

As 128 composicoes avaliadas usam arestas atomicas que estao no treino. Nenhuma falta. A cross-entropy dos bracos supervisionou 586 tokens de resposta em 208 passos. Os 136 tokens continuam sendo a KL antiga do SDFT, outra perda.

## Proximo experimento, um so

Treinar so os experts, a partir do X0S, com o mesmo grafo de desenvolvimento, e medir composicao pelo juiz relacional, nao pelo exact match. O donor ja chega a 94/128 quando os fatos estao no prompt. Os pesos estao em 0/128. Nao abrir o teste selado e nao mexer no backbone.

PARAR.
