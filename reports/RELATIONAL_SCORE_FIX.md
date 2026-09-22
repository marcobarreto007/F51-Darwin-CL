# RELATIONAL SCORE FIX

A geracao para na primeira quebra de linha ou em `Question:`. O exact match usa so esse trecho. Limite 24 tokens. Suite historica e teste selado nao foram alterados. Sem treino.

| Categoria | Exact antigo, geracao inteira, 12 tokens | Exact do trecho, parada explicita |
|---|---|---|
| paraphrase | 47/112 | 105/112 |
| inverse | 33/112 | 44/112 |
| composition | 3/128 | 3/128 |
| false premise | 5/192 | 5/192 |

## Escada

| Degrau | Exato | Trecho |
|---|---|---|
| copiar | True | Lina. |
| aresta | True | Lina |
| inverter | True | Nex-41 |
| duas_arestas | False | Nex-41 maker is Lina and Bolt-7 city is Oslo. |
| falsa | False | No, the maker of Nex-41 is Lina. |

## Erros que continuam

### paraphrase

- Esperado `Oslo`. Trecho `Bolt-7 is located in Oslo, Norway.`. Limite: False.
- Esperado `Oslo`. Trecho `Bolt-7 is located in Oslo, Norway.`. Limite: False.
- Esperado `Oslo`. Trecho `Bolt-7 is filed in the United States Patent and Trademark Office (USPTO).`. Limite: False.
- Esperado `Oslo`. Trecho `Bolt-7 is located in Oslo.`. Limite: False.
- Esperado `kiln`. Trecho `The tool listed for Kite-3 is kiln.`. Limite: False.

### inverse

- Esperado `Nex-41`. Trecho `Nex-41 tool is chisel.`. Limite: False.
- Esperado `Nex-41`. Trecho `Nex-41 maker is Lina.`. Limite: False.
- Esperado `Nex-41`. Trecho `Nex-41 tool is chisel.`. Limite: False.
- Esperado `Nex-41`. Trecho `Lina`. Limite: False.
- Esperado `Nex-41`. Trecho `Nex-41 maker is Lina.`. Limite: False.

### composition

- Esperado `Lina and Vera`. Trecho `Nex-41 maker is Lina and Kite-3 maker is Vera.`. Limite: False.
- Esperado `Lina and Vera`. Trecho `Nex-41 maker is Lina, Kite-3 maker is Vera.`. Limite: False.
- Esperado `Lina and Bern`. Trecho `Nex-41 maker is Lina and Kite-3 city is Bern.`. Limite: False.
- Esperado `Lina and Bern`. Trecho `Nex-41 maker is Lina, and Kite-3 city is Bern.`. Limite: False.
- Esperado `Lina and ivory`. Trecho `Nex-41 maker is Lina and Kite-3 color is ivory.`. Limite: False.

### false_premise

- Esperado `false`. Trecho `No, the record does not say that the maker of Nex-41 is Omar. The maker of Nex-4`. Limite: True.
- Esperado `false`. Trecho `False. Nex-41 maker is Lina.`. Limite: False.
- Esperado `false`. Trecho `No, Nex-41 maker is Lina, not Omar.`. Limite: False.
- Esperado `false`. Trecho `No, the correct maker for Nex-41 is Lina, not Omar.`. Limite: False.
- Esperado `false`. Trecho `No, the claim is false. Nex-41 maker is Lina, not Omar.`. Limite: False.

PARAR. Sem novo treino.
