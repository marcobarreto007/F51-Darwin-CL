# RELATIONAL CONTROL AUDIT

Decisao: **PROMPT_OR_FORMAT_FAIL**.

O controle nao passa. A escada mostra o motivo: uma aresta explicita sai certa e depois o modelo continua a escrever, entao o exact match da geracao inteira falha. Inversao fecha 3/3. Duas arestas nao fecham a string "A and B": uma estoura o limite de 12 tokens, outra troca o segundo objeto, outra usa virgula em vez de "and". Premissa falsa vira "No, ..." em vez da palavra "false".

A queda de 3/4 para 3/128 e de avaliador, nao de um novo treino. Nas quatro perguntas antigas, o matcher por substring em 16 tokens acerta 3/4. O exact match contra "A and B" acerta 0/4 nas mesmas saidas. A suite nova pede exact match em 128 composicoes, com limite 12.

Sem treino. ARM 3 nao executado. Teste final selado nao consultado.

## Reproducao do donor com fatos

Reproducao exata dos publicados: True.

| Categoria | Exato |
|---|---|
| paraphrase | 47/112 |
| inverse | 33/112 |
| composition | 3/128 |
| false_premise | 5/192 |

## Vinte erros por categoria

### paraphrase

- Motivo: formato_texto_extra. Solver: Lina. Label: Lina. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Maker entry for Nex-
  Tokens de entrada: 158. Limite: 12. EOS: False.
  Saida:  Lina  Question: Color of Orb-12: 
  Shift ok: True. Primeiro token da resposta: 444.
- Motivo: formato_texto_extra. Solver: Lina. Label: Lina. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: State the maker of N
  Tokens de entrada: 159. Limite: 12. EOS: False.
  Saida:  Lina  Question: What is the color of Orb-
  Shift ok: True. Primeiro token da resposta: 444.
- Motivo: formato_texto_extra. Solver: Lina. Label: Lina. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Recorded maker of Ne
  Tokens de entrada: 158. Limite: 12. EOS: False.
  Saida:  Lina  Question: What is the color of the Orb
  Shift ok: True. Primeiro token da resposta: 444.
- Motivo: formato_texto_extra. Solver: Riga. Label: Riga. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: City entry for Nex-4
  Tokens de entrada: 158. Limite: 12. EOS: False.
  Saida:  Riga  Question: Color entry for Orb-12
  Shift ok: True. Primeiro token da resposta: 431.
- Motivo: formato_texto_extra. Solver: Riga. Label: Riga. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Where is Nex-41 file
  Tokens de entrada: 158. Limite: 12. EOS: False.
  Saida:  Riga  Question: What is the color of Orb-
  Shift ok: True. Primeiro token da resposta: 431.
- Motivo: formato_texto_extra. Solver: Riga. Label: Riga. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: State the city of Ne
  Tokens de entrada: 159. Limite: 12. EOS: False.
  Saida:  Riga  Question: What is the color of Nex-
  Shift ok: True. Primeiro token da resposta: 431.
- Motivo: formato_texto_extra. Solver: Riga. Label: Riga. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Recorded city of Nex
  Tokens de entrada: 158. Limite: 12. EOS: False.
  Saida:  Riga  Question: Recorded color of Orb-12
  Shift ok: True. Primeiro token da resposta: 431.
- Motivo: formato_texto_extra. Solver: amber. Label: amber. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Color entry for Nex-
  Tokens de entrada: 158. Limite: 12. EOS: False.
  Saida:  Amber  What is the maker of Orb-12? 
  Shift ok: True. Primeiro token da resposta: 67605.
- Motivo: formato_texto_extra. Solver: amber. Label: amber. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: What hue is filed fo
  Tokens de entrada: 160. Limite: 12. EOS: False.
  Saida:  Amber  Question: What is the color of the bolt-
  Shift ok: True. Primeiro token da resposta: 67605.
- Motivo: formato_texto_extra. Solver: amber. Label: amber. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: State the color of N
  Tokens de entrada: 159. Limite: 12. EOS: False.
  Saida:  Amber  Question: What is the maker of Orb-1
  Shift ok: True. Primeiro token da resposta: 67605.
- Motivo: formato_texto_extra. Solver: amber. Label: amber. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Recorded color of Ne
  Tokens de entrada: 158. Limite: 12. EOS: False.
  Saida:  Amber  What is the color of the Orb-12
  Shift ok: True. Primeiro token da resposta: 67605.
- Motivo: formato_texto_extra. Solver: chisel. Label: chisel. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Tool entry for Nex-4
  Tokens de entrada: 158. Limite: 12. EOS: False.
  Saida:  chisel  Tool entry for Orb-12: Answer
  Shift ok: True. Primeiro token da resposta: 521.
- Motivo: formato_texto_extra. Solver: chisel. Label: chisel. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: What implement is fi
  Tokens de entrada: 160. Limite: 12. EOS: False.
  Saida:  Chisel  Question: What is the color of the Orb
  Shift ok: True. Primeiro token da resposta: 521.
- Motivo: formato_texto_extra. Solver: chisel. Label: chisel. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: State the tool of Ne
  Tokens de entrada: 159. Limite: 12. EOS: False.
  Saida:  Chisel  Question: What is the color of Orb-
  Shift ok: True. Primeiro token da resposta: 521.
- Motivo: formato_texto_extra. Solver: Omar. Label: Omar. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: The person credited 
  Tokens de entrada: 160. Limite: 12. EOS: False.
  Saida:  Omar  Question: The color of the tool used in Bolt
  Shift ok: True. Primeiro token da resposta: 53558.
- Motivo: formato_texto_extra. Solver: Omar. Label: Omar. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Maker entry for Bolt
  Tokens de entrada: 157. Limite: 12. EOS: False.
  Saida:  Omar  Question: Maker entry for Kite-3: 
  Shift ok: True. Primeiro token da resposta: 53558.
- Motivo: formato_texto_extra. Solver: Omar. Label: Omar. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: State the maker of B
  Tokens de entrada: 158. Limite: 12. EOS: False.
  Saida:  Omar  Question: What is the color of the tool used
  Shift ok: True. Primeiro token da resposta: 53558.
- Motivo: formato_texto_extra. Solver: Omar. Label: Omar. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Recorded maker of Bo
  Tokens de entrada: 157. Limite: 12. EOS: False.
  Saida:  Omar  Recorded maker of Kite-3? Answer
  Shift ok: True. Primeiro token da resposta: 53558.
- Motivo: formato_texto_extra. Solver: Oslo. Label: Oslo. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Identify the city li
  Tokens de entrada: 159. Limite: 12. EOS: True.
  Saida:  Bolt-7 is located in Oslo, Norway.
  Shift ok: True. Primeiro token da resposta: 57858.
- Motivo: formato_texto_extra. Solver: Oslo. Label: Oslo. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Which city contains 
  Tokens de entrada: 157. Limite: 12. EOS: True.
  Saida:  Bolt-7 is located in Oslo, Norway.
  Shift ok: True. Primeiro token da resposta: 57858.

### inverse

- Motivo: formato_texto_extra. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Identify the item fi
  Tokens de entrada: 159. Limite: 12. EOS: True.
  Saida:  Nex-41 tool is chisel.
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: formato_texto_extra. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Which record lists m
  Tokens de entrada: 157. Limite: 12. EOS: True.
  Saida:  Nex-41 maker is Lina.
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: formato_texto_extra. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Find the item with m
  Tokens de entrada: 158. Limite: 12. EOS: True.
  Saida:  Nex-41 tool is chisel.
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: limite_de_geracao. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Item entry for maker
  Tokens de entrada: 157. Limite: 12. EOS: False.
  Saida:  Lina  Item entry for tool chisel: Answer:
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: formato_texto_extra. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: State the item for m
  Tokens de entrada: 158. Limite: 12. EOS: False.
  Saida:  Nex-41 maker is Lina.  Question: State
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: formato_texto_extra. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Which record lists c
  Tokens de entrada: 157. Limite: 12. EOS: False.
  Saida:  Nex-41 city is Riga.  Question: Which
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: formato_texto_extra. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Item entry for city 
  Tokens de entrada: 157. Limite: 12. EOS: False.
  Saida:  Nex-41  Item entry for color Ivory: Answer
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: formato_texto_extra. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: State the item for c
  Tokens de entrada: 158. Limite: 12. EOS: False.
  Saida:  Nex-41  Question: State the color of the
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: formato_texto_extra. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Identify the item fi
  Tokens de entrada: 158. Limite: 12. EOS: True.
  Saida:  Nex-41 tool is chisel.
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: formato_texto_extra. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Which record lists c
  Tokens de entrada: 156. Limite: 12. EOS: False.
  Saida:  Nex-41 color is amber.  Question: Which record
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: limite_de_geracao. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Item entry for color
  Tokens de entrada: 156. Limite: 12. EOS: False.
  Saida:  Orb-12  Item entry for color cobalt: 
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: limite_de_geracao. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Who or what is tied 
  Tokens de entrada: 159. Limite: 12. EOS: False.
  Saida:  Orb-12 maker Hugo  Question: What is the
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: formato_texto_extra. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: State the item for c
  Tokens de entrada: 157. Limite: 12. EOS: False.
  Saida:  Nex-41 color is amber.  Question: State the
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: formato_texto_extra. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Identify the item fi
  Tokens de entrada: 159. Limite: 12. EOS: False.
  Saida:  The item filed under tool chisel is Nex-41
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: formato_texto_extra. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Which record lists t
  Tokens de entrada: 157. Limite: 12. EOS: True.
  Saida:  Nex-41 tool is chisel.
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: formato_texto_extra. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Find the item with t
  Tokens de entrada: 158. Limite: 12. EOS: True.
  Saida:  Nex-41 tool is chisel.
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: limite_de_geracao. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Item entry for tool 
  Tokens de entrada: 157. Limite: 12. EOS: False.
  Saida:  Orb-12 tool chisel  Item entry for tool
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: eos_cedo. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Who or what is tied 
  Tokens de entrada: 160. Limite: 12. EOS: True.
  Saida:  Lina
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: limite_de_geracao. Solver: Nex-41. Label: Nex-41. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: State the item for t
  Tokens de entrada: 158. Limite: 12. EOS: False.
  Saida:  Orb-12 tool is chisel.  Question: What
  Shift ok: True. Primeiro token da resposta: 87829.
- Motivo: formato_texto_extra. Solver: Bolt-7. Label: Bolt-7. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Identify the item fi
  Tokens de entrada: 158. Limite: 12. EOS: False.
  Saida:  The item filed under maker Omar is the bolt-7 tool
  Shift ok: True. Primeiro token da resposta: 45947.

### composition

- Motivo: limite_de_geracao. Solver: Lina and Vera. Label: Lina and Vera. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Name the maker of Ne
  Tokens de entrada: 167. Limite: 12. EOS: False.
  Saida:  Nex-41 maker is Lina and Kite-
  Shift ok: True. Primeiro token da resposta: 444.
- Motivo: limite_de_geracao. Solver: Lina and Vera. Label: Lina and Vera. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Give both the maker 
  Tokens de entrada: 168. Limite: 12. EOS: False.
  Saida:  Nex-41 maker is Lina, Kite-
  Shift ok: True. Primeiro token da resposta: 444.
- Motivo: limite_de_geracao. Solver: Lina and Bern. Label: Lina and Bern. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Name the maker of Ne
  Tokens de entrada: 167. Limite: 12. EOS: False.
  Saida:  Nex-41 maker is Lina and Kite-
  Shift ok: True. Primeiro token da resposta: 444.
- Motivo: limite_de_geracao. Solver: Lina and Bern. Label: Lina and Bern. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Give both the maker 
  Tokens de entrada: 168. Limite: 12. EOS: False.
  Saida:  Nex-41 maker is Lina, and Kite
  Shift ok: True. Primeiro token da resposta: 444.
- Motivo: limite_de_geracao. Solver: Lina and ivory. Label: Lina and ivory. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Name the maker of Ne
  Tokens de entrada: 167. Limite: 12. EOS: False.
  Saida:  Nex-41 maker is Lina and Kite-
  Shift ok: True. Primeiro token da resposta: 444.
- Motivo: limite_de_geracao. Solver: Lina and ivory. Label: Lina and ivory. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Give both the maker 
  Tokens de entrada: 168. Limite: 12. EOS: False.
  Saida:  Nex-41 maker is Lina, Kite-
  Shift ok: True. Primeiro token da resposta: 444.
- Motivo: limite_de_geracao. Solver: Lina and kiln. Label: Lina and kiln. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Name the maker of Ne
  Tokens de entrada: 167. Limite: 12. EOS: False.
  Saida:  Nex-41 maker is Lina and Kite-
  Shift ok: True. Primeiro token da resposta: 444.
- Motivo: limite_de_geracao. Solver: Lina and kiln. Label: Lina and kiln. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Give both the maker 
  Tokens de entrada: 168. Limite: 12. EOS: False.
  Saida:  Nex-41 maker is Lina, and Kite
  Shift ok: True. Primeiro token da resposta: 444.
- Motivo: eos_cedo. Solver: Riga and Vera. Label: Riga and Vera. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Name the city of Nex
  Tokens de entrada: 167. Limite: 12. EOS: True.
  Saida:  Riga, Vera
  Shift ok: True. Primeiro token da resposta: 431.
- Motivo: formato_texto_extra. Solver: Riga and Vera. Label: Riga and Vera. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Give both the city o
  Tokens de entrada: 168. Limite: 12. EOS: False.
  Saida:  Nex-41 is Riga and Vera is the maker
  Shift ok: True. Primeiro token da resposta: 431.
- Motivo: limite_de_geracao. Solver: Riga and Bern. Label: Riga and Bern. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Name the city of Nex
  Tokens de entrada: 167. Limite: 12. EOS: False.
  Saida:  Nex-41 is Riga, and Kite-
  Shift ok: True. Primeiro token da resposta: 431.
- Motivo: limite_de_geracao. Solver: Riga and Bern. Label: Riga and Bern. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Give both the city o
  Tokens de entrada: 168. Limite: 12. EOS: False.
  Saida:  Nex-41 is Riga and Kite-3
  Shift ok: True. Primeiro token da resposta: 431.
- Motivo: limite_de_geracao. Solver: Riga and ivory. Label: Riga and ivory. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Name the city of Nex
  Tokens de entrada: 167. Limite: 12. EOS: False.
  Saida:  Nex-41 is Riga, and Kite-
  Shift ok: True. Primeiro token da resposta: 431.
- Motivo: limite_de_geracao. Solver: Riga and ivory. Label: Riga and ivory. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Give both the city o
  Tokens de entrada: 168. Limite: 12. EOS: False.
  Saida:  Nex-41 is Riga and Kite-3
  Shift ok: True. Primeiro token da resposta: 431.
- Motivo: limite_de_geracao. Solver: Riga and kiln. Label: Riga and kiln. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Name the city of Nex
  Tokens de entrada: 167. Limite: 12. EOS: False.
  Saida:  Nex-41 is Riga, and Kite-
  Shift ok: True. Primeiro token da resposta: 431.
- Motivo: limite_de_geracao. Solver: Riga and kiln. Label: Riga and kiln. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Give both the city o
  Tokens de entrada: 168. Limite: 12. EOS: False.
  Saida:  Nex-41 is Riga and Kite-3
  Shift ok: True. Primeiro token da resposta: 431.
- Motivo: limite_de_geracao. Solver: amber and Vera. Label: amber and Vera. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Name the color of Ne
  Tokens de entrada: 167. Limite: 12. EOS: False.
  Saida:  Nex-41 color is amber and the maker of K
  Shift ok: True. Primeiro token da resposta: 67605.
- Motivo: limite_de_geracao. Solver: amber and Vera. Label: amber and Vera. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Give both the color 
  Tokens de entrada: 168. Limite: 12. EOS: False.
  Saida:  Nex-41 color is amber, and Kite-
  Shift ok: True. Primeiro token da resposta: 67605.
- Motivo: limite_de_geracao. Solver: amber and Bern. Label: amber and Bern. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Name the color of Ne
  Tokens de entrada: 167. Limite: 12. EOS: False.
  Saida:  Nex-41 color is amber and Kite-3
  Shift ok: True. Primeiro token da resposta: 67605.
- Motivo: limite_de_geracao. Solver: amber and Bern. Label: amber and Bern. Arestas no prompt: [True, True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Give both the color 
  Tokens de entrada: 168. Limite: 12. EOS: False.
  Saida:  Nex-41 color is amber, and Kite-
  Shift ok: True. Primeiro token da resposta: 67605.

### false_premise

- Motivo: limite_de_geracao. Solver: false. Label: false. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Does the record say 
  Tokens de entrada: 165. Limite: 12. EOS: False.
  Saida:  No, the record does not say that the maker of Nex
  Shift ok: True. Primeiro token da resposta: 895.
- Motivo: formato_texto_extra. Solver: false. Label: false. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Claim check: Nex-41 
  Tokens de entrada: 165. Limite: 12. EOS: True.
  Saida:  False. Nex-41 maker is Lina.
  Shift ok: True. Primeiro token da resposta: 895.
- Motivo: limite_de_geracao. Solver: false. Label: false. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Verify whether Nex-4
  Tokens de entrada: 161. Limite: 12. EOS: False.
  Saida:  No, Nex-41 maker is Lina, not
  Shift ok: True. Primeiro token da resposta: 895.
- Motivo: limite_de_geracao. Solver: false. Label: false. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Check this filing on
  Tokens de entrada: 164. Limite: 12. EOS: False.
  Saida:  No, the correct maker for Nex-41 is L
  Shift ok: True. Primeiro token da resposta: 895.
- Motivo: formato_texto_extra. Solver: false. Label: false. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Audit the claim that
  Tokens de entrada: 162. Limite: 12. EOS: False.
  Saida:  No, the claim is false. Nex-41 maker
  Shift ok: True. Primeiro token da resposta: 895.
- Motivo: limite_de_geracao. Solver: false. Label: false. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Is it recorded that 
  Tokens de entrada: 162. Limite: 12. EOS: False.
  Saida:  No, it is not recorded that Nex-41 maker
  Shift ok: True. Primeiro token da resposta: 895.
- Motivo: eos_cedo. Solver: INDETERMINADO. Label: INDETERMINADO. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Which motto belongs 
  Tokens de entrada: 159. Limite: 12. EOS: True.
  Saida:  "To be, to know, to create."
  Shift ok: True. Primeiro token da resposta: 1964.
- Motivo: limite_de_geracao. Solver: INDETERMINADO. Label: INDETERMINADO. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: State the motto file
  Tokens de entrada: 160. Limite: 12. EOS: False.
  Saida:  The motto filed for Nex-41 is "To be
  Shift ok: True. Primeiro token da resposta: 1964.
- Motivo: limite_de_geracao. Solver: INDETERMINADO. Label: INDETERMINADO. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: What slogan is store
  Tokens de entrada: 160. Limite: 12. EOS: False.
  Saida:  "Nex-41 is the best"  Question:
  Shift ok: True. Primeiro token da resposta: 1964.
- Motivo: eos_cedo. Solver: INDETERMINADO. Label: INDETERMINADO. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Name the motto of Ne
  Tokens de entrada: 159. Limite: 12. EOS: True.
  Saida:  Lina
  Shift ok: True. Primeiro token da resposta: 1964.
- Motivo: eos_cedo. Solver: INDETERMINADO. Label: INDETERMINADO. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: The motto entry of N
  Tokens de entrada: 160. Limite: 12. EOS: True.
  Saida:  "The best of all possible worlds."
  Shift ok: True. Primeiro token da resposta: 1964.
- Motivo: eos_cedo. Solver: INDETERMINADO. Label: INDETERMINADO. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Which catchphrase is
  Tokens de entrada: 161. Limite: 12. EOS: True.
  Saida:  "I'm a chisel."
  Shift ok: True. Primeiro token da resposta: 1964.
- Motivo: limite_de_geracao. Solver: false. Label: false. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Does the record say 
  Tokens de entrada: 165. Limite: 12. EOS: False.
  Saida:  No, the record does not say that the city of Nex
  Shift ok: True. Primeiro token da resposta: 895.
- Motivo: formato_texto_extra. Solver: false. Label: false. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Claim check: Nex-41 
  Tokens de entrada: 165. Limite: 12. EOS: True.
  Saida:  False. Nex-41 has city Riga.
  Shift ok: True. Primeiro token da resposta: 895.
- Motivo: limite_de_geracao. Solver: false. Label: false. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Verify whether Nex-4
  Tokens de entrada: 161. Limite: 12. EOS: False.
  Saida:  No, Nex-41 does not have city Oslo.
  Shift ok: True. Primeiro token da resposta: 895.
- Motivo: eos_cedo. Solver: false. Label: false. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Check this filing on
  Tokens de entrada: 164. Limite: 12. EOS: True.
  Saida:  No, Nex-41 is not from Oslo.
  Shift ok: True. Primeiro token da resposta: 895.
- Motivo: limite_de_geracao. Solver: false. Label: false. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Audit the claim that
  Tokens de entrada: 162. Limite: 12. EOS: False.
  Saida:  No, the claim is incorrect. Nex-41 has
  Shift ok: True. Primeiro token da resposta: 895.
- Motivo: limite_de_geracao. Solver: false. Label: false. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Is it recorded that 
  Tokens de entrada: 162. Limite: 12. EOS: False.
  Saida:  No, it is not recorded that Nex-41 city
  Shift ok: True. Primeiro token da resposta: 895.
- Motivo: eos_cedo. Solver: INDETERMINADO. Label: INDETERMINADO. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: Which motto belongs 
  Tokens de entrada: 159. Limite: 12. EOS: True.
  Saida:  "To be, to know, to create."
  Shift ok: True. Primeiro token da resposta: 1964.
- Motivo: limite_de_geracao. Solver: INDETERMINADO. Label: INDETERMINADO. Arestas no prompt: [True].
  Prompt: Known facts: | Bolt-7 city is Oslo. | Bolt-7 color is cobalt. | Bolt-7 maker is Omar. | Bolt-7 tool is spindle. | Kite-3 city is Bern. | Kite-3 color is ivory. | Kite-3 maker is Vera. | Kite-3 tool is kiln. | Nex-41 city is Riga. | Nex-41 color is amber. | Nex-41 maker is Lina. | Nex-41 tool is chisel. | Orb-12 city is Lyon. | Orb-12 color is scarlet. | Orb-12 maker is Hugo. | Orb-12 tool is loom. |  | Question: State the motto file
  Tokens de entrada: 160. Limite: 12. EOS: False.
  Saida:  The motto filed for Nex-41 is "To be
  Shift ok: True. Primeiro token da resposta: 1964.

## Escada, mesmo avaliador

| Degrau | Exato | Saida |
|---|---|---|
| copiar | False |  Lina.  The answer written here is Lina. Question |
| copiar | True |  Oslo. |
| copiar | False |  The answer written here is false. Question: Is the following |
| aresta | False |  Lina  Question: What is the color of Orb- |
| aresta | False |  Bolt-7 is filed in Oslo. |
| aresta | False |  Ivory  Question: What is the color of the tool for |
| inverter | True |  Nex-41 |
| inverter | True |  Bolt-7 |
| inverter | True |  Kite-3 |
| duas_arestas | False |  Nex-41 maker is Lina and Bolt-7 |
| duas_arestas | False |  Ivory, kiln |
| duas_arestas | False |  Omar, Riga |
| falsa | False |  No, the maker of Nex-41 is Lina |
| falsa | False |  No, the city of Bolt-7 is Oslo.  Question |
| falsa | False |  "To be, to know, to create."  Question: |

## Formato antigo e formato novo

- Substring 16: False. Exato contra 'A and B': False.
   The maker of GRAV-X9 is Marco, and the inventor of Zyph
- Substring 16: True. Exato contra 'A and B': False.
   The fictional capital is Paris, and the fictional animal is a tiger.
- Substring 16: True. Exato contra 'A and B': False.
   The fictional metal is **copper**, and the fictional color is **purple**
- Substring 16: True. Exato contra 'A and B': False.
   The fictional planet is Saturn, and the fictional instrument is piano.

- Nova pergunta no formato antigo, substring: False, exact: False.
   The maker of Nex-41 is Lina, and the maker of K
- Nova pergunta no formato antigo, substring: True, exact: False.
   Nex-41 maker is Lina and Kite-3 maker is Vera
- Nova pergunta no formato antigo, substring: False, exact: False.
   The maker of Nex-41 is Lina, and the city of K
- Nova pergunta no formato antigo, substring: True, exact: False.
   Nex-41 maker is Lina and Kite-3 city is Bern
- Nova pergunta no formato antigo, substring: False, exact: False.
   The maker of Nex-41 is Lina, and the color of K
- Nova pergunta no formato antigo, substring: True, exact: False.
   Nex-41 maker is Lina and Kite-3 color is ivory
- Nova pergunta no formato antigo, substring: False, exact: False.
   The maker of Nex-41 is Lina, and the tool of K
- Nova pergunta no formato antigo, substring: False, exact: False.
   Nex-41 maker is Lina and Kite-3 tool is kil

## Arestas e tokens

Composicoes avaliadas sem fato extra no prompt de treino: 128.
Dessas, com alguma aresta atomica ausente do treino: 0.
Exemplos de treino: 104.
Tokens supervisionados da cross-entropy, uma passagem: 293.
Duas epocas, o orcamento dos bracos: 586.
Os 136 tokens sao a KL antiga do SDFT, nao esta perda.

PARAR. Sem novo treino.
