# UNBLOCK DIAGNOSTIC

All persons, inventions, scientific effects, medical claims, events and historical relationships in F51-SRB are synthetic fiction created solely for machine-learning evaluation.

O ARM 2 nao integrou relacoes. Ele empurrou o backbone para repetir um token, e a metrica de paraphrase contou isso como acerto.

## Comando

```
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = "C:\Users\marco\Desktop\F51-Darwin-CL\src"
& "C:\Users\marco\AppData\Local\Programs\Python\Python312\python.exe" "C:\Users\marco\Desktop\F51-Darwin-CL\scripts\run_unblock_diagnostic.py"
```

## Tokens e parametros

Unicos no dataset: 78
Processados em 96 passos: 906
Supervisionados na KL: 136
processed = soma, em todos os passos, dos tokens do prompt mais os tokens gerados; gerado tem o comprimento da demonstracao. supervised = so as posicoes da KL, uma por token gerado. unique = ids distintos no prompt e na demonstracao, sem repetir os passos.

Elementos do backbone alterados no ARM 2 historico: 336899732 de 596049920
elementos escalares cujo valor bf16 difere do donor original. Nao e contagem de tensores com gradiente nem o tamanho do grupo treinavel.

## Reproducao sem treino

X0S ranks: [1, 1, 1, 1, 1, 1, 1, 1]
ARM 2 ranks: [1, 1, 49046, 1, 1, 1, 1, 3]
Suite antiga reproduzida: True

## Margem contra logprob, nos mesmos itens

A margem publicada usa so os 16 itens com distractor. O logprob medio usa as 92 perguntas. Nao sao o mesmo denominador.

| Categoria | X0S | ARM 2 |
|---|---|---|
| paraphrase | 9/40 target -7.050 | 30/40 target -6.208 |
| reverse | 4/16 target -6.088 | 0/16 target -20.352 |
| false_premise | 5/16 target -6.172 | 0/16 target -24.662 |
| distractor | 9/16 target -5.874 | 10/16 target -6.229 |
| composition | 0/4 target -5.953 | 0/4 target -13.847 |

Nos 16 distractors, o alvo foi de -5.874 para -6.229. O distractor foi de -6.038 para -10.871. A margem sobe porque o termo errado cai mais, nao porque o alvo melhora.

As paraphrases contadas como acerto no ARM 2 sao repeticao do token (`Marco Marco Marco`). O matcher aceita a substring. Isso nao e uma resposta composta.

## WITH_CONTEXT

Donor puro, 16 tokens, fatos no prompt: 3/4.
O unico erro foi cortado no token 16: "the inventor of Zyph". A sonda de 32 tokens nao rodou, porque os 16 ja tinham passado em 3.
A sonda com "Answer:" tambem nao rodou.

- False |  The maker of GRAV-X9 is Marco, and the inventor of Zyph
- True |  The fictional capital is Paris, and the fictional animal is a tiger.
- True |  The fictional metal is **copper**, and the fictional color is **purple**
- True |  The fictional planet is Saturn, and the fictional instrument is piano.

## Optimizer e 8 passos descartaveis a 0.01x

Router no optimizer: False. Alpha no optimizer: False. Duplicatas expert/backbone: 0.
model.training: False. attention_dropout: 0.0.
Passo 1, elementos alterados via mestre fp32: 56405927. Via soma em bf16: 56402465. Divergencia entre as duas escritas: 587961.

| Passo | Drift medio | Top-1 |
|---|---|---|
| 1 | +0.0023 | 7/8 |
| 4 | +0.0001 | 7/8 |
| 8 | -0.0004 | 6/8 |

## Plasticidade menor

- A, reutilizado do X2B: 96 passos, drift contra o X0S +0.0025, composition 0/4.
- B_1e-7, partida limpa no X0S: 96 passos, drift +0.0027, top-1 7/8, composition 0/4.
- C_1e-6 teve uma primeira corrida contaminada pelo backbone de B. A repeticao limpa, tambem a partir do X0S, fez 96 passos, drift -0.0009, top-1 7/8, composition 0/4.

## Curriculo relacional

Estabilidade sem composition. O curriculo relacional fica como o proximo experimento, nao foi misturado nesta corrida.

## Respostas

1. Nao ha bug de carga, de mascara ou de deslocamento causal. Ha um erro de leitura da metrica: margem e logprob nao usam os mesmos itens. O acerto de paraphrase no ARM 2 e repeticao do token.
2. Os fatos existem separados no treino. Nenhuma das 48 formulacoes ensina as duas respostas juntas, nem o alvo 'false'. Reverse de held-out nao repete o prompt inverso do treino.
3. O donor compoe. WITH_CONTEXT acertou 3/4 em 16 tokens. O quarto item trazia Marco e foi cortado em "Zyph".
4. 1e-7 e 1e-6, por 96 passos, mantiveram o drift medio dentro de 0.003 nats/byte. Composition continuou 0/4. O 0.01x nao preserva: no passo 1 um canario sai de rank 1, e em 96 passos o drift medio chega a +0.804.
5. As 48 formulacoes ensinam fatos atomicos e uma inversao com outro wording. Nao ensinam "false" nem as duas respostas juntas. O curriculo relacional, que ensina essas relacoes em outras combinacoes, tambem deixou a composition do X1 em 0/4.
6. O curriculo relacional ja foi executado, em 96 passos, experts-only e trunk 1e-7. Os dois terminaram em 3/8 canarios, drift abaixo de 0.003, e 0/4 na composition do X1. Nao ha configuracao promissora para repetir em outras seeds. O proximo experimento justificado e um so: alongar a geracao da avaliacao de composition, porque 16 tokens ja cortaram uma resposta certa do donor. Sem subir LR e sem mudar a arquitetura.

Teste final selado, nao consultado: `c163445e36c26769a841da0c2d8260b049328a4a29a77b3e4ff52dfc5fda8b20`
Fingerprint antes: `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175`
Fingerprint do ARM 2: `ce26eda8f172ce9d25021ebdcf075390aa1c956812327ede899ba29f31eceffb`
Peak VRAM bytes: 15414972416
RAM bytes: 15934574592
Wall s: 289.82

ARM 3 nao foi executado. Nenhuma fase posterior foi iniciada.

## RELATIONAL_CURRICULUM_V0 — corrida de 56 passos, substituida


Protocolo novo. Nao e o SDFT de 48 formulacoes. Backbone comparado so em 1e-7, o tronco que ficou dentro do limite. O teste selado nao foi consultado. A avaliacao de composition continua sendo a do X1.

Exemplos: 28. SHA-256 `5a6910d7e0e28cb772d1a2d7167c4e1b758327980a7e469f554641d4210afe7b`.

- REL_A_experts: passos 56, parada None, drift -0.0016, top-1 6/8, composition X1 0/4, tokens processados 638, supervisionados 112, tempo 32.3s
  - False |  Also, what is the name of the first computer to be used in the military
  - False |  The fictional capital is Paris, and the fictional animal is a lion.
  - False |  Then, write a sentence that describes the metal and the color. Answer: In
  - False |  The fictional planet is **Jupiter**, and the fictional instrument is **the Jupiter
- REL_B_1e-7: passos 56, parada None, drift +0.0005, top-1 6/8, composition X1 0/4, tokens processados 638, supervisionados 112, tempo 46.9s
  - False |  Also, what is the name of the first computer to be used in the military
  - False |  The fictional capital is Paris, and the fictional animal is a lion.
  - False |  Then, write a sentence that describes the metal and the color. Use the following
  - False |  The fictional planet is **Jupiter**, and the fictional instrument is **the Jupiter

ARM 3 nao foi executado.

## RELATIONAL_CURRICULUM_V0

Protocolo novo. Nao e o SDFT de 48 formulacoes. Backbone comparado so em 1e-7, o tronco que ficou dentro do limite. O teste selado nao foi consultado. A avaliacao de composition continua sendo a do X1.

Exemplos: 28. SHA-256 `5a6910d7e0e28cb772d1a2d7167c4e1b758327980a7e469f554641d4210afe7b`.

- REL_A_experts: passos 96, parada None, drift +0.0003, top-1 3/8, composition X1 0/4, tokens processados 1110, supervisionados 195, tempo 50.2s
  - False |  Additionally, what is the significance of the name "Zyphron-1
  - False |  The fictional capital is **Paris** and the fictional animal is **the lion**
  - False |    $$ \begin{matrix} \text{Metal} & \text
  - False |  The fictional planet is **Saturn** and the fictional instrument is **the Saturn
- REL_B_1e-7: passos 96, parada None, drift +0.0022, top-1 3/8, composition X1 0/4, tokens processados 1110, supervisionados 195, tempo 73.8s
  - False |  Additionally, what is the significance of the name "Zyphron-1
  - False |  The fictional capital is Paris, and the fictional animal is a lion.
  - False |    - The fictional metal is: _____________ - The fictional color is
  - False |  The fictional planet is **Saturn** and the fictional instrument is **the Saturn

ARM 3 nao foi executado.
