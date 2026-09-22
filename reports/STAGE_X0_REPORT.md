# STAGE X0 — CAN THE BRANCH MEMORIZE AT ALL?

All persons, inventions, scientific effects, medical claims, events and historical relationships in F51-SRB are synthetic fiction created solely for machine-learning evaluation.

Status: **STAGE X0 complete**. Conclusão: **CAPACITY_OR_OPTIMIZATION_FAIL**.

X1 não começou. SDFT não foi implementado. A corrida A e a 4R não foram reescritas.

## Régua, escrita antes do run

Oito fatos. A avaliação usa o prefixo exato da frase de treino e pede o objeto. Não há paráfrase. Passa se o exact match no treino for pelo menos 0.875, isto é, 7 de 8. Abaixo disso a branch não memorizou, e o estágio para.

Arquitetura igual à aprovada: layer 13, 8 experts SwiGLU, top-2, router treinável, donor congelado, alpha fora do otimizador. AdamW, lr 1e-3, clip 1.0, weight decay 0, 40 épocas, batch 1.

Primeiro alpha: 1e-6, o valor que a 4R julgou negligente. Se falhar, um braço de diagnóstico em 1e-4, que a calibração já mostrou mover logits. Sem mudar layer, largura nem norma.

## Resultado

| Alpha | Época | Loss de treino | Exact match | Logprob do alvo | Margem |
|---|---|---|---|---|---|
| donor, sem branch | 0 | | 0.0 | -6.846 | 0.514 |
| 1e-6 | 10 | 6.090 | 0.0 | -6.851 | 0.518 |
| 1e-6 | 40 | 6.091 | 0.0 | -6.847 | 0.539 |
| 1e-4 | 10 | 4.763 | 0.0 | -6.753 | 0.520 |
| 1e-4 | 40 | 2.039 | 0.0 | -5.160 | 0.730 |

Em 1e-6 o loss não sai de 6.09. Em 1e-4 o loss cai de 4.76 para 2.04 e a logprob do objeto sobe cerca de 1.7 nats. O exact match fica 0 nos dois. A continuação greedy, nos oito prefixos, não contém NANOR-51, GRAV-X9, Zyphron-11 nem os outros alvos. Várias continuações são uma sequência de zeros.

Fingerprint do donor nos dois braços: `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175`.

## O que isso isola

Não é falha de generalização. A pergunta era a própria frase. Em 1e-6 a branch não chega a baixar o loss. Em 1e-4 o loss desce e o texto greedy continua errado. O sinal existe e não vira o token pedido.

Não mexi na layer 13, na largura 3072 nem na norma. Esses três continuam hipóteses, não medidas deste estágio. O que foi medido é alpha e o otimizador já usado: abrir o alpha cem vezes baixa o loss e não produz a string.

## Métodos revisados, não implementados

Fichas em `reports/methods/`, lidas do checkout do mini-AGI e dos HTML do arXiv, não de um blog de resumo, exceto PC-ALM, que não tem arXiv confirmado:

- `MINI_AGI.md`
- `SDFT.md`
- `OPCD.md`
- `OEL.md`
- `IN_PLACE_TTT.md`
- `TTCD.md`
- `NESTED_LEARNING_HOPE.md`
- `PC_ALM.md`

Nenhum deles entrou no treino.

## Scoreboard

`reports/CONTINUAL_SCOREBOARD.csv` ganhou as linhas da corrida A, dos braços B e C da 4R, e as duas tentativas do X0. Nada foi apagado.

## Parar

Sem memorização, SDFT não entra. O próximo estágio, X1, só começa com `APPROVED STAGE X0`.
