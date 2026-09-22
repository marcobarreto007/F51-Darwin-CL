# PHASE 3 REPORT — plastic expert branch V0

Status: **PHASE 3 complete**. Gate: **PLASTIC_BRANCH_MECHANICAL_PASS**.

Não houve `optimizer.step`. Não houve domínio novo. Não houve dataset F. A Fase 4 não começou. `artifacts/baseline/v1/reference.json` não foi reescrito. SHA-256 continua `2c9904f3412d14bc0e2a3c13562ff33ebff5441decfb089b3a085a4cda5c13fb`.

Política de métrica: EXACT_SCORED_UTF8. Donor: `Qwen/Qwen3-0.6B-Base`, revisão `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.

## A. Ponto de inserção

Layer índice 13, das 28 (0 a 27). `(28 // 2) - 1 = 13`. Quatorze layers do donor rodam antes. Quatorze rodam depois. A branch soma na saída da layer 13. A layer original continua lá dentro, com o nome `donor_layer`. O arquivo do Transformers não foi editado.

Anatomia: `reports/QWEN3_ANATOMY.md`.

## B. Diagrama

```
hidden
  |
Qwen3DecoderLayer 13
  input_layernorm -> attention -> residual
  post_attention_layernorm -> SwiGLU MLP -> residual
  |
base
  |
  +-- RMSNorm sem peso -- router linear 1024 -> 8
  |                         top-2
  |                         8 experts SwiGLU, largura 3072
  |
delta
  |
base + alpha * delta
  |
layers 14..27 congeladas
  |
model.norm -> lm_head
```

`alpha` começa em 0 exato, em fp32. O RMSNorm da branch não tem peso. O eps é 1e-6, o mesmo do config.

## C. Parâmetros adicionados

| Grupo | Count | Trainable |
|---|---|---|
| donor | 596049920 | 0 |
| experts | 75497472 | 75497472 |
| router | 8192 | 8192 |
| alpha | 1 | 1 |

`total_added_params` = 75505665. Isso é 12.6677% do donor. Cada expert copia a largura do MLP do Qwen: 3 × 1024 × 3072. Oito deles. O custo está à vista.

## D. VRAM

| Estado | Bytes | GiB |
|---|---|---|
| Donor carregado | 1192114688 | 1.11 |
| Donor + branch, um forward curto | 1354591232 | 1.26 |

A diferença é cerca de 155 MB. Cabe na 5060 Ti. Não é o pico da suíte inteira da Fase 2, que incluiu a geração. É o peso mais um forward com a branch ligada.

## E. Paridade com alpha = 0

Dez documentos da suíte v1, os mesmos arquivos. Donor ao vivo contra Darwin-CL com alpha 0, e os dois contra o `reference.json` da Fase 2.

| Medida | Resultado |
|---|---|
| logits iguais, bit a bit | 10/10 |
| max abs logit delta | 0 |
| mean abs logit delta | 0 |
| KL | 0 |
| top-1 agreement | 1 |
| delta de NLL | 0 |
| delta de nats/byte | 0 |
| delta de bits/byte | 0 |
| delta contra a NLL da v1 | 0 |
| greedy idêntico, inclusive o texto salvo na v1 | 10/10 |

## F. Gradiente

Com alpha = 0 e loss de próximo token:

- gradiente dos experts = 0
- gradiente do router = 0
- gradiente de alpha > 0
- nenhum parâmetro do donor recebe gradiente

Com alpha = 1e-4, sem `optimizer.step`:

- gradiente dos experts > 0
- gradiente do router > 0
- donor continua sem gradiente
- os pesos dos experts e do router ficaram iguais ao clone de antes do backward

Alpha voltou a 0 no fim do teste. Não ficou em 1e-4.

## G. Fingerprint do donor

`d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175`

É o hash da Fase 2. Continua igual com a branch instalada e depois de removê-la. A instalação só renomeia o caminho interno da layer 13 para `donor_layer`. O hash estável desfaz esse prefixo e ignora `.branch.`.

## H. Checkpoint

`artifacts/phase3/plastic_branch_v0.pt` guarda metadados, experts, router e alpha. Não guarda os 596M do donor.

Recusa carregar se a revisão difere, se o fingerprint difere, ou se o SHA-256 do conteúdo não bate. Os três casos foram testados. O load correto devolve 8 experts e top-k 2.

## I. Testes

`python -m unittest tests.test_plastic_branch -v`

5 testes, todos passaram:

- checkpoint recusa donor errado e hash ruim
- fingerprint da Fase 2
- 8 experts, top-k 2, shape `(2, 5, 1024)`, bf16, cuda
- alpha 0 bloqueia gradiente da branch e alpha pequeno abre
- remover a branch devolve os mesmos logits

A suíte completa está em `scripts/run_phase3_gate.py`. Saída: `artifacts/phase3/gate.json`.

## J. Limitações

- Uma layer só. O resto do donor não tem branch.
- 12,7% de parâmetros a mais, todos residentes. Sem paging.
- Nenhum conhecimento novo foi ensinado. Alpha 0 é paridade, não aprendizado.
- O router aprende pelo softmax dos dois escolhidos. O índice discreto do top-k não tem gradiente.
- A suíte continua curta: 10 documentos.
- ROME, MEMIT, SDFT, OPCD, TTT e o leitor contínuo do mini-AGI não foram implementados.

## Removibilidade

Três passos no mesmo objeto: donor puro, donor com branch em alpha 0, branch retirada. Os logits do primeiro e do terceiro são iguais, e também iguais aos do segundo. A branch não grava nada nos pesos do Qwen.

## FINAL MECHANICAL AUDIT

Status: **PLASTIC_BRANCH_MECHANICAL_PASS_CONFIRMED**. A arquitetura não mudou. Não houve `optimizer.step`. A Fase 4 não começou.

Sonda, a mesma frase curta dos testes: `The sum of 17 and 28 is 45.` Loss de próximo token. Alpha aberto usado: `1e-4`. Depois o alpha voltou a 0. Os pesos da branch foram comparados com um clone feito antes do backward e continuaram iguais.

### Alpha = 0

| Grupo | Params | Com grad | Grad ausente | L1 | L2 | max abs |
|---|---|---|---|---|---|---|
| DONOR | 596049920 | 0 | 310 | 0 | 0 | 0 |
| EXPERTS | 75497472 | 24 | 0 | 0 | 0 | 0 |
| ROUTER | 8192 | 1 | 0 | 0 | 0 | 0 |
| ALPHA | 1 | 1 | 0 | 0.031982421875 | 0.031982421875 | 0.031982421875 |

O donor não entra no grafo: `requires_grad` é falso, então o gradiente está ausente. Experts e router entram no grafo por causa de `alpha * delta`, mas com alpha exato em 0 o gradiente que chega neles é o tensor zero. Alpha recebe sinal: a loss depende de `base + alpha * delta`, e `delta` não é zero. L1 = L2 = max abs porque é um escalar.

### Alpha = 1e-4

| Grupo | Params | Com grad | Grad ausente | L1 | L2 | max abs |
|---|---|---|---|---|---|---|
| DONOR | 596049920 | 0 | 310 | 0 | 0 | 0 |
| EXPERTS | 75497472 | 24 | 0 | 0.6519827460870147 | 0.0002343792807146266 | 6.4373016357421875e-06 |
| ROUTER | 8192 | 1 | 0 | 0.0015964135527610779 | 4.402926513655362e-05 | 1.6808509826660156e-05 |
| ALPHA | 1 | 1 | 0 | 0.032470703125 | 0.032470703125 | 0.032470703125 |

Nenhum desses valores foi escrito nos pesos.

### Remoção

A = donor original. C = branch retirada. Dez documentos da suíte v1.

| Comparação | Resultado |
|---|---|
| logits | iguais bit a bit |
| max abs delta | 0 |
| NLL | delta 0 |
| nats/byte | delta 0 |
| greedy | iguais |
| fingerprint | `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175` nos dois |

### Checkpoint

O arquivo válido guarda só a branch e os metadados. SHA-256 do conteúdo: `0fa97c507f28114c4a81ba5fb564f3c6c21fdd1de236391d7151621f37be976d`.

| Caso | Resultado |
|---|---|
| revisão e fingerprint corretos | LOAD PASS, 8 experts, top-k 2 |
| revisão adulterada | LOAD REJECT |
| fingerprint adulterado | LOAD REJECT |
| SHA-256 adulterado | LOAD REJECT |

Nenhum outro donor foi baixado. A adulteração foi só na metadata de uma cópia temporária.

### Inventário

Cinco métodos de `tests/test_plastic_branch.py`, todos PASS, mais as medições deste audit. Nenhum FAIL.

| Teste | Resultado | O que prova |
|---|---|---|
| `test_shape_dtype_device_and_counts` | PASS | donor frozen, 8 experts, top-k 2, shape `(2, 5, 1024)`, dtype bf16, device cuda, alpha fp32 |
| `test_donor_revision_and_fingerprint` | PASS | revisão pinada e fingerprint da Fase 2 com a branch instalada |
| `test_alpha_zero_blocks_branch_gradients_and_small_alpha_opens_them` | PASS | alpha 0 isola experts e router; alpha 1e-4 propaga; donor sem gradiente; pesos não mudam |
| `test_removal_restores_the_same_logits` | PASS | remover a branch devolve os logits do donor |
| `test_checkpoint_rejects_wrong_donor_and_bad_hash` | PASS | load correto, revisão errada, fingerprint errado, hash errado |
| `audit_phase3_evidence.py` | PASS | normas completas, A == C na suíte inteira, os quatro casos de checkpoint |

TOTAL TESTS: 6  
PASSED: 6  
FAILED: 0

Artefato: `artifacts/phase3/final_mechanical_audit.json`.

## Próximo passo

Fase 4, só depois de `APPROVED PHASE 3`: ensinar um domínio novo com o donor congelado, experts, router e alpha. Sem growth, poda ou paging.
