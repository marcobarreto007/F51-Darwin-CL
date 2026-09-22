# PHASE 2 REPORT — donor baseline

Status: **PHASE 2 complete**. Gate: **DONOR_BASELINE_PASS**.

Nenhum treino. Nenhum expert, adapter ou LoRA. Os pesos não mudaram. A Fase 3 não começou.

## Correções de auditoria, registradas antes desta fase

1. A frase "Darwin antigo não tem essas cinco" foi substituída, no relatório da Fase 1 e no arquivo mestre, por: Darwin antigo não implementa esses cinco mecanismos como a mesma stack integrada e causalmente validada de continual learning. Existem mecanismos parcialmente relacionados no Darwin antigo, mas não equivalentes ao pipeline do mini-AGI.

2. A métrica primária entre modelos é nats/byte, nos mesmos arquivos UTF-8.

```
NATS_PER_BYTE = total_negative_log_likelihood / total_UTF8_bytes_evaluated
BITS_PER_BYTE = NATS_PER_BYTE / ln(2)
```

Loss por token ficou registrada só como métrica interna.

## DONOR JUSTIFICATION

O texto completo, escrito antes do download dos pesos, está em `reports/DONOR_JUSTIFICATION.md`. Resumo do que foi verificado na API antes de baixar `model.safetensors`:

| Campo | Qwen/Qwen3-0.6B-Base | Qwen/Qwen3-1.7B-Base, não baixado |
|---|---|---|
| Revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| License | Apache-2.0 | Apache-2.0 |
| Arquitetura | Qwen3ForCausalLM, 28 layers, hidden 1024, intermediate 3072, 16 heads, 8 kv heads, vocab 151936 | mesma família, hidden 2048, intermediate 6144 |
| dtype | bfloat16 | bfloat16 |
| Tokenizer | Qwen2Tokenizer, sem BOS, eos `<|endoftext|>` | igual |
| Pesos | `model.safetensors`, 1192135096 bytes | 3441185608 bytes |
| VRAM de pesos | cerca de 1.19 GB | cerca de 3.44 GB |

Base, e não Instruct, porque o chat template seria uma variável a mais na loss de texto cru. 0.6B, e não 1.7B, porque fica na ordem de tamanho do mini-AGI (cerca de 540M no disco) e cabe na 5060 Ti com folga. O 1.7B continua candidato futuro. Não é o donor do V0.

`numel()` depois do load: **596049920** parâmetros. O teto estimado pelo tamanho do arquivo era 596067548. A diferença é o cabeçalho do safetensors.

## Como nats/byte trata borda, BOS, EOS e padding

Código: `src/darwin_cl/eval/boundaries.py` e `src/darwin_cl/eval/metrics.py`.

- Cada documento é pontuado sozinho. A NLL não atravessa a fronteira do próximo documento.
- `add_special_tokens=False`. Este tokenizer tem `add_bos_token=false`. BOS não entra.
- O primeiro token é só contexto. Não entra na NLL e os bytes dele não entram no denominador. A coluna `unscored_boundary_bytes` mostra quantos bytes do arquivo ficaram de fora por isso.
- EOS não é acrescentado. O id 151643 só seria pontuado se o arquivo cru o contivesse. Estes arquivos não contêm.
- Não há padding. Token marcado `pad` ou `special` não soma NLL e não soma bytes.
- O denominador é o comprimento UTF-8 dos caracteres cobertos pelos tokens pontuados, não o número de caracteres.

Teste unitário: `tests/test_nats_per_byte.py`, 5 testes, todos passaram. O caso `áx` confirma que o prefixo não pontuado (á, 2 bytes) sai do denominador e que um pad artificial não entra.

## Ambiente

| Item | Valor |
|---|---|
| dtype | bfloat16 |
| device | cuda:0, NVIDIA GeForce RTX 5060 Ti |
| seed | 0 |
| geração | greedy, `do_sample=False`, 16 tokens novos |
| SDPA | flash e mem-efficient desligados; backend math ligado |
| transformers | 4.57.3 |
| torch | 2.8.0+cu129 |
| CUDA | 12.9 |
| pico de VRAM | 1263550464 bytes, cerca de 1.18 GB |
| tempo de parede | 100.67 s, incluindo o download |
| fingerprint antes | `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175` |
| fingerprint depois | o mesmo hash |
| treino | false |

Cada documento foi pontuado duas vezes e gerado duas vezes. NLL e texto greedy bateram. Se não batessem, o script saía com erro.

## Suíte v1

Arquivos em `benchmarks/baseline/v1/`. Não existe domínio F.

| Conjunto | SHA-256 do arquivo | tokens | tokens pontuados | bytes avaliados | NLL | nats/token (interno) | nats/byte | bits/byte |
|---|---|---|---|---|---|---|---|---|
| A language | `dc859c331524effa1ea3c7bac56652a0e561a927137cb3dd25141963316171cb` | 66 | 64 | 245 | 221.7544 | 3.4649 | 0.905120 | 1.305812 |
| B code | `169fd78fcedc3333bc862254dadd1f43f947889e946d09a9267174bc5081ea7c` | 56 | 54 | 200 | 39.0181 | 0.7226 | 0.195091 | 0.281456 |
| C mathematics | `acf001e5581ef116b8d12c80c6ac963441109061cfa7cbdd2df0f87096c1cf9d` | 31 | 29 | 71 | 56.5275 | 1.9492 | 0.796162 | 1.148619 |
| D factual | `fb370b576f07a1d1b9e82ae0c1dca82e5eec235f47296869c460a2953d225eb2` | 19 | 17 | 89 | 24.2197 | 1.4247 | 0.272132 | 0.392603 |
| E reasoning | `28c3bc0cc5890ccfe8699796d73533232a4705fb927183c82fdaa872e537a05f` | 51 | 49 | 189 | 108.3985 | 2.2122 | 0.573537 | 0.827439 |

Estes nats/byte são "reproduced by F51" no donor congelado. Não são comparação com o mini-AGI. A suíte é curta de propósito. O código deu 0.195 nats/byte porque os trechos são curtos e previsíveis, não porque o modelo tenha sido medido num corpus de código.

Greedy, primeira continuação de cada item:

- A1: `, e a água estava quente. O rio passava debaixo da`
- A2: ` and wood smoke. The air was thick with the scent of pine and wood smoke`
- B1: `    for i in range(1, n+1):\n        result *= i`
- B2: `    return text == text[::-1]\n\ndef is_palindrome2(text):\n   `
- C1: ` 45. What is the sum of 17 and 28`
- C2: ` A. A second rectangle has width 6 and height 12. What`
- D1: ` Paris. The capital of Germany is Berlin. The capital of Italy is Rome.`
- D2: ` 0°C and boils at 100°C. If a 1`
- E1: ` is not in the red box. This is a valid argument. What is the`
- E2: ` is off. Which of the following, if true, most seriously weakens the`

Artefato: `artifacts/baseline/v1/reference.json`  
SHA-256: `2c9904f3412d14bc0e2a3c13562ff33ebff5441decfb089b3a085a4cda5c13fb`

## Política de comparação

`reports/COMPARISON_POLICY.md`.

- "reported by mini-AGI": o que o README dele publica, inclusive +0.0067 nat. Não foi medido aqui.
- "reproduced by F51": só o que esta máquina rodou. Nesta fase, a tabela acima.
- Os pesos treinados do mini-AGI não estão no clone. Não dá para reproduzir o número publicado carregando um checkpoint dele.

## Backlog

`reports/RESEARCH_BACKLOG.md` lista SDFT, OPCD, Online Experiential Learning, In-Place TTT, TTCD, Nested Learning / HOPE e PC-ALM. Nenhum foi implementado.

## Gate

| Condição | Resultado |
|---|---|
| donor carrega | sim, revision fixa |
| pesos não mudam | fingerprint igual |
| generation funciona | sim, repetida |
| NLL funciona | sim, repetida |
| nats/byte passa testes | 5/5 |
| datasets têm hash | sim |
| outputs reproduzíveis | sim, segunda passagem idêntica |
| baseline salvo | sim |
| nenhum treino | sim |

## Auditoria da métrica

Feita depois do gate, sem iniciar a Fase 3. Relatório: `reports/PHASE_2_METRIC_AUDIT.md`.

A fórmula da v1 já era EXACT_SCORED_UTF8: o shift causal prevê t1..tn, e os bytes de t0 ficam fora do denominador. Dois processos novos deram delta 0 em NLL, nats/byte, bits/byte e greedy, e o mesmo fingerprint de antes. `artifacts/baseline/v1/reference.json` não foi reescrito. O hash continua `2c9904f3412d14bc0e2a3c13562ff33ebff5441decfb089b3a085a4cda5c13fb`. Não existe v2.

A média por documento não substitui o agregado. O agregado oficial continua sendo a soma das NLL dividida pela soma dos bytes pontuados. Em A language isso é 0.905120. A média não ponderada dos dois documentos é 0.907186.

## Próximo passo

Fase 3, só depois de `APPROVED PHASE 2`: banco residual de 8 experts, gate perto de zero, e a medição de paridade contra este baseline. Sem crescimento, poda ou paging.
