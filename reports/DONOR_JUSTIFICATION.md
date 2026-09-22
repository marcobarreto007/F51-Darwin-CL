# DONOR JUSTIFICATION

Escrito antes do download dos pesos. Nesta etapa só foram lidos metadados da Hugging Face: `config.json`, `tokenizer_config.json` e o tamanho dos arquivos. `model.safetensors` não foi baixado ainda.

## Candidato primário, para o V0

| Campo | Valor |
|---|---|
| Repo | `Qwen/Qwen3-0.6B-Base` |
| Revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| License | Apache-2.0 |
| Arquitetura | `Qwen3ForCausalLM`, `model_type=qwen3` |
| dtype declarado | bfloat16 |
| hidden size | 1024 |
| layers | 28 |
| intermediate | 3072 |
| heads / kv heads | 16 / 8 |
| vocab | 151936 |
| contexto máximo | 32768 posições (`max_position_embeddings`); tokenizer `model_max_length` 131072 |
| RoPE theta | 1000000 |
| RMSNorm eps | 1e-6 |
| ativação | silu |
| embeddings amarrados | `tie_word_embeddings=true` |
| tokenizer | `Qwen2Tokenizer` |
| bos | ausente; `add_bos_token=false` |
| eos / pad do tokenizer | `<|endoftext|>`, id 151643 |
| arquivo de pesos | `model.safetensors`, 1192135096 bytes |
| soma dos arquivos do repo | 1203641805 bytes, cerca de 1.12 GiB |
| parâmetros esperados | no máximo 596067548, porque 1192135096 / 2 é o teto em BF16; o cabeçalho do safetensors reduz um pouco. O `numel()` exato entra no relatório só depois do load |
| VRAM estimada em BF16 | cerca de 1.19 GB só de pesos. A suíte desta fase é curta, então a ativação fica bem abaixo de 1 GB. A RTX 5060 Ti tinha 15449 MiB livres na Fase 0 |

## Por que Base e não Instruct

O benchmark de linguagem mede o texto cru. Um modelo Instruct passa por chat template, e esse template vira uma variável a mais na loss. O Base não exige esse template. A geração do baseline também não usa `apply_chat_template`.

## Por que 0.6B e não 1.7B

O mini-AGI publicado neste commit ocupa cerca de 540M de parâmetros no disco, com uma fração residente. O 0.6B fica na mesma ordem. O 1.7B é cerca de três vezes maior em disco e em VRAM de pesos, e atrasa cada iteração da Fase 3 em diante. O 0.6B cabe na 5060 Ti com folga para o banco plástico que ainda não existe. O 1.7B não é o donor do V0.

## Candidato futuro, não baixado e não executado

| Campo | Valor |
|---|---|
| Repo | `Qwen/Qwen3-1.7B-Base` |
| Revision | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| License | Apache-2.0 |
| Arquitetura | a mesma família: hidden 2048, 28 layers, intermediate 6144, vocab 151936, BF16 |
| `model.safetensors` | 3441185608 bytes, cerca de 3.21 GiB, teto de cerca de 1.72B parâmetros |
| Papel | practical track, só com nova aprovação |

## Riscos da comparação com o mini-AGI

- A métrica primária entre modelos é nats/byte nos mesmos arquivos UTF-8. Loss por token fica interna.
- O 0.6B já vem treinado. O mini-AGI publicado parte do zero. Comparar loss absoluta sem declarar essa vantagem de partida é injusto. O que se compara no aprendizado, mais tarde, é o delta do domínio novo e o drift dos antigos.
- 596M densos ficam todos na placa. O mini-AGI deixa a maior parte do pool em disco. Não são o mesmo uso de VRAM.
- O alfabeto dele é byte. O deste donor é BPE de 151936. O denominador em bytes existe para os dois caberem na mesma unidade, e mesmo assim o primeiro token de cada documento não é alvo de predição. Essa borda está documentada no código e não pode ser silenciosa.
- Números do README dele são "reported by mini-AGI". Só o que esta máquina medir é "reproduced by F51". Os pesos treinados dele não estão no repositório clonado.
- Esta fase não declara vitória. Ela só mede o donor intacto.
