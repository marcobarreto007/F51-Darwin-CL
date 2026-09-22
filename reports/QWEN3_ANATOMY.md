# Qwen3 anatomy

Lido do objeto carregado, não de um diagrama de memória.

Donor: `Qwen/Qwen3-0.6B-Base`  
Revision: `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`  
Comando: `scripts/dump_qwen3_anatomy.py`

## Classes

| Peça | Classe real |
|---|---|
| Modelo | `transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM` |
| Miolo | `Qwen3Model` |
| Decoder layer | `Qwen3DecoderLayer` |
| Attention | `Qwen3Attention` |
| MLP | `Qwen3MLP` |
| RMSNorm | `Qwen3RMSNorm` |

28 layers. `hidden_size` 1024. `intermediate_size` 3072. `rms_norm_eps` 1e-6.

## O que cada layer contém

Filhos de `model.layers[0]`, na ordem do módulo:

- `self_attn` — `Qwen3Attention`
- `mlp` — `Qwen3MLP`
- `input_layernorm` — `Qwen3RMSNorm`
- `post_attention_layernorm` — `Qwen3RMSNorm`

Attention, nomes reais:

- `self_attn.q_proj` Linear `(2048, 1024)`
- `self_attn.k_proj` Linear `(1024, 1024)`
- `self_attn.v_proj` Linear `(1024, 1024)`
- `self_attn.o_proj` Linear `(1024, 2048)`
- `self_attn.q_norm` `Qwen3RMSNorm` `(128,)`
- `self_attn.k_norm` `Qwen3RMSNorm` `(128,)`

MLP, nomes reais. É SwiGLU:

- `mlp.gate_proj` `(3072, 1024)`
- `mlp.up_proj` `(3072, 1024)`
- `mlp.down_proj` `(1024, 3072)`
- `mlp.act_fn` `SiLUActivation`

O forward da classe `Qwen3MLP` é `down_proj(silu(gate_proj(x)) * up_proj(x))`.

## Caminho residual

O forward de `Qwen3DecoderLayer` faz dois residuais, nesta ordem:

```
residual = hidden
hidden = input_layernorm(hidden)
hidden = self_attn(...)
hidden = residual + hidden

residual = hidden
hidden = post_attention_layernorm(hidden)
hidden = mlp(hidden)
hidden = residual + hidden
return hidden
```

A layer devolve um tensor, não uma tupla. Quem chama é `Qwen3Model`, e a chamada lê `decoder_layer.attention_type` para escolher a máscara. Um wrapper tem de expor esse atributo.

RMSNorm do donor também existe em `model.norm`, peso `(1024,)`. O embedding é `model.embed_tokens.weight` `(151936, 1024)`. A cabeça é `lm_head.weight`, do mesmo formato, amarrada na config.

## Onde a branch V0 entra

Índice 13, contado do zero. São 28 layers, índices 0 a 27. `(28 // 2) - 1 = 13`.

Assim, 14 layers do donor rodam antes da branch (0 a 13) e 14 layers do donor rodam depois (14 a 27). A branch soma em cima da saída da layer 13. Não substitui a layer. Não edita o arquivo do Transformers.
