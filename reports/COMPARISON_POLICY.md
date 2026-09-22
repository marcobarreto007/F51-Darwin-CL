# Comparison policy

Registrado na Fase 2. Vale para todas as fases seguintes.

1. REPORTED. Valores publicados pelo mini-AGI, incluindo o drift de +0.0067 nat no README do commit `96784b78008a22c4a1ee2c9961640c22137320ef`, são identificados como "reported by mini-AGI". Não são medições desta máquina.

2. REPRODUCED. Somente valores obtidos localmente, com comando, log e hash, são identificados como "reproduced by F51".

Os dois rótulos não se misturam numa mesma célula sem a etiqueta.

Limitação: o repositório clonado do mini-AGI não contém os pesos treinados. O README desse commit diz que os pesos ainda não foram publicados. Não há checkpoint local para reproduzir o número publicado sem um treino novo, e um treino novo não é o número publicado.

Métrica primária entre modelos, nos mesmos arquivos brutos:

```
NATS_PER_BYTE = total_negative_log_likelihood / total_UTF8_bytes_evaluated
BITS_PER_BYTE = NATS_PER_BYTE / ln(2)
```

Loss por token é métrica interna de cada modelo.
