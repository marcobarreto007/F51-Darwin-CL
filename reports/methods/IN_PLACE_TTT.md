AUTHORS
Guhao Feng, Shengjie Luo, Kai Hua, Ge Zhang, Di He, Wenhao Huang, Tianle Cai.
Lido o HTML arXiv:2604.06169. Código citado: https://github.com/ByteDance-Seed/In-Place-TTT

PAPER
In-Place Test-Time Training. ICLR 2026. arXiv:2604.06169. https://arxiv.org/abs/2604.06169

REPOSITORY
https://github.com/ByteDance-Seed/In-Place-TTT
Não clonado.

LICENSE
Preprint / ICLR. Não copiado.

CORE IDEA
Os pesos rápidos são a projeção final do MLP que já existe, em geral W_down. O alvo é o próximo token, não uma reconstrução genérica. O resto do bloco fica lento ou parado. Dá para atualizar durante a inferência sem um módulo novo do tamanho de outro modelo.

WHAT PROBLEM IT SOLVES
Adaptar um transformer já treinado sem acrescentar dezenas de milhões de parâmetros e sem retreinar do zero.

WHAT IT DOES NOT SOLVE
Não decide o que vale lembrar para o futuro. Isso é a pergunta do TTCD. Não prova que bate a nossa branch no SRB.

MINIMUM IMPLEMENTABLE MECHANISM
Uma cópia lenta de W_down e uma cópia rápida. A rápida recebe gradiente do próximo token. A lenta restaura ou recebe atualização rara. O donor original tem de poder voltar ao hash.

DEPENDENCIES
Qwen3MLP.down_proj existe e é (1024, 3072) em cada layer. Não exige a branch de 75M.

EXPECTED VRAM
Um pouco acima do donor, porque o backward atravessa o bloco e há estado do otimizador só nas matrizes rápidas. Bem menos que 75M de experts em bf16 mais Adam.

EXPECTED TRAINING COST
Por token, parecido com um passo curto de LM nas matrizes escolhidas. Mais barato que a branch inteira se só algumas layers forem rápidas.

HOW IT MAPS TO DARWIN-CL
Estágio X6, como arquitetura concorrente, não como camada em cima da branch. ARM A é a branch vencedora até lá. ARM B é o fast weight in-place.

WHAT WOULD FALSIFY IT
Nos mesmos 8 e 64 fatos, a branch ganha em acurácia held-out e em drift com custo parecido, ou o in-place muda o hash do donor sem caminho de restore.

CREDIT
Feng, Luo, Hua, Zhang, He, Huang, Cai. ByteDance Seed e Peking University.
