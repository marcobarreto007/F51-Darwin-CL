AUTHORS
Não confirmei a lista de autores numa página arXiv. O material lido é o writeup técnico da Sakana AI, repercutido em setembro de 2026.

PAPER
Augmented Lagrangian Predictive Coding: training 1000-layer networks without backpropagation. Não há id arXiv confirmado nas fontes lidas para este estágio.

REPOSITORY
Não encontrei um repositório oficial de LLM. A implementação de referência descrita é JAX, em CPU, para MLP residual.

LICENSE
Writeup da Sakana AI. Não copiado.

CORE IDEA
Cada layer tem um sistema dinâmico local. Multiplicadores de Lagrange carregam o crédito da supervisão entre vizinhos. Não há uma retropropagação global em três fases. O experimento mostrado treina MLP residual até 1000 layers em MNIST, Fashion-MNIST e afins, perto do backpropagation.

WHAT PROBLEM IT SOLVES
Crédito local em redes muito fundas, no regime em que predictive coding comum perde o sinal.

WHAT IT DOES NOT SOLVE
Não mostra transformer, não mostra linguagem, não mostra esquecimento catastrófico. Transferir isso para o Qwen é uma hipótese, não um resultado.

MINIMUM IMPLEMENTABLE MECHANISM
Uma rede pequena igual à branch, treinada com backpropagation e com um update local inspirado em PC-ALM, no mesmo problema de 8 fatos e no mesmo budget. Só então, se houver sinal estável, um protótipo no transformer.

DEPENDENCIES
Um toy que caiba em CPU ou numa fração da placa. Não depende da branch de 75M para o primeiro teste.

EXPECTED VRAM
O toy é pequeno. O paper de imagem não é o custo do 0.6B.

EXPECTED TRAINING COST
O writeup não dá custo de linguagem. O toy de 8 fatos é barato.

HOW IT MAPS TO DARWIN-CL
Estágio X12, por último. Não entra no core por ser interessante.

WHAT WOULD FALSIFY IT
No toy, o update local não alcança a acurácia do backpropagation no mesmo budget, ou é instável. Nesse caso não há protótipo no Qwen.

CREDIT
Sakana AI, pelo writeup lido. Autores nominais ficam em aberto até haver um PDF com a lista.
