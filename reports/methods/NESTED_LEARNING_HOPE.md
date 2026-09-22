AUTHORS
Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni.
Lido o HTML arXiv:2512.24695. Versão de conferência: NeurIPS 2025.

PAPER
Nested Learning: The Illusion of Deep Learning Architectures. arXiv:2512.24695. https://arxiv.org/abs/2512.24695

REPOSITORY
Não há um repositório oficial único citado no trecho lido. Não foi clonado código HOPE.

LICENSE
arXiv / NeurIPS. Não copiado.

CORE IDEA
O modelo é um conjunto de otimizações aninhadas, cada uma com o seu fluxo de contexto e a sua frequência. HOPE junta um módulo que aprende a própria regra de update com um continuum de memórias, do rápido ao lento. Adam, nesse olhar, já é uma memória associativa dos gradientes.

WHAT PROBLEM IT SOLVES
Um único ritmo de update ou aprende rápido e esquece, ou é estável e não absorve o novo.

WHAT IT DOES NOT SOLVE
Não é um módulo que se cola no Qwen. Copiar o paper inteiro não é o experimento.

MINIMUM IMPLEMENTABLE MECHANISM
Três ritmos: rápido a cada passo, médio a cada N, lento raro ou parado. Medir aquisição, drift, e o que sobra depois de zerar o rápido. Se só isso for implementado, o crédito diz que é um experimento de consolidação multi-timescale inspirado em Nested Learning / HOPE, não HOPE completo.

DEPENDENCIES
Um subconjunto rápido e um médio já existentes. Na nossa trilha, fast weights e experts. O donor continua o lento congelado.

EXPECTED VRAM
A da branch ou do fast weight já medido. O ritmo não acrescenta um segundo modelo.

EXPECTED TRAINING COST
Parecido com o treino atual, com updates mascarados na maior parte dos passos.

HOW IT MAPS TO DARWIN-CL
Estágio X9, depois de haver um rápido e um médio que já aprendem alguma coisa.

WHAT WOULD FALSIFY IT
Frequências 1, 4 e 16 não mudam o compromisso entre aquisição e retenção além do ruído, nas tarefas A, B, C, D em sequência.

CREDIT
Behrouz, Razaviyayn, Zhong, Mirrokni.
