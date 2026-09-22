# Research backlog

Nada disto foi implementado. Cada item espera a fase indicada.

NAME
Self-Distillation Fine-Tuning / SDFT

SOURCE
Shenfeld, Damani, Hübotter, Agrawal. Self-Distillation Enables Continual Learning. arXiv:2601.19897. https://arxiv.org/abs/2601.19897

IDEA
O mesmo modelo é professor, condicionado numa demonstração, e aluno, sem a demonstração. A perda destila o professor no aluno em trajetórias geradas pelo aluno.

POSSIBLE_DARWIN_CL_USE
Um professor congelado no donor, e o aluno com o banco plástico, no lugar de supervisionar só o texto novo.

PHASE_TO_REVISIT
Fase 7, ao lado do anchor KL. Não antes do baseline e do primeiro aprendizado.

NAME
On-Policy Context Distillation / OPCD

SOURCE
On-Policy Context Distillation for Language Models. arXiv:2602.12275. https://arxiv.org/abs/2602.12275

IDEA
O aluno gera a própria trajetória sem o contexto extra. A divergência reversa empurra essa trajetória para o professor que viu o contexto.

POSSIBLE_DARWIN_CL_USE
Internalizar um contexto de domínio novo nos experts, em vez de deixar o contexto só no prompt.

PHASE_TO_REVISIT
Fase 7. Não entra no V0.

NAME
Online Experiential Learning

SOURCE
Ye, Dong, Dong, Wu, Huang, Wei. Online Experiential Learning for Language Models. arXiv:2603.16856. https://arxiv.org/abs/2603.16856

IDEA
Trajetórias de uso viram conhecimento transferível, e esse conhecimento entra nos pesos por destilação on-policy, sem recompensa.

POSSIBLE_DARWIN_CL_USE
Depois que existir um loop de uso real, acumular experiência e consolidar no banco plástico.

PHASE_TO_REVISIT
Depois da Fase 8. Não há loop de uso nesta fase.

NAME
In-Place Test-Time Training

SOURCE
Feng, Luo, Hua, Zhang, Huang, He, Cai. In-Place Test-Time Training. ICLR 2026. arXiv:2604.06169. https://arxiv.org/abs/2604.06169

IDEA
Atualiza a projeção final do MLP, no próprio bloco, com alvo de próximo token, sem um módulo lateral.

POSSIBLE_DARWIN_CL_USE
Candidato a sítio plástico dentro do MLP do donor, se o banco residual da Fase 3 for insuficiente.

PHASE_TO_REVISIT
Fase 4, só como hipótese, depois que o banco residual tiver número. Não implementar agora.

NAME
Test-Time Context Distillation / TTCD

SOURCE
Learning What to Remember: Test-Time Training via Context Distillation. arXiv:2608.01672. https://arxiv.org/abs/2608.01672

IDEA
Um professor de janela longa supervisiona os pesos rápidos de um aluno de janela curta, para guardar o que serve ao futuro e não o passado inteiro.

POSSIBLE_DARWIN_CL_USE
Sinal de o que o expert novo deve memorizar quando o contexto passar do que o donor já cobre.

PHASE_TO_REVISIT
Fase 6 ou Fase 9. Não agora.

NAME
Nested Learning / HOPE

SOURCE
Behrouz, Razaviyayn, Zhong, Mirrokni. Nested Learning: The Illusion of Deep Learning Architectures. NeurIPS 2025. arXiv:2512.24695. https://arxiv.org/abs/2512.24695

IDEA
O modelo é uma pilha de otimizações com frequências diferentes. HOPE junta memória que se modifica com um continuum de memórias lentas e rápidas.

POSSIBLE_DARWIN_CL_USE
Justificativa conceitual para três velocidades, backbone lento e experts rápidos. Não é um módulo para copiar no V0.

PHASE_TO_REVISIT
Fase 5, na hora de ler a varredura de taxas. Não implementar a arquitetura HOPE.

NAME
PC-ALM

SOURCE
Sakana AI. Augmented Lagrangian Predictive Coding: training 1000-layer networks without backpropagation. Writeup público, setembro de 2026. Não foi confirmado um id arXiv nesta passagem. O experimento descrito é MLP residual em imagem, não modelo de linguagem.

IDEA
Cada camada resolve um sistema local, com multiplicador de Lagrange, e o crédito da supervisão se espalha sem uma retropropagação global.

POSSIBLE_DARWIN_CL_USE
Nenhuma no Darwin-CL V0. Não há evidência, no material lido, de que isso treine um transformer de linguagem nem de que reduza esquecimento.

PHASE_TO_REVISIT
Não revisitar antes de existir um resultado de linguagem. Fora do plano até lá.
