# Idea ledger

Ideia ainda não testada fica com RESULT = NOT TESTED e DECISION = BACKLOG. Nada abaixo afirma que uma dessas ideias ganhou do mini-AGI.

IDEA
Slow trunk, plastic experts, paging, continual reader

ORIGIN
mini-AGI

AUTHORS
Alexey Borsky

SOURCE
https://github.com/volotat/mini-AGI commit 96784b78008a22c4a1ee2c9961640c22137320ef

LICENSE
MIT

ORIGINAL CONTRIBUTION
O mesmo forward lê e treina. O tronco aprende mais devagar que os experts. O pool pagina e o Adam viaja com o expert.

F51 HYPOTHESIS
Um banco residual em cima de um donor congelado pode aprender depois sem apagar o donor no dia zero.

F51 MODIFICATION
V0 não copia o código. Uma branch só, numa layer, sem paging e sem treino.

EXPERIMENT
Fase 3, só mecânica.

RESULT
NOT TESTED como aprendizado contínuo. A mecânica da branch está na Fase 3, não neste item.

DECISION
BACKLOG

CREDIT
Alexey Borsky, mini-AGI, MIT.

IDEA
Self-Distillation Fine-Tuning / SDFT

ORIGIN
Self-Distillation Enables Continual Learning

AUTHORS
Idan Shenfeld, Mehul Damani, Jonas Hübotter, Pulkit Agrawal

SOURCE
https://arxiv.org/abs/2601.19897

LICENSE
arXiv preprint. Não copiado.

ORIGINAL CONTRIBUTION
O mesmo modelo é professor com uma demonstração e aluno sem ela. A perda é on-policy.

F51 HYPOTHESIS
Pode servir de âncora na Fase 7.

F51 MODIFICATION
Nenhuma.

EXPERIMENT
Nenhum.

RESULT
NOT TESTED

DECISION
BACKLOG

CREDIT
Shenfeld, Damani, Hübotter, Agrawal.

IDEA
On-Policy Context Distillation / OPCD e Online Experiential Learning

ORIGIN
Dois artigos ligados. OPCD internaliza contexto. OEL acumula experiência de uso e consolida com OPCD.

AUTHORS
OEL: Tianzhu Ye, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, Furu Wei. OPCD: autores da página arXiv:2602.12275. Não repito nomes que não li no PDF.

SOURCE
https://arxiv.org/abs/2602.12275 e https://arxiv.org/abs/2603.16856

LICENSE
arXiv preprints. Não copiados.

ORIGINAL CONTRIBUTION
O aluno gera a própria trajetória. O professor viu o contexto. A divergência reversa empurra o aluno.

F51 HYPOTHESIS
Útil depois que existir uso real, não no V0.

F51 MODIFICATION
Nenhuma.

EXPERIMENT
Nenhum.

RESULT
NOT TESTED

DECISION
BACKLOG

CREDIT
Os autores dos dois preprints.

IDEA
In-Place Test-Time Training

ORIGIN
In-Place Test-Time Training, ICLR 2026

AUTHORS
Guhao Feng, Shengjie Luo, Kai Hua, Ge Zhang, Wenhao Huang, Di He, Tianle Cai

SOURCE
https://arxiv.org/abs/2604.06169

LICENSE
arXiv / ICLR. Não copiado.

ORIGINAL CONTRIBUTION
Atualiza a projeção final do MLP no próprio bloco, com alvo de próximo token.

F51 HYPOTHESIS
Pode ser outro sítio plástico se o banco residual não bastar.

F51 MODIFICATION
Nenhuma. A branch V0 é um módulo ao lado, não essa projeção.

EXPERIMENT
Nenhum.

RESULT
NOT TESTED

DECISION
BACKLOG

CREDIT
Feng, Luo, Hua, Zhang, Huang, He, Cai.

IDEA
Test-Time Context Distillation / TTCD

ORIGIN
Learning What to Remember: Test-Time Training via Context Distillation

AUTHORS
Os da página arXiv:2608.01672. Nomes não foram transcritos desta passagem.

SOURCE
https://arxiv.org/abs/2608.01672

LICENSE
arXiv preprint. Não copiado.

ORIGINAL CONTRIBUTION
Um professor de janela longa supervisiona os pesos rápidos de um aluno de janela curta.

F51 HYPOTHESIS
Sinal do que guardar quando o contexto passar do donor.

F51 MODIFICATION
Nenhuma.

EXPERIMENT
Nenhum.

RESULT
NOT TESTED

DECISION
BACKLOG

CREDIT
Autores do preprint.

IDEA
Nested Learning / HOPE

ORIGIN
Nested Learning: The Illusion of Deep Learning Architectures, NeurIPS 2025

AUTHORS
Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni

SOURCE
https://arxiv.org/abs/2512.24695

LICENSE
arXiv / NeurIPS. Não copiado.

ORIGINAL CONTRIBUTION
Várias frequências de atualização no mesmo modelo. HOPE junta memória que se modifica com um continuum de memórias.

F51 HYPOTHESIS
Motivo conceitual para backbone lento e experts rápidos. Não é código para colar.

F51 MODIFICATION
Nenhuma.

EXPERIMENT
Nenhum.

RESULT
NOT TESTED

DECISION
BACKLOG

CREDIT
Behrouz, Razaviyayn, Zhong, Mirrokni.

IDEA
PC-ALM

ORIGIN
Augmented Lagrangian Predictive Coding, writeup da Sakana AI

AUTHORS
Não confirmei a lista de autores nem um id arXiv nesta passagem.

SOURCE
Writeup público de setembro de 2026. Experimento descrito em MLP de imagem, não em modelo de linguagem.

LICENSE
Não copiado.

ORIGINAL CONTRIBUTION
Crédito de supervisão por dinâmica local, sem retropropagação global.

F51 HYPOTHESIS
Nenhuma aplicação no V0.

F51 MODIFICATION
Nenhuma.

EXPERIMENT
Nenhum.

RESULT
NOT TESTED

DECISION
BACKLOG

CREDIT
Sakana AI, pelo writeup lido.

IDEA
Lineage, hash, rollback, edição de modelo

ORIGIN
F51-Darwin-SSD

AUTHORS
Marco Barreto / F51 Labs

SOURCE
references/F51-Darwin-SSD commit 70e135e811392cbaff7be6dc8fabef64d36107b8. checkpoint_root.py, circuits/rollback.py, research/rome e memit.

LICENSE
F51 Labs Proprietary. Nada foi importado. O contrato foi reescrito no checkpoint da branch.

ORIGINAL CONTRIBUTION
Âncora de identidade que não se reescreve, e rollback que confere hash.

F51 HYPOTHESIS
A branch só carrega se a revisão e o fingerprint do donor baterem.

F51 MODIFICATION
Checkpoint mínimo, só os pesos da branch, não os 596M do donor.

EXPERIMENT
Fase 3, teste de recusa. Não é o benchmark de ROME/MEMIT.

RESULT
NOT TESTED como edição de fato. A recusa do checkpoint errado foi testada na Fase 3 e está no relatório dessa fase, não como vitória deste item.

DECISION
BACKLOG para ROME, MEMIT e rollback completo. O hash da branch fica.
CREDIT
F51-Darwin-SSD, uso interno.
