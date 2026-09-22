AUTHORS
Tianzhu Ye, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, Furu Wei.
Lido o HTML arXiv:2603.16856v1. O bibtex do repositório confirma esses nomes.

PAPER
Online Experiential Learning for Language Models. arXiv:2603.16856. https://arxiv.org/abs/2603.16856

REPOSITORY
https://github.com/microsoft/LMOps/tree/main/oel
O paper também cita aka.ms/oel-code. Não clonado.

LICENSE
Preprint arXiv. Não copiado.

CORE IDEA
Dois estágios em loop. No lado do uso, trajetórias viram conhecimento transferível. No lado do treino, esse conhecimento entra nos pesos por on-policy context distillation, sem recompensa e sem acesso ao ambiente. A rodada seguinte coleta trajetórias melhores.

WHAT PROBLEM IT SOLVES
Melhorar a partir da própria experiência de uso, e testar se a experiência extraída vale mais do que destilar a conversa bruta.

WHAT IT DOES NOT SOLVE
Não é um agente geral. O paper mede jogos de texto. Não transfere sozinho para o nosso SRB.

MINIMUM IMPLEMENTABLE MECHANISM
Três rodadas curtas. Cada uma: tentar as tarefas, extrair lições das trajetórias, consolidar com OPCD, tentar de novo sem mostrar as respostas anteriores. Um braço destila a trajetória crua. O outro destila só o extrato.

DEPENDENCIES
OPCD funcionando. Sem isso o OEL não tem o passo de consolidação que o paper usa.

EXPECTED VRAM
A de um passo OPCD no 0.6B.

EXPECTED TRAINING COST
Três rodadas de tentativa mais consolidação. Maior que um OPCD único.

HOW IT MAPS TO DARWIN-CL
Estágio X5, só se X4 passar. Micro-loop no SRB, não um agente.

WHAT WOULD FALSIFY IT
O extrato não supera a destilação da trajetória crua em acurácia por rodada, no mesmo número de tokens.

CREDIT
Ye, Dong, Dong, Wu, Huang, Wei.
