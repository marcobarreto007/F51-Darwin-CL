AUTHORS
Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, Furu Wei. Microsoft Research.
Lido o HTML arXiv:2602.12275v1.

PAPER
On-Policy Context Distillation for Language Models. arXiv:2602.12275. https://arxiv.org/abs/2602.12275

REPOSITORY
https://github.com/microsoft/LMOps/tree/main/opcd
Não clonado e não copiado.

LICENSE
Preprint arXiv. Código no LMOps, licença do repositório Microsoft. Não copiado.

CORE IDEA
O student gera a trajetória sem o contexto privilegiado. A perda é KL reverso entre o student e um teacher que viu o contexto, nas posições da trajetória do student. Forward KL off-policy cobre modos demais e sofre exposure bias. O KL reverso empurra o student para o que o teacher considera provável.

WHAT PROBLEM IT SOLVES
Conhecimento que só existe no prompt passar para os pesos, de modo que a avaliação sem o prompt ainda responda.

WHAT IT DOES NOT SOLVE
Não decide sozinho o que esquecer. Não é um loop online de várias rodadas. Isso é o OEL, paper seguinte.

MINIMUM IMPLEMENTABLE MECHANISM
Contexto privilegiado com os fatos. Student gera sem esse contexto. Teacher com o contexto atribui logprob à mesma trajetória. Reverse KL. Na avaliação, o student não vê o contexto.

DEPENDENCIES
OPCD de verdade exige geração do student e KL reverso. Destilar a frase pronta off-policy não conta.

EXPECTED VRAM
Dois forwards por trajetória no 0.6B. Cabe na 5060 Ti para sequências curtas do SRB.

EXPECTED TRAINING COST
Maior que cross-entropy: um rollout mais o teacher.

HOW IT MAPS TO DARWIN-CL
Estágio X4. A branch é o que pode mudar. O donor fica congelado. O teste é: com contexto o teacher acerta, sem contexto o student erra antes, acerta depois, e erra de novo se a branch sai.

WHAT WOULD FALSIFY IT
Depois do treino, student sem contexto não fica melhor que o cross-entropy no mesmo budget, ou o ganho permanece depois de remover a branch.

CREDIT
Ye, Dong, Wu, Huang, Wei. Microsoft Research.
