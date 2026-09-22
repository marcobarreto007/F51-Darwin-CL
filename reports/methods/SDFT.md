AUTHORS
Idan Shenfeld, Mehul Damani, Jonas Hübotter, Pulkit Agrawal

PAPER
Self-Distillation Enables Continual Learning. arXiv:2601.19897. https://arxiv.org/abs/2601.19897
Lido o HTML v1 do arXiv, não um resumo de blog.

REPOSITORY
O paper aponta http://idanshenfeld.com/SDFT. Há também SDFTTrainer na documentação do TRL. Nenhum desses códigos foi copiado.

LICENSE
Preprint arXiv. Não copiado.

CORE IDEA
O mesmo modelo é professor e aluno. O professor vê o input e uma demonstração. O aluno vê só o input, gera a própria trajetória, e a perda aproxima o aluno do professor nessa trajetória. É on-policy. SFT comum treina na demonstração pronta, off-policy.

WHAT PROBLEM IT SOLVES
Aprender um comportamento novo a partir de demonstrações sem o esquecimento típico do SFT, porque o aluno treina no que ele mesmo geraria.

WHAT IT DOES NOT SOLVE
Não cria memória seletiva nem paging. Não prova nada sobre o nosso donor. Exige que o modelo já consiga usar a demonstração em contexto.

MINIMUM IMPLEMENTABLE MECHANISM
Dois forwards do mesmo peso. Teacher: prompt + demonstração. Student: gera sem a demonstração. Perda de destilação token a token na trajetória do student. Sem isso, não é SDFT.

DEPENDENCIES
Geração on-policy. Demonstrações do SRB. Donor capaz de condicionar na demonstração.

EXPECTED VRAM
Cerca do dobro de um passo de cross-entropy, porque teacher e student estão ativos. No 0.6B isso ainda cabe na 5060 Ti.

EXPECTED TRAINING COST
Um rollout por exemplo mais um forward do teacher. Mais caro que o cross-entropy atual.

HOW IT MAPS TO DARWIN-CL
Estágio X2, só depois de memorização pura. A branch treinável é o student. O teacher pode ser o mesmo objeto com contexto privilegiado e sem passo de otimização nele.

WHAT WOULD FALSIFY IT
No mesmo orçamento e nos mesmos fatos, SDFT não melhora paráfrase, relação inversa ou retenção além do cross-entropy.

CREDIT
Shenfeld, Damani, Hübotter, Agrawal.
