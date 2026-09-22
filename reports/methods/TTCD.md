AUTHORS
Zixuan Wang, Xingyu Dang, Rui-Jie Zhu, Zixin Wen, Hengyu Fu, Wenhao Chai, Jason D. Lee.
Lido o HTML arXiv:2608.01672v1. Código citado pelos autores: https://github.com/dangxingyu/ttcd

PAPER
Learning What to Remember: Test-Time Training via Context Distillation. arXiv:2608.01672. https://arxiv.org/abs/2608.01672

REPOSITORY
https://github.com/dangxingyu/ttcd
Não clonado.

LICENSE
Preprint arXiv. Não copiado.

CORE IDEA
Um teacher de janela longa e um student de janela curta veem o mesmo instante. A diferença do estado escondido diz o que o passado remoto acrescenta. Essa diferença supervisiona o fast weight, no paper a down-projection, para guardar o que ajuda a predição futura. Não é "guardar tudo" nem só o próximo token local.

WHAT PROBLEM IT SOLVES
Sob capacidade limitada, escolher o que escrever na memória paramétrica.

WHAT IT DOES NOT SOLVE
Não é um método de fatos relacionais curtos por si. O paper mede linguagem de contexto longo. Não transfere automaticamente para 8 fatos do SRB.

MINIMUM IMPLEMENTABLE MECHANISM
Dois contextos, longo e curto. Loss entre o estado do teacher parado e a função do fast weight no estado do student. No experimento sintético: 100 fatos, 10 úteis depois, contra uma estratégia que memoriza os 100.

DEPENDENCIES
Um fast weight, seja down-projection ou a branch. Sem uma memória de capacidade limitada a comparação não existe.

EXPECTED VRAM
Dois forwards curtos no 0.6B por passo de supervisão.

EXPECTED TRAINING COST
Maior que um próximo-token simples, porque há teacher de contexto longo.

HOW IT MAPS TO DARWIN-CL
Estágio X8. A pergunta é seletiva: os 10 fatos relevantes contra a interferência dos 90. Não entra antes da memorização pura.

WHAT WOULD FALSIFY IT
A estratégia seletiva não reduz interferência nem drift em relação a memorizar tudo, no mesmo orçamento de update.

CREDIT
Wang, Dang, Zhu, Wen, Fu, Chai, Lee.
