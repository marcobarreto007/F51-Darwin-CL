AUTHORS
Alexey Borsky

PAPER
Não há paper separado no checkout. O documento de autoridade é o README do repositório, commit 96784b78008a22c4a1ee2c9961640c22137320ef.

REPOSITORY
https://github.com/volotat/mini-AGI
Cópia local somente leitura: references/mini-AGI

LICENSE
MIT. Copyright (c) 2026 Alexey Borsky.

CORE IDEA
Um modelo de bytes treinado do zero. Dois blocos densos uma vez, depois um bloco repetido até 24 vezes. O pool de experts SwiGLU é paginado. Ler e treinar são o mesmo forward. O tronco, por onde passa todo caractere, aprende a 0.1× da taxa dos experts.

WHAT PROBLEM IT SOLVES
Esquecimento catastrófico quando um fluxo contínuo ensina um assunto só. No README desse commit, 524 mil caracteres de xadrez levam os assuntos não lidos de +2.2300 nats para +0.0067 nats quando o tronco está a 0.1×.

WHAT IT DOES NOT SOLVE
Não entrega um modelo de fronteira. O próprio README chama o resultado de toy e diz que o texto ainda repete. Os pesos treinados não estão publicados. Não parte de um donor pretrained.

MINIMUM IMPLEMENTABLE MECHANISM
Dois grupos de AdamW: tronco com lr × 0.1, experts com lr cheio. O mesmo forward recebe o alvo.

DEPENDENCIES
O multiplicador só existe se o código separar tronco e pool. A poda por falta de chamada é do pool paginado, não do SharedPool, que poda pelo gate.

EXPECTED VRAM
O desenho publicado cabe numa placa de 8 GB porque só o conjunto residente fica na VRAM.

EXPECTED TRAINING COST
Um passo por chunk de caracteres, do zero, por centenas de milhões de caracteres no relato do autor.

HOW IT MAPS TO DARWIN-CL
O donor fica no papel de tronco lento ou congelado. A branch é o pool. A Fase 4R já isolou o alpha. O estágio X10 é quem mede se a receita transfere para um donor pretrained.

WHAT WOULD FALSIFY IT
No nosso donor, tronco a 0.1× não reduz o drift antigo em relação a tronco congelado, no mesmo orçamento e na mesma suíte.

CREDIT
Alexey Borsky, mini-AGI, MIT.
