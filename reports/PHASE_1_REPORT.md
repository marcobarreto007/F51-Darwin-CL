# PHASE 1 REPORT — clones e decisão de arquitetura

Status: **PHASE 1 complete**. Os dois checkouts estão limpos. Nenhum arquivo dentro de `references/` foi editado. Não houve escolha de donor, download, instalação, treino, nem código de Plastic Expert Bank.

Data da inspeção: 2026-09-22T06:52:38-04:00.

## Git

| | mini-AGI | F51-Darwin-SSD |
|---|---|---|
| Path | `references/mini-AGI` | `references/F51-Darwin-SSD` |
| Remote | `https://github.com/volotat/mini-AGI.git` | `https://github.com/marcobarreto007/F51-Darwin-SSD.git` |
| Branch | `main` (`main...origin/main`) | `main` (`main...origin/main`) |
| SHA | `96784b78008a22c4a1ee2c9961640c22137320ef` | `70e135e811392cbaff7be6dc8fabef64d36107b8` |
| Assunto | `#4` | `chore: remover .github/workflows (sem scope no token)` |
| Autor do commit | Alexey Borsky | Marco Barreto |
| Data do commit | 2026-09-22T03:41:31+03:00 | 2026-08-01T14:48:32-04:00 |
| `status --porcelain` | vazio | vazio |
| Licença | MIT, Copyright (c) 2026 Alexey Borsky | F51 Labs Proprietary License, Copyright (c) 2026 F51 Labs |

Logs: `reports/logs/phase1/mini-agi-git.txt`, `reports/logs/phase1/darwin-git.txt`.

## O que foi lido

mini-AGI, no código: `minagi/model.py`, `pool.py`, `paged.py`, `plasticity.py`, `recur.py`, `stream.py`, `store.py`, `config.yaml`, `train.py`, mais `LICENSE` e o trecho do `README.md` que publica a tabela de esquecimento.

Darwin, no código: `src/f51_darwin/organism/checkpoint_root.py`, `src/f51_darwin/circuits/rollback.py`, `src/f51_darwin/darwin_x_core/moe.py`, `src/f51_darwin/heartbeat.py`, `src/f51_darwin/organism/layer_transplant.py`, `src/configs/darwin_x_1.6b_smol_transplant.yaml`, `research/memit_identity.py`, `research/rome_identity_edit.py`. A lista de testes causais e de transplante foi inventariada por nome de arquivo. Esses testes não foram executados.

## Como o mini-AGI aprende, e o que depende do quê

O passo contínuo está em `train.py` `cmd_read` e em `minagi/stream.py` `Reader.step`: o forward recebe o alvo. Não há um regime separado de fine-tune.

O tronco e o pool são grupos diferentes de AdamW. `train.py` `_split_trunk_pool` (por volta da linha 1754) separa os parâmetros. O grupo `trunk` recebe `lr * trunk_lr_mult`; o grupo `pool` recebe `lr`. Em `config.yaml`, `trunk_lr_mult` é 0.1. O README desse mesmo commit publica a medição: 524 mil caracteres só de xadrez, assuntos não lidos sobem +2.2300 nats com tronco na mesma taxa dos experts, e +0.0067 nats com tronco a 0.1× (99.84% retido contra o acaso). Essa medição não foi reexecutada aqui. É evidência publicada no repositório, não um número nosso.

O expert é um SwiGLU (`pool.py` `Expert`, projeções `w1`, `w3`, `w2`). O router de token é o `PooledMLP`. A escolha do conjunto residente não usa o embedding cru: `paged.py` `demand` olha o estado escondido dos call sites, e `recur.py` `choose_for` trata um prompt como troca de assunto.

Crescimento e poda não são a mesma regra nos dois pools, e isso importa:

- `PagedPool.add_experts` monta o expert novo por recombinação de unidades escondidas já treinadas (`recombine=16`). Clone com ruído e expert aleatório estão descritos ali como piores. O gate de nascimento default é 0.001.
- `PagedPool.prune` apaga quem não foi chamado para a placa na janela `survival`. O comentário no código diz que o gate é anti-preditivo: os gates menores são os experts mais ocupados. A poda paginada não olha o gate.
- `SharedPool.prune` faz o contrário de propósito: nesse pool todo expert está residente, o auxiliar de balanceamento uniformiza o uso, e o teste de contribuição é o gate. `SharedPool.add_experts` ainda nasce por clone mais ruído quando há `seed_from`.

A evidência publicada de esquecimento é do sistema paginado. Copiar o teste de poda do `SharedPool` para um banco paginado inverteria a regra que o próprio código marca como errada.

O Adam viaja com o expert. `store.py` `_expert_moments` grava `exp_avg` e `exp_avg_sq` por id de expert. `paged.py` `swap_to` diz que deixar o momento para trás entrega o histórico de um expert ao próximo ocupante do slot.

O controlador de taxa está em `plasticity.py` `Plasticity.observe` / `_verdict`. Não há horizonte. A taxa se move a cada avaliação de held-out, por um fator contínuo. `train.py` chama isso no lugar de cosseno. O freio `honest` do crescimento segura expert novo quando treino e held-out se separam.

A recorrência está em `recur.py` `RecurCoder.forward`: dois blocos de prelúdio, um bloco repetido até 24, parada tipo PonderNet, loss ponderada pela probabilidade de parar mais um KL. Isso é o corpo do modelo de bytes. Não é um módulo que se encaixa num donor de tokens.

Dependências, em cadeia: o multiplicador de tronco só existe porque `_split_trunk_pool` acha o pool; o Adam móvel só é correto se o swap e o `store` usam o mesmo id de expert; a poda paginada e o freio de crescimento leem a mesma staleness (`dying`); o controlador de taxa precisa de um held-out de verdade (`FolderEvaluator`); a demanda precisa do estado do trecho anterior, não do embedding. A recorrência não é pré-requisito do tronco lento nem do pool.

## O que o Darwin antigo tem, e o que não demonstrou

Contrato de linhagem: `checkpoint_root.py` `write_lineage_root_identity` recusa mudar âncoras (`model_name`, `tokenizer_id`, `creation_mode`, `created_at`, `source_checkpoint_sha256`). `structural_config_identity` é a trava de arquitetura.

Rollback: `circuits/rollback.py` verifica SHA-256 antes de carregar e publica snapshot sem substituir o destino em silêncio (`_publish_staging_no_replace`, `create_rollback_child`).

Paridade de donor: `src/configs/darwin_x_1.6b_smol_transplant.yaml` registra um build de 2026-07-28 pior que o aleatório (bpb 8.935 contra 4.908) quando RoPE, escala residual e RMSNorm não batiam com o SmolLM. Os valores que o próprio arquivo exige para paridade zero-shot são `rope_style: llama`, `residual_scale_multiplier: 4.0` (para a escala efetiva ser 1.0 em 16 camadas) e `rms_norm_eps: 1.0e-5` com upcast fp32. Isso é um contrato de montagem, não um resultado de aprendizado contínuo. O caminho aponta para SmolLM2, inclusive `research/memit_identity.py`, que abre um snapshot local `HuggingFaceTB--SmolLM2-1.7B-Instruct`. Esse snapshot não foi carregado nesta fase.

ROME e MEMIT são scripts em `research/`. Não são chamados pelo forward do `DarwinXModel`. MEMIT descreve o update rank-1 de Meng et al. e edita o SmolLM2-1.7B-Instruct num caminho local. Nenhum artefato de resultado foi aberto aqui, então esta fase não declara efeito comportamental.

MoE: `darwin_x_core/moe.py` `DeepSeekStyleMoE` tem experts fixos, eviction Nitro para CPU, apoptose, sono e fusão. Não é um pool em disco com Adam por arquivo nem poda por falta de chamada.

Aprendizado na inferência que não mexe no peso: `heartbeat.py` `TestTimeMemory.write_if_surprised` grava chave e valor com `detach`, dentro de `torch.no_grad()`. O comentário de `retrieve` diz que o slot já nasce detached. Isso não é atualização do backbone.

Órgãos (GABA, JEPA, Ghost, Soul, Senado, Heartbeat) existem no checkout de 2026-08-01. Não há, neste código, uma medição do tipo "524 mil caracteres de um assunto, drift dos outros". Não entram no Darwin-CL V0.

## Tabela

| FEATURE | MINI-AGI | DARWIN OLD | DARWIN-CL DECISION | ACTION |
|---|---|---|---|---|
| Ler e treinar no mesmo forward | `stream.py` `Reader.step`; `train.py` `cmd_read` | treino e serve separados; TTM não atualiza o peso | o passo do domínio novo usa o forward com alvo, só nos parâmetros plásticos | REIMPLEMENT |
| Tronco mais lento que o expert | `_split_trunk_pool`; `trunk_lr_mult: 0.1` | não há essa partição | Fase 4 congela o backbone; Fase 5 varre 0, 0.01, 0.03, 0.05, 0.10 | REIMPLEMENT |
| Banco de experts SwiGLU | `pool.py` `Expert` | `DeepSeekStyleMoE` dentro de cada bloco | V0: banco residual, 8 experts, top-2, fora do MoE antigo | REIMPLEMENT |
| Router sem rótulo de assunto | `PooledMLP`; `paged.demand` | gate do MoE | router do banco; o rótulo existe só na avaliação | REIMPLEMENT |
| Nascimento por recombinação e freios | `PagedPool.add_experts`; `AutoGrow` | `_create_expert`, apoptose | fora do V0 | EXPERIMENT_LATER |
| Poda | `PagedPool.prune` por falta de chamada; `SharedPool.prune` por gate | sono por magnitude | fora do V0; se vier, a regra é a do pool paginado | EXPERIMENT_LATER |
| Paging e Adam no arquivo do expert | `paged.swap_to`; `store._expert_moments` | Nitro manda expert frio para CPU | fora do V0 | EXPERIMENT_LATER |
| Recorrência adaptativa | `recur.py` `RecurCoder`, teto 24 | pilha fixa | não entra no donor | IGNORE |
| Controlador de LR por held-out | `plasticity.py` `Plasticity` | sem equivalente | Fase 6 | COPY_WITH_ATTRIBUTION |
| Linhagem e hash de checkpoint | `store.save`, um npz por expert | `checkpoint_root.py` | o contrato de identidade e de não sobrescrever âncora | ADAPT_EXISTING_DARWIN |
| Probe de esquecimento | tabela do README, 524 mil caracteres de xadrez | não há esse probe no código | Fases 5 e 12, com os números crus | REIMPLEMENT |
| Paridade com o donor | não se aplica; ele treina do zero | yaml de transplante Smol e testes `test_transplant_16b_*` | Fase 3 mede KL e top-1 antes de aprender | ADAPT_EXISTING_DARWIN |
| Rollback com SHA | não é o mecanismo dele | `circuits/rollback.py` | Fase 11 | EXPERIMENT_LATER |
| ROME | ausente | `research/rome_*.py` | trilha separada, Fase 11 | EXPERIMENT_LATER |
| MEMIT | ausente | `research/memit_identity.py` | trilha separada, Fase 11 | EXPERIMENT_LATER |
| Memória de inferência por slot | ausente | `heartbeat.py` `TestTimeMemory` | não é aprendizado de parâmetro | IGNORE |
| Órgãos, alma, senado, GABA, JEPA, Ghost | ausentes | presentes no checkout | fora desta linha | IGNORE |
| Alfabeto de byte | vocabulário 256 | token / Smol | fica o tokenizador do donor | IGNORE |
| Corpo híbrido SSD | ausente | `DarwinXBlock` | fora do V0; o corpo é o donor intacto | IGNORE |

### Decisões, com fonte

1. **Tronco a 0.1×.** Fonte: `train.py` `_split_trunk_pool` e os grupos AdamW logo abaixo; `config.yaml` `training.trunk_lr_mult`. Motivo: é a variável que o README associa ao drift de +0.0067 nats. Risco: o número foi medido em modelo de bytes, batch 1, 524 mil caracteres de xadrez. No donor o gradiente pode não estar 97,6% no tronco. Dependência: partição explícita backbone / router / expert. A Fase 5 é quem escolhe a taxa.

2. **Mesmo forward com alvo.** Fonte: `stream.py` `Reader.step`. Motivo: sem isso o "aprendizado contínuo" vira fine-tune com outro código. Risco: gerar e treinar no mesmo passo cedo demais. No V0 o passo é um script de treino curto, ainda com o mesmo `forward` do donor mais o residual. Dependência: paridade da Fase 3 antes do primeiro passo.

3. **Poda paginada ignora o gate.** Fonte: `paged.py` `prune`, por volta da linha 900. Motivo: o próprio código diz que o gate é anti-preditivo. Risco: aplicar isso no `SharedPool`, onde a regra escrita é a oposta. Dependência: só depois de existir paging e contagem de admissão. Fase 9 no mínimo.

4. **Nascimento por recombinação.** Fonte: `paged.py` `add_experts`, argumento `recombine`. Motivo: clone e aleatório estão descritos como falhas, com referência a `runs/results/birth_schemes.json` dentro do repo dele. Esse json não foi reexecutado aqui. Dependência: pool já treinado de onde tirar unidades. Fora do V0.

5. **Adam viaja.** Fonte: `store.py` `_expert_moments`; `paged.py` `swap_to`. Motivo: momento órfão contamina o próximo expert. Dependência: paging. Fora do V0.

6. **Controlador `Plasticity`.** Fonte: `plasticity.py` `observe` e `_verdict`. Motivo: é a classe mais isolada do resto do modelo de bytes. Risco: as constantes foram ajustadas em nats por caractere. Copiar o arquivo exige o aviso MIT. Dependência: held-out real. Fase 6.

7. **Linhagem Darwin.** Fonte: `checkpoint_root.py` `write_lineage_root_identity`. Motivo: âncora que não se reescreve. Risco: copiar o módulo inteiro puxa o organismo. Ação é reescrever o contrato, não importar `f51_darwin`. Dependência: nenhuma com o mini-AGI.

8. **Paridade de montagem.** Fonte: `darwin_x_1.6b_smol_transplant.yaml`, observação de 2026-07-29. Motivo: RoPE, residual e RMSNorm errados deixaram o transplante pior que o aleatório. Risco: repetir isso ao enfiar SSD ou MoE antigo no donor. Dependência: Fase 2 e Fase 3.

9. **TTM.** Fonte: `heartbeat.py` `write_if_surprised`. Motivo para ignorar: o slot é detached e o write está em `no_grad`. Não é o mecanismo que compete com o tronco lento.

10. **ROME / MEMIT.** Fonte: `research/rome_identity_edit.py`, `research/memit_identity.py`. Motivo para adiar: scripts soltos sobre um snapshot Smol, fora do forward, sem resultado carregado nesta fase. Não entram no benchmark de domínio.

## A. As cinco ideias mais fortes do mini-AGI

1. O tronco compartilhado aprende a uma fração da taxa dos experts. No commit lido, a fração publicada é 0.1, e o drift dos assuntos não lidos cai de cerca de +2.23 nats para +0.0067 nats.
2. Ler e treinar são o mesmo forward com alvo.
3. O pool paginado escolhe experts pela demanda do estado anterior, guarda o Adam no arquivo do expert, e apaga por falta de chamada, não pelo gate.
4. Expert novo nasce por recombinação de unidades já treinadas, mudo, e só se os freios (espaço, uso, rendimento, ensaio, honestidade treino/held-out) deixam.
5. A taxa de aprendizado se move pelo held-out, sem horizonte fixo (`Plasticity`).

## B. Quais delas o Darwin antigo não possui

Correção de auditoria, registrada antes da Fase 2. A frase "Darwin antigo não tem essas cinco" fica substituída por esta:

Darwin antigo não implementa esses cinco mecanismos como a mesma stack integrada e causalmente validada de continual learning. Existem mecanismos parcialmente relacionados no Darwin antigo, mas não equivalentes ao pipeline do mini-AGI.

O que existe, e por que não é equivalente:

- Taxas diferentes por grupo de parâmetro não aparecem como tronco a 0.1× contra experts, com a medição de esquecimento publicada pelo mini-AGI.
- `TestTimeMemory.write_if_surprised` grava um slot com `detach` dentro de `torch.no_grad()`. É memória lateral, não o mesmo forward com alvo atualizando o expert.
- `DeepSeekStyleMoE` roteia experts e o Nitro move expert frio para a CPU. Não é o pool paginado com Adam no arquivo, demanda pelo estado anterior e poda por falta de chamada.
- Há criação e poda de expert no MoE (`_create_expert`, `_prune_expert`, apoptose). Não é nascimento por recombinação de unidades treinadas com os cinco freios do `AutoGrow`.
- Não há um controlador que mova a taxa só pelo held-out, no lugar de um horizonte.

A linhagem e o rollback continuam úteis, e continuam sem ser essa stack.

## C. Quais mecanismos Darwin valem preservar

- A identidade de checkpoint que recusa trocar âncora (`checkpoint_root.py`).
- O rollback que confere SHA-256 antes de restaurar (`circuits/rollback.py`), para a trilha cirúrgica.
- O contrato de paridade do donor: RoPE no estilo do modelo de origem, escala residual 1 no caminho transplantado, RMSNorm com o eps do donor. Está escrito como correção de um build que ficou pior que o aleatório.

## D. Quais mecanismos Darwin abandonar nesta linha

- GABA, JEPA, Ghost, Soul, Senado, Heartbeat e o `TestTimeMemory`.
- O `DeepSeekStyleMoE` com Nitro, apoptose e sono, como se isso fosse o aprendizado contínuo.
- O corpo SSD no V0. O corpo da primeira versão é o donor intacto.
- Alfabeto de byte e o bloco repetido 24 vezes.
- Treino do zero da linhagem 1.6B.

## E. Arquitetura mínima do Darwin-CL V0

```text
donor pretrained, congelado
        |
        base = donor(x)
        plastic = top-2 de 8 SwiGLU
        y = base + gate * plastic
```

`gate` inicial perto de zero, medido por KL, acordo top-1 e delta de logit contra o donor nos mesmos prompts. Sem crescimento, sem poda, sem paging, sem ROME, sem memória externa. Backbone lr = 0 até a Fase 5.

## F. O que pode ser reutilizado literalmente sob MIT, e o que será reimplementado

Reutilização literal, quando a Fase 6 começar: `minagi/plasticity.py`, classe `Plasticity`, com o copyright de Alexey Borsky e o texto MIT em `THIRD_PARTY_NOTICES.md`. Nada disso foi copiado nesta fase.

Reimplementar, sem colar arquivo: o passo com alvo, a partição de taxas, o banco residual, o router, e mais tarde a demanda, a recombinação, a poda por staleness e o Adam por expert. `train.py`, `paged.py`, `pool.py`, `recur.py`, `stream.py` e `store.py` assumem o `RecurCoder`, o alfabeto de byte e o layout npz. Colar esses arquivos não produz o V0.

Código Darwin: não importar `f51_darwin`. O contrato de linhagem, de hash e de paridade se reescreve no pacote novo. A licença do Darwin antigo é proprietária da F51; o checkout permanece só em `references/`.

## G. Risco de benchmark injusto

Sim. Vários, e qualquer um basta para invalidar uma frase do tipo "ganhamos".

- A tabela +0.0067 nats é nats por caractere, num modelo de byte, depois de 524 mil caracteres de um assunto, com sete assuntos não lidos, batch 1, tronco a 0.1×. Loss por token de outro tokenizador não é o mesmo número.
- O Darwin-CL parte de um donor pretrained. No dia zero ele já está à frente de um modelo que ainda está no primeiro passe. Comparar loss absoluta sem declarar essa vantagem de partida é cherry-picking. A comparação justa do aprendizado é o delta do domínio novo e o drift dos antigos, na mesma unidade, no mesmo arquivo, com a mesma dose.
- Ele publicou o experimento para uma placa de 8 GB e um processo. Esta máquina tem 5060 Ti 16 GB e 3060 12 GB. Mais VRAM não pode entrar escondida como "mesma eficiência".
- Os pesos dele não estão no repositório. Rodar o código original do zero por pouco tempo não reproduz o número de quem já leu centenas de milhões de caracteres. A Fase 12 tem de dizer a dose real dos dois lados.
- Replay, KL de âncora e varredura de lambda são métodos a mais. Se entrarem na comparação, o braço tem de estar nomeado. O braço que pretende bater o +0.0067 é o que usa a mesma regra que ele mediu: tronco lento, sem replay, sem rótulo no router.
- A 3060 ainda tinha 5329 MiB ocupados sem processo de compute na Fase 0. Isso muda VRAM e tempo. Tem de ser remediado na hora do benchmark, não ajustado no texto.

## O que não foi feito

Plastic Expert Bank, escolha de donor, download, `pip install`, treino, alteração de CUDA, qualquer escrita dentro dos dois clones.

## Próximo passo

Fase 2, só depois de `APPROVED PHASE 1`: justificar um donor pequeno, baixar só então, e gravar o baseline sem treino.
