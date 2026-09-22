# PHASE 0 REPORT — inventário da máquina

Status: **PHASE 0 complete**. Nenhum código de arquitetura, nenhum clone em `references/`, nenhuma instalação.

Data local: 2026-09-22. Diretório novo: `C:\Users\marco\Desktop\F51-Darwin-CL`.

Nada foi apagado, movido, ou escrito dentro de `F51-Darwin-SSD` ou `mini-AGI`.

## Resultado

A máquina tem Windows 11 Pro build 26200, Python 3.12.10, Git 2.47.1, driver NVIDIA 595.97, toolkit CUDA 12.9 (`nvcc`) e PyTorch 2.8.0+cu129 já instalado no Python do sistema. Um tensor de 8 elementos rodou nas duas placas e devolveu soma 36.0 nas duas. Isso é o teste CUDA simples. Passou.

Restrição medida, não corrigida: a RTX 3060 reporta 5329 MiB em uso com utilização 0% e sem processo de compute listado. O teste pequeno coube mesmo assim. O orçamento livre dela agora é 6785 MiB.

## Comandos

Todos a partir de `C:\Users\marco`. Saída bruta em `reports/logs/phase0/`.

- `cmd /c ver`
- leitura de `HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion`
- `systeminfo` (recorte)
- `Get-CimInstance Win32_Processor`, `Win32_ComputerSystem`, `Win32_OperatingSystem`
- `Get-PSDrive -PSProvider FileSystem`
- `python --version`, `py -0p`, `where.exe python`
- `git --version`, `where.exe git`
- `where.exe` de `nvcc`, `cl`, `gcc`, `g++`, `cmake`, `clang`
- `nvcc --version`, `gcc --version`, `cmake --version`
- `nvidia-smi` e `nvidia-smi --query-gpu=...`
- probe de `torch` no Python do sistema e no venv `Desktop\mike\.venv`
- teste CUDA: `arange(8)+1` somado em `cuda:0` e `cuda:1`
- existência de paths de modelo e listagem do cache Hugging Face
- `git rev-parse HEAD` apenas nos clones que já existiam em `Desktop\_arch_compare` (não foi feito clone novo)

## Máquina

| Item | Valor medido |
|---|---|
| OS (`systeminfo`) | Microsoft Windows 11 Pro, 10.0.26200, x64 |
| OS (registro `ProductName`) | Windows 10 Pro, DisplayVersion 25H2, build 26200, UBR 8875 |
| `cmd /c ver` | Microsoft Windows [Version 10.0.26200.8875] |
| CPU | 12th Gen Intel Core i7-12700K, 12 cores, 20 threads, 3600 MHz |
| RAM total | 68417347584 bytes (systeminfo: 65248 MB) |
| RAM livre no momento do registro | 42707128 KB |
| Disco C | 666.52 GB usados, 281.67 GB livres |
| Disco E | 2.49 GB usados, 29.49 GB livres |
| Python no PATH | 3.12.10, `C:\Users\marco\AppData\Local\Programs\Python\Python312\python.exe` |
| Outros Pythons no launcher | CPython 3.15.0a8 e 3.12.13 via uv, em `%APPDATA%\uv\python` |
| Git | 2.47.1.windows.1 |
| Driver NVIDIA | 595.97 |
| CUDA reportado pelo driver | 13.2 |
| `nvcc` | release 12.9, V12.9.86, `C:\Users\marco\scoop\apps\cuda-12.9.1\current\bin\nvcc.exe` |
| gcc | 15.2.0 (scoop) |
| cmake | 4.4.2 (scoop) |
| `cl` (MSVC) | não está no PATH |
| `clang` | não está no PATH |

O registro ainda diz "Windows 10 Pro". O `systeminfo` diz "Windows 11 Pro" no mesmo build 26200. Os dois textos foram gravados. Não houve alteração de Windows, driver, CUDA ou BIOS.

## GPUs

Medição `nvidia-smi` em 2026-09-22 06:45:18 local.

| Índice | Nome | Compute capability | Total | Em uso | Livre | Utilização |
|---|---|---|---|---|---|---|
| 0 | NVIDIA GeForce RTX 5060 Ti | 12.0 | 16311 MiB | 602 MiB | 15449 MiB | 0% |
| 1 | NVIDIA GeForce RTX 3060 | 8.6 | 12288 MiB | 5329 MiB | 6785 MiB | 0% |

A 5060 Ti é a placa de vídeo do monitor (`Disp.A On`, driver WDDM). Os processos nela são área de trabalho (Explorer, Chrome, Cursor, e outros). Nenhum processo do tipo compute.

A 3060 não tem processo associado na tabela do `nvidia-smi`, e a utilização é 0%. Os 5329 MiB continuam contados. Tentativa anterior nesta sessão de `nvidia-smi --gpu-reset` não existe neste driver, e `Disable-PnpDevice` na 3060 falhou com "Generic failure". Fase 0 não tentou de novo e não rebootou a máquina.

## PyTorch e teste CUDA

Python do sistema, sem instalar nada:

```text
torch 2.8.0+cu129
cuda build 12.9
torch.cuda.is_available() True
device_count 2
cuda:0 NVIDIA GeForce RTX 5060 Ti cap (12, 0) sum 36.0
cuda:1 NVIDIA GeForce RTX 3060 cap (8, 6) sum 36.0
```

Soma esperada de `(arange(8)+1)` é 36. As duas placas devolveram 36.0. Exit code 0.

O venv `C:\Users\marco\Desktop\mike\.venv` tem o mesmo `2.8.0+cu129`, CUDA disponível, 2 devices. Não foi usado para o teste de soma. Não foi criado venv novo para o Darwin-CL.

## Paths existentes

| Path | Estado |
|---|---|
| `C:\Users\marco\Desktop\F51-Darwin-CL` | criado nesta fase, só `reports/` |
| `C:\Users\marco\Desktop\F51-Darwin-NITRO` | ausente |
| `C:\Users\marco\Desktop\F51-Darwin-SSD` | ausente |
| `C:\Users\marco\Desktop\F51-Nitro-MoE` | ausente |
| `C:\Users\marco\Desktop\_arch_compare\mini-AGI` | clone já existente, fora do projeto novo |
| `C:\Users\marco\Desktop\_arch_compare\F51-Darwin-SSD` | clone já existente, fora do projeto novo |
| `C:\Users\marco\qwen38-runtime\models` | GGUF locais, ver abaixo |
| `C:\Users\marco\.cache\huggingface\hub` | cache local, ver abaixo |

SHAs dos clones que já estavam em `_arch_compare` (não são o checkout de `references/`):

- mini-AGI `96784b78008a22c4a1ee2c9961640c22137320ef` remote `https://github.com/volotat/mini-AGI.git`
- F51-Darwin-SSD `70e135e811392cbaff7be6dc8fabef64d36107b8` remote `https://github.com/marcobarreto007/F51-Darwin-SSD.git`

`references/` dentro de `F51-Darwin-CL` não foi criado. Fase 1 é quem clona lá.

Arquivos em `qwen38-runtime\models` (não carregados):

| Arquivo | GB |
|---|---|
| Qwen3.8-27B-UD-IQ4_XS.gguf | 13.27 |
| Qwen3.8-27B-UD-IQ3_S.gguf | 11.21 |
| Qwen3.8-27B-UD-IQ3_XXS.gguf | 10.18 |
| mtp-Qwen3.8-27B-Q4_0.gguf | 1.28 |
| mmproj-Qwen3.8-27B-UD-F16.gguf | 0.86 |
| mmproj-F16.gguf | 0.86 |

O nome Smol não aparece na listagem de `huggingface\hub`. Há caches grandes já no disco (entre outros: Gemma 4 12B e 31B, Granite 4.2 3B/8B/30B, Qwen 3.5 4B/9B/27B, Ministral 3 8B). Nenhum deles foi carregado. Nenhum donor foi escolhido nesta fase.

## O que não foi feito

- clone
- `pip install` ou instalação global
- código em `src/darwin_cl`
- `README`, `pyproject.toml`, `LICENSE`, `THIRD_PARTY_NOTICES.md`
- download de modelo
- treino
- alteração de driver, CUDA, BIOS ou Windows

## Erros

Nenhum comando de inventário falhou com exit code diferente de zero, exceto buscas `where.exe` que reportaram ausência:

- `cl` não encontrado
- `clang` não encontrado

Isso não quebrou o teste CUDA, porque o wheel do PyTorch já vem compilado. Fica registrado para quando alguém precisar compilar extensão CUDA.

## Hashes SHA-256 dos logs

Arquivo `reports/logs/phase0_sha256.txt`.

| Arquivo | SHA-256 |
|---|---|
| cuda_test_system_python.txt | `13d122257adac262e74fe626f4fd2159018225f2d162266f3795bd125f0cefd4` |
| nvidia_smi.txt | `21269b085c56e9b3034aa1df769364e93a97fb44dab019a965fbc3950ede57ad` |
| nvidia_query.txt | `fdf0ecc8ef20080d88722cbaf5d8ce81ff98871d0ca901ad9b74c3cc1256b89f` |
| nvcc.txt | `34a46bf3fdb96baa4d9d1da7b43e2525d7da9e50668b96577ae6f0be30ecd73a` |
| python_version.txt | `6a5d9f6462ca5dfc6788cfd216ba579fffd7875776ab9d9b35659be78858f73a` |
| git_version.txt | `7af00adc72027ec7a4de0f7d321b6e3d94a02ec90a8edb291afe50ba9576a974` |
| disk.txt | `8d9020a3b32fb5a24cb353a691e3a72479d6c6af9831cdfeb9357f58661b85b5` |
| cpu.txt | `9b64b3237852fc1eff07baad1d6c7127a8c5ce12b0efce0285d5049e94280b6c` |
| model_paths.txt | `8e3fa0d6c7fd6bdc1ee8ef89ea813e568f053b5fc52a961c8c9bbb0bebc0a9a4` |
| hf_hub.txt | `02474e370b7da80da6e5cea43d7b554135a3de66d64f1d51adb7cecce4373c1f` |
| existing_miniagi_sha.txt | `ebead5a5454f6e065ca72e176a7bc85a112a17cbd1836de481c3b9e17e2f17bc` |
| existing_darwin_sha.txt | `a3ba232dad8c4179fd9b3fcfb3482fc231247cb59429bcace90d4de5626a5a23` |
| smol_present.txt | `e05b50467d0c81962a949f5df376639d0de1c8c6b7322f0da6d290d188da516e` |

O hash deste relatório é calculado depois da escrita e fica em `reports/logs/phase0_report_sha256.txt`, para o arquivo do relatório não mudar ao incluir o próprio hash.

## Próximo passo proposto

Fase 1, só depois de `APPROVED PHASE 0`: clonar mini-AGI e F51-Darwin-SSD em `F51-Darwin-CL\references\`, registrar o SHA de cada clone novo, e montar a tabela copy / reimplement / ignore. Sem código de modelo.
