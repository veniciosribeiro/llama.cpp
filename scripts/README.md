# TurboQuant Scripts

Coleção de scripts para build, teste, benchmark e validação da implementação TurboQuant no llama.cpp.

## 📋 Índice

- [Pré-requisitos](#pré-requisitos)
- [Scripts Disponíveis](#scripts-disponíveis)
- [Quick Start](#quick-start)
- [Workflow Recomendado](#workflow-recomendado)

---

## Pré-requisitos

### Sistema
- Linux/WSL2
- CUDA Toolkit 11.8+
- CMake 3.18+
- Ninja build system
- ccache (recomendado)

### Python (opcional, para alguns scripts)
```bash
sudo apt install python3 python3-pip
```

### Build do llama.cpp
```bash
cd /home/mvrdu/.goclaw/workspace/coder/ws/system/projects/llama.cpp

cmake -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_TURBOQUANT=ON \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache

cmake --build build --config Release --target llama-cli -j $(nproc)
```

---

## Scripts Disponíveis

### 🔧 Build & Setup

| Script | Descrição |
|--------|-----------|
| `generate_test_data.sh` | Gera datasets de teste (mini, small, medium, large, long-context) |

### 🧪 Testes Funcionais

| Script | Descrição |
|--------|-----------|
| `test_turboquant.sh` | Validação funcional automatizada (output, garbage, tamanho) |
| `test_long_context.sh` | Teste de contexto longo (100K tokens) |
| `compare_precision.sh` | Comparação direta: Full Precision vs TurboQuant |

### 📊 Benchmarks

| Script | Descrição |
|--------|-----------|
| `run_benchmarks.sh` | Suite completa de benchmarks (speed, memory, accuracy) |
| `monitor_vram.sh` | Monitoramento de uso de VRAM em tempo real |
| `eval_perplexity.sh` | Avaliação de accuracy via perplexidade |

---

## Quick Start

### 1. Gerar Dataset de Teste
```bash
./scripts/generate_test_data.sh small
```

### 2. Executar Teste Funcional
```bash
./scripts/test_turboquant.sh
```

### 3. Comparar com Full Precision
```bash
./scripts/compare_precision.sh
```

### 4. Rodar Benchmarks Completos
```bash
./scripts/run_benchmarks.sh
```

---

## Workflow Recomendado

### Durante Desenvolvimento

```bash
# 1. Build rápido (após mudanças)
cmake --build build --config Release --target llama-cli -j $(nproc)

# 2. Teste funcional rápido
./scripts/test_turboquant.sh

# 3. Se passar, rodar comparação
./scripts/compare_precision.sh
```

### Antes de Commit

```bash
# 1. Gerar datasets
./scripts/generate_test_data.sh all

# 2. Rodar todos os testes
./scripts/test_turboquant.sh
./scripts/test_long_context.sh

# 3. Benchmarks
./scripts/run_benchmarks.sh

# 4. Validar accuracy
./scripts/eval_perplexity.sh
```

### Para Debug

```bash
# Monitorar VRAM durante teste
./scripts/monitor_vram.sh

# Ou em outro terminal:
./build/bin/llama-cli -m model.gguf -n 100 -ctk turboquant -ctv turboquant &
./scripts/monitor_vram.sh $!
```

---

## Variáveis de Ambiente

Todos os scripts suportam variáveis de ambiente:

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `MODEL_PATH` | Caminho para modelo GGUF | `~/.cache/huggingface/...` |
| `TQ_TYPE` | Tipo de quantização KV | `turboquant` |
| `OUTPUT_FILE` | Arquivo de output | `/tmp/...` |
| `DATASET_PATH` | Dataset para perplexidade | `datasets/wikitext.txt` |

### Exemplo
```bash
MODEL_PATH=/path/to/model.gguf \
TQ_TYPE=q8_0 \
./scripts/test_turboquant.sh
```

---

## Critérios de Aceite

### Teste Funcional (`test_turboquant.sh`)
- ✅ Output não vazio
- ✅ Sem garbage (NaN, Inf, símbolos inválidos)
- ✅ Mínimo 50 palavras
- ✅ Tempo < 60s

### Longo Contexto (`test_long_context.sh`)
- ✅ 100K tokens sem OOM
- ✅ Output válido
- ✅ Tempo < 300s

### Comparação (`compare_precision.sh`)
- ✅ Overhead < 10%
- ✅ Diferença de output < 20%

### Perplexidade (`eval_perplexity.sh`)
- ✅ Degradação ≤ 1% (ideal)
- ⚠️ Degradação ≤ 5% (aceitável)
- ❌ Degradação > 5% (investigar)

---

## Output dos Scripts

### Logs
- `/tmp/turboquant_test_output.txt` - Teste funcional
- `/tmp/long_context_test.txt` - Longo contexto
- `/tmp/turboquant_comparison/` - Comparação (baseline.txt, turboquant.txt)
- `/tmp/perplexity_eval/` - Perplexidade
- `/tmp/turboquant_benchmarks/` - Benchmarks completos

### Relatórios
Os scripts geram relatórios em Markdown em:
- `/tmp/turboquant_benchmarks/report_YYYYMMDD_HHMMSS.md`

---

## Troubleshooting

### "Binário não encontrado"
```bash
# Compile o binário necessário
cmake --build build --target llama-cli
cmake --build build --target llama-perplexity
cmake --build build --target bench_turboquant
```

### "Modelo não encontrado"
```bash
# Defina MODEL_PATH
export MODEL_PATH=/caminho/para/seu/modelo.gguf

# Ou baixe um modelo
huggingface-cli download unsloth/Qwen3.5-9B-GGUF
```

### "ccache não encontrado"
```bash
# Instale ccache ou remova as flags do build
sudo apt install ccache

# Ou compile sem ccache:
cmake -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release
```

### "Out of memory"
```bash
# Reduza ctx-size
./build/bin/llama-cli ... --ctx-size 2048 ...

# Ou use batch menor
```

---

## Performance Esperada

Baseado no paper TurboQuant (Llama-3.1-8B):

| Métrica | Full Precision | TurboQuant | Economia |
|---------|---------------|------------|----------|
| Bits/dim | 16 bits | 3.5 bits | 78% |
| KV Cache (100K) | 51.2 GB | 11.2 GB | 40 GB |
| Accuracy | 50.06 | 50.06 | 0% |
| Overhead | - | 0.12ms | < 1% |

---

## Contribuindo

### Adicionando Novos Scripts

1. Siga o padrão de cores e output
2. Use variáveis de ambiente para configurações
3. Adicione documentação neste README
4. Implemente critérios de aceite claros

### Padrão de Cores
```bash
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
```

---

## Referências

- [TurboQuant Paper](https://arxiv.org/abs/2504.19874)
- [QJL Paper](https://arxiv.org/abs/2406.03482)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Documentação TurboQuant](../docs/TURBOQUANT_PLAN.md)

---

**Última atualização:** 2026-04-08  
**Responsável:** coder (turboquant-expert skill)
