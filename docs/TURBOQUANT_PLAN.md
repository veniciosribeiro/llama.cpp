# Plano de Desenvolvimento - TurboQuant

## 🎯 Objetivo

Implementar compressão de KV cache no llama.cpp usando o algoritmo **TurboQuant** (ICLR 2026), alcançando ~78% de redução de memória com perda mínima de accuracy.

**Meta:** Suportar contexto de 100K+ tokens em GPUs consumer (24GB VRAM).

---

## 📊 Benchmarks Alvo

Baseado nos resultados do paper TurboQuant (Llama-3.1-8B, LongBench):

| Métrica | Full Precision | TurboQuant | Economia |
|---------|---------------|------------|----------|
| Bits por dimensão | 16 bits | **3.5 bits** | 78% |
| KV Cache (100K tokens) | 51.2 GB | **11.2 GB** | 40 GB |
| Accuracy (LongBench) | 50.06 | **50.06** | 0% perda |
| Overhead quantização | - | 0.12ms | < 1% latency |

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    llama.cpp (turboquant branch)            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  TurboQuant │  │   QJL Core  │  │  PolarQuant (opt)   │  │
│  │  Quantizer  │  │  (1-bit JL) │  │  (transformação)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │                │                      │            │
│         └────────────────┼──────────────────────┘            │
│                          │                                   │
│              ┌───────────▼───────────┐                       │
│              │   KV Cache Manager    │                       │
│              │  (compress/decompress)│                       │
│              └───────────┬───────────┘                       │
│                          │                                   │
│         ┌────────────────┼────────────────┐                  │
│         │                │                │                  │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐          │
│  │   Attention │  │   Memory    │  │  Benchmark  │          │
│  │   Kernel    │  │   Allocator │  │   Suite     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Build Rápido (Primeira Vez)

```bash
cd /home/mvrdu/.goclaw/workspace/coder/ws/system/projects/llama.cpp

# Configurar (com ccache para rebuilds rápidos)
cmake -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_TURBOQUANT=ON \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache

# Compilar
cmake --build build --config Release --target llama-cli -j $(nproc)
```

**Sem ccache:** Remova as 3 flags `-DCMAKE_*_COMPILER_LAUNCHER=ccache`

### Teste Funcional

```bash
./build/bin/llama-cli \
  -m /mnt/c/Users/mvrdu/.cache/huggingface/hub/model/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-Q8_0.gguf \
  -n 50 \
  --ctx-size 6000 \
  -ctk turboquant \
  -ctv turboquant \
  -fa on \
  -p "Hello, how are you?"
```

### Validação Automatizada

```bash
./scripts/test_turboquant.sh
```

---

## 📋 Fases de Desenvolvimento

### **Fase 1: Fundação (Semanas 1-2)**

#### 1.1. Estrutura do Projeto
- [ ] Criar diretório `src/quantization/` no llama.cpp
- [ ] Configurar build system (CMake) para novos módulos
- [ ] Criar headers públicos em `include/llama/quant/`
- [ ] Setup de testes unitários (Google Test)

#### 1.2. QJL Core (Pré-requisito do TurboQuant)
- [ ] Implementar `qjl.hpp` - matriz JL e operações básicas
- [ ] Implementar `qjl_quantize()` - projeção + sign
- [ ] Implementar `qjl_inner_product()` - estimador assimétrico
- [ ] Criar testes unitários para QJL
- [ ] Benchmark: validar tempo < 0.07ms (d=1536, A100)

**Arquivos:**
```
src/quantization/qjl.cpp
src/quantization/qjl.hpp
tests/test_qjl.cpp
```

#### 1.3. Codebook MSE
- [ ] Implementar Lloyd-Max algorithm para codebook ótimo
- [ ] Implementar quantização uniforme (fallback)
- [ ] Criar pré-computação de codebooks (offline)
- [ ] Serialização de codebooks (binário)

**Arquivos:**
```
src/quantization/codebook_mse.cpp
src/quantization/codebook_mse.hpp
tools/precompute_codebooks.cpp
```

---

### **Fase 2: TurboQuant Core (Semanas 3-4)**

#### 2.1. Quantizer Principal
- [ ] Implementar `turboquant.hpp` - classe principal
- [ ] Implementar rotação aleatória (matriz ortogonal)
- [ ] Implementar estágio MSE (b-1 bits)
- [ ] Implementar estágio QJL residual (1 bit)
- [ ] Implementar empacotamento de bits

**Arquivos:**
```
src/quantization/turboquant.cpp
src/quantization/turboquant.hpp
```

#### 2.2. Dequantização
- [ ] Implementar `turboquant_dequantize()`
- [ ] Reconstrução MSE a partir de índices
- [ ] Reconstrução QJL do residual
- [ ] Aplicar rotação inversa

#### 2.3. Produto Interno Assimétrico
- [ ] Implementar `turboquant_inner_product()`
- [ ] Otimizar para evitar reconstrução completa
- [ ] Suporte a batch (múltiplos queries)

**Arquivos:**
```
src/quantization/turboquant_inner_product.cpp
tests/test_turboquant.cpp
```

---

### **Fase 3: Integração KV Cache (Semanas 5-6)**

#### 3.1. KV Cache Manager
- [ ] Criar `kv_cache_quant.hpp` - interface de compressão
- [ ] Implementar compressão por camada/head
- [ ] Gerenciar memória comprimida
- [ ] Cache de vetores descomprimidos (hot cache)

**Arquivos:**
```
src/llama/kv_cache_quant.cpp
src/llama/kv_cache_quant.hpp
```

#### 3.2. Integração com Attention
- [ ] Modificar `llama_attention()` para usar KV quantizado
- [ ] Hook de descompressão on-demand
- [ ] Otimizar layout de memória (contiguidade)
- [ ] Suporte a Flash Attention (se aplicável)

**Arquivos:**
```
src/llama/attention.cpp (modificar)
include/llama/attention.h (modificar)
```

#### 3.3. Memory Allocator
- [ ] Alocador específico para KV comprimido
- [ ] Pool de memória por camada
- [ ] Deallocation automático
- [ ] Stats de uso de memória

**Arquivos:**
```
src/llama/kv_allocator.cpp
src/llama/kv_allocator.hpp
```

---

### **Fase 4: Otimização GPU (Semanas 7-8)**

#### 4.1. CUDA Kernels
- [ ] Kernel `qjl_quantize_kernel()` - quantização batch
- [ ] Kernel `turboquant_quantize_kernel()`
- [ ] Kernel `qjl_inner_product_kernel()` - produto interno
- [ ] Otimizar acesso à memória global (coalescing)

**Arquivos:**
```
src/quantization/qjl.cu
src/quantization/turboquant.cu
include/llama/quant/qjl_cuda.h
```

#### 4.2. Tensor Cores
- [ ] Usar mixed-precision (FP16 para projeção JL)
- [ ] Kernel `qjl_quantize_tensor_core()`
- [ ] Benchmark: validar speedup 2-3× vs FP32

#### 4.3. Memory Layout
- [ ] Layout NHWC para melhor coalescing
- [ ] Prefetching de codebooks
- [ ] Shared memory para codebook MSE

---

### **Fase 5: PolarQuant (Opcional, Semanas 9-10)**

#### 5.1. Transformação Polar
- [ ] Implementar `polarquant.hpp`
- [ ] Transformação polar recursiva
- [ ] Quantização de ângulos e magnitudes
- [ ] Transformação inversa

**Arquivos:**
```
src/quantization/polarquant.cpp
src/quantization/polarquant.hpp
tests/test_polarquant.cpp
```

#### 5.2. Integração
- [ ] Interface unificada com TurboQuant
- [ ] Benchmark comparativo
- [ ] Switch runtime entre métodos

---

### **Fase 6: Benchmarks e Validação (Semanas 11-12)**

#### 6.1. Benchmark Suite
- [ ] Criar `bench_turboquant.cpp`
- [ ] Medir: tempo de quantização, dequantização, produto interno
- [ ] Medir: uso de memória, throughput
- [ ] Comparar com KIVI, KVQuant, PQ

**Arquivos:**
```
examples/bench_turboquant.cpp
scripts/run_benchmarks.sh
```

#### 6.2. Validação de Accuracy
- [ ] Integrar com llama.cpp main
- [ ] Rodar em LongBench (ou subset)
- [ ] Medir perplexidade
- [ ] Comparar com full precision

**Arquivos:**
```
examples/main.cpp (modificar para suporte TurboQuant)
scripts/eval_accuracy.sh
```

#### 6.3. Testes de Estresse
- [ ] Contexto 100K tokens
- [ ] Batch size variável
- [ ] Múltiplas GPUs (se aplicável)
- [ ] Memory leak detection

---

## 📦 Estrutura de Arquivos Final

```
llama.cpp/
├── src/
│   ├── quantization/
│   │   ├── qjl.cpp, qjl.hpp
│   │   ├── qjl.cu, qjl_cuda.h
│   │   ├── turboquant.cpp, turboquant.hpp
│   │   ├── turboquant.cu, turboquant_cuda.h
│   │   ├── polarquant.cpp, polarquant.hpp (opcional)
│   │   ├── codebook_mse.cpp, codebook_mse.hpp
│   │   └── CMakeLists.txt
│   │
│   └── llama/
│       ├── kv_cache_quant.cpp, kv_cache_quant.hpp
│       ├── kv_allocator.cpp, kv_allocator.hpp
│       └── attention.cpp (modificado)
│
├── include/
│   └── llama/
│       └── quant/
│           ├── qjl.h
│           ├── turboquant.h
│           └── polarquant.h
│
├── tests/
│   ├── test_qjl.cpp
│   ├── test_turboquant.cpp
│   └── test_polarquant.cpp
│
├── examples/
│   ├── bench_turboquant.cpp
│   └── main.cpp (modificado)
│
├── tools/
│   └── precompute_codebooks.cpp
│
└── scripts/
    ├── run_benchmarks.sh
    ├── eval_accuracy.sh
    └── compare_methods.py
```

---

## 🔧 Dependências

### Build-time
- CMake 3.18+
- CUDA Toolkit 11.8+ (para GPU)
- GCC 9+ ou Clang 10+
- ccache (opcional, recomendado para rebuilds rápidos)
- Ninja build system

### Runtime (opcional)
- torchac (entropy coding) - se implementar compressão adicional
- Faiss (product quantization) - para comparação

---

## 🛠️ Comandos de Build

### Build Padrão (Produção)

```bash
# Configuração
cmake -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_TURBOQUANT=ON \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache

# Build
cmake --build build --config Release --target llama-cli -j $(nproc)
```

### Build Sem ccache (se não instalado)

```bash
cmake -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_TURBOQUANT=ON

cmake --build build --config Release --target llama-cli -j $(nproc)
```

### Build para Desenvolvimento (com debug symbols)

```bash
cmake -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGGML_NATIVE=ON \
  -DGGML_TURBOQUANT=ON \
  -DGGML_TURBOQUANT_DEBUG=ON

cmake --build build --config RelWithDebInfo --target llama-cli -j $(nproc)
```

### Notas sobre CUDA Architectures

| GPU | Compute Capability | Flag |
|-----|-------------------|------|
| RTX 30xx (Ampere) | 86 | `-DCMAKE_CUDA_ARCHITECTURES=86` |
| RTX 40xx (Ada) | 89 | `-DCMAKE_CUDA_ARCHITECTURES=89` |
| A100 (Ampere) | 80 | `-DCMAKE_CUDA_ARCHITECTURES=80` |

**Para desenvolvimento local:** Use apenas a arquitetura da sua GPU para builds mais rápidos (~3×).

**Para distribuição:** Use múltiplas arquiteturas: `"80;86;89"`

### Notas sobre ccache

**Instalação (Ubuntu/Debian):**
```bash
sudo apt install ccache
```

**Benefícios:**
- Build inicial: sem diferença
- Rebuilds: **5-10× mais rápido** (aproveita cache de objetos compilados)

**Uso opcional:** Se não instalado, remova as flags `-DCMAKE_*_COMPILER_LAUNCHER=ccache`

---

## 🧪 Comandos de Teste e Validação

### Teste Funcional Básico

```bash
./build/bin/llama-cli \
  -m ${MODEL_PATH:-/mnt/c/Users/mvrdu/.cache/huggingface/hub/model/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-Q8_0.gguf} \
  -n 50 \
  --ctx-size 6000 \
  -ctk ${TQ_TYPE:-q8_0} \
  -ctv ${TQ_TYPE:-q8_0} \
  -fa on \
  -p "Hello, how are you?"
```

### Teste de Validação Automatizada

```bash
#!/bin/bash
# scripts/test_turboquant.sh

set -e

MODEL_PATH=${MODEL_PATH:-"/mnt/c/Users/mvrdu/.cache/huggingface/hub/model/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-Q8_0.gguf"}
OUTPUT_FILE="/tmp/turboquant_test_output.txt"

echo "=== TurboQuant Validation Test ==="
echo "Model: $MODEL_PATH"
echo "KV Cache Type: ${TQ_TYPE:-q8_0}"
echo ""

# Executar teste
./build/bin/llama-cli \
  -m "$MODEL_PATH" \
  -n 100 \
  --ctx-size 6000 \
  -ctk "${TQ_TYPE:-q8_0}" \
  -ctv "${TQ_TYPE:-q8_0}" \
  -fa on \
  -p "Explain quantum entanglement in simple terms." \
  | tee "$OUTPUT_FILE"

# Validações
echo ""
echo "=== Validação ==="

# 1. Verificar se output não está vazio
if [ ! -s "$OUTPUT_FILE" ]; then
  echo "❌ FALHA: Output vazio"
  exit 1
fi
echo "✅ Output não vazio"

# 2. Verificar se não contém padrões de garbage
if grep -qE "|NaN|Inf|" "$OUTPUT_FILE"; then
  echo "❌ FALHA: Detectado garbage no output"
  exit 1
fi
echo "✅ Sem garbage detectado"

# 3. Verificar se contém texto coerente (pelo menos 50 palavras)
WORD_COUNT=$(wc -w < "$OUTPUT_FILE")
if [ "$WORD_COUNT" -lt 50 ]; then
  echo "❌ FALHA: Output muito curto ($WORD_COUNT palavras)"
  exit 1
fi
echo "✅ Output com tamanho adequado ($WORD_COUNT palavras)"

# 4. Verificar tempo de execução
# (implementar com time ou métricas internas)

echo ""
echo "✅ Todos os testes passaram!"
```

### Teste de Longo Contexto (100K tokens)

```bash
#!/bin/bash
# scripts/test_long_context.sh

./build/bin/llama-cli \
  -m "$MODEL_PATH" \
  -n 200 \
  --ctx-size 100000 \
  -ctk "${TQ_TYPE:-turboquant}" \
  -ctv "${TQ_TYPE:-turboquant}" \
  -fa on \
  --memory-f32 \
  -p "Summarize the following document: $(cat docs/long_document.txt)" \
  2>&1 | tee /tmp/long_context_test.txt

# Monitorar memória durante execução
# nvtop -p $(pgrep llama-cli) &
```

### Teste Comparativo (Full Precision vs TurboQuant)

```bash
#!/bin/bash
# scripts/compare_precision.sh

PROMPT="Explain the theory of relativity."

echo "=== Full Precision ==="
time ./build/bin/llama-cli \
  -m "$MODEL_PATH" \
  -n 100 \
  --ctx-size 6000 \
  -ctk f16 \
  -ctv f16 \
  -fa on \
  -p "$PROMPT" \
  | tee /tmp/baseline.txt

echo ""
echo "=== TurboQuant ==="
time ./build/bin/llama-cli \
  -m "$MODEL_PATH" \
  -n 100 \
  --ctx-size 6000 \
  -ctk turboquant \
  -ctv turboquant \
  -fa on \
  -p "$PROMPT" \
  | tee /tmp/turboquant.txt

echo ""
echo "=== Comparação ==="
echo "Diff de output:"
diff /tmp/baseline.txt /tmp/turboquant.txt || true
```

### Monitoramento de Memória GPU

```bash
#!/bin/bash
# scripts/monitor_vram.sh

echo "Monitoring VRAM usage..."
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

### Validação de Accuracy (Perplexidade)

```bash
#!/bin/bash
# scripts/eval_perplexity.sh

# Usar llama-perplexity para medir perplexidade
./build/bin/llama-perplexity \
  -m "$MODEL_PATH" \
  -f datasets/wikitext.txt \
  -ctk "${TQ_TYPE:-turboquant}" \
  -ctv "${TQ_TYPE:-turboquant}" \
  | tee /tmp/perplexity_turboquant.txt

# Comparar com baseline
./build/bin/llama-perplexity \
  -m "$MODEL_PATH" \
  -f datasets/wikitext.txt \
  -ctk f16 \
  -ctv f16 \
  | tee /tmp/perplexity_baseline.txt
```

---

## 📊 Critérios de Aceite para Testes

| Teste | Critério de Sucesso |
|-------|---------------------|
| Funcional básico | Output coerente, ≥50 palavras, sem garbage |
| Longo contexto | 100K tokens sem OOM, output válido |
| Performance | Overhead < 1% vs baseline |
| Memória | Redução ≥ 75% no KV cache |
| Accuracy | Perplexidade ≤ 1% pior que baseline |
| Estresse | 1000 iterações sem crash/memory leak |

---

## 📈 Métricas de Sucesso

### Performance
- [ ] Tempo de quantização: < 0.12ms (d=1536, A100)
- [ ] Tempo de produto interno: < 0.10ms
- [ ] Overhead total: < 1% de latency
- [ ] Throughput: > 95% do full precision

### Memória
- [ ] Redução do KV cache: ≥ 75%
- [ ] Memory footprint: ≤ 12 GB para 100K tokens (8B model)
- [ ] Zero memory leaks (validar com Valgrind)

### Accuracy
- [ ] LongBench score: ≥ 99% do full precision
- [ ] Perplexidade: ≤ 1% degradação
- [ ] Zero regressões em testes existentes

### Code Quality
- [ ] Cobertura de testes: ≥ 80%
- [ ] Zero warnings de compilação
- [ ] Documentação completa de APIs
- [ ] Examples funcionais

---

## 🚀 Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| Implementação QJL muito lenta | Baixa | Alto | Usar Tensor Cores, otimizar kernels CUDA |
| Perda de accuracy > 5% | Média | Alto | Ajustar bit-width, usar codebooks melhores |
| Memory leaks | Baixa | Médio | Valgrind em cada commit, RAII |
| Integração com attention quebra | Média | Alto | Testes unitários extensivos, feature flags |
| CUDA incompatível com GPUs antigas | Alta | Médio | Fallback para CPU, detectar compute capability |

---

## 📅 Cronograma Estimado

| Fase | Duração | Entregáveis |
|------|---------|-------------|
| 1. Fundação | 2 semanas | QJL funcional, codebooks, testes |
| 2. TurboQuant Core | 2 semanas | Quant/dequant completo |
| 3. KV Cache Integration | 2 semanas | KV cache comprimido |
| 4. GPU Optimization | 2 semanas | CUDA kernels otimizados |
| 5. PolarQuant (opt) | 2 semanas | Método alternativo |
| 6. Benchmarks | 2 semanas | Validação completa |

**Total:** 10-12 semanas (2.5-3 meses)

---

## 👥 Divisão de Tarefas Sugerida

### Agente A (Core)
- QJL implementation
- TurboQuant quant/dequant
- Codebook MSE

### Agente B (Integration)
- KV Cache Manager
- Attention integration
- Memory allocator

### Agente C (GPU/Optimization)
- CUDA kernels
- Tensor Core optimization
- Benchmarks

---

## 📝 Próximos Passos Imediatos

1. **Hoje:** Validar estrutura de diretórios no repo
2. **Dia 1:** Implementar QJL básico (CPU)
3. **Dia 2:** Testes unitários QJL
4. **Dia 3-4:** Codebook MSE + Lloyd-Max
5. **Dia 5:** Primeira versão do TurboQuant

---

## 🔗 Referências

- [TurboQuant Paper](https://arxiv.org/abs/2504.19874)
- [QJL Paper](https://arxiv.org/abs/2406.03482)
- [PolarQuant Paper](https://arxiv.org/abs/2502.02617)
- [QJL GitHub](https://github.com/amirzandieh/QJL)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

---

**Status:** 📋 Planejamento completo  
**Última atualização:** 2026-04-08  
**Responsável:** coder (turboquant-expert skill)
