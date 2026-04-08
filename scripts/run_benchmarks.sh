#!/bin/bash
# TurboQuant Benchmark Suite - Suite completa de benchmarks
#
# Uso: ./scripts/run_benchmarks.sh
#
# Executa todos os benchmarks e gera relatório consolidado

set -e

# Configurações
MODEL_PATH="${MODEL_PATH:-/mnt/c/Users/mvrdu/.cache/huggingface/hub/model/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-Q8_0.gguf}"
OUTPUT_DIR="/tmp/turboquant_benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

echo "========================================"
echo "  TurboQuant Benchmark Suite"
echo "========================================"
echo ""
echo "📦 Model:  $MODEL_PATH"
echo "📁 Output: $OUTPUT_DIR"
echo "🕐 Time:   $TIMESTAMP"
echo ""

# Criar diretório
mkdir -p "$OUTPUT_DIR"

# Verificar pré-requisitos
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ ERRO: Modelo não encontrado${NC}"
    exit 1
fi

if [ ! -f "./build/bin/llama-cli" ]; then
    echo -e "${RED}❌ ERRO: Binário llama-cli não encontrado${NC}"
    exit 1
fi

# ============================================
# Relatório final
# ============================================
REPORT_FILE="$OUTPUT_DIR/report_$TIMESTAMP.md"

cat > "$REPORT_FILE" << EOF
# TurboQuant Benchmark Report

**Data:** $(date)
**Model:** $MODEL_PATH

## Resumo Executivo

EOF

echo -e "${MAGENTA}📋 Executando benchmarks...${NC}"
echo ""

# ============================================
# Benchmark 1: Quantização Speed
# ============================================
echo -e "${CYAN}[1/5] Quantization Speed${NC}"

# Script de benchmark de quantização
BENCH_Q_OUTPUT="$OUTPUT_DIR/bench_quant.txt"

echo "⏳ Medindo tempo de quantização..."

# Usar exemplo de benchmark se existir
if [ -f "./build/bin/bench_turboquant" ]; then
    ./build/bin/bench_turboquant \
        --dim 1536 \
        --bits 4 \
        --iterations 100 \
        | tee "$BENCH_Q_OUTPUT"
else
    echo "   (bench_turboquant não compilado, pulando...)"
    echo "   Compile com: cmake --build build --target bench_turboquant"
fi

echo ""

# ============================================
# Benchmark 2: Produto Interno Speed
# ============================================
echo -e "${CYAN}[2/5] Inner Product Speed${NC}"

BENCH_IP_OUTPUT="$OUTPUT_DIR/bench_inner_product.txt"

echo "⏳ Medindo tempo de produto interno..."

if [ -f "./build/bin/bench_turboquant" ]; then
    ./build/bin/bench_turboquant \
        --bench inner-product \
        --dim 1536 \
        --iterations 1000 \
        | tee "$BENCH_IP_OUTPUT"
else
    echo "   (bench_turboquant não compilado, pulando...)"
fi

echo ""

# ============================================
# Benchmark 3: Throughput (tokens/s)
# ============================================
echo -e "${CYAN}[3/5] Inference Throughput${NC}"

BENCH_TPUT_OUTPUT="$OUTPUT_DIR/bench_throughput.txt"

echo "⏳ Medindo throughput de inferência..."

./build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -n 50 \
    --ctx-size 2048 \
    -ctk turboquant \
    -ctv turboquant \
    -fa on \
    --temp 0.0 \
    -p "The quick brown fox" \
    2>&1 | grep -E "tokens per second|eval time" | tee "$BENCH_TPUT_OUTPUT" || echo "   (métricas não encontradas)"

echo ""

# ============================================
# Benchmark 4: Memory Usage
# ============================================
echo -e "${CYAN}[4/5] Memory Usage${NC}"

BENCH_MEM_OUTPUT="$OUTPUT_DIR/bench_memory.txt"

echo "⏳ Medindo uso de memória..."

# Rodar com monitoring
(
    echo "=== Full Precision ==="
    ./build/bin/llama-cli -m "$MODEL_PATH" -n 10 --ctx-size 4096 -ctk f16 -ctv f16 -fa on -p "Test" 2>&1 | grep -i "memory\|VRAM\|size" || echo "N/A"
    
    echo ""
    echo "=== TurboQuant ==="
    ./build/bin/llama-cli -m "$MODEL_PATH" -n 10 --ctx-size 4096 -ctk turboquant -ctv turboquant -fa on -p "Test" 2>&1 | grep -i "memory\|VRAM\|size" || echo "N/A"
) | tee "$BENCH_MEM_OUTPUT"

echo ""

# ============================================
# Benchmark 5: Accuracy (Quick Perplexity)
# ============================================
echo -e "${CYAN}[5/5] Quick Accuracy Check${NC}"

BENCH_ACC_OUTPUT="$OUTPUT_DIR/bench_accuracy.txt"

echo "⏳ Medindo perplexidade rápida..."

# Dataset mínimo
DATASET_MINI="$OUTPUT_DIR/mini_dataset.txt"
cat > "$DATASET_MINI" << 'EOF'
The quick brown fox jumps over the lazy dog.
Artificial intelligence is transforming technology.
Machine learning models require large datasets.
EOF

(
    echo "=== Full Precision ==="
    ./build/bin/llama-perplexity -m "$MODEL_PATH" -f "$DATASET_MINI" -ctk f16 -ctv f16 2>&1 | grep "perplexity" || echo "N/A"
    
    echo ""
    echo "=== TurboQuant ==="
    ./build/bin/llama-perplexity -m "$MODEL_PATH" -f "$DATASET_MINI" -ctk turboquant -ctv turboquant 2>&1 | grep "perplexity" || echo "N/A"
) | tee "$BENCH_ACC_OUTPUT"

echo ""

# ============================================
# Gerar Relatório
# ============================================
echo -e "${MAGENTA}📊 Gerando relatório...${NC}"
echo ""

cat >> "$REPORT_FILE" << 'EOF'
## Benchmarks Executados

### 1. Quantization Speed
EOF

if [ -f "$BENCH_Q_OUTPUT" ]; then
    echo "~~~" >> "$REPORT_FILE"
    tail -20 "$BENCH_Q_OUTPUT" >> "$REPORT_FILE"
    echo "~~~" >> "$REPORT_FILE"
else
    echo "*Dados não disponíveis*" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << 'EOF'

### 2. Inner Product Speed
EOF

if [ -f "$BENCH_IP_OUTPUT" ]; then
    echo "~~~" >> "$REPORT_FILE"
    tail -20 "$BENCH_IP_OUTPUT" >> "$REPORT_FILE"
    echo "~~~" >> "$REPORT_FILE"
else
    echo "*Dados não disponíveis*" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << 'EOF'

### 3. Inference Throughput
EOF

if [ -f "$BENCH_TPUT_OUTPUT" ]; then
    echo "~~~" >> "$REPORT_FILE"
    cat "$BENCH_TPUT_OUTPUT" >> "$REPORT_FILE"
    echo "~~~" >> "$REPORT_FILE"
else
    echo "*Dados não disponíveis*" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << 'EOF'

### 4. Memory Usage
EOF

if [ -f "$BENCH_MEM_OUTPUT" ]; then
    echo "~~~" >> "$REPORT_FILE"
    cat "$BENCH_MEM_OUTPUT" >> "$REPORT_FILE"
    echo "~~~" >> "$REPORT_FILE"
else
    echo "*Dados não disponíveis*" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << 'EOF'

### 5. Quick Accuracy Check
EOF

if [ -f "$BENCH_ACC_OUTPUT" ]; then
    echo "~~~" >> "$REPORT_FILE"
    cat "$BENCH_ACC_OUTPUT" >> "$REPORT_FILE"
    echo "~~~" >> "$REPORT_FILE"
else
    echo "*Dados não disponíveis*" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

## Arquivos de Output

- \`bench_quant.txt\` - Benchmark de quantização
- \`bench_inner_product.txt\` - Benchmark de produto interno
- \`bench_throughput.txt\` - Throughput de inferência
- \`bench_memory.txt\` - Uso de memória
- \`bench_accuracy.txt\` - Check de accuracy

## Próximos Passos

1. Analisar resultados acima
2. Comparar com benchmarks do paper TurboQuant
3. Identificar gargalos de performance
4. Otimizar kernels CUDA se necessário

---

*Gerado por \`run_benchmarks.sh\` em $(date)*
EOF

echo -e "${GREEN}✅ Benchmarks concluídos!${NC}"
echo ""
echo "📁 Resultados salvos em: $OUTPUT_DIR/"
echo "📄 Relatório: $REPORT_FILE"
echo ""
echo "📊 Arquivos gerados:"
ls -la "$OUTPUT_DIR/" | grep -E "\.txt|\.md"
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Para visualizar o relatório:"
echo "  cat $REPORT_FILE"
echo ""
echo "Ou abra no navegador (se tiver pandoc):"
echo "  pandoc $REPORT_FILE -o ${REPORT_FILE%.md}.html"
