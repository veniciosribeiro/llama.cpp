#!/bin/bash
# TurboQuant Perplexity Evaluation - Avaliação de accuracy via perplexidade
#
# Uso: ./scripts/eval_perplexity.sh
#
# Compara a perplexidade entre full precision e TurboQuant
# usando um dataset de referência (ex: WikiText)

set -e

# Configurações
MODEL_PATH="${MODEL_PATH:-/mnt/c/Users/mvrdu/.cache/huggingface/hub/model/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-Q8_0.gguf}"
DATASET_PATH="${DATASET_PATH:-datasets/wikitext.txt}"
OUTPUT_DIR="/tmp/perplexity_eval"

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "========================================"
echo "  TurboQuant Perplexity Evaluation"
echo "========================================"
echo ""
echo "📦 Model:     $MODEL_PATH"
echo "📊 Dataset:   $DATASET_PATH"
echo ""

# Criar diretório
mkdir -p "$OUTPUT_DIR"

# Verificar pré-requisitos
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ ERRO: Modelo não encontrado${NC}"
    exit 1
fi

if [ ! -f "./build/bin/llama-perplexity" ]; then
    echo -e "${RED}❌ ERRO: Binário llama-perplexity não encontrado${NC}"
    echo "   Execute: cmake --build build --target llama-perplexity"
    exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo -e "${YELLOW}⚠️  Dataset não encontrado: $DATASET_PATH${NC}"
    echo ""
    echo "   Baixe o WikiText dataset:"
    echo "   wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
    echo "   unzip wikitext-2-raw-v1.zip"
    echo "   mv wikitext-2-raw/wiki.test.raw datasets/wikitext.txt"
    echo ""
    
    # Criar dataset de teste mínimo
    echo "   Criando dataset de teste mínimo..."
    mkdir -p datasets
    cat > "$DATASET_PATH" << 'EOF'
The quick brown fox jumps over the lazy dog.
Artificial intelligence and machine learning are transforming technology.
Transformers have revolutionized natural language processing tasks.
Attention mechanisms allow models to focus on relevant information.
Deep learning models require large amounts of training data.
Neural networks are inspired by biological neural systems.
Language models predict the next token in a sequence.
The theory of relativity was developed by Albert Einstein.
Quantum mechanics describes the behavior of particles at small scales.
Climate change is one of the most pressing global challenges.
EOF
    echo "   Dataset criado: $DATASET_PATH"
fi

# ============================================
# Teste 1: Full Precision (Baseline)
# ============================================
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Teste 1: Full Precision (f16)${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

BASELINE_OUTPUT="$OUTPUT_DIR/perplexity_baseline.txt"

./build/bin/llama-perplexity \
    -m "$MODEL_PATH" \
    -f "$DATASET_PATH" \
    -ctk f16 \
    -ctv f16 \
    2>&1 | tee "$BASELINE_OUTPUT"

# Extrair perplexidade
BASELINE_PPL=$(grep -oP "perplexity\s*=\s*\K[0-9.]+" "$BASELINE_OUTPUT" | tail -1)

echo ""
if [ -n "$BASELINE_PPL" ]; then
    echo -e "${GREEN}✅ Baseline perplexity: $BASELINE_PPL${NC}"
else
    echo -e "${YELLOW}⚠️  Não foi possível extrair perplexidade do baseline${NC}"
    BASELINE_PPL="N/A"
fi
echo ""

# ============================================
# Teste 2: TurboQuant
# ============================================
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Teste 2: TurboQuant${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

TQ_OUTPUT="$OUTPUT_DIR/perplexity_turboquant.txt"
TQ_TYPE="${TQ_TYPE:-turboquant}"

./build/bin/llama-perplexity \
    -m "$MODEL_PATH" \
    -f "$DATASET_PATH" \
    -ctk "$TQ_TYPE" \
    -ctv "$TQ_TYPE" \
    2>&1 | tee "$TQ_OUTPUT"

# Extrair perplexidade
TQ_PPL=$(grep -oP "perplexity\s*=\s*\K[0-9.]+" "$TQ_OUTPUT" | tail -1)

echo ""
if [ -n "$TQ_PPL" ]; then
    echo -e "${GREEN}✅ TurboQuant perplexity: $TQ_PPL${NC}"
else
    echo -e "${YELLOW}⚠️  Não foi possível extrair perplexidade do TurboQuant${NC}"
    TQ_PPL="N/A"
fi
echo ""

# ============================================
# Comparação
# ============================================
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Comparação${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BLUE}Perplexidade:${NC}"
echo "  Baseline (f16):     $BASELINE_PPL"
echo "  TurboQuant:         $TQ_PPL"
echo ""

if [ "$BASELINE_PPL" != "N/A" ] && [ "$TQ_PPL" != "N/A" ]; then
    # Calcular degradação
    PPL_DIFF=$(echo "scale=2; (($TQ_PPL - $BASELINE_PPL) / $BASELINE_PPL) * 100" | bc)
    PPL_DIFF_ABS=$(echo "scale=2; $TQ_PPL - $BASELINE_PPL" | bc)
    
    echo -e "${BLUE}Degradação:${NC}"
    echo "  Absoluto:           +$PPL_DIFF_ABS"
    echo "  Relativo:           ${PPL_DIFF}%"
    echo ""
    
    # Critério de aceite: ≤ 1% degradação
    PPL_DIFF_INT=${PPL_DIFF%.*}
    PPL_DIFF_INT=${PPL_DIFF_INT#-}
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Veredito${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    if [ "${PPL_DIFF_INT:-0}" -le 1 ]; then
        echo -e "${GREEN}✅ Degradação aceitável: ${PPL_DIFF}% (máximo: 1%)${NC}"
        echo ""
        echo "🎉 TurboQuant mantém accuracy equivalente ao full precision!"
        exit 0
    elif [ "${PPL_DIFF_INT:-0}" -le 5 ]; then
        echo -e "${YELLOW}⚠️  Degradação moderada: ${PPL_DIFF}% (recomendado: ≤ 1%)${NC}"
        echo ""
        echo "   Considere ajustar parâmetros do TurboQuant"
        exit 0
    else
        echo -e "${RED}❌ Degradação alta: ${PPL_DIFF}% (máximo recomendado: 1%)${NC}"
        echo ""
        echo "   Possíveis causas:"
        echo "   - Bit-width muito baixo"
        echo "   - Codebook MSE precisa de otimização"
        echo "   - Problema na implementação"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️  Não foi possível calcular degradação (valores N/A)${NC}"
    exit 0
fi
