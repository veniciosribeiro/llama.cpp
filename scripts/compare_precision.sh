#!/bin/bash
# TurboQuant vs Full Precision - Comparação direta de performance e accuracy
#
# Uso: ./scripts/compare_precision.sh
#
# Executa o mesmo prompt com full precision (f16) e TurboQuant,
# comparando tempo, output e qualidade.

set -e

# Configurações
MODEL_PATH="${MODEL_PATH:-/mnt/c/Users/mvrdu/.cache/huggingface/hub/model/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-Q8_0.gguf}"
PROMPT="${PROMPT:-Explain the theory of relativity in simple terms.}"
OUTPUT_DIR="/tmp/turboquant_comparison"

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "========================================"
echo "  TurboQuant vs Full Precision"
echo "  Comparação Direta"
echo "========================================"
echo ""
echo "📦 Model:  $MODEL_PATH"
echo "📝 Prompt: $PROMPT"
echo ""

# Criar diretório de output
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
# Teste 1: Full Precision (Baseline)
# ============================================
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Teste 1: Full Precision (f16)${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

BASELINE_OUTPUT="$OUTPUT_DIR/baseline.txt"
BASELINE_TIME_OUTPUT="$OUTPUT_DIR/baseline_time.txt"

START_TIME=$(date +%s.%N)
time ./build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -n 100 \
    --ctx-size 6000 \
    -ctk f16 \
    -ctv f16 \
    -fa on \
    -p "$PROMPT" \
    2>&1 | tee "$BASELINE_OUTPUT"
END_TIME=$(date +%s.%N)

BASELINE_TIME=$(echo "$END_TIME - $START_TIME" | bc)
echo "$BASELINE_TIME" > "$BASELINE_TIME_OUTPUT"

BASELINE_WORDS=$(wc -w < "$BASELINE_OUTPUT" | tr -d ' ')

echo ""
echo -e "${GREEN}✅ Baseline completo${NC}"
echo "   Tempo: ${BASELINE_TIME}s"
echo "   Palavras: $BASELINE_WORDS"
echo ""

# ============================================
# Teste 2: TurboQuant
# ============================================
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Teste 2: TurboQuant${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

TQ_OUTPUT="$OUTPUT_DIR/turboquant.txt"
TQ_TIME_OUTPUT="$OUTPUT_DIR/turboquant_time.txt"
TQ_TYPE="${TQ_TYPE:-turboquant}"

START_TIME=$(date +%s.%N)
time ./build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -n 100 \
    --ctx-size 6000 \
    -ctk "$TQ_TYPE" \
    -ctv "$TQ_TYPE" \
    -fa on \
    -p "$PROMPT" \
    2>&1 | tee "$TQ_OUTPUT"
END_TIME=$(date +%s.%N)

TQ_TIME=$(echo "$END_TIME - $START_TIME" | bc)
echo "$TQ_TIME" > "$TQ_TIME_OUTPUT"

TQ_WORDS=$(wc -w < "$TQ_OUTPUT" | tr -d ' ')

echo ""
echo -e "${GREEN}✅ TurboQuant completo${NC}"
echo "   Tempo: ${TQ_TIME}s"
echo "   Palavras: $TQ_WORDS"
echo ""

# ============================================
# Comparação
# ============================================
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Comparação${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Calcular overhead
if [ "$(echo "$BASELINE_TIME > 0" | bc)" -eq 1 ]; then
    OVERHEAD=$(echo "scale=2; (($TQ_TIME - $BASELINE_TIME) / $BASELINE_TIME) * 100" | bc)
    SPEEDUP=$(echo "scale=2; $BASELINE_TIME / $TQ_TIME" | bc)
else
    OVERHEAD="N/A"
    SPEEDUP="N/A"
fi

echo -e "${BLUE}Performance:${NC}"
echo "  Baseline (f16):     ${BASELINE_TIME}s"
echo "  TurboQuant:         ${TQ_TIME}s"
echo "  Overhead:           ${OVERHEAD}%"
echo "  Speedup:            ${SPEEDUP}x"
echo ""

# Comparar outputs
echo -e "${BLUE}Output:${NC}"
echo "  Baseline palavras:  $BASELINE_WORDS"
echo "  TurboQuant palavras: $TQ_WORDS"

if [ "$BASELINE_WORDS" -gt 0 ]; then
    WORD_DIFF=$(echo "scale=2; (($TQ_WORDS - $BASELINE_WORDS) / $BASELINE_WORDS) * 100" | bc)
    echo "  Diferença:          ${WORD_DIFF}%"
fi
echo ""

# Diff de conteúdo (linhas diferentes)
echo -e "${BLUE}Similaridade de conteúdo:${NC}"
DIFF_LINES=$(diff "$BASELINE_OUTPUT" "$TQ_OUTPUT" 2>&1 | grep -c "^[<>]" || echo "0")
echo "  Linhas diferentes:  $DIFF_LINES"

if [ "$DIFF_LINES" -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Exemplo de diferenças (primeiras 5):${NC}"
    diff "$BASELINE_OUTPUT" "$TQ_OUTPUT" 2>&1 | grep "^[<>]" | head -10
fi
echo ""

# ============================================
# Veredito
# ============================================
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Veredito${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Critérios de sucesso
SUCCESS=true

# Overhead < 10%
OVERHEAD_INT=${OVERHEAD%.*}
if [ "${OVERHEAD_INT:-0}" -gt 10 ]; then
    echo -e "${RED}❌ Overhead muito alto: ${OVERHEAD}% (máximo: 10%)${NC}"
    SUCCESS=false
else
    echo -e "${GREEN}✅ Overhead aceitável: ${OVERHEAD}%${NC}"
fi

# Output tamanho similar (±20%)
if [ "$WORD_DIFF" != "N/A" ]; then
    WORD_DIFF_INT=${WORD_DIFF%.*}
    WORD_DIFF_INT=${WORD_DIFF_INT#-} # Remover sinal negativo
    if [ "${WORD_DIFF_INT:-0}" -gt 20 ]; then
        echo -e "${RED}❌ Diferença de output muito grande: ${WORD_DIFF}% (máximo: 20%)${NC}"
        SUCCESS=false
    else
        echo -e "${GREEN}✅ Output similar: ${WORD_DIFF}% de diferença${NC}"
    fi
fi

echo ""
if [ "$SUCCESS" = true ]; then
    echo -e "${GREEN}✅ TurboQuant passou nos critérios de comparação!${NC}"
    echo ""
    echo "📁 Outputs salvos em: $OUTPUT_DIR/"
    echo "   - baseline.txt"
    echo "   - turboquant.txt"
    exit 0
else
    echo -e "${RED}❌ TurboQuant não atendeu todos os critérios${NC}"
    exit 1
fi
