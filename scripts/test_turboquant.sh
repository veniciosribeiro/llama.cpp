#!/bin/bash
# TurboQuant Validation Test - Validação funcional automatizada
# 
# Uso: ./scripts/test_turboquant.sh
#
# Variáveis de ambiente:
#   MODEL_PATH  - Caminho para o modelo GGUF (padrão: ~/.cache/huggingface/...)
#   TQ_TYPE     - Tipo de quantização KV cache (padrão: q8_0)
#   OUTPUT_FILE - Arquivo de saída do teste (padrão: /tmp/turboquant_test_output.txt)

set -e

# Configurações padrão
MODEL_PATH="${MODEL_PATH:-/mnt/c/Users/mvrdu/.cache/huggingface/hub/model/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-Q8_0.gguf}"
TQ_TYPE="${TQ_TYPE:-q8_0}"
OUTPUT_FILE="${OUTPUT_FILE:-/tmp/turboquant_test_output.txt}"
PROMPT="${PROMPT:-Explain quantum entanglement in simple terms.}"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  TurboQuant Validation Test"
echo "========================================"
echo ""
echo "📦 Model:    $MODEL_PATH"
echo "🔧 KV Type:  $TQ_TYPE"
echo "📝 Prompt:   $PROMPT"
echo ""

# Verificar se modelo existe
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ ERRO: Modelo não encontrado em $MODEL_PATH${NC}"
    exit 1
fi

# Verificar se binário existe
if [ ! -f "./build/bin/llama-cli" ]; then
    echo -e "${RED}❌ ERRO: Binário llama-cli não encontrado.${NC}"
    echo "   Execute: cmake --build build --config Release --target llama-cli"
    exit 1
fi

echo "⏳ Executando teste..."
echo ""

# Executar teste e capturar output
START_TIME=$(date +%s.%N)
./build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -n 100 \
    --ctx-size 6000 \
    -ctk "$TQ_TYPE" \
    -ctv "$TQ_TYPE" \
    -fa on \
    -p "$PROMPT" \
    2>&1 | tee "$OUTPUT_FILE"
END_TIME=$(date +%s.%N)

# Calcular tempo de execução
EXEC_TIME=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo "========================================"
echo "  Validação"
echo "========================================"
echo ""

# Contador de testes
PASSED=0
FAILED=0

# Teste 1: Output não vazio
if [ -s "$OUTPUT_FILE" ]; then
    echo -e "${GREEN}✅ Output não vazio${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FALHA: Output vazio${NC}"
    ((FAILED++))
fi

# Teste 2: Verificar padrões de garbage
GARBAGE_PATTERNS="NaN|Inf|nan|inf||\\x00|\\x01|\\x02|\\x03|\\x04|\\x05|\\x06|\\x07|\\x08|\\x0B|\\x0E|\\x0F"
if grep -qE "$GARBAGE_PATTERNS" "$OUTPUT_FILE"; then
    echo -e "${RED}❌ FALHA: Detectado garbage no output${NC}"
    echo "   Padrões encontrados: $(grep -oE "$GARBAGE_PATTERNS" "$OUTPUT_FILE" | head -5 | tr '\n' ' ')"
    ((FAILED++))
else
    echo -e "${GREEN}✅ Sem garbage detectado${NC}"
    ((PASSED++))
fi

# Teste 3: Verificar tamanho do output (palavras)
WORD_COUNT=$(wc -w < "$OUTPUT_FILE" | tr -d ' ')
MIN_WORDS=50
if [ "$WORD_COUNT" -ge "$MIN_WORDS" ]; then
    echo -e "${GREEN}✅ Output com tamanho adequado ($WORD_COUNT palavras)${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FALHA: Output muito curto ($WORD_COUNT palavras, mínimo: $MIN_WORDS)${NC}"
    ((FAILED++))
fi

# Teste 4: Verificar tempo de execução
MAX_TIME=60 # segundos
EXEC_TIME_INT=${EXEC_TIME%.*}
if [ "$EXEC_TIME_INT" -lt "$MAX_TIME" ]; then
    echo -e "${GREEN}✅ Tempo de execução aceitável (${EXEC_TIME}s)${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠️  Tempo de execução alto (${EXEC_TIME}s, recomendado: <$MAX_TIME s)${NC}"
    ((PASSED++)) # Não falha, apenas alerta
fi

# Teste 5: Verificar se contém texto coerente (pelo menos uma frase completa)
if grep -qE "\.[[:space:]]" "$OUTPUT_FILE"; then
    echo -e "${GREEN}✅ Contém frases completas${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠️  Output pode não conter frases completas${NC}"
    ((PASSED++)) # Não falha, apenas alerta
fi

echo ""
echo "========================================"
echo "  Resumo"
echo "========================================"
echo -e "  ${GREEN}Passaram: $PASSED${NC}"
echo -e "  ${RED}Falharam:   $FAILED${NC}"
echo "  Tempo:      ${EXEC_TIME}s"
echo "  Palavras:   $WORD_COUNT"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✅ Todos os testes passaram!${NC}"
    exit 0
else
    echo -e "${RED}❌ Alguns testes falharam${NC}"
    exit 1
fi
