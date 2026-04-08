#!/bin/bash
# TurboQuant Long Context Test - Teste de contexto longo (100K tokens)
#
# Uso: ./scripts/test_long_context.sh
#
# Variáveis de ambiente:
#   MODEL_PATH     - Caminho para o modelo GGUF
#   TQ_TYPE        - Tipo de quantização KV cache (padrão: turboquant)
#   CTX_SIZE       - Tamanho do contexto (padrão: 100000)
#   TEST_DOCUMENT  - Documento de teste para sumarização

set -e

# Configurações
MODEL_PATH="${MODEL_PATH:-/mnt/c/Users/mvrdu/.cache/huggingface/hub/model/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-Q8_0.gguf}"
TQ_TYPE="${TQ_TYPE:-turboquant}"
CTX_SIZE="${CTX_SIZE:-100000}"
TEST_DOCUMENT="${TEST_DOCUMENT:-}"
OUTPUT_FILE="/tmp/long_context_test.txt"

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================"
echo "  TurboQuant Long Context Test"
echo "========================================"
echo ""
echo "📦 Model:     $MODEL_PATH"
echo "🔧 KV Type:   $TQ_TYPE"
echo "📏 Context:   $CTX_SIZE tokens"
echo ""

# Verificar pré-requisitos
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ ERRO: Modelo não encontrado${NC}"
    exit 1
fi

if [ ! -f "./build/bin/llama-cli" ]; then
    echo -e "${RED}❌ ERRO: Binário llama-cli não encontrado${NC}"
    exit 1
fi

# Gerar documento de teste se não fornecido
if [ -z "$TEST_DOCUMENT" ]; then
    echo -e "${YELLOW}⚠️  Nenhum documento fornecido, gerando texto de teste...${NC}"
    
    # Gerar ~10K palavras de texto repetitivo para teste
    TEST_FILE="/tmp/test_document.txt"
    for i in {1..100}; do
        echo "This is a test document for long context evaluation. " >> "$TEST_FILE"
        echo "The quick brown fox jumps over the lazy dog. " >> "$TEST_FILE"
        echo "Artificial intelligence and machine learning are transforming technology. " >> "$TEST_FILE"
        echo "Transformers have revolutionized natural language processing. " >> "$TEST_FILE"
        echo "Attention mechanisms allow models to focus on relevant information. " >> "$TEST_FILE"
    done
    
    TEST_DOCUMENT="$TEST_FILE"
    echo "   Documento gerado: $TEST_DOCUMENT ($(wc -w < "$TEST_DOCUMENT") palavras)"
fi

# Prompt de teste
PROMPT="Summarize the following document in 3-5 sentences:\n\n$(cat "$TEST_DOCUMENT" | head -c 50000)"

echo ""
echo "⏳ Executando teste de contexto longo..."
echo "   (Isso pode levar alguns minutos)"
echo ""

# Iniciar monitoring de memória em background (se nvtop disponível)
if command -v nvtop &> /dev/null; then
    echo -e "${BLUE}📊 Monitoring VRAM com nvtop...${NC}"
    nvtop -p $(pgrep -f "llama-cli" || echo "0") > /tmp/vram_log.txt 2>&1 &
    NVTOP_PID=$!
fi

START_TIME=$(date +%s.%N)

# Executar teste
./build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -n 200 \
    --ctx-size "$CTX_SIZE" \
    -ctk "$TQ_TYPE" \
    -ctv "$TQ_TYPE" \
    -fa on \
    --memory-f32 \
    -p "$PROMPT" \
    2>&1 | tee "$OUTPUT_FILE"

END_TIME=$(date +%s.%N)
EXEC_TIME=$(echo "$END_TIME - $START_TIME" | bc)

# Parar monitoring
if [ ! -z "${NVTOP_PID:-}" ]; then
    kill $NVTOP_PID 2>/dev/null || true
fi

echo ""
echo "========================================"
echo "  Resultados"
echo "========================================"
echo ""

# Validações
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

# Teste 2: Sem erros de memória (OOM)
if grep -qi "out of memory\|OOM\|CUDA out of memory\|std::bad_alloc" "$OUTPUT_FILE"; then
    echo -e "${RED}❌ FALHA: Erro de memória detectado${NC}"
    ((FAILED++))
else
    echo -e "${GREEN}✅ Sem erros de memória${NC}"
    ((PASSED++))
fi

# Teste 3: Tempo de execução
EXEC_TIME_INT=${EXEC_TIME%.*}
MAX_TIME=300 # 5 minutos
if [ "$EXEC_TIME_INT" -lt "$MAX_TIME" ]; then
    echo -e "${GREEN}✅ Tempo de execução: ${EXEC_TIME}s${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠️  Tempo muito alto: ${EXEC_TIME}s (recomendado: <$MAX_TIME s)${NC}"
    ((PASSED++))
fi

# Teste 4: Output contém sumarização
WORD_COUNT=$(wc -w < "$OUTPUT_FILE" | tr -d ' ')
if [ "$WORD_COUNT" -ge 30 ]; then
    echo -e "${GREEN}✅ Output com tamanho adequado ($WORD_COUNT palavras)${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FALHA: Output muito curto ($WORD_COUNT palavras)${NC}"
    ((FAILED++))
fi

echo ""
echo "========================================"
echo "  Resumo"
echo "========================================"
echo -e "  ${GREEN}Passaram: $PASSED${NC}"
echo -e "  ${RED}Falharam:   $FAILED${NC}"
echo "  Tempo:      ${EXEC_TIME}s"
echo "  Contexto:   $CTX_SIZE tokens"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✅ Teste de contexto longo concluído com sucesso!${NC}"
    exit 0
else
    echo -e "${RED}❌ Alguns testes falharam${NC}"
    exit 1
fi
