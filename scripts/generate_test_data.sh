#!/bin/bash
# TurboQuant Test Data Generator - Gera datasets para testes
#
# Uso: ./scripts/generate_test_data.sh [TIPO] [TAMANHO]
#
# Tipos: mini, small, medium, large, long-context
# Tamanhos: número de palavras (padrão varia por tipo)

set -e

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TIPO="${1:-small}"
TAMANHO="${2:-}"

OUTPUT_DIR="datasets"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  TurboQuant Test Data Generator"
echo "========================================"
echo ""
echo "📝 Tipo:    $TIPO"
echo "📏 Tamanho: ${TAMANHO:-auto}"
echo ""

# Texto base para geração
TEXT_SAMPLES=(
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence and machine learning are transforming technology. "
    "Transformers have revolutionized natural language processing. "
    "Attention mechanisms allow models to focus on relevant information. "
    "Deep learning models require large amounts of training data. "
    "Neural networks are inspired by biological neural systems. "
    "Language models predict the next token in a sequence. "
    "The theory of relativity was developed by Albert Einstein. "
    "Quantum mechanics describes the behavior of particles at small scales. "
    "Climate change is one of the most pressing global challenges. "
    "The human brain contains approximately 86 billion neurons. "
    "Machine learning algorithms learn patterns from data. "
    "Computer vision enables machines to interpret visual information. "
    "Reinforcement learning uses rewards to train agents. "
    "Natural language understanding is a key AI challenge. "
)

generate_text() {
    local target_words=$1
    local output_file=$2
    
    echo "   Gerando $target_words palavras..."
    
    > "$output_file"
    local word_count=0
    local sample_idx=0
    
    while [ $word_count -lt $target_words ]; do
        echo -n "${TEXT_SAMPLES[$sample_idx]}" >> "$output_file"
        word_count=$(wc -w < "$output_file" | tr -d ' ')
        sample_idx=$(( (sample_idx + 1) % ${#TEXT_SAMPLES[@]} ))
    done
    
    echo "   ✅ Gerado: $output_file ($word_count palavras)"
}

case "$TIPO" in
    mini)
        TAMANHO=${TAMANHO:-50}
        FILE="$OUTPUT_DIR/test_mini.txt"
        generate_text $TAMANHO "$FILE"
        ;;
    
    small)
        TAMANHO=${TAMANHO:-500}
        FILE="$OUTPUT_DIR/test_small.txt"
        generate_text $TAMANHO "$FILE"
        ;;
    
    medium)
        TAMANHO=${TAMANHO:-5000}
        FILE="$OUTPUT_DIR/test_medium.txt"
        generate_text $TAMANHO "$FILE"
        ;;
    
    large)
        TAMANHO=${TAMANHO:-50000}
        FILE="$OUTPUT_DIR/test_large.txt"
        generate_text $TAMANHO "$FILE"
        ;;
    
    long-context)
        TAMANHO=${TAMANHO:-100000}
        FILE="$OUTPUT_DIR/test_long_context.txt"
        generate_text $TAMANHO "$FILE"
        echo ""
        echo -e "${YELLOW}⚠️  Dataset grande (${TAMANHO} palavras)${NC}"
        echo "   Use com: --ctx-size $(($TAMANHO * 2))"
        ;;
    
    wikitext)
        # Download do WikiText-2
        FILE="$OUTPUT_DIR/wikitext.txt"
        if [ ! -f "$FILE" ]; then
            echo "   Baixando WikiText-2 dataset..."
            wget -q https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
            unzip -q wikitext-2-raw-v1.zip
            mv wikitext-2-raw/wiki.test.raw "$FILE"
            rm -rf wikitext-2-raw-v1.zip wikitext-2-raw/
            echo "   ✅ WikiText-2 baixado: $FILE"
        else
            echo "   ✅ WikiText-2 já existe: $FILE"
        fi
        ;;
    
    all)
        echo -e "${BLUE}Gerando todos os datasets...${NC}"
        echo ""
        $0 mini
        $0 small
        $0 medium
        $0 large
        $0 wikitext
        echo ""
        echo -e "${GREEN}✅ Todos os datasets gerados!${NC}"
        exit 0
        ;;
    
    *)
        echo -e "${RED}❌ Tipo desconhecido: $TIPO${NC}"
        echo ""
        echo "Tipos válidos:"
        echo "  mini         - 50 palavras (teste rápido)"
        echo "  small        - 500 palavras (teste funcional)"
        echo "  medium       - 5K palavras (teste de accuracy)"
        echo "  large        - 50K palavras (teste de contexto)"
        echo "  long-context - 100K palavras (teste extremo)"
        echo "  wikitext     - WikiText-2 dataset (benchmark)"
        echo "  all          - Gerar todos"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Dataset gerado com sucesso!${NC}"
echo ""
echo "📁 Arquivo: $FILE"
echo "📊 Tamanho: $(wc -w < "$FILE" | tr -d ' ') palavras"
echo "💾 Bytes:   $(wc -c < "$FILE" | tr -d ' ')"
echo ""
echo "Uso sugerido:"
echo "  ./build/bin/llama-cli -m model.gguf -f $FILE -n 100"
