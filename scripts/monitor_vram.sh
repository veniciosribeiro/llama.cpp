#!/bin/bash
# TurboQuant Memory Monitoring - Monitoramento de uso de memória GPU
#
# Uso: ./scripts/monitor_vram.sh [PID]
#
# Monitora o uso de VRAM durante a execução do llama-cli
# e compara entre full precision e TurboQuant.

set -e

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "========================================"
echo "  TurboQuant VRAM Monitor"
echo "========================================"
echo ""

# Verificar se nvidia-smi está disponível
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ ERRO: nvidia-smi não encontrado${NC}"
    echo "   Instale drivers NVIDIA ou CUDA toolkit"
    exit 1
fi

# Verificar se há processo llama-cli rodando
LLAMA_PID=$(pgrep -f "llama-cli" | head -1)

if [ -n "$LLAMA_PID" ]; then
    echo -e "${GREEN}✅ Processo llama-cli detectado: PID $LLAMA_PID${NC}"
    TARGET_PID="$LLAMA_PID"
else
    echo -e "${YELLOW}⚠️  Nenhum processo llama-cli rodando${NC}"
    echo ""
    echo "   Inicie o llama-cli em outro terminal, ou"
    echo "   execute este script após iniciar um teste."
    echo ""
    echo "   Exemplo:"
    echo "   ./build/bin/llama-cli -m model.gguf -n 100 -ctk turboquant -ctv turboquant &"
    echo "   ./scripts/monitor_vram.sh"
    echo ""
    TARGET_PID=""
fi

echo ""
echo "📊 Monitorando VRAM (Ctrl+C para parar)..."
echo ""

# Header
printf "%-10s %-15s %-15s %-15s %-10s\n" "TIME" "USED (MB)" "FREE (MB)" "TOTAL (MB)" "USAGE %"
printf "%-10s %-15s %-15s %-15s %-10s\n" "----" "---------" "---------" "---------" "------"

# Log file
LOG_FILE="/tmp/vram_monitor_$(date +%Y%m%d_%H%M%S).log"

START_TIME=$(date +%s)

# Monitoring loop
while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    TIME_STR=$(date +%H:%M:%S)
    
    # Obter stats da GPU
    GPU_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits)
    
    # Parse (assume single GPU)
    USED=$(echo "$GPU_INFO" | awk -F', ' '{print $1}')
    TOTAL=$(echo "$GPU_INFO" | awk -F', ' '{print $2}')
    FREE=$(echo "$GPU_INFO" | awk -F', ' '{print $3}')
    UTIL=$(echo "$GPU_INFO" | awk -F', ' '{print $4}')
    
    # Calcular porcentagem
    if [ "$TOTAL" -gt 0 ]; then
        USAGE_PCT=$(echo "scale=1; ($USED / $TOTAL) * 100" | bc)
    else
        USAGE_PCT="N/A"
    fi
    
    # Print
    printf "%-10s %-15s %-15s %-15s %-10s\n" "$TIME_STR" "$USED" "$FREE" "$TOTAL" "${USAGE_PCT}%"
    
    # Log
    echo "$ELAPSED,$USED,$FREE,$TOTAL,$USAGE_PCT,$UTIL" >> "$LOG_FILE"
    
    # Se tiver PID alvo, verificar se processo ainda existe
    if [ -n "$TARGET_PID" ]; then
        if ! kill -0 "$TARGET_PID" 2>/dev/null; then
            echo ""
            echo -e "${YELLOW}⚠️  Processo llama-cli terminou${NC}"
            echo ""
            echo "📁 Log salvo em: $LOG_FILE"
            echo ""
            
            # Gerar resumo
            echo "========================================"
            echo "  Resumo"
            echo "========================================"
            
            if [ -f "$LOG_FILE" ]; then
                MAX_USED=$(cut -d',' -f2 "$LOG_FILE" | sort -n | tail -1)
                AVG_USED=$(cut -d',' -f2 "$LOG_FILE" | awk '{sum+=$1} END {print int(sum/NR)}')
                echo "  VRAM Máximo:  ${MAX_USED} MB"
                echo "  VRAM Médio:   ${AVG_USED} MB"
                echo "  Duração:      ${ELAPSED}s"
                echo "  Amostras:     $(wc -l < "$LOG_FILE")"
            fi
            
            exit 0
        fi
    fi
    
    sleep 1
done
