#!/usr/bin/env bash
# Bateria "energia por tarefa": cada modelo conclui o resumo naturalmente (limite alto de saída).
# Entrada fixa de 4.000 caracteres (~1.000-1.200 tokens, cerca de uma página de texto normativo).
# Fator adicional: modo de raciocínio do gemma4 desligado (nao) x ligado (sim).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
OUT=resultados/tarefa
mkdir -p "$OUT"
rodar() {  # modelo think
  echo "=== $1 think=$2 $(date +%H:%M:%S)"
  python medir_inferencia.py --modelos "$1" --think "$2" --reps 10 --chars-entrada 4000 \
    --num-predict 4096 --num-ctx 8192 --saida "$OUT/${1}_think-$2"
}
rodar llama3.2 padrao
rodar gemma4 nao
rodar gemma4 sim
echo "=== FIM $(date +%H:%M:%S)"
