#!/usr/bin/env bash
# Matriz completa: 2 modelos x 3 tamanhos de entrada x 2 tamanhos de saída x 10 repetições.
# Entradas de 1.000 / 4.000 / 12.000 caracteres (~300 / ~1.200 / ~3.600 tokens); saídas de até 128 / 512 tokens.
# num_ctx 8192 comporta a maior entrada somada à maior saída.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
OUT=resultados/matriz
mkdir -p "$OUT"
for modelo in llama3.2 gemma4; do
  for chars in 1000 4000 12000; do
    for saida in 128 512; do
      rodada="${modelo}_in${chars}_out${saida}"
      echo "=== $rodada $(date +%H:%M:%S)"
      python medir_inferencia.py --modelos "$modelo" --reps 10 --chars-entrada "$chars" \
        --num-predict "$saida" --num-ctx 8192 --saida "$OUT/$rodada"
    done
  done
done
echo "=== FIM $(date +%H:%M:%S)"
