#!/usr/bin/env bash
# Full RTN sweep: 9 models x {Q,K,V,O} x every layer x {4,3} bits.
# Produces the 2,144 RTN measurements reported in the paper.
#
# Resumable: re-run the same command after an interruption and it will
# continue from the last completed layer.

set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="${OUTPUT_DIR:-results/raw}"
mkdir -p "$OUTPUT_DIR"

MODELS=(
  gpt-j-6b
  llama1-7b
  llama2-7b
  llama3-8b
  mistral-7b
  opt-1.3b
  opt-6.7b
  qwen2.5-1.5b
  qwen2.5-7b
)

for model in "${MODELS[@]}"; do
  echo "==================================================================="
  echo ">> RTN sweep: ${model}"
  echo "==================================================================="
  python src/quant_sensitivity_experiment.py \
    --models "${model}" \
    --method rtn \
    --bits 4 3 \
    --group_size 128 \
    --n_tokens 16384 \
    --block_size 1024 \
    --eval_datasets wikitext2 \
    --output_dir "${OUTPUT_DIR}"
done

echo "RTN sweep complete. Results in ${OUTPUT_DIR}/"
