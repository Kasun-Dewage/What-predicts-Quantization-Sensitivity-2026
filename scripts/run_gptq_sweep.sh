#!/usr/bin/env bash
# GPTQ sweep: 7 models x {Q,K,V,O} x every layer x {4,3} bits.
# Produces the 1,664 GPTQ measurements reported in the paper.
# Calibration: 2,048 tokens from C4, group size 128.

set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="${OUTPUT_DIR:-results/raw}"
mkdir -p "$OUTPUT_DIR"

MODELS=(
  gpt-j-6b
  llama2-7b
  llama3-8b
  mistral-7b
  opt-1.3b
  opt-6.7b
  qwen2.5-1.5b
)

for model in "${MODELS[@]}"; do
  echo "==================================================================="
  echo ">> GPTQ sweep: ${model}"
  echo "==================================================================="
  python src/quant_sensitivity_experiment.py \
    --models "${model}" \
    --method gptq \
    --bits 4 3 \
    --group_size 128 \
    --n_tokens 16384 \
    --block_size 1024 \
    --n_calib_tokens 2048 \
    --calib_source c4 \
    --eval_datasets wikitext2 \
    --output_dir "${OUTPUT_DIR}"
done

echo "GPTQ sweep complete. Results in ${OUTPUT_DIR}/"
