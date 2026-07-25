#!/usr/bin/env bash
# Regenerate every figure and LaTeX table reported in the paper from the
# JSON files in results/raw/.

set -euo pipefail
cd "$(dirname "$0")/.."

RAW_DIR="${RAW_DIR:-results/raw}"
FIG_DIR="${FIG_DIR:-results/figures}"
TABLE_DIR="${TABLE_DIR:-results/tables}"

mkdir -p "${FIG_DIR}" "${TABLE_DIR}"

echo ">> Analyzing RTN results"
python src/analyze_quant_results.py \
  --input "${RAW_DIR}/*_quant_sensitivity_rtn_g128.json" \
  --bits 4 3 \
  --n_boot 2000 \
  --output_dir "${FIG_DIR}"

if compgen -G "${RAW_DIR}/*_quant_sensitivity_gptq_g128.json" > /dev/null; then
  echo ">> Analyzing GPTQ results"
  python src/analyze_quant_results.py \
    --input "${RAW_DIR}/*_quant_sensitivity_gptq_g128.json" \
    --bits 4 3 \
    --n_boot 2000 \
    --output_dir "${FIG_DIR}/gptq"
else
  echo ">> No GPTQ results found, skipping."
fi

# Collect LaTeX tables into results/tables/
find "${FIG_DIR}" -name '*.tex' -exec cp {} "${TABLE_DIR}/" \; 2>/dev/null || true

echo "Figures  -> ${FIG_DIR}/"
echo "Tables   -> ${TABLE_DIR}/"
