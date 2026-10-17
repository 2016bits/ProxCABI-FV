#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/plan_a/multiseed_report}"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m proxcabi.causal multiseed-report \
  --outputs-dir outputs \
  --multiseed-dir outputs/fever_multiseed \
  --variants "${VARIANTS:-full,no_cf,no_z,no_w}" \
  --seeds "${SEEDS:-13,21,42}" \
  --output-dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m proxcabi.causal paired-significance \
  --outputs-dir outputs \
  --multiseed-dir outputs/fever_multiseed \
  --variants "${SIGNIFICANCE_VARIANTS:-no_cf,no_z,no_w}" \
  --seeds "${SEEDS:-13,21,42}" \
  --output-dir "${OUTPUT_DIR}"

echo "Report: ${OUTPUT_DIR}/multiseed_results.md"
echo "Significance: ${OUTPUT_DIR}/paired_significance.md"
