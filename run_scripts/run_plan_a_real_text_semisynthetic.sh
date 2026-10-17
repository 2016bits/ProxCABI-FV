#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"
RESPONSES="${RESPONSES:-outputs/plan_a/paper_experiments/real_exploratory/responses/responses.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/plan_a/real_text_semisynthetic}"
cd "${PROJECT_DIR}"

"${PYTHON_BIN}" -m proxcabi.causal real-text-semisynthetic \
  --responses "${RESPONSES}" --output-dir "${OUTPUT_DIR}" \
  --n "${N:-20000}" --seeds "${SEEDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}" \
  --proxy-strengths "${PROXY_STRENGTHS:-0.55,0.65,0.75,0.85,0.95}" \
  --bootstrap-reps "${BOOTSTRAP_REPS:-30}"

"${PYTHON_BIN}" -m proxcabi.causal real-text-sensitivity \
  --responses "${RESPONSES}" --output-dir "${OUTPUT_DIR}" \
  --n "${N:-20000}" --seeds "${SEEDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}"

"${PYTHON_BIN}" -m proxcabi.causal neural-real-text-semisynthetic \
  --responses "${RESPONSES}" --output-dir "${OUTPUT_DIR}" \
  --n "${N:-20000}" --seeds "${NEURAL_SEEDS:-0,1,2,3,4}" \
  --proxy-strengths "${PROXY_STRENGTHS:-0.55,0.65,0.75,0.85,0.95}" \
  --epochs "${NEURAL_EPOCHS:-300}" --patience "${NEURAL_PATIENCE:-50}"

echo "Report: ${OUTPUT_DIR}/semisynthetic_results.md"
echo "Sensitivity: ${OUTPUT_DIR}/sensitivity_results.md"
echo "Neural bridge: ${OUTPUT_DIR}/neural_semisynthetic_results.md"
