#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m proxcabi.causal counterfactual-human-template \
  --data-dir data --train-items "${TRAIN_ITEMS:-300}" --dev-items "${DEV_ITEMS:-100}" \
  --annotators "${ANNOTATORS:-3}" --seed "${SEED:-2027}" \
  --output-dir "${OUTPUT_DIR:-outputs/plan_a/counterfactual_human_evaluation}"
