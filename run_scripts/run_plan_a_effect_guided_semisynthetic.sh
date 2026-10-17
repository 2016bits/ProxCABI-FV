#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"
cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m proxcabi.causal effect-guided-semisynthetic \
  --responses "${RESPONSES:-outputs/plan_a/paper_experiments/real_exploratory/responses/responses.csv}" \
  --output-dir "${OUTPUT_DIR:-outputs/plan_a/effect_guided_semisynthetic}" \
  --n "${N:-40000}" \
  --seeds "${SEEDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}" \
  --proxy-accuracy "${PROXY_ACCURACY:-0.85}" --budget-groups "${BUDGET_GROUPS:-2}"
