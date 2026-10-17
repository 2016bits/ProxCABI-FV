#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"
cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m proxcabi.causal counterfactual-quality \
  --train data/FEVER/converted_data/cf_train_hard_nli.json \
  --dev data/FEVER/converted_data/cf_dev_hard_nli.json \
  --output-dir outputs/plan_a/counterfactual_quality
