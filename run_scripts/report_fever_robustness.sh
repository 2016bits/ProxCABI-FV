#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m proxcabi.causal robustness-report \
  --data-dir data --multiseed-dir outputs/fever_multiseed \
  --variants "${VARIANTS:-full,no_cf,no_z,no_w,erm,cf_only,contrastive_only,reweight,group_dro}" \
  --seeds "${SEEDS:-13,21,42}" \
  --error-sample-size "${ERROR_SAMPLE_SIZE:-100}" \
  --output-dir "${OUTPUT_DIR:-outputs/plan_a/robustness_report}"
