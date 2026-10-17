#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m proxcabi.causal efficiency-report \
  --multiseed-dir outputs/fever_multiseed \
  --variants "${VARIANTS:-full,erm,cf_only,contrastive_only,reweight,group_dro}" \
  --seeds "${SEEDS:-13,21,42}" \
  --output-dir "${OUTPUT_DIR:-outputs/plan_a/efficiency_report}"
