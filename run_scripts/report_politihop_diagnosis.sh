#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m proxcabi.causal politihop-diagnosis \
  --data-dir data \
  --predictions outputs/plan_a/politihop_error_analysis/predictions_PolitiHop_symmetric.jsonl \
  --output-dir outputs/plan_a/politihop_error_analysis
