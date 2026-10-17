#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"
cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m proxcabi.causal www-report \
  --plan-a-dir outputs/plan_a \
  --todo docs/WWW_EXPERIMENT_TODO.md \
  --output-dir outputs/plan_a/www_submission
echo "Report: outputs/plan_a/www_submission/www_experiment_report.md"
