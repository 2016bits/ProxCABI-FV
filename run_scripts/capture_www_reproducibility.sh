#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m proxcabi.causal reproducibility-manifest \
  --project-dir "${PROJECT_DIR}" \
  --output-dir "${OUTPUT_DIR:-outputs/plan_a/reproducibility}"
