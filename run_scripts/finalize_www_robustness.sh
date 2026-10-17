#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"

cd "${PROJECT_DIR}"
test -f outputs/plan_a/baseline_multiseed_report/paired_significance.json
PYTHON_BIN="${PYTHON_BIN}" bash run_scripts/report_fever_robustness.sh
PYTHON_BIN="${PYTHON_BIN}" bash run_scripts/capture_www_reproducibility.sh
PYTHON_BIN="${PYTHON_BIN}" bash run_scripts/report_www_experiments.sh
