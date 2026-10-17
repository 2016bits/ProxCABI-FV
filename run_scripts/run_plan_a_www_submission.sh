#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-7}"
PIDS=()

cd "${PROJECT_DIR}"
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
wait_all() {
  local failed=0
  local pid
  for pid in "${PIDS[@]}"; do
    wait "${pid}" || failed=1
  done
  PIDS=()
  return "${failed}"
}
trap cleanup INT TERM

GPU="${GPU_A}" SEEDS="13 21 42" VARIANTS="full no_cf" PYTHON_BIN="${PYTHON_BIN}" \
  bash run_scripts/run_fever_multiseed_stage1.sh &
PIDS+=("$!")
GPU="${GPU_B}" SEEDS="13 21 42" VARIANTS="no_z no_w" PYTHON_BIN="${PYTHON_BIN}" \
  bash run_scripts/run_fever_multiseed_stage1.sh &
PIDS+=("$!")
wait_all

PYTHON_BIN="${PYTHON_BIN}" bash run_scripts/report_fever_multiseed.sh

GPU="${GPU_A}" SEEDS="13 21 42" VARIANTS="erm cf_only contrastive_only" \
  REPORT_AT_END=0 PYTHON_BIN="${PYTHON_BIN}" \
  bash run_scripts/run_fever_multiseed_stage2_baselines.sh &
PIDS+=("$!")
GPU="${GPU_B}" SEEDS="13 21 42" VARIANTS="reweight group_dro" \
  REPORT_AT_END=0 PYTHON_BIN="${PYTHON_BIN}" \
  bash run_scripts/run_fever_multiseed_stage2_baselines.sh &
PIDS+=("$!")
wait_all

VARIANTS="full,erm,cf_only,contrastive_only,reweight,group_dro" \
SIGNIFICANCE_VARIANTS="erm,cf_only,contrastive_only,reweight,group_dro" \
OUTPUT_DIR="outputs/plan_a/baseline_multiseed_report" PYTHON_BIN="${PYTHON_BIN}" \
  bash run_scripts/report_fever_multiseed.sh
PYTHON_BIN="${PYTHON_BIN}" bash run_scripts/report_fever_efficiency.sh
PYTHON_BIN="${PYTHON_BIN}" bash run_scripts/capture_www_reproducibility.sh
PYTHON_BIN="${PYTHON_BIN}" bash run_scripts/report_www_experiments.sh

echo "WWW experiment pipeline complete."
