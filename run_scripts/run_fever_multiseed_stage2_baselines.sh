#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"
SEEDS="${SEEDS:-13 21 42}"
VARIANTS="${VARIANTS:-erm cf_only contrastive_only reweight group_dro}"
GPU="${GPU:-0}"

cd "${PROJECT_DIR}"
for seed in ${SEEDS}; do
  for variant in ${VARIANTS}; do
    run_dir="outputs/fever_multiseed/${variant}/seed_${seed}"
    if [[ -f "${run_dir}/best_model.pt" && -f "${run_dir}/eval_FEVER_dev.json" && \
          -f "${run_dir}/eval_FEVER_symmetric.json" && \
          -s "${run_dir}/predictions_FEVER_symmetric.jsonl" ]]; then
      echo "[Matched baseline stage] skip completed variant=${variant} seed=${seed}"
      continue
    fi
    echo
    echo "[Matched baseline stage] variant=${variant} seed=${seed} gpu=${GPU}"
    GPU="${GPU}" PYTHON_BIN="${PYTHON_BIN}" \
      bash run_scripts/train_fever_deberta_multiseed.sh "${variant}" "${seed}"
  done
done

if [[ "${REPORT_AT_END:-1}" == "1" ]]; then
  VARIANTS="full,erm,cf_only,contrastive_only,reweight,group_dro" \
  SIGNIFICANCE_VARIANTS="erm,cf_only,contrastive_only,reweight,group_dro" \
  SEEDS="13,21,42" PYTHON_BIN="${PYTHON_BIN}" \
  OUTPUT_DIR="outputs/plan_a/baseline_multiseed_report" \
    bash run_scripts/report_fever_multiseed.sh

  PYTHON_BIN="${PYTHON_BIN}" bash run_scripts/report_www_experiments.sh
fi
