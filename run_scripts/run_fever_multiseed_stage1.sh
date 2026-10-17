#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS="${SEEDS:-13 21 42}"
VARIANTS="${VARIANTS:-full no_cf no_z no_w}"
GPU="${GPU:-0}"

cd "${PROJECT_DIR}"
for seed in ${SEEDS}; do
  for variant in ${VARIANTS}; do
    run_dir="outputs/fever_multiseed/${variant}/seed_${seed}"
    if [[ -f "${run_dir}/best_model.pt" && -f "${run_dir}/eval_FEVER_dev.json" && \
          -f "${run_dir}/eval_FEVER_symmetric.json" ]]; then
      echo "[Multi-seed stage 1] skip completed variant=${variant} seed=${seed}"
      continue
    fi
    echo
    echo "[Multi-seed stage 1] variant=${variant} seed=${seed} gpu=${GPU}"
    GPU="${GPU}" bash run_scripts/train_fever_deberta_multiseed.sh "${variant}" "${seed}"
  done
done

GPU="${GPU}" PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}" \
  bash run_scripts/backfill_fever_multiseed_predictions.sh
