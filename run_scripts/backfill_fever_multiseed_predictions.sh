#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-13 21 42}"
VARIANTS="${VARIANTS:-full no_cf no_z no_w}"
export CUDA_VISIBLE_DEVICES="${GPU}"
cd "${PROJECT_DIR}"

# Multiple per-seed GPU queues can finish at nearly the same time.  Serialize
# legacy prediction backfill so they never write the same JSONL concurrently.
mkdir -p outputs/fever_multiseed
exec 9>outputs/fever_multiseed/.prediction_backfill.lock
flock 9

legacy_dir() {
  case "$1" in
    full) echo "outputs/fever_deberta_v3_base_hard_nli_prox_contrastive" ;;
    no_cf) echo "outputs/fever_deberta_ablation_no_cf" ;;
    no_z) echo "outputs/fever_deberta_ablation_no_z" ;;
    no_w) echo "outputs/fever_deberta_ablation_no_w" ;;
  esac
}

for seed in ${SEEDS}; do
  for variant in ${VARIANTS}; do
    if [[ "${seed}" == "13" ]]; then
      candidate="outputs/fever_multiseed/${variant}/seed_${seed}"
      if [[ -f "${candidate}/best_model.pt" ]]; then
        run_dir="${candidate}"
      else
        run_dir="$(legacy_dir "${variant}")"
      fi
    else
      run_dir="outputs/fever_multiseed/${variant}/seed_${seed}"
    fi
    [[ -f "${run_dir}/best_model.pt" ]] || continue
    for split in dev symmetric; do
      prediction_file="${run_dir}/predictions_FEVER_${split}.jsonl"
      if [[ -s "${prediction_file}" ]]; then
        continue
      fi
      echo "[Prediction backfill] variant=${variant} seed=${seed} split=${split}"
      "${PYTHON_BIN}" -m proxcabi evaluate \
        --checkpoint-dir "${run_dir}" --data-dir data --dataset FEVER --split "${split}" \
        --batch-size "${EVAL_BATCH_SIZE:-16}" --calibration-split dev \
        --calibration-grid-steps 10 --save-predictions
    done
  done
done
