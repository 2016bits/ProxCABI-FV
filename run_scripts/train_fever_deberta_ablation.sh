#!/usr/bin/env bash
set -euo pipefail

ABLATION="${1:-${ABLATION:-}}"
if [[ -z "${ABLATION}" ]]; then
  echo "Usage: $0 {no_z|no_w|no_bridge_prox|no_cf|random_cf|easy_cf}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

OUTPUT_DIR="outputs/fever_deberta_ablation_${ABLATION}"
CF_ARGS=(--counterfactual-split cf_train_hard_nli --max-counterfactual-samples 40000)
EXTRA_ARGS=()

case "${ABLATION}" in
  no_z)
    EXTRA_ARGS=(--disable-z-proxy)
    ;;
  no_w)
    EXTRA_ARGS=(
      --disable-w-proxy
      --proxy-weight 0.0
      --bridge-weight 0.0
      --contrastive-heads fact,g
      --selection-head g
    )
    ;;
  no_bridge_prox)
    EXTRA_ARGS=(
      --proxy-weight 0.0
      --bridge-weight 0.0
      --contrastive-heads fact,g
      --selection-head g
    )
    ;;
  no_cf)
    CF_ARGS=()
    ;;
  random_cf)
    CF_ARGS=(--counterfactual-split cf_train_random --max-counterfactual-samples 15000)
    ;;
  easy_cf)
    CF_ARGS=(--counterfactual-split cf_train_easy --max-counterfactual-samples 15000)
    ;;
  *)
    echo "Unknown ablation: ${ABLATION}" >&2
    exit 2
    ;;
esac

python -m proxcabi train \
  --data-dir data \
  --dataset FEVER \
  --output-dir "${OUTPUT_DIR}" \
  --backbone microsoft/deberta-v3-base \
  --train-split train \
  --eval-split dev \
  "${CF_ARGS[@]}" \
  --epochs 2 \
  --lr 8e-6 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --eval-batch-size 16 \
  --contrastive-weight 0.05 \
  --contrastive-margin 1.0 \
  "${EXTRA_ARGS[@]}"

python -m proxcabi evaluate \
  --checkpoint-dir "${OUTPUT_DIR}" \
  --data-dir data \
  --dataset FEVER \
  --split dev \
  --batch-size 16 \
  --calibration-split dev \
  --calibration-grid-steps 10

python -m proxcabi evaluate \
  --checkpoint-dir "${OUTPUT_DIR}" \
  --data-dir data \
  --dataset FEVER \
  --split symmetric \
  --batch-size 16 \
  --calibration-split dev \
  --calibration-grid-steps 10
