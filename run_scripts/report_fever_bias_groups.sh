#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

python -m proxcabi bias-report \
  --checkpoint-dir "${CHECKPOINT_DIR:-outputs/fever_deberta_v3_base_hard_nli_prox_contrastive}" \
  --data-dir "${DATA_DIR:-data}" \
  --dataset "${DATASET:-FEVER}" \
  --split "${SPLIT:-symmetric}" \
  --output-dir "${OUTPUT_DIR:-outputs/bias_report}" \
  --train-split "${TRAIN_SPLIT:-train}" \
  --calibration-split "${CALIBRATION_SPLIT:-dev}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --max-length "${MAX_LENGTH:-256}" \
  --lmi-top-k-per-label "${LMI_TOP_K_PER_LABEL:-32}" \
  --lmi-min-count "${LMI_MIN_COUNT:-5}" \
  --min-group-size "${MIN_GROUP_SIZE:-20}"
