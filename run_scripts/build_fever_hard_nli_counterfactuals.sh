#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
TEACHER_MODEL="${TEACHER_MODEL:-MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli}"
TEACHER_MIN_CONFIDENCE="${TEACHER_MIN_CONFIDENCE:-0.65}"

python -m proxcabi build-fever-counterfactuals \
  --data-dir data \
  --source-split train \
  --output-split cf_train_hard_nli \
  --max-groups 12000 \
  --seed 31 \
  --mode hard \
  --teacher-model "${TEACHER_MODEL}" \
  --teacher-min-confidence "${TEACHER_MIN_CONFIDENCE}" \
  --teacher-batch-size 64

python -m proxcabi build-fever-counterfactuals \
  --data-dir data \
  --source-split dev \
  --output-split cf_dev_hard_nli \
  --max-groups 1500 \
  --seed 37 \
  --mode hard \
  --teacher-model "${TEACHER_MODEL}" \
  --teacher-min-confidence "${TEACHER_MIN_CONFIDENCE}" \
  --teacher-batch-size 64
