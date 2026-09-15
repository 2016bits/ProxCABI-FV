#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

python -m proxcabi train \
  --data-dir data \
  --dataset FEVER \
  --output-dir outputs/fever_roberta_hard_cf_prox_contrastive \
  --backbone roberta-base \
  --train-split train \
  --eval-split dev \
  --counterfactual-split cf_train_hard \
  --max-counterfactual-samples 40000 \
  --epochs 3 \
  --lr 1e-5 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --contrastive-weight 0.05 \
  --contrastive-margin 1.0

python -m proxcabi evaluate \
  --checkpoint-dir outputs/fever_roberta_hard_cf_prox_contrastive \
  --data-dir data \
  --dataset FEVER \
  --split dev \
  --batch-size 32 \
  --calibration-split dev \
  --calibration-grid-steps 10

python -m proxcabi evaluate \
  --checkpoint-dir outputs/fever_roberta_hard_cf_prox_contrastive \
  --data-dir data \
  --dataset FEVER \
  --split symmetric \
  --batch-size 32 \
  --calibration-split dev \
  --calibration-grid-steps 10
