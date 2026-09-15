#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

python -m proxcabi train \
  --data-dir data \
  --dataset FEVER \
  --output-dir outputs/fever_deberta_v3_base_hard_nli_prox_contrastive \
  --backbone microsoft/deberta-v3-base \
  --train-split train \
  --eval-split dev \
  --counterfactual-split cf_train_hard_nli \
  --max-counterfactual-samples 40000 \
  --epochs 2 \
  --lr 8e-6 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --eval-batch-size 16 \
  --contrastive-weight 0.05 \
  --contrastive-margin 1.0

python -m proxcabi evaluate \
  --checkpoint-dir outputs/fever_deberta_v3_base_hard_nli_prox_contrastive \
  --data-dir data \
  --dataset FEVER \
  --split dev \
  --batch-size 16 \
  --calibration-split dev \
  --calibration-grid-steps 10

python -m proxcabi evaluate \
  --checkpoint-dir outputs/fever_deberta_v3_base_hard_nli_prox_contrastive \
  --data-dir data \
  --dataset FEVER \
  --split symmetric \
  --batch-size 16 \
  --calibration-split dev \
  --calibration-grid-steps 10
