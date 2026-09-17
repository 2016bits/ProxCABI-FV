#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

OUT="${OUT:-outputs/fever_deberta_v3_base_hard_nli_bias_group_dro}"

python -m proxcabi train \
  --data-dir data \
  --dataset FEVER \
  --output-dir "${OUT}" \
  --backbone microsoft/deberta-v3-base \
  --train-split train \
  --eval-split dev \
  --counterfactual-split cf_train_hard_nli \
  --max-counterfactual-samples "${MAX_COUNTERFACTUAL_SAMPLES:-40000}" \
  --epochs "${EPOCHS:-2}" \
  --lr "${LR:-8e-6}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-2}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-16}" \
  --contrastive-weight "${CONTRASTIVE_WEIGHT:-0.05}" \
  --contrastive-margin "${CONTRASTIVE_MARGIN:-1.0}" \
  --bias-reweighting \
  --bias-reweight-factor "${BIAS_REWEIGHT_FACTOR:-1.5}" \
  --group-dro-weight "${GROUP_DRO_WEIGHT:-0.2}" \
  --bias-features "${BIAS_FEATURES:-claim_negation,claim_label_lmi,claim_evidence_lexical_overlap}"

python -m proxcabi evaluate \
  --checkpoint-dir "${OUT}" \
  --data-dir data \
  --dataset FEVER \
  --split dev \
  --batch-size "${EVAL_BATCH_SIZE:-16}" \
  --calibration-split dev \
  --calibration-grid-steps 10

python -m proxcabi evaluate \
  --checkpoint-dir "${OUT}" \
  --data-dir data \
  --dataset FEVER \
  --split symmetric \
  --batch-size "${EVAL_BATCH_SIZE:-16}" \
  --calibration-split dev \
  --calibration-grid-steps 10

CHECKPOINT_DIR="${OUT}" bash run_scripts/report_fever_bias_groups.sh
