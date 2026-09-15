#!/usr/bin/env bash
set -euo pipefail

python -m proxcabi build-fever-counterfactuals \
  --data-dir data \
  --source-split train \
  --output-split cf_train_hard \
  --max-groups 20000 \
  --seed 23 \
  --mode hard

python -m proxcabi build-fever-counterfactuals \
  --data-dir data \
  --source-split dev \
  --output-split cf_dev_hard \
  --max-groups 2000 \
  --seed 29 \
  --mode hard
