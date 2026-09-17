#!/usr/bin/env bash
set -euo pipefail

python -m proxcabi build-fever-counterfactuals \
  --data-dir data \
  --source-split train \
  --output-split cf_train_easy \
  --max-groups 3750 \
  --seed 41 \
  --mode basic

python -m proxcabi build-fever-counterfactuals \
  --data-dir data \
  --source-split dev \
  --output-split cf_dev_easy \
  --max-groups 182 \
  --seed 43 \
  --mode basic

python -m proxcabi build-fever-random-counterfactuals \
  --data-dir data \
  --source-split train \
  --output-split cf_train_random \
  --max-groups 7500 \
  --seed 47

python -m proxcabi build-fever-random-counterfactuals \
  --data-dir data \
  --source-split dev \
  --output-split cf_dev_random \
  --max-groups 364 \
  --seed 53
