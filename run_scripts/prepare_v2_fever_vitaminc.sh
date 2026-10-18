#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/v2/fever_vitaminc_joint.yaml}"

python -m proxcabi v2-build-vitaminc-pairs \
  --input data/VitaminC/converted_data/train.jsonl \
  --output outputs/v2/frozen_data/vitaminc_official_train_pairs.jsonl \
  --split-origin train

python -m proxcabi v2-build-vitaminc-pairs \
  --input data/VitaminC/converted_data/dev.jsonl \
  --output outputs/v2/frozen_data/vitaminc_official_dev_pairs.jsonl \
  --split-origin dev

python -m proxcabi v2-fit-proxies \
  --config "$config" \
  --output-dir outputs/v2/proxy_artifacts/fever_vitaminc_joint
