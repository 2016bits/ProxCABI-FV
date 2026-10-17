#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/v2/fever_pilot.yaml}"
run_dir="${2:-outputs/v2/pilot/M1_seed13}"

test -f outputs/v2/frozen_data/data_audit.json
test -f outputs/v2/proxy_artifacts/fever/proxy_builder_v2.json

python -m pytest -q tests/v2

python -m proxcabi v2-train \
  --config "$config" \
  --variant M1 \
  --seed 13 \
  --output-dir "$run_dir"

python -m proxcabi v2-evaluate \
  --checkpoint "$run_dir/best_model.pt" \
  --config "$config" \
  --split dev_report \
  --output-dir "$run_dir/eval_id" \
  --head standardized \
  --max-samples 96
