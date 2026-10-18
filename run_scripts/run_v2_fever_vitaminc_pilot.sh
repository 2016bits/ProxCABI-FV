#!/usr/bin/env bash
set -euo pipefail

test -f outputs/v2/frozen_data/vitaminc_official_train_pairs.jsonl
test -f outputs/v2/proxy_artifacts/fever_vitaminc_joint/proxy_builder_v2.json

python -m pytest -q tests/v2

for variant in M1 M3; do
  run_dir="outputs/v2/joint/pilot/${variant}_seed13"
  python -m proxcabi v2-train \
    --config configs/v2/fever_vitaminc_joint_pilot.yaml \
    --variant "$variant" \
    --seed 13 \
    --output-dir "$run_dir"

  python -m proxcabi v2-evaluate \
    --checkpoint "$run_dir/best_model.pt" \
    --config configs/v2/fever_vitaminc_joint_pilot.yaml \
    --dataset FEVER \
    --split dev_report \
    --output-dir "$run_dir/eval_fever_id"

  python -m proxcabi v2-evaluate \
    --checkpoint "$run_dir/best_model.pt" \
    --config configs/v2/fever_vitaminc_joint_pilot.yaml \
    --dataset VitaminC \
    --split test \
    --max-samples 256 \
    --output-dir "$run_dir/eval_vitaminc_test"

  python -m proxcabi v2-evaluate-pairs \
    --checkpoint "$run_dir/best_model.pt" \
    --config configs/v2/fever_vitaminc_joint_pilot.yaml \
    --dataset VitaminC \
    --split dev \
    --pair-pool outputs/v2/frozen_data/vitaminc_official_dev_pairs.jsonl \
    --max-samples 256 \
    --max-pair-samples 1024 \
    --output-dir "$run_dir/eval_vitaminc_pairs"
done

python -m proxcabi v2-report \
  --runs-dir outputs/v2/joint/pilot \
  --output-dir outputs/v2/joint/pilot_tables
