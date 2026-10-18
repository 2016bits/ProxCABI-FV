#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/v2/fever_vitaminc_joint.yaml}"
output_root="${2:-outputs/v2/joint/main}"
seeds="${V2_SEEDS:-13 21 42}"
variants="${V2_VARIANTS:-M1 M3}"

if [[ ! -f outputs/v2/frozen_data/vitaminc_official_train_pairs.jsonl ]] || \
   [[ ! -f outputs/v2/proxy_artifacts/fever_vitaminc_joint/proxy_builder_v2.json ]]; then
  run_scripts/prepare_v2_fever_vitaminc.sh "$config"
fi

python -m pytest -q tests/v2

for seed in $seeds; do
  for variant in $variants; do
    run_dir="$output_root/${variant}/seed_${seed}"
    python -m proxcabi v2-train \
      --config "$config" \
      --variant "$variant" \
      --seed "$seed" \
      --output-dir "$run_dir"

    python -m proxcabi v2-evaluate \
      --checkpoint "$run_dir/best_model.pt" \
      --config "$config" \
      --dataset FEVER \
      --split dev_report \
      --output-dir "$run_dir/eval_fever_id"

    python -m proxcabi v2-evaluate \
      --checkpoint "$run_dir/best_model.pt" \
      --config "$config" \
      --dataset FEVER \
      --split symmetric \
      --output-dir "$run_dir/eval_fever_symmetric"

    python -m proxcabi v2-evaluate \
      --checkpoint "$run_dir/best_model.pt" \
      --config "$config" \
      --dataset VitaminC \
      --split test \
      --output-dir "$run_dir/eval_vitaminc_test"

    python -m proxcabi v2-evaluate-pairs \
      --checkpoint "$run_dir/best_model.pt" \
      --config "$config" \
      --dataset VitaminC \
      --split dev \
      --pair-pool outputs/v2/frozen_data/vitaminc_official_dev_pairs.jsonl \
      --output-dir "$run_dir/eval_vitaminc_pairs"
  done
done

python -m proxcabi v2-report \
  --runs-dir "$output_root" \
  --output-dir outputs/v2/joint/paper_tables
