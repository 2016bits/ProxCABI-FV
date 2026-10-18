#!/usr/bin/env bash
set -euo pipefail

runs_root="${1:-outputs/v2/joint/main}"

while IFS= read -r checkpoint; do
  run_dir="${checkpoint%/best_model.pt}"
  config="$run_dir/config_resolved.yaml"
  if [[ ! -f "$config" ]]; then
    echo "Missing resolved config for $checkpoint" >&2
    exit 1
  fi
  for split in test_real test_synthetic; do
    python -m proxcabi v2-evaluate \
      --checkpoint "$checkpoint" \
      --config "$config" \
      --dataset VitaminC \
      --split "$split" \
      --output-dir "$run_dir/eval_vitaminc_${split}"
  done
done < <(find "$runs_root" -type f -name best_model.pt | sort)

python -m proxcabi v2-report \
  --runs-dir "$runs_root" \
  --output-dir outputs/v2/joint/paper_tables
