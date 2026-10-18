#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
base_config="${2:-configs/v2/fever_vitaminc_joint.yaml}"
output_root="${3:-outputs/v2/joint/ablations}"
generated_dir="$output_root/generated_configs"

if [[ -z "$mode" ]]; then
  echo "Usage: $0 {base_only|budget_curve|lambda_sweep} [base_config] [output_root]" >&2
  exit 2
fi
if [[ ! -f outputs/v2/frozen_data/vitaminc_official_train_pairs.jsonl ]] || \
   [[ ! -f outputs/v2/proxy_artifacts/fever_vitaminc_joint/proxy_builder_v2.json ]]; then
  echo "Run run_scripts/prepare_v2_fever_vitaminc.sh first." >&2
  exit 1
fi
mkdir -p "$generated_dir"

render_config() {
  local output_config="$1"
  local pair_budget="$2"
  local lambda_semantic="$3"
  local experiment_name="$4"
  python - "$base_config" "$output_config" "$pair_budget" "$lambda_semantic" "$experiment_name" <<'PY'
import sys
from pathlib import Path
import yaml

source, destination, budget, semantic_weight, name = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
config["project"] = f"ProxCABI-FV-FEVER-VitaminC-{name}"
config["sampling"]["mode"] = "uniform"
config["sampling"]["pair_budget"] = int(budget)
config["sampling"]["inv_sem_ratio"] = 0.0
config["loss"]["lambda_invariance"] = 0.0
config["loss"]["lambda_semantic"] = float(semantic_weight)
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

train_and_evaluate() {
  local config="$1"
  local variant="$2"
  local seed="$3"
  local run_dir="$4"

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

  for split in test test_real test_synthetic; do
    python -m proxcabi v2-evaluate \
      --checkpoint "$run_dir/best_model.pt" \
      --config "$config" \
      --dataset VitaminC \
      --split "$split" \
      --output-dir "$run_dir/eval_vitaminc_${split}"
  done

  python -m proxcabi v2-evaluate-pairs \
    --checkpoint "$run_dir/best_model.pt" \
    --config "$config" \
    --dataset VitaminC \
    --split dev \
    --pair-pool outputs/v2/frozen_data/vitaminc_official_dev_pairs.jsonl \
    --output-dir "$run_dir/eval_vitaminc_pairs"
}

case "$mode" in
  base_only)
    # Joint FEVER+VitaminC training without any official pair augmentation.
    seeds="${V2_SEEDS:-13 21 42}"
    config="$generated_dir/base_only.yaml"
    render_config "$config" 0 0.1 base-only
    for seed in $seeds; do
      train_and_evaluate \
        "$config" M1 "$seed" "$output_root/base_only/M1/seed_${seed}"
    done
    report_root="$output_root/base_only"
    ;;
  budget_curve)
    # B=5000 is supplied by the main experiment, so only complementary budgets run here.
    seeds="${V2_SEEDS:-13}"
    budgets="${V2_BUDGETS:-1000 3000 10000}"
    variants="${V2_VARIANTS:-M1 M3}"
    for budget in $budgets; do
      config="$generated_dir/budget_${budget}.yaml"
      render_config "$config" "$budget" 0.1 "budget-${budget}"
      for seed in $seeds; do
        for variant in $variants; do
          train_and_evaluate \
            "$config" "$variant" "$seed" \
            "$output_root/budget_curve/B${budget}/${variant}/seed_${seed}"
        done
      done
    done
    report_root="$output_root/budget_curve"
    ;;
  lambda_sweep)
    # lambda=0.1 is supplied by the main M3 experiment.
    seeds="${V2_SEEDS:-13}"
    lambdas="${V2_LAMBDAS:-0.05 0.2}"
    for semantic_weight in $lambdas; do
      tag="${semantic_weight//./p}"
      config="$generated_dir/lambda_${tag}.yaml"
      render_config "$config" 5000 "$semantic_weight" "lambda-${semantic_weight}"
      for seed in $seeds; do
        train_and_evaluate \
          "$config" M3 "$seed" \
          "$output_root/lambda_sweep/lambda_${tag}/M3/seed_${seed}"
      done
    done
    report_root="$output_root/lambda_sweep"
    ;;
  *)
    echo "Unknown mode: $mode (expected base_only, budget_curve, or lambda_sweep)" >&2
    exit 2
    ;;
esac

if [[ "${V2_SKIP_REPORT:-0}" != "1" ]]; then
  python -m proxcabi v2-report \
    --runs-dir "$report_root" \
    --output-dir "$report_root/paper_tables"
fi
