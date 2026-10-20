#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "Usage: $0 CONFIG VARIANT SEED RUN_DIR HEAD INVARIANT_POOL SEMANTIC_POOL" >&2
  exit 2
fi
config="$1"; variant="$2"; seed="$3"; run_dir="$4"; head="$5"; invariant_pool="$6"; semantic_pool="$7"
python_bin="${PYTHON_BIN:-python}"

if [[ -f "$run_dir/run_manifest.json" ]] && \
   ! "$python_bin" -c '
import hashlib,json,sys
manifest=json.load(open(sys.argv[1]))
digest=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest()
ok=(manifest.get("status")=="complete" and manifest.get("config_sha256")==digest
    and str(manifest.get("variant"))==sys.argv[3] and int(manifest.get("seed",-1))==int(sys.argv[4]))
raise SystemExit(0 if ok else 1)
' "$run_dir/run_manifest.json" "$config" "$variant" "$seed"; then
  archive="${run_dir}.incomplete.$(date +%Y%m%d_%H%M%S)"
  echo "Preserving incomplete run as $archive" >&2
  mv "$run_dir" "$archive"
elif [[ ! -f "$run_dir/run_manifest.json" && -d "$run_dir" && -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  archive="${run_dir}.incomplete.$(date +%Y%m%d_%H%M%S)"
  echo "Preserving incomplete run as $archive" >&2
  mv "$run_dir" "$archive"
fi
if [[ ! -f "$run_dir/run_manifest.json" ]]; then
  "$python_bin" -m proxcabi v2-train --config "$config" --variant "$variant" --seed "$seed" --output-dir "$run_dir"
fi

# CLEVER uses the independently trained ERM fusion checkpoint plus this run's
# claim-only checkpoint. Alpha=1.0 is frozen before any report/test evaluation.
if [[ "$(basename "$config")" == baselines__clever__seed_*.yaml ]]; then
  fusion_config="$(dirname "$config")/baselines__erm__seed_${seed}.yaml"
  fusion_checkpoint="outputs/v2/paper/runs/baselines/erm/seed_${seed}/best_model.pt"
  claim_checkpoint="$run_dir/best_model.pt"
  [[ -f "$fusion_checkpoint" ]] || { echo "Missing matched ERM fusion checkpoint: $fusion_checkpoint" >&2; exit 1; }
  for spec in "FEVER dev_report fever_id" "FEVER symmetric fever_symmetric" "VitaminC test vitaminc_test" "VitaminC test_real vitaminc_real" "VitaminC test_synthetic vitaminc_synthetic"; do
    read -r dataset split tag <<<"$spec"
    if [[ ! -f "$run_dir/eval_${tag}/eval_${split}.json" ]]; then
      "$python_bin" -m proxcabi v2-evaluate-clever \
        --fusion-checkpoint "$fusion_checkpoint" --claim-checkpoint "$claim_checkpoint" \
        --fusion-config "$fusion_config" --claim-config "$config" --alpha 1.0 \
        --dataset "$dataset" --split "$split" --output-dir "$run_dir/eval_${tag}"
    fi
  done
  if [[ -n "$invariant_pool" && -f "$invariant_pool" && ! -f "$run_dir/eval_pairs_invariant/eval_pairs_fever_dev.json" ]]; then
    "$python_bin" -m proxcabi v2-evaluate-clever-pairs \
      --fusion-checkpoint "$fusion_checkpoint" --claim-checkpoint "$claim_checkpoint" \
      --fusion-config "$fusion_config" --claim-config "$config" --alpha 1.0 \
      --dataset FEVER --split dev --pair-pool "$invariant_pool" --output-dir "$run_dir/eval_pairs_invariant"
  fi
  if [[ -n "$semantic_pool" && -f "$semantic_pool" && ! -f "$run_dir/eval_pairs_semantic/eval_pairs_vitaminc_dev.json" ]]; then
    "$python_bin" -m proxcabi v2-evaluate-clever-pairs \
      --fusion-checkpoint "$fusion_checkpoint" --claim-checkpoint "$claim_checkpoint" \
      --fusion-config "$fusion_config" --claim-config "$config" --alpha 1.0 \
      --dataset VitaminC --split dev --pair-pool "$semantic_pool" --output-dir "$run_dir/eval_pairs_semantic"
  fi
  exit 0
fi

for spec in "FEVER dev_report fever_id" "FEVER symmetric fever_symmetric" "VitaminC test vitaminc_test" "VitaminC test_real vitaminc_real" "VitaminC test_synthetic vitaminc_synthetic"; do
  read -r dataset split tag <<<"$spec"
  if [[ ! -f "$run_dir/eval_${tag}/eval_${split}.json" ]]; then
    "$python_bin" -m proxcabi v2-evaluate --checkpoint "$run_dir/best_model.pt" --config "$config" \
      --dataset "$dataset" --split "$split" --head "$head" --output-dir "$run_dir/eval_${tag}"
  fi
done
if [[ -n "$invariant_pool" && -f "$invariant_pool" && ! -f "$run_dir/eval_pairs_invariant/eval_pairs_fever_dev.json" ]]; then
  "$python_bin" -m proxcabi v2-evaluate-pairs --checkpoint "$run_dir/best_model.pt" --config "$config" \
    --dataset FEVER --split dev --pair-pool "$invariant_pool" --head "$head" --output-dir "$run_dir/eval_pairs_invariant"
fi
if [[ -n "$semantic_pool" && -f "$semantic_pool" && ! -f "$run_dir/eval_pairs_semantic/eval_pairs_vitaminc_dev.json" ]]; then
  "$python_bin" -m proxcabi v2-evaluate-pairs --checkpoint "$run_dir/best_model.pt" --config "$config" \
    --dataset VitaminC --split dev --pair-pool "$semantic_pool" --head "$head" --output-dir "$run_dir/eval_pairs_semantic"
fi

# E4 inference-head and reference sensitivity: same Full checkpoint, no retraining.
if [[ "$(basename "$config")" == components__full__seed_*.yaml ]]; then
  for comparison in "fact train_empirical" "g train_empirical" "standardized train_uniform"; do
    read -r comparison_head reference <<<"$comparison"
    tag="head_${comparison_head}_${reference}"
    for spec in "FEVER dev_report fever_id" "FEVER symmetric fever_symmetric" "VitaminC test vitaminc_test"; do
      read -r dataset split eval_tag <<<"$spec"
      output="$run_dir/eval_${tag}_${eval_tag}"
      if [[ ! -f "$output/eval_${split}.json" ]]; then
        "$python_bin" -m proxcabi v2-evaluate --checkpoint "$run_dir/best_model.pt" --config "$config" \
          --dataset "$dataset" --split "$split" --head "$comparison_head" --reference "$reference" \
          --output-dir "$output"
      fi
    done
    if [[ -f "$invariant_pool" && ! -f "$run_dir/eval_${tag}_pairs_invariant/eval_pairs_fever_dev.json" ]]; then
      "$python_bin" -m proxcabi v2-evaluate-pairs --checkpoint "$run_dir/best_model.pt" --config "$config" \
        --dataset FEVER --split dev --pair-pool "$invariant_pool" --head "$comparison_head" \
        --reference "$reference" --output-dir "$run_dir/eval_${tag}_pairs_invariant"
    fi
    if [[ -f "$semantic_pool" && ! -f "$run_dir/eval_${tag}_pairs_semantic/eval_pairs_vitaminc_dev.json" ]]; then
      "$python_bin" -m proxcabi v2-evaluate-pairs --checkpoint "$run_dir/best_model.pt" --config "$config" \
        --dataset VitaminC --split dev --pair-pool "$semantic_pool" --head "$comparison_head" \
        --reference "$reference" --output-dir "$run_dir/eval_${tag}_pairs_semantic"
    fi
  done
fi

