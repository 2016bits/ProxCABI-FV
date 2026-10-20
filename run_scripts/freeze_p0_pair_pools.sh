#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
annotation_dir="${ANNOTATION_DIR:-outputs/v2/paper/annotations}"
train_csv="${TRAIN_ANNOTATIONS_CSV:-$annotation_dir/train_invariant_tasks.csv}"
eval_csv="${EVAL_ANNOTATIONS_CSV:-$annotation_dir/eval_invariant_tasks.csv}"
train_jsonl="$annotation_dir/train_invariant_decisions.jsonl"
eval_jsonl="$annotation_dir/eval_invariant_decisions.jsonl"

if [[ ! -f "$train_csv" || ! -f "$eval_csv" ]]; then
  "$python_bin" scripts/paper_v2/build_pair_candidates.py --output-dir "$annotation_dir"
  echo "Annotation packets created in $annotation_dir. Complete both CSV files and rerun this command." >&2
  exit 2
fi

"$python_bin" scripts/paper_v2/annotations_csv_to_jsonl.py --input "$train_csv" --output "$train_jsonl"
"$python_bin" scripts/paper_v2/annotations_csv_to_jsonl.py --input "$eval_csv" --output "$eval_jsonl"
"$python_bin" scripts/paper_v2/assemble_pair_pools.py \
  --train-annotations "$train_jsonl" --eval-annotations "$eval_jsonl" \
  --candidate-dir "$annotation_dir" --output-dir outputs/v2/paper/pairs
"$python_bin" scripts/paper_v2/validate_pairs.py \
  --train outputs/v2/paper/pairs/pairs_train_verified.jsonl \
  --invariant outputs/v2/paper/pairs/pairs_eval_invariant.jsonl \
  --semantic outputs/v2/paper/pairs/pairs_eval_semantic.jsonl --strict

