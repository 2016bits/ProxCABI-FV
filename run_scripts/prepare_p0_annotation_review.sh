#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${PYTHON_BIN:-python}"
annotation_dir="${ANNOTATION_DIR:-outputs/v2/paper/annotations}"
ui_dir="${REVIEW_UI_DIR:-$annotation_dir/review_ui}"

if [[ ! -f "$annotation_dir/train_invariant_tasks.csv" || ! -f "$annotation_dir/eval_invariant_tasks.csv" ]]; then
  "$python_bin" scripts/paper_v2/build_pair_candidates.py --output-dir "$annotation_dir"
fi

mkdir -p "$ui_dir"
"$python_bin" scripts/paper_v2/build_annotation_review_ui.py \
  --input "$annotation_dir/train_invariant_tasks.csv" \
  --output "$ui_dir/train_invariant_review.html" \
  --title "P0 Train Invariant：300 条生成器审计"
"$python_bin" scripts/paper_v2/build_annotation_review_ui.py \
  --input "$annotation_dir/eval_invariant_tasks.csv" \
  --output "$ui_dir/eval_invariant_review.html" \
  --title "P0 Eval Invariant：154 条逐条核查"

echo "Review pages:"
echo "  $ui_dir/train_invariant_review.html"
echo "  $ui_dir/eval_invariant_review.html"
