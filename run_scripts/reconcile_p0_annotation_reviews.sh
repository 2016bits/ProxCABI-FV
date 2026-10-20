#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  cat >&2 <<'EOF'
Usage:
  bash run_scripts/reconcile_p0_annotation_reviews.sh \
    REVIEWER_A_TRAIN.csv REVIEWER_A_EVAL.csv \
    REVIEWER_B_TRAIN.csv REVIEWER_B_EVAL.csv
EOF
  exit 2
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${PYTHON_BIN:-python}"
annotation_dir="${ANNOTATION_DIR:-outputs/v2/paper/annotations}"
review_dir="${REVIEW_UI_DIR:-$annotation_dir/review_ui}"
adjudication_dir="$review_dir/adjudication"

mkdir -p "$adjudication_dir"

compare_packet() {
  local packet="$1"
  local reviewer_a="$2"
  local reviewer_b="$3"

  "$python_bin" scripts/paper_v2/compare_annotation_reviews.py \
    --tasks "$annotation_dir/${packet}_invariant_tasks.csv" \
    --first "$reviewer_a" \
    --second "$reviewer_b" \
    --output "$adjudication_dir/${packet}_combined_for_adjudication.csv" \
    --report "$adjudication_dir/${packet}_agreement_report.json"

  "$python_bin" scripts/paper_v2/build_annotation_review_ui.py \
    --input "$adjudication_dir/${packet}_combined_for_adjudication.csv" \
    --output "$adjudication_dir/${packet}_adjudication.html" \
    --mode adjudication \
    --title "P0 ${packet^}：双人分歧仲裁"
}

compare_packet train "$1" "$3"
compare_packet eval "$2" "$4"

cat <<EOF
Agreement reports:
  $adjudication_dir/train_agreement_report.json
  $adjudication_dir/eval_agreement_report.json
Adjudication pages:
  $adjudication_dir/train_adjudication.html
  $adjudication_dir/eval_adjudication.html
EOF
