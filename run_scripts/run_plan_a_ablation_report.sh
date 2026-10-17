#!/usr/bin/env bash
set -euo pipefail

# Build the paper-ready Plan A ablation tables from completed experiment outputs.
# Set RETRAIN_FEVER_ABLATIONS=1 only when all FEVER ablation checkpoints should be
# retrained; this is GPU-intensive and overwrites their result directories.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLAN_A_DIR="${PLAN_A_DIR:-outputs/plan_a}"
OUTPUT_DIR="${OUTPUT_DIR:-${PLAN_A_DIR}/ablation_report}"
MULTIDATASET_METRICS="${MULTIDATASET_METRICS:-${PLAN_A_DIR}/multidataset_paper/report/multidataset_metrics.json}"
RUN_TESTS="${RUN_TESTS:-1}"
RETRAIN_FEVER_ABLATIONS="${RETRAIN_FEVER_ABLATIONS:-0}"

cd "${PROJECT_DIR}"

run_step() {
  local title="$1"
  shift
  echo
  echo "[Plan A ablation] ${title}"
  "$@"
}

if [[ "${RUN_TESTS}" == "1" ]]; then
  run_step "Regression tests" "${PYTHON_BIN}" -m pytest tests/ -q
fi

if [[ "${RETRAIN_FEVER_ABLATIONS}" == "1" ]]; then
  run_step "Retrain all FEVER ablations" bash run_scripts/run_fever_deberta_ablation_suite.sh
fi

run_step "Validate required completed outputs" "${PYTHON_BIN}" - \
  "${PLAN_A_DIR}" "${MULTIDATASET_METRICS}" <<'PY'
import sys
from pathlib import Path

plan_a = Path(sys.argv[1])
multidataset = Path(sys.argv[2])
required = [
    plan_a / "discrete_smoke/metrics.json",
    plan_a / "discrete_smoke/identification_metrics.json",
    plan_a / "neural_smoke/metrics.json",
    multidataset,
]
run_names = [
    "fever_deberta_v3_base_hard_nli_prox_contrastive",
    "fever_deberta_ablation_no_z",
    "fever_deberta_ablation_no_w",
    "fever_deberta_ablation_no_bridge_prox",
    "fever_deberta_ablation_no_cf",
    "fever_deberta_ablation_random_cf",
    "fever_deberta_ablation_easy_cf",
]
for name in run_names:
    root = Path("outputs") / name
    required.extend([
        root / "train_summary.json",
        root / "eval_FEVER_dev.json",
        root / "eval_FEVER_symmetric.json",
    ])
missing = [str(path) for path in required if not path.is_file()]
if missing:
    print("Missing inputs required for the ablation report:", file=sys.stderr)
    print("\n".join(f"  - {path}" for path in missing), file=sys.stderr)
    raise SystemExit(2)
print(f"Validated {len(required)} input files.")
PY

run_step "Build paper-ready ablation tables" \
  "${PYTHON_BIN}" -m proxcabi.causal ablation-report \
  --outputs-dir outputs \
  --plan-a-dir "${PLAN_A_DIR}" \
  --multidataset-metrics "${MULTIDATASET_METRICS}" \
  --output-dir "${OUTPUT_DIR}"

echo
echo "[Plan A ablation] Completed successfully."
echo "Report: ${OUTPUT_DIR}/ablation_results.md"
echo "FEVER table: ${OUTPUT_DIR}/fever_structural_ablation.csv"
echo "Synthetic table: ${OUTPUT_DIR}/synthetic_ablation.csv"
echo "Cross-dataset heads: ${OUTPUT_DIR}/cross_dataset_head_comparison.csv"
