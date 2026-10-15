#!/usr/bin/env bash
set -euo pipefail

# Reproducible paper-reporting pipeline for the current Plan A scope.
# It never trains or selects a model on symmetric-FEVER.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/fever_deberta_v3_base_hard_nli_prox_contrastive}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/plan_a/paper_experiments}"
PAIR_SPLIT="${PAIR_SPLIT:-cf_dev_hard_nli}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_SYNTHETIC="${RUN_SYNTHETIC:-1}"
REPORT_ONLY="${REPORT_ONLY:-0}"
EVALUATION_TORCH_VERSION="${EVALUATION_TORCH_VERSION:-}"
EVALUATION_TRANSFORMERS_VERSION="${EVALUATION_TRANSFORMERS_VERSION:-}"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

run_step() {
  local title="$1"
  shift
  echo
  echo "[Paper pipeline] ${title}"
  "$@"
}

if [[ "${REPORT_ONLY}" != "1" && "${RUN_TESTS}" == "1" ]]; then
  run_step "Regression tests" "${PYTHON_BIN}" -m pytest tests/ -q
fi

if [[ "${REPORT_ONLY}" != "1" && "${RUN_SYNTHETIC}" == "1" ]]; then
  run_step "Synthetic identification, stress, neural bridge, and gates" \
    env RUN_TESTS=0 PYTHON_BIN="${PYTHON_BIN}" bash run_scripts/run_plan_a_all.sh
fi

if [[ "${REPORT_ONLY}" != "1" ]]; then
run_step "Frozen-verifier counterfactual paired evaluation" \
  env PYTHON_BIN="${PYTHON_BIN}" CHECKPOINT_DIR="${CHECKPOINT_DIR}" DATA_DIR="${DATA_DIR}" \
  DATASET=FEVER PAIR_SPLIT="${PAIR_SPLIT}" \
  OUTPUT_DIR="${OUTPUT_DIR}/real_exploratory" BATCH_SIZE="${BATCH_SIZE}" \
  MAX_LENGTH="${MAX_LENGTH}" BOOTSTRAP_REPS=1000 RUN_STANDARD_EVAL=0 \
  bash run_scripts/run_plan_a_real_exploratory.sh

run_step "FEVER dev evaluation (no self-calibrated score)" \
  "${PYTHON_BIN}" -m proxcabi evaluate \
  --checkpoint-dir "${CHECKPOINT_DIR}" --output-dir "${OUTPUT_DIR}/evaluation" \
  --data-dir "${DATA_DIR}" --dataset FEVER --split dev \
  --batch-size "${BATCH_SIZE}" --max-length "${MAX_LENGTH}"

run_step "symmetric-FEVER evaluation calibrated only on FEVER dev" \
  "${PYTHON_BIN}" -m proxcabi evaluate \
  --checkpoint-dir "${CHECKPOINT_DIR}" --output-dir "${OUTPUT_DIR}/evaluation" \
  --data-dir "${DATA_DIR}" --dataset FEVER --split symmetric \
  --batch-size "${BATCH_SIZE}" --max-length "${MAX_LENGTH}" \
  --calibration-split dev --fever-symmetric-group-decode

run_step "Claim-bias group report on symmetric-FEVER" \
  "${PYTHON_BIN}" -m proxcabi bias-report \
  --checkpoint-dir "${CHECKPOINT_DIR}" --data-dir "${DATA_DIR}" \
  --dataset FEVER --split symmetric --train-split train --calibration-split dev \
  --output-dir "${OUTPUT_DIR}/bias_report" --batch-size "${BATCH_SIZE}" \
  --max-length "${MAX_LENGTH}" --min-group-size 20
fi

run_step "Build paper-ready Markdown, CSV, and JSON tables" \
  "${PYTHON_BIN}" -m proxcabi.causal paper-report \
  --output-dir "${OUTPUT_DIR}/report" \
  --paired-dir "${OUTPUT_DIR}/real_exploratory/paired_report" \
  --eval-dir "${OUTPUT_DIR}/evaluation" \
  --bias-report "${OUTPUT_DIR}/bias_report/FEVER_symmetric_bias_report.json"

run_step "Write reproducibility manifest" "${PYTHON_BIN}" - \
  "${OUTPUT_DIR}" "${CHECKPOINT_DIR}" "${REPORT_ONLY}" \
  "${EVALUATION_TORCH_VERSION}" "${EVALUATION_TRANSFORMERS_VERSION}" <<'PY'
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy
import torch
import transformers

from proxcabi.causal.io import file_sha256
from proxcabi.data import split_path

output_dir, checkpoint_dir = Path(sys.argv[1]), Path(sys.argv[2])
report_only = sys.argv[3] == "1"
evaluation_torch_override, evaluation_transformers_override = sys.argv[4], sys.argv[5]
try:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
except Exception:
    commit = None
manifest = {
    "git_commit": commit,
    "report_environment": {
        "python": platform.python_version(), "numpy": numpy.__version__,
        "torch": torch.__version__, "transformers": transformers.__version__,
    },
    "checkpoint": str(checkpoint_dir),
    "checkpoint_sha256": file_sha256(checkpoint_dir / "best_model.pt"),
    "model_selection_used_symmetric_fever": False,
    "symmetric_calibration_split": "FEVER dev",
    "paired_estimate_kind": "real_exploratory",
    "real_strict_ate_reported": False,
}
response_manifest_path = output_dir / "real_exploratory" / "responses" / "manifest.json"
response_manifest = json.loads(response_manifest_path.read_text(encoding="utf-8"))
previous_manifest_path = output_dir / "run_manifest.json"
previous_manifest = (json.loads(previous_manifest_path.read_text(encoding="utf-8"))
                     if previous_manifest_path.exists() else {})
previous_evaluation = previous_manifest.get("evaluation_environment", {})
manifest["evaluation_environment"] = {
    "python": response_manifest.get("python"),
    "numpy": response_manifest.get("numpy"),
    "torch": evaluation_torch_override or previous_evaluation.get("torch") or (
        None if report_only else torch.__version__
    ),
    "transformers": evaluation_transformers_override or (
        previous_evaluation.get("transformers") or
        (None if report_only else transformers.__version__)
    ),
}
status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
manifest["git_dirty"] = bool(status.strip())
source_digest = hashlib.sha256()
source_roots = [Path("proxcabi"), Path("run_scripts"), Path("configs/plan_a"), Path("tests")]
source_files = sorted(
    path for root in source_roots if root.exists() for path in root.rglob("*")
    if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc"}
)
for path in source_files:
    source_digest.update(str(path).encode())
    source_digest.update(b"\0")
    source_digest.update(path.read_bytes())
    source_digest.update(b"\0")
manifest["source_tree_sha256"] = source_digest.hexdigest()
manifest["data_sha256"] = {
    split: file_sha256(split_path(Path("data"), "FEVER", split))
    for split in ("train", "dev", "symmetric", "cf_dev_hard_nli")
}
(output_dir / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo
echo "[Paper pipeline] Completed successfully."
echo "Main table: ${OUTPUT_DIR}/report/paper_results.md"
echo "Machine-readable metrics: ${OUTPUT_DIR}/report/paper_metrics.json"
echo "Reproducibility manifest: ${OUTPUT_DIR}/run_manifest.json"
