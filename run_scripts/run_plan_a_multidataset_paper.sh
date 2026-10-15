#!/usr/bin/env bash
set -euo pipefail

# Full, no-subsampling evaluation for HoVer, PolitiHop, and VitaminC.
# Test splits are evaluated only after checkpoints have been fixed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/plan_a/multidataset_paper}"
HOVER_CHECKPOINT="${HOVER_CHECKPOINT:-outputs/hover_roberta}"
POLITIHOP_CHECKPOINT="${POLITIHOP_CHECKPOINT:-outputs/politihop_roberta}"
VITAMINC_CHECKPOINT="${VITAMINC_CHECKPOINT:-outputs/vitaminc_roberta}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
RUN_TESTS="${RUN_TESTS:-1}"
REPORT_ONLY="${REPORT_ONLY:-0}"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}/evaluation"

run_step() {
  local title="$1"
  shift
  echo
  echo "[Multi-dataset] ${title}"
  "$@"
}

evaluate_split() {
  local checkpoint="$1"
  local dataset="$2"
  local split="$3"
  run_step "${dataset}/${split}" \
    "${PYTHON_BIN}" -m proxcabi evaluate \
    --checkpoint-dir "${checkpoint}" --output-dir "${OUTPUT_DIR}/evaluation" \
    --data-dir "${DATA_DIR}" --dataset "${dataset}" --split "${split}" \
    --batch-size "${BATCH_SIZE}" --max-length "${MAX_LENGTH}"
}

if [[ "${REPORT_ONLY}" != "1" ]]; then
  if [[ "${RUN_TESTS}" == "1" ]]; then
    run_step "Regression tests" "${PYTHON_BIN}" -m pytest tests/ -q
  fi

  evaluate_split "${HOVER_CHECKPOINT}" HOVER dev
  evaluate_split "${HOVER_CHECKPOINT}" HOVER test

  evaluate_split "${POLITIHOP_CHECKPOINT}" PolitiHop dev
  evaluate_split "${POLITIHOP_CHECKPOINT}" PolitiHop test
  evaluate_split "${POLITIHOP_CHECKPOINT}" PolitiHop symmetric

  evaluate_split "${VITAMINC_CHECKPOINT}" VitaminC test_real
  evaluate_split "${VITAMINC_CHECKPOINT}" VitaminC test_synthetic
fi

run_step "Build cross-dataset paper tables" \
  "${PYTHON_BIN}" -m proxcabi.causal multidataset-report \
  --eval-dir "${OUTPUT_DIR}/evaluation" --output-dir "${OUTPUT_DIR}/report"

run_step "Write reproducibility manifest" "${PYTHON_BIN}" - \
  "${OUTPUT_DIR}" "${DATA_DIR}" "${HOVER_CHECKPOINT}" \
  "${POLITIHOP_CHECKPOINT}" "${VITAMINC_CHECKPOINT}" <<'PY'
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

output, data_dir = Path(sys.argv[1]), Path(sys.argv[2])
checkpoints = {
    "HOVER": Path(sys.argv[3]), "PolitiHop": Path(sys.argv[4]), "VitaminC": Path(sys.argv[5]),
}
try:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
except Exception:
    commit = None
status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
source_digest = hashlib.sha256()
for root in (Path("proxcabi"), Path("run_scripts"), Path("configs/plan_a"), Path("tests")):
    if not root.exists():
        continue
    for path in sorted(path for path in root.rglob("*") if path.is_file()
                       and "__pycache__" not in path.parts and path.suffix != ".pyc"):
        source_digest.update(str(path).encode()); source_digest.update(b"\0")
        source_digest.update(path.read_bytes()); source_digest.update(b"\0")
splits = {
    "HOVER": ("dev", "test"),
    "PolitiHop": ("dev", "test", "symmetric"),
    "VitaminC": ("test_real", "test_synthetic"),
}
manifest = {
    "git_commit": commit, "git_dirty": bool(status.strip()),
    "source_tree_sha256": source_digest.hexdigest(),
    "environment": {
        "python": platform.python_version(), "numpy": numpy.__version__,
        "torch": torch.__version__, "transformers": transformers.__version__,
    },
    "checkpoint_sha256": {
        dataset: file_sha256(path / "best_model.pt") for dataset, path in checkpoints.items()
    },
    "data_sha256": {
        f"{dataset}/{split}": file_sha256(split_path(data_dir, dataset, split))
        for dataset, dataset_splits in splits.items() for split in dataset_splits
    },
    "subsampling": False,
    "test_used_for_training_or_model_selection": False,
    "calibration": "none",
}
(output / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo
echo "[Multi-dataset] Completed successfully."
echo "Paper table: ${OUTPUT_DIR}/report/multidataset_results.md"
echo "Full metrics: ${OUTPUT_DIR}/report/multidataset_results.csv"
echo "Subgroups: ${OUTPUT_DIR}/report/multidataset_subgroups.csv"
