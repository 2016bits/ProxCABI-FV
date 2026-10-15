#!/usr/bin/env bash
set -euo pipefail

# Frozen-verifier paired analysis. This deliberately reports exploratory
# sensitivity metrics rather than a proximal observational ATE.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/fever_deberta_v3_base_hard_nli_prox_contrastive}"
DATA_DIR="${DATA_DIR:-data}"
DATASET="${DATASET:-FEVER}"
PAIR_SPLIT="${PAIR_SPLIT:-cf_dev_hard_nli}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/plan_a/real_exploratory_fever}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_GROUPS="${MAX_GROUPS:-}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-1000}"
RUN_STANDARD_EVAL="${RUN_STANDARD_EVAL:-0}"

cd "${PROJECT_DIR}"

collect=(
  "${PYTHON_BIN}" -m proxcabi.causal collect-responses
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --data-dir "${DATA_DIR}"
  --dataset "${DATASET}"
  --split "${PAIR_SPLIT}"
  --output-dir "${OUTPUT_DIR}/responses"
  --head fact
  --outcome nll
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
)
if [[ -n "${MAX_GROUPS}" ]]; then
  collect+=(--max-groups "${MAX_GROUPS}")
fi

echo "[Plan A exploratory] Collect frozen-verifier responses"
"${collect[@]}"

echo "[Plan A exploratory] Evaluate paired effect, CCR, and SSR"
"${PYTHON_BIN}" -m proxcabi.causal evaluate-pairs \
  --input "${OUTPUT_DIR}/responses/responses.csv" \
  --output-dir "${OUTPUT_DIR}/paired_report" \
  --bootstrap-reps "${BOOTSTRAP_REPS}"

if [[ "${RUN_STANDARD_EVAL}" == "1" ]]; then
  echo "[Plan A exploratory] Evaluate FEVER dev"
  "${PYTHON_BIN}" -m proxcabi evaluate \
    --checkpoint-dir "${CHECKPOINT_DIR}" --data-dir "${DATA_DIR}" \
    --dataset FEVER --split dev --calibration-split dev

  echo "[Plan A exploratory] Evaluate symmetric-FEVER"
  "${PYTHON_BIN}" -m proxcabi evaluate \
    --checkpoint-dir "${CHECKPOINT_DIR}" --data-dir "${DATA_DIR}" \
    --dataset FEVER --split symmetric --calibration-split dev \
    --fever-symmetric-group-decode
fi

echo "[Plan A exploratory] Completed"
echo "Report: ${OUTPUT_DIR}/paired_report/summary.md"
