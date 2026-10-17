#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-${VARIANT:-}}"
SEED="${2:-${SEED:-}}"
if [[ -z "${VARIANT}" || -z "${SEED}" ]]; then
  echo "Usage: $0 {full|no_cf|no_z|no_w|no_bridge_prox|erm|cf_only|contrastive_only|reweight|group_dro} SEED" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"
GPU="${GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/fever_multiseed}"
OUTPUT_DIR="${OUTPUT_ROOT}/${VARIANT}/seed_${SEED}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
RUN_TRAIN=1
if [[ -e "${OUTPUT_DIR}/best_model.pt" && "${FORCE:-0}" != "1" ]]; then
  if [[ -f "${OUTPUT_DIR}/train_summary.json" ]]; then
    echo "Reusing completed training run and resuming evaluation: ${OUTPUT_DIR}"
    RUN_TRAIN=0
  else
    echo "Refusing to treat a partial checkpoint as a completed run: ${OUTPUT_DIR}" >&2
    echo "Archive the partial directory or set FORCE=1 for an intentional retrain." >&2
    exit 2
  fi
fi
mkdir -p "${OUTPUT_DIR}"

CF_ARGS=(--counterfactual-split cf_train_hard_nli --max-counterfactual-samples 40000)
EXTRA_ARGS=()
case "${VARIANT}" in
  full)
    ;;
  no_cf)
    CF_ARGS=()
    ;;
  no_z)
    EXTRA_ARGS=(--disable-z-proxy)
    ;;
  no_w)
    EXTRA_ARGS=(--disable-w-proxy --proxy-weight 0.0 --bridge-weight 0.0 \
      --contrastive-heads fact,g --selection-head g)
    ;;
  no_bridge_prox)
    EXTRA_ARGS=(--proxy-weight 0.0 --bridge-weight 0.0 \
      --contrastive-heads fact,g --selection-head g)
    ;;
  erm)
    CF_ARGS=()
    EXTRA_ARGS=(--proxy-weight 0.0 --bridge-weight 0.0 --g-weight 0.0 \
      --contrastive-weight 0.0 --contrastive-heads fact --selection-head fact)
    ;;
  cf_only)
    EXTRA_ARGS=(--proxy-weight 0.0 --bridge-weight 0.0 --g-weight 0.0 \
      --contrastive-weight 0.0 --contrastive-heads fact --selection-head fact)
    ;;
  contrastive_only)
    EXTRA_ARGS=(--proxy-weight 0.0 --bridge-weight 0.0 --g-weight 0.0 \
      --contrastive-heads fact --selection-head fact)
    ;;
  reweight)
    EXTRA_ARGS=(--proxy-weight 0.0 --bridge-weight 0.0 --g-weight 0.0 \
      --contrastive-weight 0.0 --contrastive-heads fact --selection-head fact \
      --bias-reweighting --bias-reweight-factor 1.5)
    ;;
  group_dro)
    EXTRA_ARGS=(--proxy-weight 0.0 --bridge-weight 0.0 --g-weight 0.0 \
      --contrastive-weight 0.0 --contrastive-heads fact --selection-head fact \
      --bias-reweighting --bias-reweight-factor 1.5 --group-dro-weight 0.2 \
      --group-dro-heads fact)
    ;;
  *)
    echo "Unknown variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in PYTHON_BIN; refusing an accidental CPU training run")
print(f"Using {torch.cuda.get_device_name(0)} with torch={torch.__version__}, cuda={torch.version.cuda}")
PY

if [[ "${RUN_TRAIN}" == "1" ]]; then
  "${PYTHON_BIN}" -m proxcabi train \
    --data-dir data --dataset FEVER --output-dir "${OUTPUT_DIR}" \
    --backbone microsoft/deberta-v3-base --train-split train --eval-split dev \
    "${CF_ARGS[@]}" --epochs 2 --lr 8e-6 --batch-size 8 \
    --gradient-accumulation-steps 2 --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --contrastive-weight 0.05 --contrastive-margin 1.0 --seed "${SEED}" \
    "${EXTRA_ARGS[@]}"
fi

for split in dev symmetric; do
  if [[ -f "${OUTPUT_DIR}/eval_FEVER_${split}.json" && \
        -s "${OUTPUT_DIR}/predictions_FEVER_${split}.jsonl" ]]; then
    echo "Skipping completed evaluation: ${OUTPUT_DIR}, split=${split}"
    continue
  fi
  "${PYTHON_BIN}" -m proxcabi evaluate \
    --checkpoint-dir "${OUTPUT_DIR}" --data-dir data --dataset FEVER --split "${split}" \
    --batch-size "${EVAL_BATCH_SIZE}" --calibration-split dev --calibration-grid-steps 10 \
    --save-predictions
done

echo "Completed ${VARIANT}, seed=${SEED}: ${OUTPUT_DIR}"
