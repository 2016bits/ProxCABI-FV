#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python_bin="${PYTHON_BIN:-/data/yangjun/.conda/envs/proxcabi/bin/python}"
gpus="${GPUS:-0,1,2,3}"
jobs_per_gpu="${JOBS_PER_GPU:-1}"
ifs_backup="$IFS"; IFS=',' read -r -a gpu_array <<<"$gpus"; IFS="$ifs_backup"
max_parallel="${MAX_PARALLEL:-$(( ${#gpu_array[@]} * jobs_per_gpu ))}"
pair_dir="outputs/v2/paper/pairs"

[[ -x "$python_bin" ]] || { echo "Python environment not found: $python_bin" >&2; exit 2; }

# Fail before scheduling: train.device=auto would otherwise silently use CPU.
CUDA_VISIBLE_DEVICES="${gpu_array[0]}" "$python_bin" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to PyTorch; refusing to start P0 training on CPU")
x = torch.zeros(1, device="cuda")
print(f"CUDA preflight OK: {torch.cuda.get_device_name(0)}")
PY

"$python_bin" scripts/paper_v2/audit_protocol.py \
  --config configs/paper_v2/protocol.yaml --output-dir outputs/v2/paper/protocol
"$python_bin" scripts/paper_v2/generate_experiment_matrix.py \
  --config configs/paper_v2/protocol.yaml --output-dir outputs/v2/paper/task_matrix --dry-run

run_method() {
  local method="$1"
  echo "Running P0 baseline: $method"
  PYTHON_BIN="$python_bin" P0_FAMILY=baselines P0_NAME="$method" \
    MAX_PARALLEL="$max_parallel" JOBS_PER_GPU="$jobs_per_gpu" GPUS="$gpus" \
    bash run_scripts/run_p0_parallel.sh
}

# CLEVER requires the matching ERM fusion checkpoint, so finish/reuse ERM first.
run_method erm

# Run every other dependency-free method in one shared GPU queue.  Keeping
# CrossAug out of this temporary matrix prevents its missing pair pool from
# turning an otherwise successful baseline run into a failed scheduler run.
dependency_jobs="$(mktemp)"
trap 'rm -f "$dependency_jobs"' EXIT
awk -F '\t' 'NR == 1 || ($2 == "baselines" && \
  ($3 == "vitaminc_trained" || $3 == "bias_aware_regularization" || \
   $3 == "group_dro" || $3 == "clever"))' \
  outputs/v2/paper/task_matrix/jobs.tsv > "$dependency_jobs"
PYTHON_BIN="$python_bin" MAX_PARALLEL="$max_parallel" JOBS_PER_GPU="$jobs_per_gpu" \
  GPUS="$gpus" bash run_scripts/run_p0_parallel.sh "$dependency_jobs"

train_pool="$pair_dir/pairs_train_verified.jsonl"
invariant_pool="$pair_dir/pairs_eval_invariant.jsonl"
semantic_pool="$pair_dir/pairs_eval_semantic.jsonl"
if [[ -s "$train_pool" && -s "$invariant_pool" && -s "$semantic_pool" ]]; then
  "$python_bin" scripts/paper_v2/validate_pairs.py \
    --train "$train_pool" --invariant "$invariant_pool" \
    --semantic "$semantic_pool" --strict
  GPUS="$gpus" PYTHON_BIN="$python_bin" bash run_scripts/prepare_p0_selection_pools.sh
  run_method crossaug
else
  echo "SKIP CrossAug: verified pair pools are not frozen." >&2
  echo "Complete outputs/v2/paper/annotations/{train,eval}_invariant_tasks.csv," >&2
  echo "then run: PYTHON_BIN=$python_bin bash run_scripts/freeze_p0_pair_pools.sh" >&2
fi

"$python_bin" scripts/paper_v2/build_paper_reports.py \
  --input outputs/v2/paper --output outputs/v2/paper/reports
echo "P0 baseline reports:"
echo "  outputs/v2/paper/reports/main_fever_sym.{csv,tex}"
echo "  outputs/v2/paper/reports/main_vitaminc.{csv,tex}"
