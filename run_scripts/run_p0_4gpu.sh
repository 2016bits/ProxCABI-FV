#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python_bin="${PYTHON_BIN:-python}"
gpus="${GPUS:-0,1,2,3}"
jobs_per_gpu="${JOBS_PER_GPU:-2}"
ifs_backup="$IFS"; IFS=',' read -r -a gpu_array <<<"$gpus"; IFS="$ifs_backup"
max_parallel=$(( ${#gpu_array[@]} * jobs_per_gpu ))

if pgrep -f 'run_scripts/(run_p0_parallel|run_p0_4gpu|handoff_p0_runner)\.sh|python .*proxcabi v2-(train|evaluate)' \
    | grep -vw -e "$$" -e "$PPID" >/dev/null; then
  echo "Another P0 scheduler/train process is active; refusing to start a duplicate." >&2
  exit 1
fi

mkdir -p outputs/v2/paper/logs
"$python_bin" scripts/paper_v2/audit_protocol.py \
  --config configs/paper_v2/protocol.yaml --output-dir outputs/v2/paper/protocol

if [[ ! -f outputs/v2/paper/annotations/candidate_manifest.json ]]; then
  "$python_bin" scripts/paper_v2/build_pair_candidates.py
fi

required_pools=(
  outputs/v2/paper/pairs/pairs_train_verified.jsonl
  outputs/v2/paper/pairs/pairs_eval_invariant.jsonl
  outputs/v2/paper/pairs/pairs_eval_semantic.jsonl
)
missing_pool=0
for path in "${required_pools[@]}"; do
  [[ -s "$path" ]] || missing_pool=1
done
if [[ "$missing_pool" == "1" ]]; then
  echo "Verified pools are not frozen. Complete these annotation CSVs:" >&2
  echo "  outputs/v2/paper/annotations/train_invariant_tasks.csv" >&2
  echo "  outputs/v2/paper/annotations/eval_invariant_tasks.csv" >&2
  echo "Then run: bash run_scripts/freeze_p0_pair_pools.sh" >&2
  exit 2
fi

"$python_bin" scripts/paper_v2/validate_pairs.py \
  --train "${required_pools[0]}" --invariant "${required_pools[1]}" \
  --semantic "${required_pools[2]}" --strict
"$python_bin" -m proxcabi v2-diagnose --config configs/v2/fever_vitaminc_joint.yaml \
  --output-dir outputs/v2/paper/proxy_health
"$python_bin" scripts/paper_v2/generate_experiment_matrix.py \
  --config configs/paper_v2/protocol.yaml --output-dir outputs/v2/paper/task_matrix --dry-run

echo "Checking the three selector warm-up checkpoints."
P0_FAMILY=components P0_NAME=base_only MAX_PARALLEL="${#gpu_array[@]}" JOBS_PER_GPU=1 GPUS="$gpus" \
  bash run_scripts/run_p0_parallel.sh

# Adapted CLEVER reuses the matched ERM claim--evidence model as its fusion
# branch, so freeze those three checkpoints before the dependency-free grid.
echo "Checking the three matched ERM fusion checkpoints required by CLEVER."
P0_FAMILY=baselines P0_NAME=erm MAX_PARALLEL="${#gpu_array[@]}" JOBS_PER_GPU=1 GPUS="$gpus" \
  bash run_scripts/run_p0_parallel.sh

if [[ ! -f outputs/v2/paper/selection/pools/selected_dual_budget_seed_13.jsonl ||
      ! -f outputs/v2/paper/selection/pools/selected_dual_budget_seed_21.jsonl ||
      ! -f outputs/v2/paper/selection/pools/selected_dual_budget_seed_42.jsonl ]]; then
  GPUS="$gpus" bash run_scripts/prepare_p0_selection_pools.sh
fi

echo "Starting P0 grid: GPUs=$gpus, jobs_per_gpu=$jobs_per_gpu, max_parallel=$max_parallel"
MAX_PARALLEL="$max_parallel" GPUS="$gpus" bash run_scripts/run_p0_parallel.sh

"$python_bin" scripts/paper_v2/build_paper_reports.py \
  --input outputs/v2/paper --output outputs/v2/paper/reports

