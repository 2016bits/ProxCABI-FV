#!/usr/bin/env bash
set -euo pipefail

action="${1:-start}"
remote_host="${REMOTE_HOST:-3090}"
remote_project="${REMOTE_PROJECT:-/data/yangjun/fact/ProxCABI_p0}"
remote_python="${REMOTE_PYTHON:-/data/yangjun/.conda/envs/NodeRAG/bin/python}"
gpus="${GPUS:-0,1,2,3}"
jobs_per_gpu="${JOBS_PER_GPU:-1}"
max_parallel="${MAX_PARALLEL:-4}"

case "$action" in
  start)
    ssh -o BatchMode=yes "$remote_host" bash -s -- \
      "$remote_project" "$remote_python" "$gpus" "$jobs_per_gpu" "$max_parallel" <<'REMOTE'
set -euo pipefail
project="$1"; python_bin="$2"; gpus="$3"; jobs_per_gpu="$4"; max_parallel="$5"
cd "$project"

pid_file="outputs/v2/paper/remote_3090_runner.pid"
console_log="outputs/v2/paper/remote_3090_console.log"
if [[ -f "$pid_file" ]]; then
  old_pid="$(cat "$pid_file")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "P0 baseline runner is already active: PID=$old_pid"
    echo "Log: $project/$console_log"
    exit 0
  fi
fi

[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 2; }
CUDA_VISIBLE_DEVICES="${gpus%%,*}" "$python_bin" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable on the remote node")
print("CUDA preflight OK:", torch.cuda.get_device_name(0))
PY

# The 3090 host has no scheduler. Refuse a fresh launch if another compute job
# is already present instead of competing with it.
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "A GPU compute process is already active; refusing to start a new queue." >&2
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader >&2
  exit 3
fi

mkdir -p outputs/v2/paper/logs
nohup env \
  HF_HOME=/data/yangjun/.cache/huggingface \
  TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTHON_BIN="$python_bin" GPUS="$gpus" \
  JOBS_PER_GPU="$jobs_per_gpu" MAX_PARALLEL="$max_parallel" \
  bash run_scripts/run_p0_baselines.sh \
  > "$console_log" 2>&1 < /dev/null &
runner_pid=$!
echo "$runner_pid" > "$pid_file"
echo "Started P0 baseline runner: PID=$runner_pid"
echo "Log: $project/$console_log"
REMOTE
    ;;
  status)
    ssh -o BatchMode=yes "$remote_host" bash -s -- "$remote_project" <<'REMOTE'
set -euo pipefail
cd "$1"
pid_file="outputs/v2/paper/remote_3090_runner.pid"
if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "runner=active pid=$(cat "$pid_file")"
else
  echo "runner=inactive"
fi
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
tail -n 30 outputs/v2/paper/remote_3090_console.log 2>/dev/null || true
REMOTE
    ;;
  logs)
    ssh -o BatchMode=yes "$remote_host" \
      "cd '$remote_project' && tail -f outputs/v2/paper/remote_3090_console.log"
    ;;
  *)
    echo "Usage: $0 {start|status|logs}" >&2
    exit 2
    ;;
esac
