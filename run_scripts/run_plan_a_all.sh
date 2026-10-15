#!/usr/bin/env bash
set -euo pipefail

# Complete CPU-safe Plan A reference pipeline.
# Optional real-data run:
#   REAL_CONFIG=configs/plan_a/verifier_observational.yaml \
#   REAL_OUTPUT_DIR=outputs/plan_a/real_strict \
#   bash run_scripts/run_plan_a_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TESTS="${RUN_TESTS:-1}"
REAL_CONFIG="${REAL_CONFIG:-}"
REAL_OUTPUT_DIR="${REAL_OUTPUT_DIR:-outputs/plan_a/real_strict}"

cd "${PROJECT_DIR}"

run_step() {
  local title="$1"
  shift
  echo
  echo "[Plan A] ${title}"
  "$@"
}

if [[ "${RUN_TESTS}" == "1" ]]; then
  run_step "Run unit and regression tests" "${PYTHON_BIN}" -m pytest tests/ -q
fi

run_step "Check legacy CLI compatibility" "${PYTHON_BIN}" -m proxcabi --help
run_step "Check Plan A CLI" "${PYTHON_BIN}" -m proxcabi.causal --help

run_step "Run discrete synthetic identification pipeline" \
  bash run_scripts/plan_a_synthetic_smoke.sh

run_step "Run synthetic stress grid" \
  bash run_scripts/plan_a_synthetic_stress.sh

run_step "Generate neural-bridge synthetic observations" \
  "${PYTHON_BIN}" -m proxcabi.causal synthetic-generate \
  --config configs/plan_a/synthetic_neural_smoke.yaml

run_step "Fit real-valued neural moment bridge" \
  "${PYTHON_BIN}" -m proxcabi.causal fit-bridge \
  --config configs/plan_a/synthetic_neural_smoke.yaml \
  --input outputs/plan_a/synthetic_neural_smoke/observations.npz \
  --output-dir outputs/plan_a/neural_smoke \
  --method neural

run_step "Evaluate neural identification against oracle truth" \
  "${PYTHON_BIN}" -m proxcabi.causal evaluate-identification \
  --pred-dir outputs/plan_a/neural_smoke \
  --oracle outputs/plan_a/synthetic_neural_smoke/oracle_truth.json

run_step "Build stage-wise effect-guided weights" \
  "${PYTHON_BIN}" -m proxcabi.causal effect-guided \
  --pred-dir outputs/plan_a/discrete_smoke \
  --output-dir outputs/plan_a/effect_guided_smoke

run_step "Validate generated metrics and safety gates" "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path


def read(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


discrete = read("outputs/plan_a/discrete_smoke/metrics.json")
identification = read("outputs/plan_a/discrete_smoke/identification_metrics.json")
diagnostics = read("outputs/plan_a/discrete_smoke/bridge_diagnostics.json")
assumptions = read("outputs/plan_a/discrete_smoke/assumptions.json")
stress = read("outputs/plan_a/synthetic_stress/metrics.json")["scenarios"]
neural = read("outputs/plan_a/neural_smoke/metrics.json")
neural_identification = read("outputs/plan_a/neural_smoke/identification_metrics.json")
weights = read("outputs/plan_a/effect_guided_smoke/effect_group_weights.json")

checks = {
    "discrete_identification_gate": discrete["passed_identification_gate"] is True,
    "discrete_abs_error_below_0.06": identification["abs_error"] < 0.06,
    "oracle_u_abs_error_below_0.02": abs(identification["oracle_u_adjustment"] - identification["tau_true"]) < 0.02,
    "all_bridge_matrices_full_rank": diagnostics["all_required_matrices_full_column_rank"] is True,
    "heldout_moment_rmse_below_0.03": diagnostics["heldout"]["conditional_moment_rmse"] < 0.03,
    "bootstrap_has_successful_refits": diagnostics["bootstrap_successful_reps"] > 0,
    "synthetic_assumptions_designed_valid": assumptions["overall_status"] == "designed_valid",
    "valid_stress_bias_below_0.06": stress["valid"]["mean_abs_bias"] < 0.06,
    "weak_z_gate_fails": stress["weak_z_051"]["failure_rate"] > 0.0,
    "weak_w_gate_fails": stress["weak_w_051"]["failure_rate"] > 0.0,
    "invalid_z_is_more_biased": stress["invalid_z"]["mean_abs_bias"] > stress["valid"]["mean_abs_bias"],
    "invalid_w_is_more_biased": stress["invalid_w"]["mean_abs_bias"] > stress["valid"]["mean_abs_bias"],
    "neural_is_synthetic_oracle": neural["estimate_kind"] == "synthetic_oracle",
    "neural_abs_error_below_0.06": neural_identification["abs_error"] < 0.06,
    "effect_weights_passed_gate": weights["passed_identification_gate"] is True,
}

failed = [name for name, passed in checks.items() if not passed]
summary = {
    "checks": checks,
    "discrete_tau_hat": discrete["ate"],
    "neural_tau_hat": neural["ate"],
    "tau_true": identification["tau_true"],
    "status": "passed" if not failed else "failed",
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
if failed:
    raise SystemExit("Plan A validation failed: " + ", ".join(failed))
PY

if [[ -n "${REAL_CONFIG}" ]]; then
  run_step "Fit gated real-data bridge" \
    "${PYTHON_BIN}" -m proxcabi.causal fit-bridge \
    --config "${REAL_CONFIG}" \
    --output-dir "${REAL_OUTPUT_DIR}"

  run_step "Build real-data effect-guided weights" \
    "${PYTHON_BIN}" -m proxcabi.causal effect-guided \
    --pred-dir "${REAL_OUTPUT_DIR}" \
    --output-dir "${REAL_OUTPUT_DIR}/effect_guided"
else
  echo
  echo "[Plan A] Real strict stage skipped: set REAL_CONFIG after supplying auditable proxy/treatment protocols."
fi

echo
echo "[Plan A] All requested stages completed successfully."
