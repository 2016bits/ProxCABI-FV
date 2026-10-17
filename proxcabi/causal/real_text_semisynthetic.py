from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .discrete_bridge import DiscreteOutcomeBridge
from .errors import CausalError
from .estimate import (bootstrap_discrete_effect, naive_effect, oracle_u_adjustment,
                       standardized_discrete_adjustment)
from .io import atomic_json, atomic_text, file_sha256
from .synthetic import BinarySCMConfig, generate_binary_scm


def run_real_text_semisynthetic(
    responses_csv: Path,
    output_dir: Path,
    *,
    n: int = 20_000,
    seeds: list[int] | None = None,
    proxy_strengths: list[float] | None = None,
    beta: float = 0.3,
    delta: float = 0.8,
    text_heterogeneity: float = 0.15,
    noise_std: float = 0.1,
    bootstrap_reps: int = 30,
) -> dict[str, Any]:
    seeds = seeds or list(range(20))
    proxy_strengths = proxy_strengths or [0.55, 0.65, 0.75, 0.85, 0.95]
    anchors = _load_real_anchors(responses_csv)
    rows: list[dict[str, Any]] = []
    for strength in proxy_strengths:
        for seed in seeds:
            observed, oracle = generate_binary_scm(BinarySCMConfig(
                n=n, seed=seed, z_accuracy=strength, w_accuracy=strength,
                beta=beta, delta=delta, noise_std=0.0,
            ))
            rng = np.random.default_rng(seed + 100_003)
            anchor_values = rng.choice(anchors, size=n, replace=True)
            eps = rng.normal(0.0, noise_std, n)
            observed["r"] = (
                beta * observed["a"] + delta * oracle["u"]
                + text_heterogeneity * anchor_values + eps
            )
            result: dict[str, Any] = {
                "proxy_accuracy": strength,
                "seed": seed,
                "n": n,
                "tau_true": beta,
                "naive": naive_effect(observed["a"], observed["r"]),
                "z_only": standardized_discrete_adjustment(
                    observed["a"], observed["z"], observed["r"]
                ),
                "w_only": standardized_discrete_adjustment(
                    observed["a"], observed["w"], observed["r"]
                ),
                "ipw_z": _proxy_ipw(observed["a"], observed["z"], observed["r"]),
                "oracle_u": oracle_u_adjustment(
                    observed["a"], oracle["u"], observed["r"]
                ),
                "proximal": None,
                "proximal_ci_low": None,
                "proximal_ci_high": None,
                "proximal_ci_covered": None,
                "identification_passed": False,
            }
            try:
                bridge = DiscreteOutcomeBridge(min_cell=20, min_singular_value=1e-3).fit(
                    *(observed[key] for key in ("a", "z", "w", "l", "r"))
                )
                result["proximal"] = bridge.estimate_ate(observed["w"], observed["l"]).ate
                _, low, high, successful = bootstrap_discrete_effect(
                    observed, reps=bootstrap_reps, seed=seed + 200_003,
                    bridge_factory=lambda: DiscreteOutcomeBridge(
                        min_cell=20, min_singular_value=1e-3
                    ),
                )
                result["bootstrap_successful_reps"] = successful
                result["proximal_ci_low"] = low
                result["proximal_ci_high"] = high
                result["proximal_ci_covered"] = (
                    bool(low <= beta <= high) if low is not None and high is not None else None
                )
                result["identification_passed"] = True
            except (CausalError, ValueError) as exc:
                result["failure"] = str(exc)
            rows.append(result)

    summaries = _summaries(rows, beta)
    report = {
        "design": {
            "kind": "semi_synthetic_known_effect_with_real_text_response_anchors",
            "real_causal_identification_claimed": False,
            "responses_csv": str(responses_csv),
            "responses_sha256": file_sha256(responses_csv),
            "num_unique_real_anchors": len(anchors),
            "n_per_run": n,
            "seeds": seeds,
            "proxy_strengths": proxy_strengths,
            "tau_true": beta,
            "latent_confounder_effect": delta,
            "real_text_heterogeneity_scale": text_heterogeneity,
            "noise_std": noise_std,
            "bootstrap_reps": bootstrap_reps,
        },
        "runs": rows,
        "summaries": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "semisynthetic_metrics.json", report)
    _write_csv(output_dir / "semisynthetic_runs.csv", rows)
    _write_csv(output_dir / "semisynthetic_summary.csv", summaries)
    atomic_text(output_dir / "semisynthetic_results.md", _markdown(report))
    return report


def run_real_text_sensitivity(
    responses_csv: Path,
    output_dir: Path,
    *,
    n: int = 20_000,
    seeds: list[int] | None = None,
    beta: float = 0.3,
) -> dict[str, Any]:
    seeds = seeds or list(range(20))
    anchors = _load_real_anchors(responses_csv)
    scenarios = [
        {"scenario": "baseline", "dimension": "reference"},
        {"scenario": "overlap_0.30_0.70", "dimension": "overlap", "p_a_u0": 0.30, "p_a_u1": 0.70},
        {"scenario": "overlap_0.10_0.90", "dimension": "overlap", "p_a_u0": 0.10, "p_a_u1": 0.90},
        {"scenario": "overlap_0.05_0.95", "dimension": "overlap", "p_a_u0": 0.05, "p_a_u1": 0.95},
        *[{"scenario": f"invalid_z_{value:.2f}", "dimension": "z_exclusion",
           "direct_z_effect": value} for value in (0.10, 0.20, 0.40)],
        *[{"scenario": f"invalid_w_{value:.2f}", "dimension": "w_treatment",
           "direct_w_to_a": value} for value in (0.10, 0.20, 0.40)],
        *[{"scenario": f"sample_n_{value}", "dimension": "sample_size", "n": value}
          for value in (2000, 5000, 10000)],
        {"scenario": "shuffled_z", "dimension": "negative_control", "shuffle": "z"},
        {"scenario": "shuffled_w", "dimension": "negative_control", "shuffle": "w"},
        {"scenario": "shuffled_both", "dimension": "negative_control", "shuffle": "both"},
    ]
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            run_n = int(scenario.get("n", n))
            cfg = BinarySCMConfig(
                n=run_n, seed=seed, z_accuracy=0.85, w_accuracy=0.85,
                p_a_u0=float(scenario.get("p_a_u0", 0.20)),
                p_a_u1=float(scenario.get("p_a_u1", 0.80)),
                beta=beta, delta=0.8, noise_std=0.0,
                direct_z_effect=float(scenario.get("direct_z_effect", 0.0)),
                direct_w_to_a=float(scenario.get("direct_w_to_a", 0.0)),
            )
            observed, oracle = generate_binary_scm(cfg)
            rng = np.random.default_rng(seed + 100_003)
            observed["r"] = (
                beta * observed["a"] + 0.8 * oracle["u"]
                + cfg.direct_z_effect * observed["z"]
                + 0.15 * rng.choice(anchors, size=run_n, replace=True)
                + rng.normal(0.0, 0.1, run_n)
            )
            shuffle = scenario.get("shuffle")
            if shuffle in {"z", "both"}:
                observed["z"] = rng.permutation(observed["z"])
            if shuffle in {"w", "both"}:
                observed["w"] = rng.permutation(observed["w"])
            row = {
                "scenario": scenario["scenario"], "dimension": scenario["dimension"],
                "seed": seed, "n": run_n, "tau_true": beta,
                "naive": naive_effect(observed["a"], observed["r"]),
                "proximal": None, "identification_passed": False,
                "assumptions_designed_valid": (
                    bool(oracle["assumptions_designed_valid"]) and shuffle is None
                ),
            }
            try:
                bridge = DiscreteOutcomeBridge(min_cell=20, min_singular_value=1e-3).fit(
                    *(observed[key] for key in ("a", "z", "w", "l", "r"))
                )
                row["proximal"] = bridge.estimate_ate(observed["w"], observed["l"]).ate
                row["identification_passed"] = True
            except (CausalError, ValueError) as exc:
                row["failure"] = str(exc)
            rows.append(row)
    summaries = []
    for scenario in scenarios:
        subset = [row for row in rows if row["scenario"] == scenario["scenario"]]
        for method in ("naive", "proximal"):
            values = np.asarray([row[method] for row in subset if row.get(method) is not None])
            errors = values - beta
            summaries.append({
                "scenario": scenario["scenario"], "dimension": scenario["dimension"],
                "method": method, "successful_runs": len(values),
                "pass_rate": (float(np.mean([row["identification_passed"] for row in subset]))
                              if method == "proximal" else None),
                "mean_tau_hat": float(values.mean()) if len(values) else None,
                "bias": float(errors.mean()) if len(values) else None,
                "rmse": float(np.sqrt(np.mean(errors ** 2))) if len(values) else None,
            })
    report = {
        "design": {"kind": "real_text_semisynthetic_sensitivity", "tau_true": beta,
                   "seeds": seeds, "real_causal_identification_claimed": False,
                   "responses_sha256": file_sha256(responses_csv)},
        "runs": rows, "summaries": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "sensitivity_metrics.json", report)
    _write_csv(output_dir / "sensitivity_runs.csv", rows)
    _write_csv(output_dir / "sensitivity_summary.csv", summaries)
    lines = ["# Real-text semi-synthetic sensitivity", "",
             "| Dimension | Scenario | Method | Runs | Gate pass | Bias | RMSE |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        gate = "—" if row["pass_rate"] is None else f"{row['pass_rate']:.2f}"
        bias = "—" if row["bias"] is None else f"{row['bias']:.6f}"
        rmse = "—" if row["rmse"] is None else f"{row['rmse']:.6f}"
        lines.append(f"| {row['dimension']} | {row['scenario']} | {row['method']} | "
                     f"{row['successful_runs']} | {gate} | {bias} | {rmse} |")
    atomic_text(output_dir / "sensitivity_results.md", "\n".join(lines) + "\n")
    return report


def run_neural_real_text_semisynthetic(
    responses_csv: Path,
    output_dir: Path,
    *,
    n: int = 20_000,
    seeds: list[int] | None = None,
    proxy_strengths: list[float] | None = None,
    beta: float = 0.3,
    epochs: int = 300,
    patience: int = 50,
) -> dict[str, Any]:
    from .moment_bridge import NeuralMomentBridge

    seeds = seeds or [0, 1, 2, 3, 4]
    proxy_strengths = proxy_strengths or [0.55, 0.65, 0.75, 0.85, 0.95]
    anchors = _load_real_anchors(responses_csv)
    rows = []
    for strength in proxy_strengths:
        for seed in seeds:
            observed, oracle = generate_binary_scm(BinarySCMConfig(
                n=n, seed=seed, z_accuracy=strength, w_accuracy=strength,
                beta=beta, delta=0.8, noise_std=0.0,
            ))
            rng = np.random.default_rng(seed + 100_003)
            observed["r"] = (
                beta * observed["a"] + 0.8 * oracle["u"]
                + 0.15 * rng.choice(anchors, size=n, replace=True)
                + rng.normal(0.0, 0.1, n)
            )
            order = rng.permutation(n)
            cut = int(0.8 * n)
            train, valid = order[:cut], order[cut:]
            bridge = NeuralMomentBridge(
                epochs=epochs, patience=patience, seed=seed,
                emb_dim=8, hidden_dim=32, device="cpu",
            ).fit(
                *(observed[key][train] for key in ("a", "z", "w", "l", "r")),
                val=tuple(observed[key][valid] for key in ("a", "z", "w", "l", "r")),
            )
            tau = bridge.estimate_ate(observed["w"], observed["l"]).ate
            diagnostics = bridge.diagnose()
            rows.append({
                "proxy_accuracy": strength, "seed": seed, "n": n,
                "tau_true": beta, "tau_hat": tau, "error": tau - beta,
                "absolute_error": abs(tau - beta),
                "epochs_completed": diagnostics["epochs_completed"],
                "heldout_conditional_moment_mse": diagnostics["heldout_conditional_moment_mse"],
            })
    summaries = []
    for strength in proxy_strengths:
        values = np.asarray([row["tau_hat"] for row in rows
                             if row["proxy_accuracy"] == strength])
        errors = values - beta
        summaries.append({
            "proxy_accuracy": strength, "method": "neural_moment_proximal",
            "successful_runs": len(values), "mean_tau_hat": float(values.mean()),
            "bias": float(errors.mean()), "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        })
    report = {
        "design": {"kind": "neural_real_text_semisynthetic", "tau_true": beta,
                   "seeds": seeds, "proxy_strengths": proxy_strengths,
                   "epochs": epochs, "patience": patience,
                   "real_causal_identification_claimed": False,
                   "responses_sha256": file_sha256(responses_csv)},
        "runs": rows, "summaries": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "neural_semisynthetic_metrics.json", report)
    _write_csv(output_dir / "neural_semisynthetic_runs.csv", rows)
    _write_csv(output_dir / "neural_semisynthetic_summary.csv", summaries)
    lines = ["# Neural real-text semi-synthetic recovery", "",
             "| Proxy accuracy | Seeds | Mean tau | Bias | RMSE | Std |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        std = "—" if row["std"] is None else f"{row['std']:.6f}"
        lines.append(f"| {row['proxy_accuracy']:.2f} | {row['successful_runs']} | "
                     f"{row['mean_tau_hat']:.6f} | {row['bias']:.6f} | "
                     f"{row['rmse']:.6f} | {std} |")
    atomic_text(output_dir / "neural_semisynthetic_results.md", "\n".join(lines) + "\n")
    build_combined_semisynthetic_report(output_dir)
    return report


def build_combined_semisynthetic_report(output_dir: Path) -> list[dict[str, Any]]:
    discrete_path = output_dir / "semisynthetic_metrics.json"
    neural_path = output_dir / "neural_semisynthetic_metrics.json"
    if not discrete_path.is_file() or not neural_path.is_file():
        return []
    discrete = json.loads(discrete_path.read_text(encoding="utf-8"))
    neural = json.loads(neural_path.read_text(encoding="utf-8"))
    rows = list(discrete["summaries"]) + list(neural["summaries"])
    rows.sort(key=lambda row: (float(row["proxy_accuracy"]), str(row["method"])))
    _write_csv(output_dir / "combined_semisynthetic_summary.csv", rows)
    lines = ["# Combined real-text semi-synthetic recovery", "",
             "Discrete results use 20 seeds; neural moment results use five fixed seeds. "
             "The true ATE is 0.3.", "",
             "| Proxy accuracy | Method | Seeds | Bias | RMSE |",
             "|---:|---|---:|---:|---:|"]
    for row in rows:
        if row.get("bias") is None or row.get("rmse") is None:
            continue
        lines.append(f"| {float(row['proxy_accuracy']):.2f} | {row['method']} | "
                     f"{row['successful_runs']} | {float(row['bias']):.6f} | "
                     f"{float(row['rmse']):.6f} |")
    atomic_text(output_dir / "combined_semisynthetic_results.md", "\n".join(lines) + "\n")
    return rows


def _load_real_anchors(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing real response anchors: {path}")
    by_source: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("contrast_role") == "orig_support":
                by_source[str(row["source_id"])] = float(row["response"])
    if len(by_source) < 20:
        raise ValueError("at least 20 unique orig_support response anchors are required")
    values = np.log1p(np.asarray(list(by_source.values()), dtype=np.float64))
    scale = float(values.std())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("real response anchors have no finite variation")
    return np.clip((values - values.mean()) / scale, -3.0, 3.0)


def _proxy_ipw(a: np.ndarray, proxy: np.ndarray, r: np.ndarray) -> float:
    propensity = np.empty(len(a), dtype=np.float64)
    for value in np.unique(proxy):
        mask = proxy == value
        propensity[mask] = np.mean(a[mask])
    propensity = np.clip(propensity, 0.01, 0.99)
    treated_weights = a / propensity
    control_weights = (1 - a) / (1 - propensity)
    mu1 = np.sum(treated_weights * r) / np.sum(treated_weights)
    mu0 = np.sum(control_weights * r) / np.sum(control_weights)
    return float(mu1 - mu0)


def _summaries(rows: list[dict[str, Any]], truth: float) -> list[dict[str, Any]]:
    methods = ["naive", "z_only", "w_only", "ipw_z", "proximal", "oracle_u"]
    result = []
    for strength in sorted({float(row["proxy_accuracy"]) for row in rows}):
        subset = [row for row in rows if float(row["proxy_accuracy"]) == strength]
        for method in methods:
            values = np.asarray([row[method] for row in subset if row.get(method) is not None], dtype=float)
            errors = values - truth
            coverage_values = [row["proximal_ci_covered"] for row in subset
                               if row.get("proximal_ci_covered") is not None]
            result.append({
                "proxy_accuracy": strength,
                "method": method,
                "successful_runs": len(values),
                "mean_tau_hat": float(values.mean()) if len(values) else None,
                "bias": float(errors.mean()) if len(values) else None,
                "rmse": float(np.sqrt(np.mean(errors ** 2))) if len(values) else None,
                "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                "ci_coverage": (float(np.mean(coverage_values))
                                if method == "proximal" and coverage_values else None),
            })
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Real-text semi-synthetic causal recovery", "",
             "Real frozen-verifier responses provide outcome heterogeneity; treatment, latent "
             "confounding, proxies, and the true effect are controlled. This does not claim "
             "causal identification on observational FEVER.", "",
             "| Proxy accuracy | Method | Runs | Mean tau | Bias | RMSE | 95% CI coverage |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for row in report["summaries"]:
        coverage = "—" if row["ci_coverage"] is None else f"{row['ci_coverage']:.3f}"
        values = [row[key] for key in ("mean_tau_hat", "bias", "rmse")]
        formatted = ["—" if value is None else f"{value:.6f}" for value in values]
        lines.append(
            f"| {row['proxy_accuracy']:.2f} | {row['method']} | {row['successful_runs']} | "
            f"{formatted[0]} | {formatted[1]} | {formatted[2]} | {coverage} |"
        )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
