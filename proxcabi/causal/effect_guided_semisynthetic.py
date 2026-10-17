from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .discrete_bridge import DiscreteOutcomeBridge
from .estimate import standardized_discrete_adjustment
from .io import atomic_json, atomic_text, file_sha256
from .real_text_semisynthetic import _load_real_anchors


BETAS = np.asarray([0.05, 0.15, 0.35, 0.65], dtype=np.float64)
P_A_U0 = np.asarray([0.05, 0.20, 0.30, 0.40], dtype=np.float64)
P_A_U1 = np.asarray([0.95, 0.80, 0.70, 0.60], dtype=np.float64)


def run_effect_guided_semisynthetic(responses_csv: Path, output_dir: Path, *,
                                    n: int = 40_000, seeds: list[int] | None = None,
                                    proxy_accuracy: float = 0.85,
                                    budget_groups: int = 2) -> dict[str, Any]:
    seeds = seeds or list(range(20))
    anchors = _load_real_anchors(responses_csv)
    if not 1 <= budget_groups < len(BETAS):
        raise ValueError("budget_groups must be between 1 and number of groups minus one")
    oracle_selected = set(np.argsort(BETAS)[-budget_groups:].tolist())
    oracle_utility = float(0.25 * BETAS[list(oracle_selected)].sum())
    rows, estimates = [], []
    for seed in seeds:
        observed = _generate(n, seed, proxy_accuracy, anchors)
        rng = np.random.default_rng(seed + 77_777)
        order = rng.permutation(n)
        cut = int(0.8 * n)
        train, target = order[:cut], order[cut:]
        bridge = DiscreteOutcomeBridge(min_cell=20, min_singular_value=1e-3).fit(
            *(observed[key][train] for key in ("a", "z", "w", "l", "r"))
        )
        proximal = bridge.estimate_cate(observed["w"][target], observed["l"][target])
        scores = {
            "proximal": {group: proximal[group].ate for group in range(len(BETAS))},
            "naive": _group_effects(observed, train, None),
            "z_only": _group_effects(observed, train, "z"),
            "oracle": {group: float(BETAS[group]) for group in range(len(BETAS))},
        }
        for method, method_scores in scores.items():
            selected = set(sorted(method_scores, key=method_scores.get)[-budget_groups:])
            rows.append(_strategy_row(seed, method, selected, oracle_selected, oracle_utility))
        random_selected = set(rng.choice(len(BETAS), size=budget_groups, replace=False).tolist())
        rows.append(_strategy_row(seed, "random", random_selected, oracle_selected, oracle_utility))
        uniform_utility = float((budget_groups / len(BETAS)) * 0.25 * BETAS.sum())
        rows.append({"seed": seed, "method": "uniform", "selected_groups": "fractional_all",
                     "captured_true_utility": uniform_utility,
                     "utility_fraction_of_oracle": uniform_utility / oracle_utility,
                     "regret": oracle_utility - uniform_utility,
                     "exact_oracle_selection": False})
        for method, method_scores in scores.items():
            for group, estimate in method_scores.items():
                estimates.append({"seed": seed, "method": method, "group": group,
                                  "tau_true": float(BETAS[group]), "tau_hat": float(estimate),
                                  "absolute_error": abs(float(estimate) - float(BETAS[group]))})
    summaries = []
    for method in ("proximal", "naive", "z_only", "random", "uniform", "oracle"):
        subset = [row for row in rows if row["method"] == method]
        regret = np.asarray([row["regret"] for row in subset])
        summaries.append({
            "method": method, "runs": len(subset),
            "mean_captured_utility": float(np.mean([row["captured_true_utility"] for row in subset])),
            "mean_utility_fraction_of_oracle": float(np.mean(
                [row["utility_fraction_of_oracle"] for row in subset])),
            "mean_regret": float(regret.mean()),
            "std_regret": float(regret.std(ddof=1)) if len(regret) > 1 else None,
            "exact_oracle_selection_rate": float(np.mean(
                [row["exact_oracle_selection"] for row in subset])),
        })
    report = {"design": {
        "kind": "semi_synthetic_effect_guided_group_targeting",
        "real_causal_identification_claimed": False, "n_per_run": n, "seeds": seeds,
        "proxy_accuracy": proxy_accuracy, "group_true_effects": BETAS.tolist(),
        "group_p_a_u0": P_A_U0.tolist(), "group_p_a_u1": P_A_U1.tolist(),
        "budget_groups": budget_groups, "oracle_selected_groups": sorted(oracle_selected),
        "oracle_utility": oracle_utility, "responses_sha256": file_sha256(responses_csv),
    }, "runs": rows, "group_estimates": estimates, "summaries": summaries}
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "effect_guided_metrics.json", report)
    _write_csv(output_dir / "effect_guided_runs.csv", rows)
    _write_csv(output_dir / "effect_guided_group_estimates.csv", estimates)
    _write_csv(output_dir / "effect_guided_summary.csv", summaries)
    atomic_text(output_dir / "effect_guided_results.md", _markdown(report))
    return report


def _generate(n: int, seed: int, accuracy: float, anchors: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    l = rng.integers(0, len(BETAS), size=n, dtype=np.int64)
    u = rng.binomial(1, 0.5, n).astype(np.int64)
    z = np.bitwise_xor(u, rng.binomial(1, 1 - accuracy, n)).astype(np.int64)
    w = np.bitwise_xor(u, rng.binomial(1, 1 - accuracy, n)).astype(np.int64)
    a = rng.binomial(1, np.where(u == 1, P_A_U1[l], P_A_U0[l])).astype(np.int64)
    r = (BETAS[l] * a + 0.8 * u + 0.15 * rng.choice(anchors, n, replace=True)
         + rng.normal(0.0, 0.1, n))
    return {"a": a, "z": z, "w": w, "l": l, "r": r}


def _group_effects(observed: dict[str, np.ndarray], indices: np.ndarray,
                   proxy: str | None) -> dict[int, float]:
    result = {}
    for group in range(len(BETAS)):
        selected = indices[observed["l"][indices] == group]
        if proxy is None:
            a, r = observed["a"][selected], observed["r"][selected]
            result[group] = float(r[a == 1].mean() - r[a == 0].mean())
        else:
            result[group] = standardized_discrete_adjustment(
                observed["a"][selected], observed[proxy][selected], observed["r"][selected])
    return result


def _strategy_row(seed: int, method: str, selected: set[int], oracle: set[int],
                  oracle_utility: float) -> dict[str, Any]:
    utility = float(0.25 * BETAS[list(selected)].sum())
    return {"seed": seed, "method": method,
            "selected_groups": ",".join(str(value) for value in sorted(selected)),
            "captured_true_utility": utility,
            "utility_fraction_of_oracle": utility / oracle_utility,
            "regret": oracle_utility - utility,
            "exact_oracle_selection": selected == oracle}


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Semi-synthetic effect-guided targeting", "",
             "A fixed two-of-four group budget is allocated using each score. Utility is "
             "evaluated with known heterogeneous effects. This is not an observational "
             "FEVER causal claim or end-to-end verifier result.", "",
             "| Method | Runs | Utility / oracle | Mean regret | Exact oracle selection |",
             "|---|---:|---:|---:|---:|"]
    for row in report["summaries"]:
        lines.append(f"| {row['method']} | {row['runs']} | "
                     f"{row['mean_utility_fraction_of_oracle']:.3f} | "
                     f"{row['mean_regret']:.6f} | {row['exact_oracle_selection_rate']:.3f} |")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
