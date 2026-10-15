from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .assumptions import synthetic_assumptions
from .discrete_bridge import DiscreteOutcomeBridge
from .errors import CausalError
from .estimate import (bootstrap_discrete_effect, naive_effect, oracle_u_adjustment,
                       standardized_discrete_adjustment, with_uncertainty)
from .io import (atomic_json, atomic_text, atomic_yaml, load_observations, load_yaml,
                 manifest, save_observations)
from .synthetic import BinarySCMConfig, generate_binary_scm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m proxcabi.causal", description="Plan A causal tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synthetic-generate", help="Generate observed SCM data and separate oracle truth")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--seed", type=int)

    p = sub.add_parser("fit-bridge", help="Fit a discrete outcome bridge and estimate ATE")
    p.add_argument("--config", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--method", choices=["discrete", "neural"])
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--bootstrap-reps", type=int)

    p = sub.add_parser("evaluate-identification", help="Compare an estimate with synthetic oracle truth")
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--seed", type=int, default=13)

    p = sub.add_parser("synthetic-stress", help="Run Plan A SCM stress grid")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--seed", type=int)

    p = sub.add_parser("diagnose", help="Run observable support/rank diagnostics")
    p.add_argument("--config", type=Path)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=13)

    p = sub.add_parser("effect-guided", help="Convert gated CATE estimates into auditable group weights")
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--max-weight", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=13)

    p = sub.add_parser("collect-responses", help="Collect frozen-verifier responses for exploratory pairs")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--dataset", default="FEVER")
    p.add_argument("--split", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--head", choices=["fact"], default="fact")
    p.add_argument("--outcome", choices=["nll", "correct_prob"], default="nll")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-groups", type=int)
    p.add_argument("--seed", type=int, default=13)

    p = sub.add_parser("evaluate-pairs", help="Evaluate CCR/SSR and exploratory paired effects")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap-reps", type=int, default=1000)
    p.add_argument("--confidence-level", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=13)

    p = sub.add_parser("paper-report", help="Aggregate Plan A outputs into paper tables")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--discrete-dir", type=Path, default=Path("outputs/plan_a/discrete_smoke"))
    p.add_argument("--neural-dir", type=Path, default=Path("outputs/plan_a/neural_smoke"))
    p.add_argument("--stress-dir", type=Path, default=Path("outputs/plan_a/synthetic_stress"))
    p.add_argument("--paired-dir", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--bias-report", type=Path, required=True)

    p = sub.add_parser("multidataset-report", help="Aggregate HoVer/PolitiHop/VitaminC paper tables")
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "synthetic-generate":
            result = _synthetic_generate(args)
        elif args.command == "fit-bridge":
            result = _fit_bridge(args)
        elif args.command == "evaluate-identification":
            result = _evaluate_identification(args)
        elif args.command == "synthetic-stress":
            result = _synthetic_stress(args)
        elif args.command == "diagnose":
            result = _diagnose(args)
        elif args.command == "effect-guided":
            result = _effect_guided(args)
        elif args.command == "collect-responses":
            result = _collect_responses(args)
        elif args.command == "evaluate-pairs":
            result = _evaluate_pairs(args)
        elif args.command == "paper-report":
            result = _paper_report(args)
        else:
            result = _multidataset_report(args)
    except (CausalError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


def _resolved(config_path: Path | None, output_dir: Path | None, seed: int | None) -> dict[str, Any]:
    config = load_yaml(config_path) if config_path else {}
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    if seed is not None:
        config["seed"] = seed
    return config


def _run_dir(config: dict[str, Any], fallback: str) -> Path:
    path = Path(config.get("output_dir", fallback))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_run_metadata(output: Path, config: dict[str, Any], inputs: list[Path] | None = None) -> None:
    atomic_yaml(output / "config_resolved.yaml", config)
    atomic_json(output / "manifest.json", manifest(config, inputs))


def _synthetic_generate(args: argparse.Namespace) -> dict[str, Any]:
    config = _resolved(args.config, args.output_dir, args.seed)
    if config.get("mode", "synthetic_strict") != "synthetic_strict":
        raise ValueError("synthetic-generate requires mode: synthetic_strict")
    scm_values = dict(config.get("scm", {}))
    scm_values["seed"] = int(config.get("seed", scm_values.get("seed", 13)))
    scm = BinarySCMConfig(**scm_values)
    observed, oracle = generate_binary_scm(scm)
    output = _run_dir(config, "outputs/plan_a/synthetic")
    save_observations(output / "observations.npz", observed)
    oracle_file = {key: value for key, value in oracle.items() if key != "u"}
    oracle_file["u_file"] = "oracle_u.npy"
    np.save(output / "oracle_u.npy", oracle["u"], allow_pickle=False)
    atomic_json(output / "oracle_truth.json", oracle_file)
    assumptions = synthetic_assumptions(oracle["settings"])
    atomic_json(output / "assumptions.json", assumptions)
    _write_run_metadata(output, config)
    metrics = {
        "n": scm.n, "tau_true": oracle["tau_true"], "naive_difference": naive_effect(observed["a"], observed["r"]),
        "estimate_kind": "synthetic_oracle", "assumption_status": assumptions["overall_status"],
    }
    atomic_json(output / "metrics.json", metrics)
    return {"output_dir": str(output), **metrics}


def _bridge_settings(config: dict[str, Any]) -> dict[str, Any]:
    bridge = config.get("bridge", {})
    return {"ridge": float(bridge.get("ridge", 1e-6)), "min_cell": int(bridge.get("min_cell", 20)),
            "min_singular_value": float(bridge.get("min_singular_value", 1e-3))}


def _fit_bridge(args: argparse.Namespace) -> dict[str, Any]:
    config = _resolved(args.config, args.output_dir, args.seed)
    configured_input = config.get("input") or config.get("causal_data", {}).get("observations_path")
    input_path = args.input or (Path(configured_input) if configured_input else None)
    if input_path is None:
        raise ValueError("--input or config input is required")
    output = _run_dir(config, "outputs/plan_a/discrete")
    mode = str(config.get("mode", "synthetic_strict"))
    if input_path.suffix.lower() == ".csv":
        from .causal_data import load_causal_csv
        protocol: dict[str, Any] = {}
        protocol_path = config.get("causal_data", {}).get("proxy_protocol_path")
        if protocol_path:
            protocol = load_yaml(Path(protocol_path))
        observed, _ = load_causal_csv(
            input_path, mode=mode, proxy_protocol=protocol,
            fit_split=config.get("causal_data", {}).get("fit_split"),
        )
        treatment_path = config.get("causal_data", {}).get("treatment_protocol_path")
        if mode == "real_strict" and (not treatment_path or not Path(treatment_path).exists()):
            raise ValueError("real_strict requires an existing treatment_protocol_path")
    else:
        if mode == "real_strict":
            raise ValueError("real_strict requires an auditable CSV with split and semantic_group columns")
        observed = load_observations(input_path)
    seed = int(config.get("seed", 13))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(observed["a"]))
    cut = max(1, int(0.8 * len(order)))
    train_idx, heldout_idx = order[:cut], order[cut:]
    settings = _bridge_settings(config)
    method = args.method or config.get("bridge", {}).get("method", "discrete")
    if method == "neural":
        from .moment_bridge import NeuralMomentBridge
        neural = config.get("bridge", {}).get("neural", {})
        bridge = NeuralMomentBridge(seed=seed, **neural).fit(
            *(observed[k][train_idx] for k in ("a", "z", "w", "l", "r")),
            val=tuple(observed[k][heldout_idx] for k in ("a", "z", "w", "l", "r")),
        )
    else:
        bridge = DiscreteOutcomeBridge(**settings).fit(*(observed[k][train_idx] for k in ("a", "z", "w", "l", "r")))
    point = bridge.estimate_ate(observed["w"], observed["l"])
    report = config.get("report", {})
    reps = int(args.bootstrap_reps if args.bootstrap_reps is not None else report.get("bootstrap_reps", 0))
    if method == "discrete":
        se, low, high, successful = bootstrap_discrete_effect(
            observed, reps=reps, seed=seed, confidence_level=float(report.get("confidence_level", 0.95)),
            bridge_factory=lambda: DiscreteOutcomeBridge(**settings),
        )
    else:
        se, low, high, successful = None, None, None, 0
    estimate = with_uncertainty(point, se, low, high)
    diagnostics = bridge.diagnose()
    if method == "discrete":
        diagnostics["heldout"] = bridge.heldout_moments(*(observed[k][heldout_idx] for k in ("a", "z", "w", "l", "r")))
    else:
        import torch
        torch.save({"model_state": bridge.model.state_dict(), "maps": bridge.maps,
                    "diagnostics": diagnostics}, output / "estimator.pt")
    diagnostics["bootstrap_successful_reps"] = successful
    from .diagnostics import observable_support_diagnostics
    diagnostics["observable_support"] = observable_support_diagnostics(
        observed["a"], observed["z"], observed["w"], observed["l"]
    )
    comparisons = {
        "naive_difference": naive_effect(observed["a"], observed["r"]),
        "single_proxy_z": standardized_discrete_adjustment(observed["a"], observed["z"], observed["r"]),
        "single_proxy_w": standardized_discrete_adjustment(observed["a"], observed["w"], observed["r"]),
    }
    estimate_kind = ("synthetic_oracle" if mode == "synthetic_strict" else
                     "exploratory" if mode == "real_exploratory" else "identified_under_assumptions")
    metrics = {**estimate.to_dict(), **comparisons, "estimate_kind": estimate_kind}
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "bridge_diagnostics.json", diagnostics)
    assumption_report = (synthetic_assumptions(config.get("scm", {})) if mode == "synthetic_strict" else {
        "mode": mode, "overall_status": "untested" if mode == "real_exploratory" else "argued",
        "passed_observable_identification_gate": True,
        "warning": "Rank and moment diagnostics do not prove proxy exclusion or completeness.",
    })
    atomic_json(output / "assumptions.json", assumption_report)
    _write_effect_tables(output, bridge, observed, estimate, diagnostics)
    _write_run_metadata(output, config, [input_path])
    return {"output_dir": str(output), **metrics}


def _write_effect_tables(output: Path, bridge: Any, observed: dict[str, np.ndarray],
                         estimate: Any, diagnostics: dict[str, Any]) -> None:
    with (output / "effect_estimates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(estimate.to_dict()))
        writer.writeheader(); writer.writerow(estimate.to_dict())
    subgroup_rows = []
    for lv in np.unique(observed["l"]):
        mask = observed["l"] == lv
        effect = bridge.estimate_ate(observed["w"][mask], observed["l"][mask])
        subgroup_rows.append({"l": int(lv), **effect.to_dict()})
    with (output / "subgroup_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subgroup_rows[0]))
        writer.writeheader(); writer.writerows(subgroup_rows)
    moment_rows = diagnostics.get("heldout", {}).get("cells", [])
    if moment_rows:
        with (output / "moment_residuals.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(moment_rows[0]))
            writer.writeheader(); writer.writerows(moment_rows)
    atomic_text(output / "summary.md", (
        "# Plan A effect estimate\n\n"
        f"- Method: `{estimate.method}`\n"
        f"- ATE: `{estimate.ate:.8f}`\n"
        f"- Mu(0): `{estimate.mu0:.8f}`\n"
        f"- Mu(1): `{estimate.mu1:.8f}`\n"
        f"- Observable identification gate: `{estimate.passed_identification_gate}`\n\n"
        "Rank, support, and moment diagnostics are empirical checks; they do not prove "
        "the proxy exclusion restrictions or completeness.\n"
    ))


def _evaluate_identification(args: argparse.Namespace) -> dict[str, Any]:
    metrics = json.loads((args.pred_dir / "metrics.json").read_text(encoding="utf-8"))
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    tau_true, tau_hat = float(oracle["tau_true"]), float(metrics["ate"])
    result = {"tau_true": tau_true, "tau_hat": tau_hat, "abs_error": abs(tau_hat - tau_true),
              "squared_error": (tau_hat - tau_true) ** 2,
              "assumptions_designed_valid": bool(oracle.get("assumptions_designed_valid", False))}
    u_path = args.oracle.parent / str(oracle.get("u_file", "oracle_u.npy"))
    observation_path = args.oracle.parent / "observations.npz"
    if u_path.exists() and observation_path.exists():
        observed = load_observations(observation_path)
        u = np.load(u_path, allow_pickle=False)
        result["oracle_u_adjustment"] = oracle_u_adjustment(observed["a"], u, observed["r"])
    output = args.output_dir or args.pred_dir
    atomic_json(output / "identification_metrics.json", result)
    return result


def _stress_scenarios(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    configured = config.get("scenarios")
    if configured:
        return [(str(item["name"]), {k: v for k, v in item.items() if k != "name"}) for item in configured]
    return [
        ("valid", {}), ("weak_z_055", {"z_accuracy": 0.55}), ("weak_z_051", {"z_accuracy": 0.51}),
        ("weak_w_055", {"w_accuracy": 0.55}), ("weak_w_051", {"w_accuracy": 0.51}),
        ("invalid_z", {"direct_z_effect": 0.2}), ("invalid_w", {"direct_w_to_a": 0.2}),
        ("no_confounding", {"delta": 0.0}), ("no_effect", {"beta": 0.0}),
        ("scarce_overlap", {"p_a_u0": 0.02, "p_a_u1": 0.98}),
    ]


def _synthetic_stress(args: argparse.Namespace) -> dict[str, Any]:
    config = _resolved(args.config, args.output_dir, args.seed)
    output = _run_dir(config, "outputs/plan_a/synthetic_stress")
    base = dict(config.get("scm", {}))
    seeds = [int(x) for x in config.get("seeds", [11, 13, 17])]
    settings = _bridge_settings(config)
    rows: list[dict[str, Any]] = []
    for name, override in _stress_scenarios(config):
        for seed in seeds:
            cfg = BinarySCMConfig(**{**base, **override, "seed": seed})
            obs, oracle = generate_binary_scm(cfg)
            row: dict[str, Any] = {"scenario": name, "seed": seed, "tau_true": oracle["tau_true"],
                                   "naive": naive_effect(obs["a"], obs["r"]),
                                   "assumption_status": oracle["assumption_status"], "identified": False}
            try:
                bridge = DiscreteOutcomeBridge(**settings).fit(*(obs[k] for k in ("a", "z", "w", "l", "r")))
                estimate = bridge.estimate_ate(obs["w"], obs["l"])
                diag = bridge.diagnose()
                row.update({"identified": True, "tau_hat": estimate.ate,
                            "abs_bias": abs(estimate.ate - oracle["tau_true"]),
                            "min_singular_value": min(c["min_singular_value"] for c in diag["cells"].values()),
                            "condition_number": max(c["condition_number"] for c in diag["cells"].values()),
                            "rank": min(c["rank"] for c in diag["cells"].values()), "failure": ""})
            except CausalError as exc:
                row.update({"tau_hat": None, "abs_bias": None, "min_singular_value": None,
                            "condition_number": None, "rank": None, "failure": str(exc)})
            rows.append(row)
    fieldnames = list(rows[0])
    with (output / "effect_estimates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for name, _ in _stress_scenarios(config):
        group = [row for row in rows if row["scenario"] == name]
        errors = [float(row["abs_bias"]) for row in group if row["identified"]]
        summary[name] = {"runs": len(group), "successful": len(errors), "failure_rate": 1 - len(errors) / len(group),
                         "mean_abs_bias": float(np.mean(errors)) if errors else None,
                         "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else None}
    result = {"estimate_kind": "synthetic_oracle", "scenarios": summary}
    atomic_json(output / "metrics.json", result)
    atomic_json(output / "assumptions.json", {"note": "Per-row assumption_status distinguishes designed-valid and known violations."})
    _write_run_metadata(output, config)
    return {"output_dir": str(output), **result}


def _diagnose(args: argparse.Namespace) -> dict[str, Any]:
    config = _resolved(args.config, args.output_dir, args.seed)
    observed = load_observations(args.input)
    bridge = DiscreteOutcomeBridge(**_bridge_settings(config))
    try:
        bridge.fit(*(observed[k] for k in ("a", "z", "w", "l", "r")))
    except CausalError:
        diagnostics = bridge.diagnose()
        atomic_json(args.output_dir / "bridge_diagnostics.json", diagnostics)
        raise
    diagnostics = bridge.diagnose()
    atomic_json(args.output_dir / "bridge_diagnostics.json", diagnostics)
    _write_run_metadata(args.output_dir, config, [args.input])
    return diagnostics


def _effect_guided(args: argparse.Namespace) -> dict[str, Any]:
    from .effect_guided import effect_guided_group_weights

    metrics = json.loads((args.pred_dir / "metrics.json").read_text(encoding="utf-8"))
    assumptions = json.loads((args.pred_dir / "assumptions.json").read_text(encoding="utf-8"))
    if not metrics.get("passed_identification_gate"):
        raise ValueError("effect-guided weighting requires a passed identification gate")
    if assumptions.get("overall_status") in {"known_violated", "untested"}:
        raise ValueError(f"effect-guided weighting blocked by assumption status: {assumptions.get('overall_status')}")
    with (args.pred_dir / "subgroup_effects.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cate = {row["l"]: float(row["ate"]) for row in rows}
    weights = effect_guided_group_weights(cate, args.kappa, args.max_weight)
    result = {
        "method": "proximal_effect_guided", "group_key": "l", "weights": weights,
        "source_estimates": str(args.pred_dir / "subgroup_effects.csv"),
        "passed_identification_gate": True, "seed": args.seed,
        "note": "Weights guide a later training stage; they are not a differentiable ATE penalty.",
    }
    atomic_json(args.output_dir / "effect_group_weights.json", result)
    return {"output_dir": str(args.output_dir), **result}


def _collect_responses(args: argparse.Namespace) -> dict[str, Any]:
    from proxcabi.data import load_samples

    from .verifier_adapter import FrozenVerifierAdapter

    samples = load_samples(args.data_dir, args.dataset, args.split)
    samples = [sample for sample in samples if (sample.metadata or {}).get("contrast_group")]
    if args.max_groups is not None:
        chosen = []
        seen = set()
        for sample in samples:
            group = str(sample.metadata["contrast_group"])
            if group not in seen and len(seen) >= args.max_groups:
                continue
            seen.add(group)
            chosen.append(sample)
        samples = chosen
    if not samples:
        raise ValueError("split contains no counterfactual contrast groups")
    adapter = FrozenVerifierAdapter.from_checkpoint(
        args.checkpoint_dir, device=args.device, max_length=args.max_length,
        batch_size=args.batch_size, head=args.head,
    )
    probabilities = adapter.predict_proba(
        [sample.claim for sample in samples], [sample.evidence for sample in samples]
    )
    labels = np.asarray([sample.label_id for sample in samples], dtype=int)
    correct_probability = probabilities[np.arange(len(samples)), labels]
    responses = (-np.log(np.maximum(correct_probability, 1e-8)) if args.outcome == "nll"
                 else correct_probability)
    predictions = probabilities.argmax(axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "responses.csv"
    fields = [
        "row_id", "sample_id", "source_id", "split", "contrast_group", "contrast_role",
        "edit_type", "label_id", "prediction", "correct", "response", "correct_probability",
        "prob_supports", "prob_refutes", "prob_nei", "nli_teacher_confidence", "claim", "evidence",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (sample, probs) in enumerate(zip(samples, probabilities)):
            metadata = sample.metadata or {}
            writer.writerow({
                "row_id": index, "sample_id": sample.sample_id,
                "source_id": metadata.get("source_id", ""), "split": args.split,
                "contrast_group": metadata.get("contrast_group", ""),
                "contrast_role": metadata.get("contrast_role", ""),
                "edit_type": metadata.get("edit_type", "unknown"), "label_id": sample.label_id,
                "prediction": int(predictions[index]), "correct": int(predictions[index] == labels[index]),
                "response": float(responses[index]), "correct_probability": float(correct_probability[index]),
                "prob_supports": float(probs[0]), "prob_refutes": float(probs[1]),
                "prob_nei": float(probs[2]),
                "nli_teacher_confidence": metadata.get("nli_teacher_confidence", ""),
                "claim": sample.claim, "evidence": sample.evidence,
            })
    config = {
        "mode": "real_exploratory", "seed": args.seed, "dataset": args.dataset,
        "split": args.split, "checkpoint_dir": str(args.checkpoint_dir), "head": args.head,
        "outcome": args.outcome, "device": args.device, "batch_size": args.batch_size,
        "max_length": args.max_length, "max_groups": args.max_groups,
    }
    atomic_yaml(args.output_dir / "config_resolved.yaml", config)
    atomic_json(args.output_dir / "manifest.json", {
        **manifest(config, [args.checkpoint_dir / "best_model.pt"]),
        "checkpoint_sha256": adapter.checkpoint_hash,
    })
    atomic_json(args.output_dir / "assumptions.json", {
        "mode": "real_exploratory", "overall_status": "untested",
        "passed_identification_gate": False, "estimate_kind": "exploratory",
        "warning": "Counterfactual pairs measure frozen-model sensitivity, not a proximal observational ATE.",
    })
    result = {
        "output_dir": str(args.output_dir), "responses_path": str(output_path),
        "num_rows": len(samples),
        "num_groups": len({str(sample.metadata["contrast_group"]) for sample in samples}),
        "checkpoint_sha256": adapter.checkpoint_hash, "outcome_definition": args.outcome,
        "estimate_kind": "exploratory",
    }
    atomic_json(args.output_dir / "collection_summary.json", result)
    return result


def _evaluate_pairs(args: argparse.Namespace) -> dict[str, Any]:
    from .exploratory import build_paired_report, write_paired_report

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics, pairs, groups = build_paired_report(
        rows, seed=args.seed, bootstrap_reps=args.bootstrap_reps,
        confidence_level=args.confidence_level,
    )
    write_paired_report(args.output_dir, metrics, pairs, groups)
    config = {
        "mode": "real_exploratory", "seed": args.seed, "input": str(args.input),
        "bootstrap_reps": args.bootstrap_reps, "confidence_level": args.confidence_level,
    }
    atomic_yaml(args.output_dir / "config_resolved.yaml", config)
    atomic_json(args.output_dir / "manifest.json", manifest(config, [args.input]))
    return {"output_dir": str(args.output_dir), **metrics}


def _paper_report(args: argparse.Namespace) -> dict[str, Any]:
    from .paper_report import build_paper_report

    report = build_paper_report(
        output_dir=args.output_dir, discrete_dir=args.discrete_dir,
        neural_dir=args.neural_dir, stress_dir=args.stress_dir,
        paired_dir=args.paired_dir, eval_dir=args.eval_dir,
        bias_report=args.bias_report,
    )
    return {"output_dir": str(args.output_dir), "num_metrics": len(report["rows"]),
            "num_bias_features": len(report["bias_groups"])}


def _multidataset_report(args: argparse.Namespace) -> dict[str, Any]:
    from .multidataset_report import build_multidataset_report

    report = build_multidataset_report(args.eval_dir, args.output_dir)
    return {"output_dir": str(args.output_dir), "num_results": len(report["results"]),
            "num_subgroups": len(report["subgroups"])}
