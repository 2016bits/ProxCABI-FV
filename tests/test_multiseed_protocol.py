from __future__ import annotations

import json

from proxcabi.causal.multiseed_report import (
    COMMON_PROTOCOL_FIELDS,
    VARIANT_PROTOCOL_FIELDS,
    _protocol_audit,
)


def _summary(seed: int) -> dict[str, object]:
    value = {field: None for field in (*COMMON_PROTOCOL_FIELDS, *VARIANT_PROTOCOL_FIELDS)}
    value.update({
        "seed": seed,
        "backbone": "model",
        "epochs": 2,
        "learning_rate": 8e-6,
        "batch_size": 8,
        "counterfactual_split": "cf",
        "num_counterfactual_samples": 10,
        "batches_per_epoch": 20,
        "optimizer_steps_per_epoch": 10,
        "scheduled_optimizer_steps": 20,
    })
    return value


def test_protocol_audit_detects_cross_seed_schedule_mismatch(tmp_path) -> None:
    multiseed = tmp_path / "multiseed"
    runs = []
    for seed in (13, 21):
        root = multiseed / "full" / f"seed_{seed}"
        root.mkdir(parents=True)
        summary = _summary(seed)
        if seed == 21:
            summary["scheduled_optimizer_steps"] = 21
        (root / "train_summary.json").write_text(json.dumps(summary))
        runs.append({"variant": "full", "seed": seed})

    audit = _protocol_audit(tmp_path, multiseed, runs)

    assert audit["passed"] is False
    assert any(row["field"] == "scheduled_optimizer_steps" for row in audit["mismatches"])


def test_protocol_audit_accepts_seed_only_difference(tmp_path) -> None:
    multiseed = tmp_path / "multiseed"
    runs = []
    for seed in (13, 21, 42):
        root = multiseed / "full" / f"seed_{seed}"
        root.mkdir(parents=True)
        (root / "train_summary.json").write_text(json.dumps(_summary(seed)))
        runs.append({"variant": "full", "seed": seed})

    assert _protocol_audit(tmp_path, multiseed, runs)["passed"] is True
