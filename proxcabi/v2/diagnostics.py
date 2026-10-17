from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence

from proxcabi.data import FactSample, load_samples

from .structured_proxies import StructuredProxyBuilder


def proxy_diagnostics(
    proxy: StructuredProxyBuilder,
    base_train: Sequence[FactSample],
    observation_sets: Dict[str, Sequence[FactSample]],
) -> Dict[str, object]:
    """Report interpretable W coverage without fitting on observation sets."""

    if len(base_train) != proxy.fit_sample_count:
        raise ValueError("base train count does not match frozen proxy artifact")
    result: Dict[str, object] = {
        "schema_version": proxy.schema_version,
        "fit_sample_count": proxy.fit_sample_count,
        "base_train_source_sha256": proxy.base_train_source_sha256,
        "z_dim": proxy.z_dim,
        "num_w_classes": proxy.num_w_classes,
        "pi0": proxy.pi0,
        "pi0_entropy": _entropy(proxy.pi0),
        "sets": {},
    }
    for name, samples in {"base_train": base_train, **observation_sets}.items():
        counts = Counter(proxy.encode_w(sample) for sample in samples)
        total = len(samples)
        distribution = [counts.get(index, 0) / total if total else 0.0
                        for index in range(proxy.num_w_classes)]
        unk_id = proxy.w_vocab.get("UNK")
        result["sets"][name] = {  # type: ignore[index]
            "num_samples": total,
            "w_counts": {str(index): counts.get(index, 0) for index in range(proxy.num_w_classes)},
            "w_distribution": distribution,
            "w_entropy": _entropy(distribution),
            "max_class_share": max(distribution, default=0.0),
            "unk_rate": counts.get(unk_id, 0) / total if total and unk_id is not None else 0.0,
            "l1_distance_from_pi0": sum(abs(a - b) for a, b in zip(distribution, proxy.pi0)),
        }
    return result


def run_proxy_diagnostics(config_path: Path, output_dir: Path) -> Dict[str, object]:
    import yaml
    from .train import _load_samples_file

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    paths, data = config["paths"], config["data"]
    proxy = StructuredProxyBuilder.load(Path(paths["proxy_artifact"]))
    base = load_samples(Path(paths["data_dir"]), str(data["dataset"]), str(data["train_split"]))
    observations = {
        "cf_pool_train_observed_only": _load_samples_file(
            Path(paths["pair_pool_train"]), str(data["dataset"]), "train"
        ),
        "dev_select_observed_only": _load_samples_file(
            Path(paths["pair_pool_val"]).parent / f"{data['select_split']}.jsonl",
            str(data["dataset"]), str(data["select_split"]),
        ),
    }
    report = proxy_diagnostics(proxy, base, observations)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "proxy_diagnostics_v2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _entropy(values: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in values if value > 0)
