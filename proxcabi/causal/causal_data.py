from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .assumptions import validate_no_split_leakage, validate_proxy_protocol


REQUIRED_COLUMNS = ("treatment", "z", "w", "l", "response", "split", "semantic_group")


def load_causal_csv(path: Path, *, mode: str, proxy_protocol: dict[str, Any] | None = None,
                    fit_split: str | None = None) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    """Load a causal observation table and apply strict provenance/leakage gates."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty causal data: {path}")
    missing = set(REQUIRED_COLUMNS) - set(rows[0])
    if missing:
        raise ValueError(f"causal data missing columns: {sorted(missing)}")
    validate_no_split_leakage(rows)
    validate_proxy_protocol(proxy_protocol or {}, mode)
    selected = rows if fit_split is None else [row for row in rows if row["split"] == fit_split]
    if not selected:
        raise ValueError(f"no rows for split {fit_split!r}")
    observed = {
        "a": np.asarray([int(row["treatment"]) for row in selected], dtype=np.int64),
        "z": np.asarray([int(row["z"]) for row in selected], dtype=np.int64),
        "w": np.asarray([int(row["w"]) for row in selected], dtype=np.int64),
        "l": np.asarray([int(row["l"]) for row in selected], dtype=np.int64),
        "r": np.asarray([float(row["response"]) for row in selected], dtype=np.float64),
    }
    return observed, selected
