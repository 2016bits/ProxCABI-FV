from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


LABEL_ALIASES = {
    "supports": "supports",
    "support": "supports",
    "supported": "supports",
    "refutes": "refutes",
    "refute": "refutes",
    "refuted": "refutes",
    "not enough info": "not enough info",
    "nei": "not enough info",
    "notenoughinfo": "not enough info",
}

LABELS = ["supports", "refutes", "not enough info"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


@dataclass
class FactSample:
    sample_id: str
    dataset: str
    split: str
    claim: str
    evidence: str
    label: str
    num_hops: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def label_id(self) -> int:
        return LABEL_TO_ID[self.label]


def normalize_label(label: Any) -> str:
    text = str(label).strip().lower().replace("_", " ")
    text = " ".join(text.split())
    if text not in LABEL_ALIASES:
        raise ValueError(f"Unknown label: {label!r}")
    return LABEL_ALIASES[text]


def _iter_json(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for row in obj:
            yield row
    elif isinstance(obj, dict):
        # Some conversion stats files are dicts; skip non-record dicts.
        if {"claim", "label", "evidence"}.issubset(obj.keys()):
            yield obj
        else:
            raise ValueError(f"{path} is not a record list")
    else:
        raise ValueError(f"Unsupported JSON root in {path}: {type(obj).__name__}")


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_records(path: Path) -> Iterator[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl(path)
    elif suffix == ".json":
        yield from _iter_json(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def _to_sample(row: Dict[str, Any], dataset: str, split: str) -> FactSample:
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"metadata": metadata}
    hop_value = row.get("num_hops", metadata.get("num_hops", -1))
    try:
        num_hops = int(hop_value)
    except (TypeError, ValueError):
        num_hops = -1
    evidence = row.get("evidence", "")
    if isinstance(evidence, list):
        evidence = " ".join(str(part) for part in evidence)
    return FactSample(
        sample_id=str(row.get("id", "")),
        dataset=dataset,
        split=split,
        claim=str(row.get("claim", "")),
        evidence=str(evidence),
        label=normalize_label(row.get("label", "")),
        num_hops=num_hops,
        metadata=metadata,
    )


def split_path(data_dir: Path, dataset: str, split: str) -> Path:
    converted = data_dir / dataset / "converted_data"
    candidates = [
        converted / f"{split}.json",
        converted / f"{split}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find split {dataset}/{split} under {converted}")


def available_splits(data_dir: Path, dataset: str) -> List[str]:
    converted = data_dir / dataset / "converted_data"
    if not converted.exists():
        return []
    splits = []
    for path in sorted(converted.iterdir()):
        if path.suffix.lower() in {".json", ".jsonl"} and path.name != "conversion_stats.json":
            splits.append(path.stem)
    return splits


def load_samples(
    data_dir: Path,
    dataset: str,
    split: str,
    max_samples: Optional[int] = None,
) -> List[FactSample]:
    path = split_path(data_dir, dataset, split)
    rows = []
    for i, row in enumerate(iter_records(path)):
        if max_samples is not None and i >= max_samples:
            break
        rows.append(_to_sample(row, dataset, split))
    return rows


def iter_samples(
    data_dir: Path,
    dataset: str,
    split: str,
    max_samples: Optional[int] = None,
) -> Iterable[FactSample]:
    path = split_path(data_dir, dataset, split)
    for i, row in enumerate(iter_records(path)):
        if max_samples is not None and i >= max_samples:
            break
        yield _to_sample(row, dataset, split)
