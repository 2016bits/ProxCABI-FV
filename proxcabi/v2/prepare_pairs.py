from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml

from proxcabi.data import FactSample, ID_TO_LABEL, load_samples, normalize_label

from .schemas import CounterfactualRecord, validate_record


DEV_NAMESPACE = "FEVER_DEV_V2"
VERIFIED_METHODS = frozenset({"human", "dataset_gold", "rule_plus_human"})


def split_dev_samples(
    samples: Sequence[FactSample],
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    namespace: str = DEV_NAMESPACE,
) -> Dict[str, List[FactSample]]:
    """Deterministically group and label-stratify a development set.

    All rows sharing ``parent_id`` stay together. Groups are ordered by a
    stable SHA-256 digest instead of Python's process-randomized hash.
    """

    if len(ratios) != 3 or any(value < 0 for value in ratios):
        raise ValueError("ratios must contain three non-negative values")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to one")
    grouped: Dict[str, List[FactSample]] = defaultdict(list)
    for sample in samples:
        parent_id = str(sample.metadata.get("parent_id", sample.sample_id)).strip()
        if not parent_id:
            raise ValueError("all dev samples need a sample_id or parent_id")
        grouped[parent_id].append(sample)
    by_label: Dict[str, List[Tuple[str, List[FactSample]]]] = defaultdict(list)
    for parent_id, rows in grouped.items():
        labels = {row.label for row in rows}
        if len(labels) != 1:
            raise ValueError(f"parent {parent_id!r} has mixed base labels: {sorted(labels)}")
        by_label[next(iter(labels))].append((parent_id, rows))

    output = {"dev_select": [], "dev_calib": [], "dev_report": []}
    for label_groups in by_label.values():
        label_groups.sort(key=lambda item: _stable_digest(f"{namespace}|{item[0]}"))
        counts = _largest_remainder_counts(len(label_groups), ratios)
        boundaries = (counts[0], counts[0] + counts[1])
        partitions = (
            label_groups[: boundaries[0]],
            label_groups[boundaries[0] : boundaries[1]],
            label_groups[boundaries[1] :],
        )
        for split_name, partition in zip(output, partitions):
            for _, rows in partition:
                output[split_name].extend(_copy_with_split(row, split_name) for row in rows)
    for rows in output.values():
        rows.sort(key=lambda sample: sample.sample_id)
    assert_disjoint_parents(output)
    return output


def convert_legacy_counterfactuals(
    base_samples: Sequence[FactSample],
    counterfactual_samples: Sequence[FactSample],
    split_origin: str,
) -> Tuple[List[CounterfactualRecord], Dict[str, object]]:
    """Convert legacy contrast rows into audited V2 candidate records.

    Teacher-only filtering is intentionally not promoted to gold verification.
    Only explicit trusted verification methods can produce ``verified=True``.
    """

    bases = {str(sample.sample_id): sample for sample in base_samples}
    records: List[CounterfactualRecord] = []
    rejection_counts: Counter[str] = Counter()
    seen_record_ids: set[str] = set()
    seen_pair_ids: set[str] = set()
    seen_text: set[Tuple[str, str, str]] = set()
    for sample in counterfactual_samples:
        metadata = sample.metadata or {}
        if not sample.sample_id or sample.sample_id in seen_record_ids:
            rejection_counts["duplicate_or_empty_record_id"] += 1
            continue
        parent_id = str(metadata.get("parent_id", metadata.get("source_id", ""))).strip()
        if parent_id not in bases:
            rejection_counts["missing_parent"] += 1
            continue
        base = bases[parent_id]
        same_claim = _normalize_text(sample.claim) == _normalize_text(base.claim)
        same_evidence = _normalize_text(sample.evidence) == _normalize_text(base.evidence)
        if same_claim and same_evidence:
            rejection_counts["original_not_counterfactual"] += 1
            continue
        if same_claim == same_evidence:
            rejection_counts["multiple_or_unknown_changed_fields"] += 1
            continue
        changed_field = "evidence" if same_claim else "claim"
        source_label, target_label = base.label_id, sample.label_id
        prefix = "invariant" if source_label == target_label else "semantic"
        transform_type = f"{prefix}_{changed_field}"
        pair_id = str(
            metadata.get("pair_id")
            or f"{metadata.get('contrast_group', parent_id)}:{metadata.get('contrast_role', sample.sample_id)}"
        ).strip()
        if not pair_id or pair_id in seen_pair_ids:
            rejection_counts["duplicate_pair_id"] += 1
            continue
        text_key = (_normalize_text(sample.claim), _normalize_text(sample.evidence), sample.label)
        if text_key in seen_text:
            rejection_counts["duplicate_text_label"] += 1
            continue
        method = str(metadata.get("verification_method", "")).strip().lower()
        if not method:
            method = "teacher_only" if metadata.get("nli_teacher_model") else "legacy_unverified"
        explicitly_verified = metadata.get("verified") is True
        verified = explicitly_verified and method in VERIFIED_METHODS
        source_hash = _sample_hash(base)
        record = CounterfactualRecord(
            record_id=sample.sample_id,
            parent_id=parent_id,
            claim=sample.claim,
            evidence=sample.evidence,
            source_label=source_label,
            target_label=target_label,
            transform_type=transform_type,
            changed_field=changed_field,
            edit_category=str(metadata.get("edit_category", metadata.get("edit_type", "unknown"))).strip() or "unknown",
            verified=verified,
            verification_method=method,
            split_origin=split_origin,
            pair_id=pair_id,
            generator_version=str(metadata.get("generator_version", metadata.get("counterfactual_mode", "legacy"))),
            source_hash=source_hash,
            evidence_source_id=str(metadata.get("evidence_source_id", metadata.get("page", ""))),
        )
        try:
            validate_record(record)
        except (TypeError, ValueError):
            rejection_counts["schema_validation"] += 1
            continue
        records.append(record)
        seen_record_ids.add(sample.sample_id)
        seen_pair_ids.add(pair_id)
        seen_text.add(text_key)

    audit = _counterfactual_audit(records, len(counterfactual_samples), rejection_counts)
    return records, audit


def prepare_from_config(config_path: Path, output_dir: Path) -> Dict[str, object]:
    """Freeze FEVER dev partitions and convert configured legacy CF pools."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data_cfg = dict(config.get("data") or {})
    paths_cfg = dict(config.get("paths") or {})
    prepare_cfg = dict(config.get("prepare") or {})
    data_dir = Path(paths_cfg.get("data_dir", "data"))
    dataset = str(data_cfg.get("dataset", "FEVER"))
    train_split = str(data_cfg.get("train_split", "train"))
    dev_split = str(prepare_cfg.get("source_dev_split", "dev"))
    cf_train_split = str(prepare_cfg.get("legacy_cf_train_split", "cf_train_hard_nli"))
    cf_dev_split = str(prepare_cfg.get("legacy_cf_dev_split", "cf_dev_hard_nli"))

    base_train = load_samples(data_dir, dataset, train_split)
    base_dev = load_samples(data_dir, dataset, dev_split)
    assert_no_cross_split_leakage({train_split: base_train, dev_split: base_dev})
    dev_parts = split_dev_samples(base_dev)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in dev_parts.items():
        _write_jsonl(output_dir / f"{split_name}.jsonl", (_sample_row(row) for row in rows))

    cf_train = load_samples(data_dir, dataset, cf_train_split)
    train_records, train_audit = convert_legacy_counterfactuals(base_train, cf_train, train_split)
    _write_jsonl(output_dir / f"{dataset.lower()}_cf_train.jsonl", (_record_row(row, dataset) for row in train_records))

    cf_dev = load_samples(data_dir, dataset, cf_dev_split)
    dev_records, dev_audit = convert_legacy_counterfactuals(base_dev, cf_dev, "dev_select")
    select_parents = {_parent_id(sample) for sample in dev_parts["dev_select"]}
    dev_records = [record for record in dev_records if record.parent_id in select_parents]
    dev_audit["accepted_after_dev_select_filter"] = len(dev_records)
    _write_jsonl(
        output_dir / f"{dataset.lower()}_cf_dev_select.jsonl",
        (_record_row(row, dataset) for row in dev_records),
    )

    audit: Dict[str, object] = {
        "schema_version": "v2.0",
        "dataset": dataset,
        "base_counts": {train_split: len(base_train), dev_split: len(base_dev)},
        "dev_split_counts": {key: len(value) for key, value in dev_parts.items()},
        "dev_split_label_counts": {
            key: dict(sorted(Counter(row.label for row in value).items()))
            for key, value in dev_parts.items()
        },
        "counterfactual_train": train_audit,
        "counterfactual_dev_select": dev_audit,
        "warning": (
            "teacher-only and legacy-rule records remain unverified candidates and are not eligible "
            "for semantic loss until independently verified"
        ),
    }
    _write_json(output_dir / "data_audit.json", audit)
    _write_json(
        output_dir / "split_manifest.json",
        {
            "namespace": DEV_NAMESPACE,
            "ratios": [0.6, 0.2, 0.2],
            "source_dev_sha256": _samples_hash(base_dev),
            "files": {
                path.name: _file_sha256(path)
                for path in sorted(output_dir.glob("*.jsonl"))
            },
        },
    )
    return audit


def fit_proxies_from_config(config_path: Path, output_dir: Path) -> Dict[str, object]:
    """Fit and persist V2 structured proxies on the configured base train only."""

    from .structured_proxies import StructuredProxyBuilder

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    paths_cfg = dict(config.get("paths") or {})
    data_cfg = dict(config.get("data") or {})
    proxy_cfg = dict(config.get("proxy") or {})
    if proxy_cfg.get("fit_on", "base_train_only") != "base_train_only":
        raise ValueError("V2 proxies must use proxy.fit_on=base_train_only")
    data_dir = Path(paths_cfg.get("data_dir", "data"))
    sources = data_cfg.get("train_sources")
    if sources:
        samples = []
        for source in sources:
            source = dict(source)
            samples.extend(load_samples(
                data_dir,
                str(source["dataset"]),
                str(source.get("split", "train")),
                max_samples=int(source["max_samples"]) if source.get("max_samples") is not None else None,
            ))
    else:
        dataset = str(data_cfg.get("dataset", "FEVER"))
        split = str(data_cfg.get("train_split", "train"))
        samples = load_samples(data_dir, dataset, split)
    builder = StructuredProxyBuilder(
        lmi_top_k=int(proxy_cfg.get("lmi_top_k", 64)),
        lmi_min_count=int(proxy_cfg.get("lmi_min_count", 5)),
        use_unknown_w=bool(proxy_cfg.get("w_use_unknown_category", True)),
    ).fit(samples, alpha=float(proxy_cfg.get("smoothing_alpha", 1.0)))
    builder.save(output_dir)
    result = {
        "output_dir": str(output_dir),
        "fit_sample_count": builder.fit_sample_count,
        "base_train_source_sha256": builder.base_train_source_sha256,
        "z_dim": builder.z_dim,
        "num_w_classes": builder.num_w_classes,
        "pi0": builder.pi0,
    }
    _write_json(output_dir / "fit_summary.json", result)
    return result


def build_vitaminc_official_pairs(
    input_path: Path,
    output_path: Path,
    split_origin: str = "train",
    max_cases: int | None = None,
) -> Dict[str, object]:
    """Recover dataset-gold VitaminC contrastive edges by official ``case_id``.

    Only rows differing in exactly claim or evidence and having different gold
    labels become semantic pairs. No invariant relation is inferred.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    label_transitions: Counter[str] = Counter()
    seen_cases: set[str] = set()
    current_case = ""
    current_rows: List[Mapping[str, object]] = []
    cases_written = 0

    def emit_case(rows: Sequence[Mapping[str, object]], handle: object) -> int:
        if not rows:
            return 0
        emitted = 0
        ordered = sorted(rows, key=lambda row: str(row.get("id", "")))
        for left_index, source in enumerate(ordered):
            for target in ordered[left_index + 1:]:
                same_claim = _normalize_text(str(source.get("claim", ""))) == _normalize_text(str(target.get("claim", "")))
                same_evidence = _normalize_text(str(source.get("evidence", ""))) == _normalize_text(str(target.get("evidence", "")))
                if same_claim == same_evidence:
                    continue
                source_label = normalize_label(source.get("label", ""))
                target_label = normalize_label(target.get("label", ""))
                if source_label == target_label:
                    counts["same_label_edges_skipped"] += 1
                    continue
                source_meta = dict(source.get("metadata") or {})
                case_id = str(source_meta["case_id"])
                changed_field = "evidence" if same_claim else "claim"
                pair_id = f"VitaminC:{case_id}:{source['id']}->{target['id']}"
                source_digest = hashlib.sha256(json.dumps(
                    [source["id"], source["claim"], source["evidence"], source_label],
                    ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                metadata = {
                    "dataset": "VitaminC",
                    "parent_id": str(source["id"]),
                    "pair_id": pair_id,
                    "case_id": case_id,
                    "transform_type": f"semantic_{changed_field}",
                    "source_label": source_label,
                    "target_label": target_label,
                    "changed_field": changed_field,
                    "edit_category": f"official_revision_{changed_field}",
                    "verified": True,
                    "verification_method": "dataset_gold",
                    "split_origin": split_origin,
                    "generator_version": "vitaminc_official_case_v1",
                    "source_hash": source_digest,
                    "evidence_source_id": str(source_meta.get("page", "")),
                    "wiki_revision_id": str(source_meta.get("wiki_revision_id", "")),
                    "revision_type": str(source_meta.get("revision_type", "unknown")),
                }
                output = {
                    "id": f"pair_{target['id']}_from_{source['id']}",
                    "claim": target["claim"],
                    "evidence": target["evidence"],
                    "label": target_label,
                    "num_hops": int(target.get("num_hops", 1)),
                    "metadata": metadata,
                }
                handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
                counts[f"semantic_{changed_field}"] += 1
                label_transitions[f"{source_label}->{target_label}"] += 1
                emitted += 1
        return emitted

    with input_path.open("r", encoding="utf-8") as source_handle, output_path.open("w", encoding="utf-8") as output_handle:
        for line in source_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str((row.get("metadata") or {}).get("case_id", ""))
            if not case_id:
                raise ValueError(f"VitaminC row {row.get('id')!r} has no case_id")
            if current_case and case_id != current_case:
                emit_case(current_rows, output_handle)
                seen_cases.add(current_case)
                cases_written += 1
                if max_cases is not None and cases_written >= max_cases:
                    current_rows = []
                    break
                if case_id in seen_cases:
                    raise ValueError(f"VitaminC case_id is not contiguous: {case_id}")
                current_rows = []
            current_case = case_id
            current_rows.append(row)
        else:
            if current_rows and (max_cases is None or cases_written < max_cases):
                emit_case(current_rows, output_handle)
                cases_written += 1
    report = {
        "schema_version": "v2.0",
        "source": str(input_path),
        "output": str(output_path),
        "split_origin": split_origin,
        "num_cases": cases_written,
        "num_pairs": counts["semantic_claim"] + counts["semantic_evidence"],
        "transform_types": {key: value for key, value in sorted(counts.items()) if key.startswith("semantic_")},
        "same_label_edges_skipped": counts["same_label_edges_skipped"],
        "label_transitions": dict(sorted(label_transitions.items())),
        "output_sha256": _file_sha256(output_path),
        "verification_method": "dataset_gold",
    }
    _write_json(output_path.parent / f"{output_path.stem}_audit.json", report)
    return report


def assert_disjoint_parents(splits: Mapping[str, Sequence[FactSample]]) -> None:
    owners: Dict[str, str] = {}
    conflicts = []
    for split_name, samples in splits.items():
        for sample in samples:
            parent = _parent_id(sample)
            previous = owners.setdefault(parent, split_name)
            if previous != split_name:
                conflicts.append(f"{parent}:{previous}/{split_name}")
    if conflicts:
        raise ValueError("parent ids cross splits: " + ", ".join(conflicts[:10]))


def assert_no_cross_split_leakage(splits: Mapping[str, Sequence[FactSample]]) -> None:
    """Reject parent or exact normalized input overlap across named splits."""

    parent_owner: Dict[str, str] = {}
    text_owner: Dict[Tuple[str, str], str] = {}
    conflicts = []
    for split_name, samples in splits.items():
        for sample in samples:
            parent = _parent_id(sample)
            text = (_normalize_text(sample.claim), _normalize_text(sample.evidence))
            if parent in parent_owner and parent_owner[parent] != split_name:
                conflicts.append(f"parent:{parent}:{parent_owner[parent]}/{split_name}")
            else:
                parent_owner[parent] = split_name
            if text in text_owner and text_owner[text] != split_name:
                conflicts.append(f"text:{sample.sample_id}:{text_owner[text]}/{split_name}")
            else:
                text_owner[text] = split_name
    if conflicts:
        raise ValueError("cross-split leakage detected: " + ", ".join(conflicts[:10]))


def _counterfactual_audit(
    records: Sequence[CounterfactualRecord],
    candidate_count: int,
    rejections: Counter[str],
) -> Dict[str, object]:
    return {
        "candidate_rows": candidate_count,
        "accepted_candidates": len(records),
        "verified_candidates": sum(record.verified for record in records),
        "rejected_rows": sum(rejections.values()),
        "rejection_reasons": dict(sorted(rejections.items())),
        "transform_types": dict(sorted(Counter(record.transform_type for record in records).items())),
        "label_transitions": dict(sorted(Counter(
            f"{ID_TO_LABEL[record.source_label]}->{ID_TO_LABEL[record.target_label]}"
            for record in records
        ).items())),
        "verification_methods": dict(sorted(Counter(record.verification_method for record in records).items())),
    }


def _record_row(record: CounterfactualRecord, dataset: str) -> Dict[str, object]:
    return {
        "id": record.record_id,
        "claim": record.claim,
        "evidence": record.evidence,
        "label": ID_TO_LABEL[record.target_label],
        "num_hops": -1,
        "metadata": {
            "dataset": dataset,
            "parent_id": record.parent_id,
            "pair_id": record.effective_pair_id,
            "transform_type": record.transform_type,
            "source_label": ID_TO_LABEL[record.source_label],
            "target_label": ID_TO_LABEL[record.target_label],
            "changed_field": record.changed_field,
            "edit_category": record.edit_category,
            "verified": record.verified,
            "verification_method": record.verification_method,
            "split_origin": record.split_origin,
            "generator_version": record.generator_version,
            "source_hash": record.source_hash,
            "evidence_source_id": record.evidence_source_id,
        },
    }


def _sample_row(sample: FactSample) -> Dict[str, object]:
    metadata = dict(sample.metadata)
    metadata["parent_id"] = _parent_id(sample)
    metadata["split_origin"] = sample.split
    return {
        "id": sample.sample_id,
        "claim": sample.claim,
        "evidence": sample.evidence,
        "label": sample.label,
        "num_hops": sample.num_hops,
        "metadata": metadata,
    }


def _copy_with_split(sample: FactSample, split_name: str) -> FactSample:
    metadata = dict(sample.metadata)
    metadata["parent_id"] = _parent_id(sample)
    metadata["source_split"] = sample.split
    return FactSample(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        split=split_name,
        claim=sample.claim,
        evidence=sample.evidence,
        label=sample.label,
        num_hops=sample.num_hops,
        metadata=metadata,
    )


def _largest_remainder_counts(total: int, ratios: Sequence[float]) -> List[int]:
    exact = [total * ratio for ratio in ratios]
    counts = [int(value) for value in exact]
    remaining = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda i: (-(exact[i] - counts[i]), i))
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def _parent_id(sample: FactSample) -> str:
    return str(sample.metadata.get("parent_id", sample.sample_id))


def _normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sample_hash(sample: FactSample) -> str:
    value = [sample.sample_id, sample.claim, sample.evidence, sample.label]
    return _stable_digest(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _samples_hash(samples: Sequence[FactSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update((_sample_hash(sample) + "\n").encode("ascii"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
