#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import iter_jsonl, metadata, sha256, write_json


def normalized(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate P0 pair pools and cross-split leakage")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--invariant", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v2/paper/pairs"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    paths = {"train": args.train, "invariant_eval": args.invariant, "semantic_eval": args.semantic}
    errors, warnings = [], []
    ids_by_split, texts_by_split = defaultdict(set), defaultdict(set)
    token_rows = defaultdict(list)
    stats = {}
    all_pair_ids = set()
    for role, path in paths.items():
        counts, transitions, groups = Counter(), Counter(), Counter()
        for line_number, row in enumerate(iter_jsonl(path), 1):
            meta = metadata(row)
            pair_id = str(meta.get("pair_id", row.get("pair_id", ""))).strip()
            source_id = str(meta.get("parent_id", row.get("source_id", ""))).strip()
            transform = str(meta.get("transform_type", row.get("transformation_type", "")))
            source_label = str(meta.get("source_label", row.get("label_original", "")))
            target_label = str(meta.get("target_label", row.get("label_changed", row.get("label", ""))))
            verification = str(meta.get("verification_method", row.get("verification_source", ""))).strip()
            if not pair_id or not source_id or not transform or not verification:
                errors.append(f"{path}:{line_number}: missing provenance fields")
            if pair_id in all_pair_ids:
                errors.append(f"duplicate pair_id across pools: {pair_id}")
            all_pair_ids.add(pair_id)
            if meta.get("verified", row.get("verification_status")) is not True:
                errors.append(f"{path}:{line_number}: pair is not explicitly verified")
            is_inv = transform.startswith("invariant")
            is_sem = transform.startswith("semantic")
            if role == "invariant_eval" and not is_inv:
                errors.append(f"{path}:{line_number}: non-invariant row in invariant pool")
            if role == "semantic_eval" and not is_sem:
                errors.append(f"{path}:{line_number}: non-semantic row in semantic pool")
            if is_inv and source_label != target_label:
                errors.append(f"{path}:{line_number}: invariant label changed")
            if is_sem and source_label == target_label:
                errors.append(f"{path}:{line_number}: semantic label did not change")
            ids_by_split[role].add(source_id)
            texts_by_split[role].add((normalized(row.get("claim")), normalized(row.get("evidence"))))
            tokens = frozenset(re.findall(r"\w+", normalized(row.get("claim", "")) + " " + normalized(row.get("evidence", ""))))
            token_rows[role].append((pair_id, tokens))
            counts["invariant" if is_inv else "semantic" if is_sem else "unknown"] += 1
            transitions[f"{source_label}->{target_label}"] += 1
            groups[str(meta.get("edit_category", row.get("pair_group", "ungrouped")))] += 1
        stats[role] = {"path": str(path), "sha256": sha256(path), "counts": dict(counts),
                       "transitions": dict(transitions), "groups": dict(groups)}
    overlaps = {}
    roles = list(paths)
    for i, left in enumerate(roles):
        for right in roles[i + 1:]:
            key = f"{left}__{right}"
            source_overlap = ids_by_split[left] & ids_by_split[right]
            text_overlap = texts_by_split[left] & texts_by_split[right]
            overlaps[key] = {"source_id_count": len(source_overlap), "normalized_text_count": len(text_overlap)}
            if source_overlap or text_overlap:
                errors.append(f"cross-pool leakage {key}: source={len(source_overlap)}, text={len(text_overlap)}")
            near = near_duplicates(token_rows[left], token_rows[right], threshold=0.9)
            overlaps[key]["near_duplicate_count_at_jaccard_0.9"] = len(near)
            overlaps[key]["near_duplicate_examples"] = near[:20]
            if near and "train" in {left, right}:
                errors.append(f"cross-pool near-duplicate leakage {key}: {len(near)} pairs at Jaccard>=0.9")
            elif near:
                warnings.append(
                    f"evaluation pools contain {len(near)} cross-dataset near-duplicates for {key}; "
                    "reported separately and not used for model selection"
                )
    report = {"status": "pass" if not errors else "fail", "stats": stats,
              "overlaps": overlaps, "errors": errors, "warnings": warnings}
    write_json(args.output_dir / "validation_report.json", report)
    write_json(args.output_dir.parent / "protocol" / "leakage_report.json", report)
    if args.strict and errors:
        raise SystemExit(f"pair validation failed with {len(errors)} errors")


def near_duplicates(
    left: list[tuple[str, frozenset[str]]],
    right: list[tuple[str, frozenset[str]]],
    threshold: float,
) -> list[dict]:
    """Candidate with deterministic MinHash bands, verify with exact token Jaccard."""

    right_bands = defaultdict(list)
    for pair_id, tokens in right:
        for band in minhash_bands(tokens):
            right_bands[band].append((pair_id, tokens))
    seen, output = set(), []
    for left_id, left_tokens in left:
        candidates = {}
        for band in minhash_bands(left_tokens):
            for right_id, right_tokens in right_bands.get(band, []):
                candidates[right_id] = right_tokens
        for right_id, right_tokens in candidates.items():
            key = (left_id, right_id)
            if key in seen:
                continue
            seen.add(key)
            union = left_tokens | right_tokens
            score = len(left_tokens & right_tokens) / len(union) if union else 1.0
            if score >= threshold:
                output.append({"left_pair_id": left_id, "right_pair_id": right_id, "jaccard": score})
    return sorted(output, key=lambda row: (-row["jaccard"], row["left_pair_id"], row["right_pair_id"]))


def minhash_bands(tokens: frozenset[str]) -> list[tuple[int, tuple[int, ...]]]:
    if not tokens:
        return [(0, (0,))]
    values = []
    for seed in range(16):
        values.append(min(
            int.from_bytes(hashlib.sha256(f"{seed}|{token}".encode()).digest()[:8], "big")
            for token in tokens
        ))
    return [(band, tuple(values[band * 4:(band + 1) * 4])) for band in range(4)]


if __name__ == "__main__":
    main()

