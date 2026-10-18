from __future__ import annotations

import json

from proxcabi.v2.prepare_pairs import build_vitaminc_official_pairs


def test_official_case_grid_becomes_four_gold_semantic_edges(tmp_path) -> None:
    source = tmp_path / "vitaminc.jsonl"
    rows = [
        ("case_1", "claim old", "evidence old", "supports"),
        ("case_2", "claim old", "evidence new", "not enough info"),
        ("case_3", "claim new", "evidence old", "not enough info"),
        ("case_4", "claim new", "evidence new", "supports"),
    ]
    with source.open("w", encoding="utf-8") as handle:
        for sample_id, claim, evidence, label in rows:
            handle.write(json.dumps({
                "id": sample_id, "claim": claim, "evidence": evidence, "label": label,
                "num_hops": 1,
                "metadata": {"case_id": "case", "page": "Page", "revision_type": "real"},
            }) + "\n")
    output = tmp_path / "pairs.jsonl"
    report = build_vitaminc_official_pairs(source, output)
    pairs = [json.loads(line) for line in output.read_text().splitlines()]
    assert report["num_pairs"] == 4
    assert report["transform_types"] == {"semantic_claim": 2, "semantic_evidence": 2}
    assert all(row["metadata"]["verified"] is True for row in pairs)
    assert all(row["metadata"]["verification_method"] == "dataset_gold" for row in pairs)
