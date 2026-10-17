import json
from pathlib import Path

train = json.loads(
    Path("data/FEVER/converted_data/train.json").read_text(encoding="utf-8")
)
pilot_parent_ids = {str(row["id"]) for row in train[:96]}

pool_path = Path("outputs/v2/frozen_data/fever_cf_train.jsonl")
output_path = Path("annotations_train.jsonl")

tasks = []
for line in pool_path.read_text(encoding="utf-8").splitlines():
    if not line:
        continue
    row = json.loads(line)
    meta = row["metadata"]
    if str(meta["parent_id"]) not in pilot_parent_ids:
        continue
    tasks.append({
        "pair_id": meta["pair_id"],
        "parent_id": meta["parent_id"],
        "claim": row["claim"],
        "evidence": row["evidence"],
        "source_label": meta["source_label"],
        "target_label": meta["target_label"],
        "transform_type": meta["transform_type"],
        "verified": False,
        "verification_method": "human",
        "notes": "TODO: manually verify"
    })

with output_path.open("w", encoding="utf-8") as handle:
    for task in tasks:
        handle.write(json.dumps(task, ensure_ascii=False) + "\n")

print(f"Wrote {len(tasks)} tasks to {output_path}")