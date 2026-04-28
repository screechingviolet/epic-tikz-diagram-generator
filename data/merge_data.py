import json
import random
from pathlib import Path

def merge_and_shuffle_constraint_data(
    input_dir: str = "constraint_data",
    output_path: str = "constraint_data/dataset_merged.jsonl",
    max_per_type: int = 200,
):
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    all_records = []

    for constraint_type in ["parallel", "perpendicular", "angle",
                            "line_tangent", "circle_tangent", "point_on_circle"]:
        path = input_dir / f"dataset_{constraint_type}.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue

        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        taken = records[:max_per_type]
        all_records.extend(taken)
        print(f"  {constraint_type}: {len(taken)}/{len(records)} records taken")

    random.shuffle(all_records)

    with open(output_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print(f"\nDone. {len(all_records)} total records → {output_path}")


if __name__ == "__main__":
    merge_and_shuffle_constraint_data(
        input_dir="constraint_data",
        output_path="constraint_data/dataset_merged.jsonl",
        max_per_type=200,
    )