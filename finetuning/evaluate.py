"""End-to-end eval: pick a dataset row, generate, score against the truth.

Usage:
    # Score base model on row 0 of dataset_simple, default NL variant
    python finetuning/evaluate.py --dataset dataset_simple

    # Score the trained adapter on row 3 of dataset_complex, variant 2
    python finetuning/evaluate.py --dataset dataset_complex --row 3 --variant 2 \
        --adapter Qwen2-0.5B-GRPO-geometry

Reuses infer.py's `load_model` + `generate` and loss.py's `check_constraints`,
so the score reported here matches what the trainer's reward function would
have given for the same completion.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "loss-fn"))

from loss import check_constraints, Confusion  # noqa: E402
from prompts import MODEL_NAME  # noqa: E402
from infer import load_model, generate  # noqa: E402

def split_completion(text: str) -> list[str]:
    """Split a completion into per-line tokens — no filtering or fixups.

    Mirrors train_grpo.py's `_split_completion` exactly so the score reported
    here matches what the trainer would have given for the same completion.
    """
    return [line.strip() for line in text.splitlines() if line.strip()]


def _print_block(title: str, lines):
    print(f"{title}:")
    for line in lines:
        print(f"  {line}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Score model output on one dataset row.")
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter dir, or omit for base model.")
    parser.add_argument("--dataset", default="dataset_simple",
                        help="Dataset name (no .jsonl). Resolved against "
                             "curriculum_data/ then constraint_data/.")
    parser.add_argument("--row", type=int, default=0,
                        help="Row index in the dataset (default 0).")
    parser.add_argument("--variant", type=int, default=0,
                        help="Which NL variant of the row to use (default 0).")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    dataset_path = None
    for parent in ("curriculum_data", "constraint_data"):
        candidate = PROJECT_ROOT / parent / f"{args.dataset}.jsonl"
        if candidate.is_file():
            dataset_path = candidate
            break
    if dataset_path is None:
        parser.error(
            f"dataset {args.dataset!r}.jsonl not found under "
            "curriculum_data/ or constraint_data/"
        )
    rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    if not 0 <= args.row < len(rows):
        parser.error(f"--row {args.row} out of range (dataset has {len(rows)} rows)")
    row = rows[args.row]
    if not 0 <= args.variant < len(row["nl_variants"]):
        parser.error(f"--variant out of range (row has {len(row['nl_variants'])} variants)")
    nl = row["nl_variants"][args.variant]
    truth_geo = row["geometry"]
    truth_constr = row["constraints"]

    print(f"[eval] dataset:  {args.dataset}.jsonl  (row {args.row}, variant {args.variant})", file=sys.stderr)
    print(f"[eval] adapter:  {args.adapter or '(none — base model only)'}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(args.adapter)
    completion = generate(model, tokenizer, nl, args.max_new_tokens, args.temperature)

    pred_geo = split_completion(completion)

    with contextlib.redirect_stdout(io.StringIO()):
        score = check_constraints(pred_geo, truth_constr)

    print()
    print("=" * 70)
    _print_block("NL prompt", [nl])
    _print_block("Ground-truth geometry", truth_geo)
    _print_block("Ground-truth constraints", truth_constr)
    _print_block("Model raw completion", completion.splitlines() or [""])
    _print_block("Lines passed to check_constraints", pred_geo or ["(none — empty completion)"])

    if score is Confusion:
        print("Score: Confusion  (truth contains an unknown constraint label)")
    else:
        # Mirror the trainer's [0, 1] mapping (score / 3) so this number is
        # comparable with what you saw during training.
        print(f"Score: {score:.3f} / 3.0   →   reward = {score/3:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
