"""train_sft_xval.py — SFT for the Gaussian xVal model.

Why this exists
---------------
The existing `train_sft.py` uses TRL's `SFTTrainer`, which assumes a
discrete-token model: it tokenizes text once, applies a `completion_only_loss`
mask, and runs CE. The xVal model needs (input_ids, values) inputs and a
combined CE + Gaussian-NLL loss, so SFTTrainer doesn't fit. This is a
from-scratch loop that does the right thing.

What it does
------------
Per dataset row (selected NL variant):
  1. Build a 3-message conversation (system, user, assistant=ground-truth
     geometry).
  2. Apply the chat template twice — once for [system, user] with
     `add_generation_prompt=True` to get the prompt, and once for the full
     three-message conversation. Verify the prompt tokens are a strict
     prefix of the full tokens; skip examples where the chat template breaks
     prefix consistency (rare, but worth catching).
  3. Encode the full text with `encode_with_values` to produce `(input_ids,
     values)` and build a labels tensor that's `-100` over the prompt
     portion so loss only flows through the assistant turn.
  4. Pad batches, run `gaussian_sft_loss(..., labels=...)`, optimize.

What it does NOT do
-------------------
  * LoRA. Add by wrapping `self.base` in `get_peft_model(...)` before
    constructing `GaussianXValModel`.
  * Resume from checkpoint.
  * Mixed precision. Fine to add — the loss is well-behaved.
  * Validation split.

Usage
-----
    python finetuning/train_sft_xval.py --dataset dataset_simple
    python finetuning/train_sft_xval.py --dataset dataset_complex \\
        --epochs 1 --batch-size 4 --lambda-num 1.0

Run this *before* `train_grpo_xval.py` — the regression head needs at least
some supervised signal before policy gradient noise on a flat reward
landscape can do anything useful.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "finetuning"))

from prompts import MODEL_NAME, SYSTEM_PROMPT  # noqa: E402
from xval import (  # noqa: E402
    GaussianXValModel,
    encode_with_values,
    gaussian_sft_loss,
    setup_tokenizer,
)


SAVE_DIR = PROJECT_ROOT / "Qwen2-0.5B-SFT-xval"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _resolve_dataset(name: str) -> Path:
    for parent in ("curriculum_data", "constraint_data"):
        candidate = PROJECT_ROOT / parent / f"{name}.jsonl"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"dataset {name!r}.jsonl not found under curriculum_data/ or constraint_data/"
    )


def load_rows(name: str) -> list[dict]:
    path = _resolve_dataset(name)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f"[data] {path.relative_to(PROJECT_ROOT)} — {len(rows)} rows")
    return rows


def build_examples(
    tokenizer, rows: list[dict], variant: int, max_length: int,
) -> list[dict]:
    """Convert each row into (input_ids, values, labels) for SFT.

    Skips rows that exceed `max_length` after tokenization, or where the
    chat template breaks prefix-consistency between prompt and full text.
    """
    examples = []
    n_skipped_len = 0
    n_skipped_prefix = 0

    for row in rows:
        if variant >= len(row["nl_variants"]):
            continue
        nl = row["nl_variants"][variant]
        completion = "\n".join(row["geometry"])

        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": nl},
        ]
        full_msgs = prompt_msgs + [
            {"role": "assistant", "content": completion},
        ]

        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False,
        )

        # Sanity: prompt text must be a strict prefix of full text. If it
        # isn't, the chat template inserted/dropped tokens at the boundary
        # and our prompt_len calculation would be off — skip.
        if not full_text.startswith(prompt_text):
            n_skipped_prefix += 1
            continue

        prompt_ids, _ = encode_with_values(prompt_text, tokenizer)
        full_ids, full_values = encode_with_values(full_text, tokenizer)

        # Stronger sanity: prompt ids must be a prefix of full ids.
        prompt_len = int(len(prompt_ids))
        if not torch.equal(full_ids[:prompt_len], prompt_ids):
            n_skipped_prefix += 1
            continue

        if int(full_ids.shape[0]) > max_length:
            n_skipped_len += 1
            continue

        labels = full_ids.clone()
        labels[:prompt_len] = -100  # mask prompt portion

        examples.append({
            "input_ids": full_ids,
            "values":    full_values,
            "labels":    labels,
            "prompt_len": prompt_len,
            "total_len":  int(full_ids.shape[0]),
        })

    print(f"[data] kept {len(examples)} / {len(rows)} rows  "
          f"(skipped {n_skipped_len} for length, {n_skipped_prefix} for prefix mismatch)")
    if not examples:
        raise RuntimeError("No usable training examples after filtering.")
    return examples


class XValSFTDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch: list[dict], pad_token_id: int) -> dict:
    """Right-pad input_ids / values / labels / attention_mask."""
    max_len = max(ex["total_len"] for ex in batch)
    B = len(batch)

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    values = torch.ones((B, max_len), dtype=torch.float32)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)

    for i, ex in enumerate(batch):
        L = ex["total_len"]
        input_ids[i, :L] = ex["input_ids"]
        values[i, :L] = ex["values"]
        labels[i, :L] = ex["labels"]
        attn[i, :L] = 1

    return {
        "input_ids":      input_ids,
        "values":         values,
        "labels":         labels,
        "attention_mask": attn,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   help="Dataset name (no .jsonl). Resolved against "
                        "curriculum_data/ then constraint_data/.")
    p.add_argument("--variant", type=int, default=0,
                   help="Which NL variant of each row to use (default 0).")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024,
                   help="Skip examples that tokenize to more than this many tokens.")
    p.add_argument("--lambda-num", type=float, default=1.0,
                   help="Weight on the Gaussian NLL term (token CE has weight 1).")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=str(SAVE_DIR))
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # ---- Tokenizer + model ----
    tokenizer = setup_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}, base={MODEL_NAME}")
    model = GaussianXValModel(MODEL_NAME, tokenizer).to(device)
    model.train()

    # ---- Data ----
    rows = load_rows(args.dataset)
    examples = build_examples(tokenizer, rows, args.variant, args.max_length)

    dataset = XValSFTDataset(examples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
        drop_last=False,
    )

    # ---- Schedule ----
    n_total = max(1, int(args.epochs * len(loader)))
    n_warmup = int(args.warmup_ratio * n_total)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=n_warmup, num_training_steps=n_total,
    )

    print(f"[init] {len(loader)} batches/epoch  →  {n_total} total steps  "
          f"(warmup {n_warmup})")

    # ---- Loop ----
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    step = 0
    epoch = 0
    done = False
    while not done:
        epoch += 1
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            values = batch["values"].to(device)
            labels = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)

            token_logits, mu, log_sigma = model(
                input_ids, values, attention_mask=attn,
            )
            shifted_labels = labels[:, 1:].contiguous()

            total, t_loss, n_loss = gaussian_sft_loss(
                token_logits, mu, log_sigma, input_ids, values,
                model.num_token_id,
                lambda_num=args.lambda_num,
                labels=shifted_labels,
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % args.log_every == 0 or step == 1:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch} step {step:4d}/{n_total}  "
                    f"loss={total.item():+.4f}  "
                    f"token={t_loss.item():.4f}  "
                    f"num_nll={n_loss.item():+.4f}  "
                    f"lr={lr_now:.2e}",
                    flush=True,
                )

            if step >= n_total:
                done = True
                break

    # ---- Save ----
    out = save_path / "final_state.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name":       MODEL_NAME,
            "vocab_size":       len(tokenizer),
            "args":             vars(args),
        },
        out,
    )
    print(f"[done] saved {out}  ({step} steps)")
    print(f"[done] to load: instantiate GaussianXValModel(MODEL_NAME, tokenizer) "
          "and call .load_state_dict(torch.load(...)['model_state_dict'])")


if __name__ == "__main__":
    main()
