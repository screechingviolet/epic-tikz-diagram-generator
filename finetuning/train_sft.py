"""
train_sft.py

Stage 1: Supervised fine-tuning on CoT geometry data.
Run this BEFORE train_grpo.py. The SFT checkpoint at
`Qwen2-0.5B-SFT-geometry/final` is then auto-detected as the GRPO starting
point — see the SFT_CHECKPOINT branch in train_grpo.py.

Defaults are tuned for a *short* warmup. The cot_data CoT is templated
(one stock sentence per constraint), so heavier SFT just locks the model
into those templates and constrains GRPO's exploration. The defaults
below — simple+medium tiers, 1 epoch, 1000-row random cap, dropout 0.1 —
aim to teach the <think>...</think> wrapper without grinding the
templates in.

Examples:
    # Default: cot_simple + cot_medium, 1000 random rows, 1 epoch
    python finetuning/train_sft.py

    # All four files, no cap (heavy warmup — usually too much)
    python finetuning/train_sft.py \
        --datasets cot_simple cot_medium cot_complex cot_merged --max-rows 0

    # Reproducible with a different seed
    python finetuning/train_sft.py --seed 7
"""

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

# ---------------------------------------------------------------------------
# Project layout — make `loss-fn/` importable (hyphenated dirs aren't on
# sys.path by default) and import the shared prompt/model constants so SFT
# and GRPO can never silently drift apart.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "loss-fn"))

from prompts import MODEL_NAME, SYSTEM_PROMPT  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# Default to the simpler tiers only — the templates are the same across
# tiers, so simple+medium are enough to teach the format. cot_complex is
# saved for GRPO so the harder examples are seen *fresh* under reward
# pressure, not after the policy has been baked into the template.
DEFAULT_COT_FILES = (
    "cot_simple",
    "cot_medium",
)

DEFAULT_MAX_ROWS = 1000  # 0 means "no cap"
DEFAULT_SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT cold-start for the geometry task."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_COT_FILES),
        help=(
            "CoT dataset names without .jsonl, resolved against cot_data/ "
            f"(default: {' '.join(DEFAULT_COT_FILES)})."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=(
            f"Cap the SFT set to this many random rows after expansion "
            f"(default: {DEFAULT_MAX_ROWS}). Pass 0 to disable the cap "
            "and train on every row."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for the row sample (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args()


args = _parse_args()


# ---------------------------------------------------------------------------
# DATASET LOADING
# ---------------------------------------------------------------------------
def load_sft_dataset(names: list[str], max_rows: int, seed: int) -> Dataset:
    """Load CoT records, expand each into one row per NL variant, and
    optionally cap to a random subset.

    Each record contributes len(nl_variants) training rows that share the
    same `full_output` target (`<think>...</think>` + primitives). This is
    intentional — it teaches the model that the same target answer should
    be reachable from any phrasing of the same description.

    If `max_rows > 0` and the expanded set exceeds it, we draw a uniform
    random sample of size `max_rows` (seeded for reproducibility). The
    sample is taken *after* expansion so the per-record NL paraphrases
    behave like ordinary training rows in the sampling distribution.
    """
    rows = []
    for name in names:
        path = PROJECT_ROOT / "cot_data" / f"{name}.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue

        records = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        print(f"  {path.name}: {len(records)} records")

        for record in records:
            full_output = record.get("full_output")
            if not full_output:
                continue
            for nl in record["nl_variants"]:
                rows.append(
                    {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": nl},
                        ],
                        "completion": [
                            {"role": "assistant", "content": full_output},
                        ],
                    }
                )

    total = len(rows)
    if max_rows and total > max_rows:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_rows)
        print(
            f"Loaded {total} expanded rows; sampled {len(rows)} "
            f"(seed={seed}) for SFT."
        )
    else:
        print(f"Loaded {total} SFT training examples (no cap applied)")
    return Dataset.from_list(rows)


dataset = load_sft_dataset(args.datasets, args.max_rows, args.seed)


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
peft_config = LoraConfig(
    # Mirrors train_grpo.py (r, alpha, target modules) so the adapter shape
    # is preserved across SFT → GRPO. Dropout is bumped to 0.1 here as a
    # cheap regulariser against the cot_data templates: each constraint
    # type produces a near-identical sentence, which the model would
    # otherwise memorise verbatim. Higher dropout makes that harder.
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Sized against measured cot_data full sequence lengths (chat template +
# system prompt + user turn + assistant <think>+primitives, ~3.5 chars/tok):
#   simple   ≤  750 tok
#   medium   ≤  900 tok
#   merged   ≤  850 tok
#   complex  ≤ 1200 tok   ← drives this setting
# 2048 covers complex max with comfortable headroom for chat-template
# variation across transformers versions.
# Anchor the output directory to the project root so train_grpo's
# SFT_CHECKPOINT auto-detection finds it regardless of which directory
# train_sft.py was launched from.
SFT_OUTPUT_DIR = PROJECT_ROOT / "Qwen2-0.5B-SFT-geometry"
SFT_FINAL_DIR = SFT_OUTPUT_DIR / "final"

sft_args = SFTConfig(
    output_dir=str(SFT_OUTPUT_DIR),
    # 1 epoch on the capped sample is the deliberate "format-only" warmup.
    # On the default ~1000-row cap that's ~60 optimizer steps at the
    # effective batch of 16 — enough to lock in <think>...</think> framing
    # without driving the policy onto the templates. Bump if you uncap.
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    # SFT can use a higher LR than GRPO since the gradient is supervised
    # and well-conditioned; 2e-4 is the standard LoRA range.
    learning_rate=2e-4,
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    max_length=2048,
    # Mask the system prompt + user turn so loss is computed only over the
    # assistant's <think>+primitives. Without this, the gradient is diluted
    # by the (already-known) chat formatting and prompt tokens.
    completion_only_loss=True,
)

# We pass `prompt` + `completion` columns directly (no `dataset_text_field`)
# so SFTTrainer applies the chat template itself and builds completion-only
# labels — flattening to a single text field would defeat the masking.
trainer = SFTTrainer(
    model=MODEL_NAME,
    args=sft_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model(str(SFT_FINAL_DIR))
print(f"SFT done. Checkpoint saved → {SFT_FINAL_DIR}")
