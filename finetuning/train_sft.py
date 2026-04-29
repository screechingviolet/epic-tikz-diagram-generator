"""Supervised fine-tune as a warm-start before GRPO.

Trains the model on (NL prompt → ground-truth geometry) pairs with standard
cross-entropy loss. The job here is *not* to make the model great at geometry
— GRPO does that — it's to break biases and establish baseline format so
GRPO has something to refine instead of explore from scratch.

Specifically useful for breaking the few-shot bias:
  * SYSTEM_PROMPT examples all place points at the origin and align lines
    with the x-axis (P0=(0,0), P1=(5,0), L0=P0–P1).
  * The dataset's ground-truth geometries use random non-axis-aligned coords
    that *do* satisfy the constraints.
  * SFT shows the model "for prompts like this, here are valid coords" — the
    model learns to vary placement instead of defaulting to the few-shot
    pattern.

Usage:
    # Baseline format / break few-shot bias on the simplest dataset
    python finetuning/train_sft.py --dataset dataset_simple

    # Constraint-specific warm-up before GRPO on the same dataset
    python finetuning/train_sft.py --dataset dataset_line_tangent \
        --resume-from Qwen2-0.5B-GRPO-geometry

The adapter saves to the same SAVE_DIR train_grpo.py uses, so you can chain:
    python finetuning/train_sft.py --dataset dataset_simple
    python finetuning/train_grpo.py --dataset dataset_simple \
        --resume-from Qwen2-0.5B-GRPO-geometry
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "finetuning"))

from prompts import MODEL_NAME, SYSTEM_PROMPT  # noqa: E402

SAVE_DIR = "Qwen2-0.5B-GRPO-geometry"
_DATASET_DIRS = ("curriculum_data", "constraint_data")


def _resolve_dataset_path(name: str) -> Path:
    """Resolve <name>.jsonl under curriculum_data/ or constraint_data/."""
    for parent in _DATASET_DIRS:
        candidate = PROJECT_ROOT / parent / f"{name}.jsonl"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No dataset {name!r}.jsonl found under "
        + " or ".join(f"{d}/" for d in _DATASET_DIRS)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT warm-start for the geometry task.")
    parser.add_argument(
        "--dataset",
        default="dataset_simple",
        help="Dataset name without .jsonl (default: dataset_simple). "
             "Resolved against curriculum_data/ then constraint_data/.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a saved adapter directory. If set, SFT continues from "
             "those weights instead of the base model.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
        help="Number of epochs (default: 1.0). One pass is usually enough — "
             "more starts memorising specific coordinates and hurts GRPO "
             "exploration later.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="LoRA SFT learning rate (default: 5e-5).",
    )
    return parser.parse_args()


args = _parse_args()
DATASET_PATH = _resolve_dataset_path(args.dataset)
RESUME_FROM = args.resume_from
print(f"[train_sft] dataset: {DATASET_PATH}")
print(f"[train_sft] epochs:  {args.epochs}")
if RESUME_FROM is not None:
    print(f"[train_sft] resume-from: {RESUME_FROM}")


def load_sft_dataset(path: Path) -> Dataset:
    """Build a conversational HF Dataset of (system + user → assistant) triples.

    Each row of the source dataset has 5 NL variants and one ground-truth
    geometry; we expand to 5 SFT rows by pairing every variant with the same
    target geometry. SFTTrainer auto-detects the conversational ("messages")
    format and applies the chat template at training time.
    """
    rows = []
    with open(path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            target = "\n".join(record["geometry"])
            for nl in record["nl_variants"]:
                rows.append(
                    {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": nl},
                            {"role": "assistant", "content": target},
                        ],
                    }
                )
    return Dataset.from_list(rows)


dataset = load_sft_dataset(DATASET_PATH)
print(f"[train_sft] dataset rows: {len(dataset)}")


# ---------------------------------------------------------------------------
# LoRA config — matches train_grpo.py so the saved adapter slots directly
# into a follow-up GRPO run via --resume-from.
# ---------------------------------------------------------------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir=SAVE_DIR,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=args.learning_rate,
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    # Loss only on the assistant turn, not the system+user prompt — we don't
    # want to teach the model to generate the system prompt back at us.
    completion_only_loss=True,
    save_strategy="no",
    report_to="none",
)


# ---------------------------------------------------------------------------
# Model construction — same branching as train_grpo.py so the SFT and GRPO
# resume paths look identical to the user.
# ---------------------------------------------------------------------------
if RESUME_FROM is not None:
    print(f"[train_sft] loading base + adapter from {RESUME_FROM!r}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model_for_trainer = PeftModel.from_pretrained(
        base, RESUME_FROM, is_trainable=True
    )
    trainer_peft_config = None
else:
    model_for_trainer = MODEL_NAME
    trainer_peft_config = peft_config

trainer = SFTTrainer(
    model=model_for_trainer,
    args=training_args,
    train_dataset=dataset,
    peft_config=trainer_peft_config,
)

trainer.train()
trainer.save_model(SAVE_DIR)
print(f"[train_sft] saved adapter to {SAVE_DIR!r}")
