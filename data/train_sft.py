"""
train_sft.py

Stage 1: Supervised fine-tuning on CoT geometry data.
Run this BEFORE train_grpo.py. The SFT checkpoint is then used as the
starting point for GRPO instead of the raw pretrained model.
"""

import json
import sys
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "loss-fn"))

# ---------------------------------------------------------------------------
# SYSTEM PROMPT  (same as GRPO so the model sees consistent formatting)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You convert a natural-language description of a geometric diagram into a "
    "list of geometric primitives. Output one primitive per line and nothing "
    "else.\n"
    "\n"
    "Primitives:\n"
    "  point(name: str, x: float, y: float)\n"
    "  line(name: str, p1: str, p2: str)\n"
    "  circle(name: str, center: str, radius: float)\n"
    "\n"
    "Think through the construction step by step inside <think>...</think> "
    "tags, then output the primitives.\n"
    "\n"
    "Example:\n"
    "Description: A circle with radius 3 and a tangent line.\n"
    "Output:\n"
    "<think>\n"
    "I need a circle and a tangent line.\n"
    "I'll place the center at A = (0, 0).\n"
    "For tangency, the line must be distance 3 from A.\n"
    "Tangent point at (3, 0), line is vertical through it.\n"
    "</think>\n"
    "point(A, 0.0000, 0.0000)\n"
    "circle(ω0, A, 3.0000)\n"
    "point(B, 3.0000, 1.0000)\n"
    "point(C, 3.0000, -1.0000)\n"
    "line(la, B, C)"
)

# ---------------------------------------------------------------------------
# DATASET LOADING
# ---------------------------------------------------------------------------

def load_sft_dataset(paths: list[str]) -> Dataset:
    rows = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue

        # read all records first so we can slice
        records = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

        # take only the latter 30%
        start = int(len(records) * 0.70)
        records = records[start:]
        print(f"  {path.name}: using {len(records)} records "
              f"(latter 30% of {start + len(records)} total)")

        for record in records:
            full_output = record.get("full_output")
            if not full_output:
                continue
            for nl in record["nl_variants"]:
                rows.append({
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": nl},
                    ],
                    "completion": [
                        {"role": "assistant", "content": full_output},
                    ],
                })

    print(f"Loaded {len(rows)} SFT training examples")
    return Dataset.from_list(rows)

COT_DATA_PATHS = [
    "cot_data/cot_simple.jsonl",
    "cot_data/cot_medium.jsonl",
    "cot_data/cot_complex.jsonl",
    "cot_data/cot_merged.jsonl",
]

dataset = load_sft_dataset(COT_DATA_PATHS)

# ---------------------------------------------------------------------------
# FORMATTING: convert chat messages to a single string the SFT trainer
# can compute loss on. Loss is only computed on the assistant turn.
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

def format_example(example):
    """Apply chat template and mark which tokens to train on."""
    messages = example["prompt"] + example["completion"]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = dataset.map(format_example)

# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

sft_args = SFTConfig(
    output_dir="Qwen2-0.5B-SFT-geometry",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,         # SFT can use higher LR than GRPO
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    dataset_text_field="text",  # matches the key we set in format_example
    max_length=512,
)

trainer = SFTTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    args=sft_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model("Qwen2-0.5B-SFT-geometry/final")
print("SFT done. Checkpoint saved → Qwen2-0.5B-SFT-geometry/final")