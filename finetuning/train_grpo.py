import json
import sys
from pathlib import Path

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

# ---------------------------------------------------------------------------
# Make the sibling `loss-fn` package importable. Its directory has a hyphen,
# so it isn't importable by default — add it to sys.path and import directly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "loss-fn"))

from loss import check_constraints, BIG_BAD_LOSS  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset loading: expand demo_dataset.jsonl into (prompt, constraints) rows.
# Each scene contributes one training example per natural-language variant.
# ---------------------------------------------------------------------------
DATASET_PATH = PROJECT_ROOT / "data" / "demo_dataset.jsonl"

SYSTEM_PROMPT = (
    "You are a geometry diagram generator. Given a natural language "
    "description of a diagram, output a list of geometry primitives in the "
    "EPIC GEOMETRY LANGUAGE.\n"
    "Available primitives (one per line, nothing else):\n"
    "  point(name, x, y)\n"
    "  line(name, point_1_name, point_2_name)\n"
    "  circle(name, center_point_name, radius)"
)


def load_geometry_dataset(path: Path) -> Dataset:
    """Return a HF Dataset with chat-style prompts and per-example constraints."""
    rows = []
    with open(path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            constraints = record["constraints"]
            for nl in record["nl_variants"]:
                rows.append(
                    {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": nl},
                        ],
                        "constraints": constraints,
                    }
                )
    return Dataset.from_list(rows)


dataset = load_geometry_dataset(DATASET_PATH)

# ---------------------------------------------------------------------------
# Custom reward function: wraps check_constraints() from loss-fn/loss.py.
# Reward is bounded in [0, 1] (see docstring) so GRPO's group-normalised
# advantages stay well-conditioned even with small num_generations.
# ---------------------------------------------------------------------------
PRIMITIVE_PREFIXES = ("point(", "line(", "circle(")
_LIST_PREFIXES = ("- ", "* ", "+ ")


def _parse_completion_to_geometry(text: str) -> list[str]:
    """Extract primitive calls (point/line/circle …) from a completion string."""
    primitives = []
    for raw in text.replace(";", "\n").splitlines():
        token = raw.strip().rstrip(",")
        if not token:
            continue
        # Strip common bullet / numbered list prefixes a chat model might emit.
        for prefix in _LIST_PREFIXES:
            if token.startswith(prefix):
                token = token[len(prefix):].strip()
                break
        if len(token) > 2 and token[0].isdigit() and token[1] in (".", ")"):
            token = token[2:].strip()
        if token.startswith(PRIMITIVE_PREFIXES) and token.endswith(")"):
            primitives.append(token)
    return primitives


def reward_constraints(completions, constraints, **kwargs):
    """Per-completion reward: fraction of truth constraints that are satisfied.

    Reward is bounded in [0, 1] on purpose. GRPO normalises advantages within
    each group of `num_generations` samples; large-magnitude outliers (e.g.
    a -100 penalty for one bad parse) blow up the z-score and drown the
    learning signal from the other samples in the group.

    Tier structure (strictly monotone — better-shaped output always scores
    at least as much as worse-shaped output):

      0.00              unparseable completion (no point/line/circle primitives)
      0.05              primitives parsed but structurally invalid (bad refs,
                        wrong arity, etc.) — loss.py raised an exception
      0.10 + 0.90 * (score/len(truth))    well-formed; floor of 0.10 so a
                                          valid-but-zero-constraints completion
                                          still beats a malformed one

    Args:
        completions: list of model completions. With a chat-style prompt these
            arrive as lists of message dicts; with a plain prompt they arrive
            as strings. Both shapes are handled.
        constraints: per-example column from the dataset — a list of lists of
            constraint strings (e.g. ["length(L0, 3.886)", "tangent(L0, C1)"]).
    """
    rewards = []
    for completion, truth in zip(completions, constraints):
        if isinstance(completion, list):
            text = "".join(part.get("content", "") for part in completion)
        else:
            text = completion

        pred_geo = _parse_completion_to_geometry(text)
        if not pred_geo:
            rewards.append(0.0)
            continue

        score = check_constraints(pred_geo, truth)
        if score == BIG_BAD_LOSS:
            # loss.check_constraints returns BIG_BAD_LOSS as an error sentinel
            # (e.g. a line referencing an undefined point). Give a tiny credit
            # for at least emitting parseable primitives so the model has a
            # gradient to climb away from total garbage.
            rewards.append(0.05)
        else:
            denom = max(len(truth), 1)
            # Floor at 0.10 so well-formed geometry always outranks malformed
            # output, even when zero truth constraints happen to be satisfied.
            rewards.append(0.10 + 0.90 * float(score) / denom)
    return rewards


# ---------------------------------------------------------------------------
# Training setup
# ---------------------------------------------------------------------------
peft_config = LoraConfig(
    # r bumped from 8 → 16 (and alpha kept at 2*r) to give the adapter a
    # bit more capacity. The geometry task needs to memorise a small but
    # non-trivial output schema, and r=8 was leaving capacity on the table.
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO-geometry",
    # ~10–15 min on a single consumer GPU. Long enough to see the reward
    # curve trend up off the floor, short enough to iterate on.
    max_steps=150,
    logging_steps=1,
    # num_generations=2 was effectively giving us one comparison per prompt,
    # so the within-group advantage was almost pure noise. 4 is the sweet
    # spot for a small-model PoC: meaningful relative ranking, still cheap.
    num_generations=4,
    # per_device_train_batch_size must be divisible by num_generations.
    # 4 samples = 1 unique prompt × 4 generations per device step.
    per_device_train_batch_size=4,
    # Bumps the effective batch to 8 samples = 2 unique prompts per
    # optimizer step, which smooths the gradient noticeably.
    gradient_accumulation_steps=2,
    # Geometry blocks reach ~80–120 tokens; 384 leaves headroom without
    # paying for tokens we won't generate.
    max_completion_length=384,
    # 1e-5 was way too conservative for LoRA on a 0.5B model. 5e-5 is the
    # standard LoRA range and gets us measurable movement inside 150 steps.
    learning_rate=5e-5,
    warmup_ratio=0.1,         # smooths the higher LR through early steps
    # Diverse generations are essential for GRPO — without spread inside
    # the group, advantages collapse to zero. 1.0 is a safe explicit value.
    temperature=1.0,
    # KL coefficient against the reference policy. Default is 0.04; making
    # it explicit so it's obvious where to dial if the policy drifts.
    beta=0.04,
    bf16=True,
    save_strategy="no",
    report_to="none",
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_constraints,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()
