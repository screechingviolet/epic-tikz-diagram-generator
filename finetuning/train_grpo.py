import contextlib
import io
import json
import sys
from pathlib import Path

from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

# ---------------------------------------------------------------------------
# Make the sibling `loss-fn` package importable. Its directory has a hyphen,
# so it isn't importable by default — add it to sys.path and import directly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "loss-fn"))

from loss import check_constraints, parse_fn, BIG_BAD_LOSS  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset loading: expand demo_dataset.jsonl into (prompt, constraints) rows.
# Each scene contributes one training example per natural-language variant.
# ---------------------------------------------------------------------------
DATASET_PATH = PROJECT_ROOT / "data" / "demo_dataset.jsonl"

SYSTEM_PROMPT = (
    "You convert a natural-language description of a geometric diagram into a "
    "list of geometric primitives. Output one primitive per line and nothing "
    "else.\n"
    "\n"
    "Primitives:\n"
    "  point(name: str, x: float, y: float)\n"
    "  line(name: str, p1: str, p2: str)              "
    "# p1, p2 must name previously defined points\n"
    "  circle(name: str, center: str, radius: float)  "
    "# center must name a previously defined point\n"
    "\n"
    "Example 1:\n"
    "Description: Two points, 5 units apart, connected by a line.\n"
    "Output:\n"
    "point(P0, 0, 0)\n"
    "point(P1, 5, 0)\n"
    "line(L0, P0, P1)\n"
    "\n"
    "Example 2:\n"
    "Description: A circle of radius 2 centered at the origin.\n"
    "Output:\n"
    "point(P0, 0, 0)\n"
    "circle(C0, P0, 2)\n"
    "\n"
    "Example 3:\n"
    "Description: A circle of radius 1 sits at one end of a line of length 3.\n"
    "Output:\n"
    "point(P0, 0, 0)\n"
    "point(P1, 3, 0)\n"
    "circle(C0, P0, 1)\n"
    "line(L0, P0, P1)"
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


class MaxRewardCallback(TrainerCallback):
    """Track the max reward seen between log events and inject it into logs.

    GRPOTrainer logs `rewards/<func>/mean` and `rewards/<func>/std` but not the
    max. We feed every reward batch into `record()` from the reward function;
    on each `on_log` call we surface the running max as
    `rewards/reward_constraints/max` (and `reward_max` to mirror the
    aggregate-style key TRL already emits).
    """

    def __init__(self):
        self._max = float("-inf")
        self._count = 0

    def record(self, rewards):
        if not rewards:
            return
        batch_max = max(rewards)
        if batch_max > self._max:
            self._max = batch_max
        self._count += len(rewards)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self._count > 0:
            logs["rewards/reward_constraints/max"] = self._max
            logs["reward_max"] = self._max
        # Reset for the next logging interval.
        self._max = float("-inf")
        self._count = 0


max_reward_cb = MaxRewardCallback()


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


def _structural_validity(pred_geo):
    """Return (all_valid, fraction_valid) over the predicted primitives.

    Walks `pred_geo` in order, building a name table, and counts how many
    primitives have arguments that resolve cleanly:

      point(name, x, y)         → x and y parse as float
      line(name, p1, p2)        → p1 and p2 are previously defined point names
      circle(name, center, r)   → center is a previously defined point name
                                  AND r parses as float

    Mirrors the rules check_constraints uses, but counts partial success
    instead of raising on the first bad reference. This lets the reward
    function award partial credit between the 0.05 and 0.10 tiers, giving
    the model a gradient to climb out of the "structurally invalid" plateau.
    """
    point_names = set()
    valid_count = 0
    total = len(pred_geo)
    if total == 0:
        return False, 0.0

    for prim in pred_geo:
        try:
            head, params = parse_fn(prim)
        except Exception:
            continue

        if head == "point" and len(params) == 3:
            try:
                float(params[1])
                float(params[2])
            except ValueError:
                continue
            point_names.add(params[0])
            valid_count += 1
        elif head == "line" and len(params) == 3:
            if params[1] in point_names and params[2] in point_names:
                valid_count += 1
        elif head == "circle" and len(params) == 3:
            try:
                float(params[2])
            except ValueError:
                continue
            if params[1] in point_names:
                valid_count += 1

    return (valid_count == total), valid_count / total


def reward_constraints(completions, constraints, **kwargs):
    """Per-completion reward: fraction of truth constraints that are satisfied.

    Reward is bounded in [0, 1] on purpose. GRPO normalises advantages within
    each group of `num_generations` samples; large-magnitude outliers (e.g.
    a -100 penalty for one bad parse) blow up the z-score and drown the
    learning signal from the other samples in the group.

    Tier structure (strictly monotone — better-shaped output always scores
    at least as much as worse-shaped output):

      0.00                              unparseable completion (no primitives)
      0.05 + 0.05 * frac_refs_valid     primitives parsed; ramp on the fraction
                                        whose refs resolve. Range (0.05, 0.10).
      0.10                              all refs resolve but truth-side lookup
                                        failed (truth references a shape the
                                        model didn't define).
      0.10 + 0.90 * (score/len(truth))  fully valid; satisfied constraints
                                        pull reward up toward 1.0.

    The 0.05 → 0.10 ramp is the key change vs the previous cliff. Models
    plateau at the 0.05 tier when format is learned but cross-refs aren't;
    the ramp gives a gradient to climb (more valid refs ⇒ higher reward)
    instead of a step the policy has to vault over in a single jump.

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

        all_valid, frac_valid = _structural_validity(pred_geo)
        if not all_valid:
            # Smooth ramp 0.05 → 0.10 on the fraction of primitives whose
            # references resolve cleanly. With frac_valid = 0 we sit at the
            # old 0.05 floor; as more refs become valid the reward rises
            # continuously toward the all-valid tier.
            rewards.append(0.05 + 0.05 * frac_valid)
            continue

        # All references resolve — safe to call check_constraints, which can
        # still raise if a *truth* constraint references a shape the model
        # didn't emit. Swallow its print spam so the trainer log stays clean.
        with contextlib.redirect_stdout(io.StringIO()):
            score = check_constraints(pred_geo, truth)
        if score == BIG_BAD_LOSS:
            # Pre-check passed but check_constraints still raised — almost
            # always a missing-shape lookup against the truth list. Award the
            # all-valid tier; nothing the model can do about which constraints
            # were chosen.
            rewards.append(0.10)
        else:
            denom = max(len(truth), 1)
            rewards.append(0.10 + 0.90 * float(score) / denom)
    max_reward_cb.record(rewards)
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
    model="Qwen/Qwen2-3B-Instruct",
    reward_funcs=reward_constraints,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    callbacks=[max_reward_cb],
)

trainer.train()
