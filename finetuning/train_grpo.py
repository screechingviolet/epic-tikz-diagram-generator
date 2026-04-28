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

from loss import check_constraints, Confusion  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset loading: expand demo_dataset.jsonl into (prompt, constraints) rows.
# Each scene contributes one training example per natural-language variant.
# ---------------------------------------------------------------------------
DATASET_PATH = PROJECT_ROOT / "curriculum_data" / "dataset_complex.jsonl"

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
    "Description: Two points P0 and P1 are 5 units apart and connected by a line L0.\n"
    "Output:\n"
    "point(P0, 0, 0)\n"
    "point(P1, 5, 0)\n"
    "line(L0, P0, P1)\n"
    "\n"
    "Example 2:\n"
    "Description: Circle C0 of radius 2.77 centered around a point P0.\n"
    "Output:\n"
    "point(P0, 0, 0)\n"
    "circle(C0, P0, 2.77)\n"
    "\n"
    "Example 3:\n"
    "Description: A line segment L0 of length 3.4823 has endpoints P0 and P1. A circle C0 is centered at P0 and passes through P1.\n"
    "Output:\n"
    "point(P0, 0, 0)\n"
    "point(P1, 3.4823, 0)\n"
    "circle(C0, P0, 3.4823)\n"
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


class LogReorderCallback(TrainerCallback):
    """Reorder per-step log entries so the most important metrics print first
    and drop redundant / always-zero keys.

    HuggingFace Trainer prints `logs` by iterating its keys, and Python dicts
    preserve insertion order, so reordering the dict in `on_log` reorders the
    printed line.

    Two transformations:
      1. Drop keys starting with any prefix in DROP_PREFIXES — these are
         either per-reward-function duplicates of the aggregate keys
         (`reward`/`reward_std`/`reward_max`) or PPO-clipping internals that
         are always zero in this GRPO setup.
      2. Reorder remaining keys: PRIORITY_KEYS first in the documented order,
         then everything else in its original position.

    Must be registered AFTER any callback that injects new keys (e.g.
    MaxRewardCallback adds `reward_max`), so the new keys are present at the
    time we reorder.
    """

    PRIORITY_KEYS = (
        # Reward signal — the thing you actually watch during training.
        "reward",
        "reward_max",
        "reward_std",
        "frac_reward_zero_std",
        # Optimization health.
        "loss",
        "kl",
        "entropy",
        "grad_norm",
        # Progress.
        "epoch",
        "step_time",
        "learning_rate",
        # Output stats.
        "completions/mean_length",
        "completions/min_length",
        "completions/max_length",
        "completions/clipped_ratio",
        "num_tokens",
    )

    DROP_PREFIXES = (
        "rewards/",     # per-reward-function metrics duplicate the aggregates
        "clip_ratio/",  # PPO-clipping internals; uniformly zero here
    )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        for key in [k for k in logs if k.startswith(self.DROP_PREFIXES)]:
            del logs[key]
        ordered = {key: logs[key] for key in self.PRIORITY_KEYS if key in logs}
        for key, value in logs.items():
            if key not in ordered:
                ordered[key] = value
        logs.clear()
        logs.update(ordered)


class CompletionPeekCallback(TrainerCallback):
    """Periodically print the best-in-batch completion so you can see what
    the model is actually generating during training.

    Use the running per-step reward number to know *whether* training is
    working; use this callback's output to diagnose *why* it isn't.

    Wired in two places: the reward function calls `record()` with each
    batch's (prompts, completions, rewards, truths), and `on_step_end` fires
    every `every_n_steps` to print the best of the most recent batch.
    """

    def __init__(self, every_n_steps: int = 10, max_chars: int = 400):
        self._every_n = every_n_steps
        self._max_chars = max_chars
        self._buffer = []  # list of (prompt_text, completion_text, reward, truth)

    @staticmethod
    def _prompt_text(p):
        # Chat-style prompt: pull the user turn's content.
        if isinstance(p, list):
            return next(
                (m.get("content", "") for m in p if m.get("role") == "user"),
                "",
            )
        return str(p)

    @staticmethod
    def _completion_text(c):
        if isinstance(c, list):
            return "".join(part.get("content", "") for part in c)
        return str(c)

    def record(self, prompts, completions, rewards, truths):
        if prompts is None:
            prompts = [None] * len(completions)
        self._buffer = [
            (self._prompt_text(p), self._completion_text(c), r, t)
            for p, c, r, t in zip(prompts, completions, rewards, truths)
        ]

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step == 0 or step % self._every_n != 0 or not self._buffer:
            return
        p, c, r, truth = max(self._buffer, key=lambda t: t[2])
        sep = "─" * 70
        prompt_line = (p[:160] + "...") if len(p) > 160 else p
        print(f"\n{sep}")
        print(f"[step {step}] best of last batch  reward={r:.3f}")
        print(f"  prompt: {prompt_line}")
        print(f"  truth constraints: {truth}")
        print("  completion:")
        body = c[: self._max_chars].splitlines() or [""]
        for line in body:
            print(f"    {line}")
        if len(c) > self._max_chars:
            print(f"    [... {len(c) - self._max_chars} more chars]")
        print(sep)


max_reward_cb = MaxRewardCallback()
log_reorder_cb = LogReorderCallback()
completion_peek_cb = CompletionPeekCallback(every_n_steps=10)


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
    """Per-completion reward: bounded [0, 1] mapping of `check_constraints`.

    Reward is bounded in [0, 1] on purpose. GRPO normalises advantages within
    each group of `num_generations` samples; large-magnitude outliers blow up
    the z-score and drown the learning signal from the other samples in the
    group.

    `check_constraints` already returns a continuous, monotone score in [0, 3]
    that encodes partial credit at every failure mode — bad parse, bad floats,
    wrong arg count, missing refs (with internal ramp on fraction-valid), and
    success (with internal ramp on fraction-of-constraints-satisfied). We map
    that to [0, 1] by dividing by 3.

    Mapping (from loss.py):
      0.000    parse error / unknown failure          (score 0)
      0.083    wrong arg count                        (score 0.25)
      0.167    bad float                              (score 0.5)
      0.250    type check failed                      (score 0.75)
      [0.333, 0.667]   parseable but missing refs    (score 1 + valid/total)
      [0.667, 1.000]   fully valid; constraint ramp  (score 2 + correct/total)

    `Confusion` is returned if the *truth* list contains an unknown constraint
    label — that's a dataset bug, not the model's fault. We award the floor of
    the success tier (0.667) so a Confusion sentinel doesn't poison the group.

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
            # No extractable primitives — short-circuit to the floor and skip
            # the (defensive) call into check_constraints.
            rewards.append(0.0)
            continue

        names = []
        for p in pred_geo:
            try:
                _, params = parse_fn(p)
                if params:
                    names.append(params[0])
            except Exception:
                pass
        duplicate_penalty = len(set(names)) / max(len(names), 1)

        # Swallow check_constraints' print spam from its exception handlers
        # so the trainer log stays clean. Guard against unexpected raises
        # too — rewards must never crash the trainer. (Known case: if pred_geo
        # contains only points, the KeyError handler in loss.py divides by
        # all_refs == 0.) On any unexpected raise, treat as the parseable
        # floor of the "missing refs" tier — model emitted *something*.
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                score = check_constraints(pred_geo, truth)
        except Exception:
            rewards.append(1.0 / 3.0)
            continue

        if score is Confusion:
            # Dataset-side failure: an unknown constraint label in truth.
            # Award the all-refs-valid floor; nothing the model can do.
            rewards.append(2.0 / 3.0)
        else:
            # score is in [0, 3]; map linearly to [0, 1] and clamp defensively.
            r = float(score) / 3.0
            r = r * (0.5 + 0.5 * duplicate_penalty)
            rewards.append(max(0.0, min(1.0, r)))

    max_reward_cb.record(rewards)
    completion_peek_cb.record(
        kwargs.get("prompts"), completions, rewards, constraints
    )
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
    callbacks=[max_reward_cb, log_reorder_cb, completion_peek_cb],
)

trainer.train()
