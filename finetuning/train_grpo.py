import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, PeftModel

# ---------------------------------------------------------------------------
# Make the sibling `loss-fn` package importable. Its directory has a hyphen,
# so it isn't importable by default — add it to sys.path and import directly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "loss-fn"))

from loss import check_constraints, Confusion  # noqa: E402

# Shared with infer.py — single source of truth for the base model string and
# the system prompt the adapter is tuned against.
from prompts import MODEL_NAME, SYSTEM_PROMPT  # noqa: E402

# ---------------------------------------------------------------------------
# Run configuration
#
# SAVE_DIR is stable across runs. The two knobs you'll want to flip on the
# CLI are the dataset and whether to resume from a saved adapter:
#
#   python finetuning/train_grpo.py
#       (default — train dataset_simple from scratch)
#
#   python finetuning/train_grpo.py --dataset dataset_medium
#
#   python finetuning/train_grpo.py --resume-from Qwen2-0.5B-GRPO-geometry
#       (continue training from a previously-saved adapter)
#
# --dataset takes a bare name (e.g. `dataset_simple`, no `.jsonl`) and is
# resolved against `curriculum_data/<name>.jsonl`.
# ---------------------------------------------------------------------------
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
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for the geometry task.")
    parser.add_argument(
        "--dataset",
        default="dataset_simple",
        help="Dataset name without .jsonl (default: dataset_simple). "
             "Resolved against curriculum_data/ then constraint_data/.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a saved adapter directory (from a previous "
             "trainer.save_model() call). If set, training continues from "
             "those weights instead of the base model.",
    )
    return parser.parse_args()


args = _parse_args()
DATASET_PATH = _resolve_dataset_path(args.dataset)
RESUME_FROM = args.resume_from
print(f"[train_grpo] dataset: {DATASET_PATH}")
if RESUME_FROM is not None:
    print(f"[train_grpo] resume-from: {RESUME_FROM}")

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
    """Trim per-step log entries down to the essentials and order them.

    HuggingFace Trainer prints `logs` by iterating its keys, and Python dicts
    preserve insertion order, so rewriting the dict in `on_log` rewrites the
    printed line. We replace the dict with *only* the keys in KEEP — every
    other key TRL emits is dropped, so the line fits in a single terminal
    row.

    The kept keys, in priority order:
      reward                  — headline metric
      reward_max              — best in batch (some sample is doing well)
      frac_reward_zero_std    — exploration health (1.0 ⇒ policy collapsed)
      loss                    — optimization health
      grad_norm               — gradient health
      kl                      — drift from the reference policy
      step_time               — wall time per step

    Dropped (and why): reward_std (covered by frac_reward_zero_std),
    entropy (similar info to kl), epoch (too granular at step level),
    learning_rate (rarely changing meaningfully), all completions/* length
    stats (typically constant), num_tokens (not actionable), and every
    `rewards/<func>/...` and `clip_ratio/...` key.

    Must be registered AFTER MaxRewardCallback, which injects `reward_max`
    into the logs.
    """

    KEEP = (
        "reward",
        "reward_max",
        "frac_reward_zero_std",
        "loss",
        "grad_norm",
        "kl",
        "step_time",
    )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        kept = {key: logs[key] for key in self.KEEP if key in logs}
        logs.clear()
        logs.update(kept)


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


def _split_completion(text: str) -> list[str]:
    """Split a completion into per-line tokens (no filtering, no fixups).

    The model is meant to emit one primitive per line and nothing else.
    Splitting only on newlines and stripping whitespace means any preamble,
    bullet markers, or stray non-primitive lines (e.g. `angle(L0, L1, 51)`)
    survive into check_constraints, which raises ParseError on unknown
    heads → reward 0. That's deliberate: we want the model to learn not to
    waste tokens on commentary or invalid forms, even if a fixup parser
    *could* recover the valid lines.
    """
    return [line.strip() for line in text.splitlines() if line.strip()]


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

        pred_geo = _split_completion(text)
        if not pred_geo:
            # No extractable primitives — short-circuit to the floor and skip
            # the (defensive) call into check_constraints.
            rewards.append(0.0)
            continue

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
    output_dir=SAVE_DIR,
    # ~10–15 min on a single consumer GPU. Long enough to see the reward
    # curve trend up off the floor, short enough to iterate on.
    max_steps=300,
    logging_steps=1,
    # num_generations=2 was effectively giving us one comparison per prompt,
    # so the within-group advantage was almost pure noise. 4 is the sweet
    # spot for a small-model PoC: meaningful relative ranking, still cheap.
    num_generations=8,
    # per_device_train_batch_size must be divisible by num_generations.
    # 4 samples = 1 unique prompt × 4 generations per device step.
    per_device_train_batch_size=8,
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
    temperature=1.2,
    # KL coefficient against the reference policy. Default is 0.04; making
    # it explicit so it's obvious where to dial if the policy drifts.
    beta=0.04,
    bf16=True,
    # Periodic crash-safety checkpoints. With max_steps=150 and save_steps=25
    # we get ~6 checkpoints over a run; save_total_limit=2 keeps only the two
    # most recent on disk so the output_dir doesn't bloat. These are full HF
    # Trainer checkpoints (preserve optimizer state + step count) and live in
    # `<output_dir>/checkpoint-{step}/`. The final `trainer.save_model()` at
    # the end writes the LoRA adapter to `<output_dir>/` (top level) — the
    # two coexist without conflict.
    save_strategy="steps",
    save_steps=25,
    save_total_limit=2,
    report_to="none",
)

# ---------------------------------------------------------------------------
# Model construction
#
# Two branches:
#   * Fresh run — pass the base-model name string and a peft_config; trl wraps
#     the model in a fresh LoRA adapter for us.
#   * Resume run — load the base model + the saved adapter ourselves (with
#     is_trainable=True so optimizer steps actually update it), then pass the
#     PeftModel directly to the trainer with peft_config=None (the adapter is
#     already attached, we don't want trl to add another one on top).
# ---------------------------------------------------------------------------
if RESUME_FROM is not None:
    print(f"[train_grpo] resuming from saved adapter at {RESUME_FROM!r}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model_for_trainer = PeftModel.from_pretrained(
        base, RESUME_FROM, is_trainable=True
    )
    trainer_peft_config = None
else:
    model_for_trainer = MODEL_NAME
    trainer_peft_config = peft_config

trainer = GRPOTrainer(
    model=model_for_trainer,
    reward_funcs=reward_constraints,
    args=training_args,
    train_dataset=dataset,
    peft_config=trainer_peft_config,
    callbacks=[max_reward_cb, log_reorder_cb, completion_peek_cb],
)

# Crash-recovery resume. Two distinct mechanisms now coexist:
#   * RESUME_FROM (above) — load a previously saved *final* adapter and start
#     training fresh from those weights (no optimizer state, step count = 0).
#     Used to continue work across separate Colab sessions.
#   * resume_from_checkpoint — pick up a *partial* run from an HF Trainer
#     checkpoint in SAVE_DIR (full optimizer state + step count). Used when
#     a single run is interrupted (e.g. Colab disconnect) and we want to
#     pick up exactly where we left off.
# If RESUME_FROM is set the user has explicitly chosen the starting weights,
# so don't second-guess them by also auto-resuming a stale checkpoint.
if RESUME_FROM is None and Path(SAVE_DIR).is_dir():
    last_checkpoint = get_last_checkpoint(SAVE_DIR)
    if last_checkpoint is not None:
        print(f"[train_grpo] resuming partial run from {last_checkpoint!r}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
else:
    trainer.train()

# Persist the final adapter. With LoRA + PEFT, trainer.save_model() writes
# only the adapter weights (~MBs), not a merged copy of the base model. Use
# the same directory in RESUME_FROM next run to continue training from here.
trainer.save_model(SAVE_DIR)
print(f"[train_grpo] saved adapter to {SAVE_DIR!r}")
