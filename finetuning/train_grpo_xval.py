"""train_grpo_xval.py — Group Relative Policy Optimization on the
Gaussian xVal model.

Why this exists
---------------
TRL's `GRPOTrainer` is built around a discrete-token policy: the importance
ratio is computed solely from token log-probs, and the trainer's generation
loop assumes `model.generate()`. The Gaussian xVal model emits a *hybrid*
action at every step: a token (categorical) plus, when the token is `<num>`,
a number (Gaussian). Subclassing TRL's trainer to inject Gaussian log-probs
into the importance ratio is more friction than writing the loop directly,
so this file is a from-scratch GRPO implementation that handles both action
components.

Algorithm
---------
For each prompt p in the dataset:
  1. Sample G completions stochastically (`gaussian_rollout`).
  2. Score each completion with `loss.check_constraints` against the
     ground-truth constraints.
  3. Group-normalize: A_i = (r_i − mean(r)) / (std(r) + eps).
  4. Recompute log-probs of every saved (token, value) pair under the
     *current* policy in a single batched forward pass.
  5. PPO-clipped surrogate objective with the per-step combined log-prob
     (token + Gaussian, when applicable):

        ratio_t = exp(new_log_p_t − old_log_p_t)
        L_t     = − min(ratio_t · A, clip(ratio_t, 1−ε, 1+ε) · A)

  6. Average over completion length, then over completions, then optimize.

What this DOES NOT include
--------------------------
  * KL-to-reference penalty. Add by holding a frozen `GaussianXValModel`,
    computing both new and reference log-probs, and adding a KL term per
    step. Token KL: discrete; Gaussian KL between two Gaussians has a
    closed form. Skipped here for clarity.
  * LoRA. Wrap the base model with `get_peft_model(...)` before
    `GaussianXValModel(...)` if you want PEFT.
  * Multi-GPU / sharding.
  * KV cache during rollout. `gaussian_rollout` is O(N²) per completion.
  * Multiple inner PPO epochs per rollout batch (currently 1).
  * Resume from checkpoint — only "save final adapter" is implemented.

Usage
-----
    python finetuning/train_grpo_xval.py --dataset dataset_simple
    python finetuning/train_grpo_xval.py --dataset dataset_complex \\
        --num-generations 8 --max-new-tokens 256 --steps 200

The expectation is that you'd SFT the Gaussian model with `gaussian_sft_loss`
before running this — RL exploration on a randomly-initialized regression
head is brutal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "loss-fn"))
sys.path.insert(0, str(PROJECT_ROOT / "finetuning"))

from loss import check_constraints  # noqa: E402
from prompts import MODEL_NAME, SYSTEM_PROMPT  # noqa: E402
from xval import (  # noqa: E402
    NUM_TOKEN,
    GaussianXValModel,
    decode_with_values,
    encode_with_values,
    gaussian_log_prob,
    gaussian_rollout,
    setup_tokenizer,
)


SAVE_DIR = PROJECT_ROOT / "Qwen2-0.5B-GRPO-xval"


# ---------------------------------------------------------------------------
# Dataset loading — same conventions as train_grpo.py
# ---------------------------------------------------------------------------

def _resolve_dataset(name: str) -> Path:
    for parent in ("curriculum_data", "constraint_data"):
        candidate = PROJECT_ROOT / parent / f"{name}.jsonl"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"dataset {name!r}.jsonl not found under curriculum_data/ or constraint_data/"
    )


def load_dataset(name: str) -> list[dict]:
    path = _resolve_dataset(name)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f"[data] {path.relative_to(PROJECT_ROOT)} — {len(rows)} rows")
    return rows


def build_prompt(tokenizer, nl: str) -> str:
    """Apply the chat template to (system, user) → raw text we then xVal-encode."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": nl},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Reward — mirrors train_grpo.py's split + score
# ---------------------------------------------------------------------------

def _split_completion(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def reward_for_completion(
    rollout: dict, tokenizer, truth_constraints: list[str],
) -> float:
    """Decode the completion using the per-step values and score it."""
    text = decode_with_values(
        rollout["completion_tokens"],
        rollout["completion_values"],
        tokenizer,
    )
    pred_geo = _split_completion(text)
    try:
        score = check_constraints(pred_geo, truth_constraints)
    except Exception:
        # Reward function should be robust, but if it raises (e.g. malformed
        # output triggers an unhandled case) treat it as zero reward.
        score = 0.0
    if not isinstance(score, (int, float)):
        score = 0.0
    return float(score)


# ---------------------------------------------------------------------------
# Padding + batched log-prob recomputation
# ---------------------------------------------------------------------------

def pad_rollouts(rollouts: list[dict], pad_token_id: int):
    """Pad a list of (variable-length) rollouts into one batched tensor pack.

    All padding is on the *right*. We track per-row prompt length so the
    completion mask can pick out the correct positions later.
    """
    max_len = max(int(r["input_ids"].shape[0]) for r in rollouts)
    B = len(rollouts)

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    values = torch.ones((B, max_len), dtype=torch.float32)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    prompt_lens = torch.zeros((B,), dtype=torch.long)
    full_lens = torch.zeros((B,), dtype=torch.long)

    for i, r in enumerate(rollouts):
        L = int(r["input_ids"].shape[0])
        input_ids[i, :L] = r["input_ids"]
        values[i, :L] = r["values"]
        attn[i, :L] = 1
        prompt_lens[i] = r["prompt_len"]
        full_lens[i] = L

    return input_ids, values, attn, prompt_lens, full_lens


def recompute_step_log_probs(
    model: GaussianXValModel,
    input_ids: torch.Tensor,
    values: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    full_lens: torch.Tensor,
):
    """Run the model once on the padded batch and pull out per-step log-probs
    of the saved (token, value) actions over the *completion* portion.

    Returns
    -------
    step_log_probs   : (B, max_completion_len) — log p(token) + 1[NUM] · log p(value)
    completion_mask  : (B, max_completion_len) — 1 at real completion steps
    """
    B, L = input_ids.shape
    token_logits, mu, log_sigma = model(
        input_ids, values, attention_mask=attention_mask,
    )

    # Position i in the shifted view predicts target i+1.
    log_softmax = F.log_softmax(token_logits[:, :-1], dim=-1)        # (B, L-1, V)
    targets = input_ids[:, 1:]                                        # (B, L-1)
    token_lp = log_softmax.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    value_lp = gaussian_log_prob(
        values[:, 1:], mu[:, :-1], log_sigma[:, :-1],
    )                                                                 # (B, L-1)
    is_num = (targets == model.num_token_id).float()
    step_lp = token_lp + is_num * value_lp                            # (B, L-1)

    # Mask: 1 at completion steps. Shifted position p corresponds to original
    # target position p+1, so completion steps have shifted index in
    # [prompt_len-1, full_len-2] inclusive.
    arange = torch.arange(L - 1, device=input_ids.device).unsqueeze(0).expand(B, -1)
    starts = prompt_lens.unsqueeze(-1) - 1
    ends = full_lens.unsqueeze(-1) - 1                                # exclusive end in shifted space
    completion_mask = ((arange >= starts) & (arange < ends)).float()  # (B, L-1)

    # Slice down to the longest completion's width to keep tensors smaller.
    max_comp = int((full_lens - prompt_lens).max().item())
    out_lp = torch.zeros((B, max_comp), device=input_ids.device, dtype=step_lp.dtype)
    out_mask = torch.zeros((B, max_comp), device=input_ids.device)
    for i in range(B):
        plen = int(prompt_lens[i].item())
        flen = int(full_lens[i].item())
        comp_len = flen - plen
        # Shifted indices [plen-1, flen-2] map to completion indices [0, comp_len-1].
        out_lp[i, :comp_len] = step_lp[i, plen - 1: flen - 1]
        out_mask[i, :comp_len] = 1.0

    return out_lp, out_mask


def stack_old_log_probs(rollouts: list[dict], max_comp: int) -> torch.Tensor:
    """Right-pad the per-step log-probs from rollout into one tensor."""
    B = len(rollouts)
    out = torch.zeros((B, max_comp), dtype=torch.float32)
    for i, r in enumerate(rollouts):
        old = r["token_log_probs"] + r["value_log_probs"]              # (comp_len,)
        out[i, : old.shape[0]] = old
    return out


# ---------------------------------------------------------------------------
# Group-relative advantage
# ---------------------------------------------------------------------------

def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score within each group.

    `rewards` shape: (num_prompts, G). Returns same shape.
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + eps)


# ---------------------------------------------------------------------------
# One optimizer step over a batch of prompts
# ---------------------------------------------------------------------------

def grpo_step(
    model: GaussianXValModel,
    tokenizer,
    optimizer,
    prompts: list[tuple[str, list[str]]],   # list of (chat-templated prompt, truth_constraints)
    n_generations: int,
    max_new_tokens: int,
    temperature: float,
    clip_eps: float,
    device: str,
):
    """Roll out, score, and apply one PPO-clipped GRPO update."""
    # ---- 1. Rollouts (no grad) ----
    rollouts: list[dict] = []
    rewards = torch.zeros((len(prompts), n_generations))
    for p_idx, (prompt_text, truth_constraints) in enumerate(prompts):
        for g in range(n_generations):
            r = gaussian_rollout(
                model, tokenizer, prompt_text,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
            r["_truth_constraints"] = truth_constraints
            r["_group"] = p_idx
            rollouts.append(r)
            rewards[p_idx, g] = reward_for_completion(r, tokenizer, truth_constraints)

    # ---- 2. Advantages (group z-score, broadcast over generations) ----
    A = group_advantages(rewards).flatten()                             # (P*G,)

    # ---- 3. Pad batch + run model once for new log-probs ----
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids, values, attn, prompt_lens, full_lens = pad_rollouts(rollouts, pad_id)
    input_ids = input_ids.to(device)
    values = values.to(device)
    attn = attn.to(device)
    prompt_lens = prompt_lens.to(device)
    full_lens = full_lens.to(device)

    new_step_lp, comp_mask = recompute_step_log_probs(
        model, input_ids, values, attn, prompt_lens, full_lens,
    )                                                                   # (B, max_comp)
    max_comp = new_step_lp.shape[1]
    old_step_lp = stack_old_log_probs(rollouts, max_comp).to(device)    # (B, max_comp)

    # ---- 4. PPO-clipped surrogate ----
    # Per-step ratio. Use a stabilized form: clamp the log-ratio rather than
    # exp'ing huge values directly.
    log_ratio = (new_step_lp - old_step_lp).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    A_b = A.to(device).unsqueeze(-1)                                    # (B, 1)
    surr1 = ratio * A_b
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A_b
    per_step_loss = -torch.min(surr1, surr2)                            # (B, max_comp)

    # Average over completion steps within each rollout, then over rollouts.
    masked = per_step_loss * comp_mask
    per_rollout_loss = masked.sum(dim=-1) / comp_mask.sum(dim=-1).clamp_min(1.0)
    loss = per_rollout_loss.mean()

    # ---- 5. Optimize ----
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # ---- 6. Diagnostics ----
    with torch.no_grad():
        diag = {
            "loss":           loss.item(),
            "reward_mean":    rewards.mean().item(),
            "reward_max":     rewards.max().item(),
            "reward_min":     rewards.min().item(),
            "reward_std":     rewards.std().item(),
            "advantage_abs":  A.abs().mean().item(),
            "ratio_mean":     ratio[comp_mask.bool()].mean().item(),
            "ratio_max":      ratio[comp_mask.bool()].max().item(),
            "frac_zero_std":  ((rewards.std(dim=-1) < 1e-6).float().mean().item()),
            "n_rollouts":     len(rollouts),
            "avg_comp_len":   comp_mask.sum().item() / comp_mask.shape[0],
        }
    return diag


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _format_diag(step: int, d: dict) -> str:
    return (
        f"step {step:4d}  "
        f"loss={d['loss']:+.4f}  "
        f"R={d['reward_mean']:.3f}  Rmax={d['reward_max']:.3f}  "
        f"|A|={d['advantage_abs']:.2f}  "
        f"ratio_max={d['ratio_max']:.2f}  "
        f"zero_std={d['frac_zero_std']:.2f}  "
        f"len={d['avg_comp_len']:.0f}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   help="Dataset name (no .jsonl). Resolved against curriculum_data/ then constraint_data/.")
    p.add_argument("--steps", type=int, default=100,
                   help="Number of GRPO updates to perform.")
    p.add_argument("--prompts-per-step", type=int, default=2,
                   help="Distinct prompts per optimizer step (each gets G generations).")
    p.add_argument("--num-generations", type=int, default=8,
                   help="Generations per prompt (group size G).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=5e-6,
                   help="Lower than typical SFT LR — RL is much noisier.")
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--variant", type=int, default=0,
                   help="Which NL variant of each row to use (default 0).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=str(SAVE_DIR))
    p.add_argument("--resume-from", default=None,
                   help="Path to a state-dict checkpoint (e.g. the SFT save: "
                        "Qwen2-0.5B-SFT-xval/final_state.pt). Strongly "
                        "recommended — the regression head needs supervised "
                        "warm-up before RL.")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # ---- Tokenizer + model ----
    tokenizer = setup_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}, base={MODEL_NAME}")
    model = GaussianXValModel(MODEL_NAME, tokenizer).to(device)
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[init] resume_from: missing={len(missing)} unexpected={len(unexpected)}")
        print(f"[init] loaded weights from {args.resume_from}")
    else:
        print("[init] WARNING: no --resume-from; the regression head is "
              "untrained. Expect ~zero rewards.")
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # ---- Dataset ----
    rows = load_dataset(args.dataset)

    # Pre-build prompts (chat-templated) once per row; pick the requested variant.
    prompt_pool = []
    for row in rows:
        if args.variant >= len(row["nl_variants"]):
            continue
        nl = row["nl_variants"][args.variant]
        prompt_pool.append((build_prompt(tokenizer, nl), row["constraints"]))
    print(f"[init] prompt pool size: {len(prompt_pool)}")

    # ---- Training loop ----
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    rng = torch.Generator().manual_seed(args.seed)
    for step in range(1, args.steps + 1):
        idxs = torch.randint(
            0, len(prompt_pool), (args.prompts_per_step,), generator=rng,
        ).tolist()
        batch = [prompt_pool[i] for i in idxs]
        diag = grpo_step(
            model, tokenizer, optimizer,
            prompts=batch,
            n_generations=args.num_generations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            clip_eps=args.clip_eps,
            device=device,
        )
        print(_format_diag(step, diag), flush=True)

    # ---- Save ----
    out = save_path / "final_state.pt"
    torch.save({"model_state_dict": model.state_dict()}, out)
    print(f"[done] saved {out}")


if __name__ == "__main__":
    main()
