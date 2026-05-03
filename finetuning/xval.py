"""xval.py — xVal-style continuous number encoding for the geometry task.

Reference: Golkar et al., "xVal: A Continuous Number Encoding for Large
Language Models" (2023). https://arxiv.org/abs/2310.02989

Idea
----
Instead of letting the BPE tokenizer splinter every number into 1–3 digit
sub-pieces (which destroys magnitude information), replace every numerical
literal in the text with a single special token `<num>` and pair the token
sequence with a parallel real-valued tensor `values` whose entries hold the
actual number at each `<num>` position (and 1.0 elsewhere).

The model then:

  * scales its input embeddings element-wise by `values` — so the `<num>`
    embedding carries its magnitude into the residual stream;
  * uses the LM head as usual to predict the next *token*;
  * uses an extra regression head (small MLP off the final hidden state) to
    predict the next *number* whenever the next token is `<num>`.

Training loss is CE over tokens + λ · MSE over numbers (masked to positions
where the gold next token is `<num>`).

What is and isn't here
----------------------
This file contains: tokenizer setup, encode/decode helpers, the XValModel
wrapper, a combined-loss helper, a single-batch generate loop, and a
smoke test that loads Qwen2-0.5B and runs one forward + one decode step.

What is intentionally NOT here:
  * Full SFT training loop — easy to add on top of `compute_loss` or
    `gaussian_sft_loss`.
  * LoRA — wrap the base model in `get_peft_model(...)` before instantiating
    the model if you want PEFT.
  * KV cache. `generate()` re-runs the full prefix every step (O(N²)). Fine
    for the smoke test, slow for long generations. Adding cache support is
    mechanical: pass `past_key_values` through the wrapped base model.

This file now provides two model variants:

  * `XValModel` — deterministic regression head, MSE loss. SFT-only.
  * `GaussianXValModel` — Gaussian regression head outputting (μ, log σ).
    Compatible with SFT (Gaussian NLL) and RL (Gaussian log-prob plugged
    into a policy-gradient importance ratio). See `train_grpo_xval.py`.

Caveats
-------
The number regex is the part most likely to surprise you. We exclude any
digit immediately preceded by a Latin letter, Greek letter (U+0370–03FF) or
another digit, so identifiers like `P0`, `γ12`, `L7` keep their digits as
ordinary text. Strings like `P-5` lose the minus (matches `5`), which is
fine for this domain because `P-5` is far more likely to be an identifier
than arithmetic. Inspect `NUM_REGEX` if your data violates these
assumptions.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "finetuning"))
from prompts import MODEL_NAME  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_TOKEN = "<num>"

# Match a decimal literal NOT preceded by a letter (Latin or Greek) or another
# digit. Letters → identifiers like P0, γ12. Digits → middle of a multi-digit
# identifier like P12. The optional leading minus is only matched if the
# minus itself is at a non-letter, non-digit boundary.
NUM_REGEX = re.compile(r"(?<![A-Za-zͰ-Ͽ\d])(-?\d+(?:\.\d+)?)")


# ---------------------------------------------------------------------------
# Tokenizer setup
# ---------------------------------------------------------------------------

def setup_tokenizer(base_model_name: str = MODEL_NAME):
    """Load the base tokenizer and add `<num>` as a single special token.

    Critical: `<num>` must tokenize to exactly one id, otherwise the parallel
    values tensor and the input_ids tensor will fall out of alignment.
    """
    tok = AutoTokenizer.from_pretrained(base_model_name)
    if NUM_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [NUM_TOKEN]})
    # Sanity check: `<num>` must be a single token after adding.
    assert len(tok(NUM_TOKEN, add_special_tokens=False)["input_ids"]) == 1, (
        f"{NUM_TOKEN!r} does not tokenize to a single id — special token "
        "addition silently failed."
    )
    return tok


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

def encode_with_values(text: str, tokenizer):
    """Replace numerical literals in `text` with `<num>` and tokenize.

    Returns
    -------
    input_ids : LongTensor of shape (seq,)
    values    : FloatTensor of shape (seq,)
        `values[i]` is the number at position i if `input_ids[i]` is the
        `<num>` token, else 1.0.
    """
    captured: list[float] = []

    def repl(match):
        captured.append(float(match.group(1)))
        return NUM_TOKEN

    masked = NUM_REGEX.sub(repl, text)

    ids = tokenizer(masked, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    num_id = tokenizer.convert_tokens_to_ids(NUM_TOKEN)

    values = torch.ones(len(ids), dtype=torch.float32)
    num_positions = (ids == num_id).nonzero(as_tuple=True)[0]
    if len(num_positions) != len(captured):
        raise ValueError(
            f"Found {len(num_positions)} `<num>` positions but captured "
            f"{len(captured)} numbers. The tokenizer likely merged `<num>` "
            f"with adjacent text. Inspect: {masked!r}"
        )
    for pos, val in zip(num_positions.tolist(), captured):
        values[pos] = val
    return ids, values


def decode_with_values(input_ids, values, tokenizer, num_format: str = "{:.4f}") -> str:
    """Reverse `encode_with_values`: substitute numbers back into text.

    Numbers are formatted with `num_format` and then trailing zeros are
    stripped (so `5.0000` → `5`, `2.7700` → `2.77`).
    """
    num_id = tokenizer.convert_tokens_to_ids(NUM_TOKEN)
    text = tokenizer.decode(input_ids, skip_special_tokens=False)
    nums = [
        float(v) for tid, v in zip(input_ids.tolist(), values.tolist())
        if tid == num_id
    ]

    parts = text.split(NUM_TOKEN)
    out = []
    for j, p in enumerate(parts):
        out.append(p)
        if j < len(parts) - 1:
            v = nums[j]
            formatted = num_format.format(v)
            if "." in formatted:
                formatted = formatted.rstrip("0").rstrip(".")
            out.append(formatted if formatted else "0")
    return "".join(out)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class XValModel(nn.Module):
    """A HF CausalLM wrapped with xVal-style continuous number encoding.

    Forward signature is `(input_ids, values, attention_mask=None)` and
    returns `(token_logits, num_pred)`. The regression head is run on every
    position; supervise it only at positions where the gold next token is
    `<num>` (see `compute_loss`).
    """

    def __init__(self, base_model_name: str, tokenizer, torch_dtype=torch.float32):
        super().__init__()
        # Pin the base model dtype explicitly. Recent transformers respects
        # `torch_dtype` from the model config, which loads Qwen2-0.5B as
        # bf16 — that fights with our float32 `values` tensor on CPU. Default
        # to float32; pass torch.bfloat16 explicitly on GPU if you want it.
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch_dtype,
        )
        # Resize input + output embeddings to include `<num>`.
        self.base.resize_token_embeddings(len(tokenizer))
        self.num_token_id = tokenizer.convert_tokens_to_ids(NUM_TOKEN)

        hidden = self.base.config.hidden_size
        # Two-layer MLP regression head. Init the last layer near zero so the
        # untrained head doesn't dominate early MSE.
        self.num_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.num_head[-1].bias)
        nn.init.normal_(self.num_head[-1].weight, std=1e-3)
        # Match head dtype to the base so the linear ops line up.
        self.num_head.to(torch_dtype)

    def forward(self, input_ids, values, attention_mask=None):
        # Look up token embeddings, then scale element-wise by `values`.
        # Cast `values` to the embedding dtype so bf16/fp16 stacks don't
        # silently get promoted to float32 by the multiplication.
        emb = self.base.get_input_embeddings()(input_ids)
        emb = emb * values.to(emb.dtype).unsqueeze(-1)

        out = self.base(
            inputs_embeds=emb,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        token_logits = out.logits                    # (B, S, V)
        last_hidden = out.hidden_states[-1]          # (B, S, H)
        num_pred = self.num_head(last_hidden).squeeze(-1)  # (B, S)
        return token_logits, num_pred


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    token_logits, num_pred, input_ids, values,
    num_token_id: int, lambda_num: float = 1.0, ignore_index: int = -100,
):
    """Combined CE + MSE loss for next-token + next-number prediction.

    Position i in the inputs predicts token i+1. The number MSE is masked to
    positions where the *next* token is `<num>` (i.e. the model is about to
    emit a number, so we want the regression head to be right at i).
    """
    # Standard next-token CE over the shifted sequence.
    logits = token_logits[:, :-1].contiguous()
    labels = input_ids[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )

    # Number MSE only at positions where the gold next token is `<num>`.
    num_pred_shifted = num_pred[:, :-1]   # (B, S-1) — head output at position i
    num_targets = values[:, 1:]            # (B, S-1) — true value at position i+1
    mask = labels == num_token_id
    if mask.any():
        num_loss = F.mse_loss(num_pred_shifted[mask], num_targets[mask])
    else:
        num_loss = torch.zeros((), device=logits.device)

    total = token_loss + lambda_num * num_loss
    return total, token_loss, num_loss


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: XValModel, tokenizer, prompt: str,
    max_new_tokens: int = 256, do_sample: bool = False, temperature: float = 1.0,
) -> str:
    """Single-prompt autoregressive decode with `<num>` handling.

    On every step, the LM head picks the next token; if that token is
    `<num>`, the regression head's prediction at the same position becomes
    the number we attach to it. The new (token, value) pair is appended to
    `(input_ids, values)` and we step forward again.

    No KV cache — each step re-encodes the full prefix. Fine for sanity
    checks; replace with cached generation for production runs.
    """
    model.eval()
    device = next(model.parameters()).device

    input_ids, values = encode_with_values(prompt, tokenizer)
    input_ids = input_ids.unsqueeze(0).to(device)
    values = values.unsqueeze(0).to(device)

    num_token_id = model.num_token_id
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        token_logits, num_pred = model(input_ids, values)
        last_logits = token_logits[0, -1]

        if do_sample and temperature > 0:
            probs = F.softmax(last_logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        else:
            next_tok = last_logits.argmax(dim=-1, keepdim=True)

        if eos_id is not None and next_tok.item() == eos_id:
            break

        if next_tok.item() == num_token_id:
            next_val = num_pred[0, -1:]   # (1,)
        else:
            next_val = torch.ones(1, device=device)

        input_ids = torch.cat([input_ids, next_tok.unsqueeze(0)], dim=1)
        values = torch.cat([values, next_val.unsqueeze(0)], dim=1)

    return decode_with_values(input_ids[0].cpu(), values[0].cpu(), tokenizer)


# ===========================================================================
# Gaussian variant: head outputs (μ, log σ); compatible with policy gradient.
# ===========================================================================

_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_log_prob(value, mu, log_sigma):
    """Log density of N(value | μ, σ²), elementwise.

    Stable form: avoids `sigma = exp(log_sigma)` when computing
    `(value - μ)/σ` by folding the exponential in directly.
    """
    z = (value - mu) * torch.exp(-log_sigma)
    return -0.5 * z * z - log_sigma - 0.5 * _LOG_2PI


class GaussianXValModel(nn.Module):
    """xVal model with a Gaussian regression head: outputs (μ, log σ).

    Treat number emission as a Gaussian action: at every position the head
    parameterises the distribution from which the next number (if any) is
    drawn. The Gaussian log-prob plugs into a policy-gradient importance
    ratio alongside the token log-prob, so RL updates can flow into both
    the head and the trunk.

    For SFT, use `gaussian_sft_loss` — Gaussian NLL is strictly more general
    than the deterministic head's MSE (with σ=1 fixed they're equivalent up
    to constants).
    """

    # Clamp `log σ` to keep training stable. σ ∈ [~0.05, ~7.4] is enough
    # range for this domain (geometry coords are O(1)–O(10)).
    LOG_SIGMA_MIN = -3.0
    LOG_SIGMA_MAX = 2.0

    def __init__(self, base_model_name: str, tokenizer, torch_dtype=torch.float32):
        super().__init__()
        # See note in XValModel: pin dtype explicitly. Recent transformers
        # respects the model config's torch_dtype, which is bf16 for Qwen2.
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch_dtype,
        )
        self.base.resize_token_embeddings(len(tokenizer))
        self.num_token_id = tokenizer.convert_tokens_to_ids(NUM_TOKEN)

        hidden = self.base.config.hidden_size
        self.num_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),  # (μ, log σ)
        )
        # Init the last layer near zero so the untrained head is approximately
        # μ ≈ 0, log σ ≈ 0 (i.e. σ ≈ 1, a reasonable prior on number scale).
        nn.init.zeros_(self.num_head[-1].bias)
        nn.init.normal_(self.num_head[-1].weight, std=1e-3)
        self.num_head.to(torch_dtype)

    def forward(self, input_ids, values, attention_mask=None):
        emb = self.base.get_input_embeddings()(input_ids)
        emb = emb * values.to(emb.dtype).unsqueeze(-1)
        out = self.base(
            inputs_embeds=emb,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        token_logits = out.logits                       # (B, S, V)
        last_hidden = out.hidden_states[-1]              # (B, S, H)
        params = self.num_head(last_hidden)              # (B, S, 2)
        mu = params[..., 0]                              # (B, S)
        log_sigma = params[..., 1].clamp(
            self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX,
        )                                                # (B, S)
        return token_logits, mu, log_sigma


def gaussian_sft_loss(
    token_logits, mu, log_sigma, input_ids, values,
    num_token_id: int, lambda_num: float = 1.0, ignore_index: int = -100,
    labels=None,
):
    """SFT loss for the Gaussian head: token CE + masked Gaussian NLL.

    Same shape contract as `compute_loss`. The number-loss term is the
    negative log-likelihood of the gold number under the head's Gaussian.

    Parameters
    ----------
    labels : LongTensor or None
        If provided, must already be shifted (shape (B, S-1)) and may contain
        `ignore_index` (-100 by default) at positions that should be excluded
        from both the token CE and the Gaussian NLL — typically the prompt
        portion of an SFT example. If None, labels are auto-derived as
        `input_ids[:, 1:]` (i.e. supervise every step).
    """
    logits = token_logits[:, :-1].contiguous()
    if labels is None:
        labels = input_ids[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )

    mu_shifted = mu[:, :-1]
    log_sigma_shifted = log_sigma[:, :-1]
    targets = values[:, 1:]

    # `labels == num_token_id` already excludes ignore_index positions
    # (since ignore_index != num_token_id), so prompt positions are skipped
    # automatically when labels carries the SFT mask.
    mask = labels == num_token_id
    if mask.any():
        log_p = gaussian_log_prob(
            targets[mask], mu_shifted[mask], log_sigma_shifted[mask],
        )
        num_loss = -log_p.mean()
    else:
        num_loss = torch.zeros((), device=logits.device)

    total = token_loss + lambda_num * num_loss
    return total, token_loss, num_loss


@torch.no_grad()
def gaussian_rollout(
    model: GaussianXValModel, tokenizer, prompt: str,
    max_new_tokens: int = 256, temperature: float = 1.0,
):
    """Stochastic rollout — samples both tokens and numbers, returns log-probs.

    This is the function `train_grpo_xval.py` uses to collect on-policy
    trajectories. The returned tensors are detached and on CPU; the GRPO
    loss recomputes log-probs under the current (differentiable) policy
    in a separate forward pass.

    Returns
    -------
    A dict with:
      input_ids        : LongTensor (full_seq,)   — prompt + completion
      values           : FloatTensor (full_seq,)
      prompt_len       : int
      completion_tokens: LongTensor (comp_len,)   — the completion only
      completion_values: FloatTensor (comp_len,)
      token_log_probs  : FloatTensor (comp_len,)  — log p(token_t) at sampling time
      value_log_probs  : FloatTensor (comp_len,)  — log p(value_t) if token == NUM else 0
      ended_with_eos   : bool
    """
    model.eval()
    device = next(model.parameters()).device

    p_ids, p_vals = encode_with_values(prompt, tokenizer)
    prompt_len = int(len(p_ids))
    input_ids = p_ids.unsqueeze(0).to(device)
    values = p_vals.unsqueeze(0).to(device)

    num_token_id = model.num_token_id
    eos_id = tokenizer.eos_token_id

    tok_lps = []
    val_lps = []
    ended = False

    for _ in range(max_new_tokens):
        token_logits, mu, log_sigma = model(input_ids, values)
        last_logits = token_logits[0, -1]
        last_mu = mu[0, -1]
        last_log_sigma = log_sigma[0, -1]

        if temperature > 0:
            scaled = last_logits / temperature
            probs = F.softmax(scaled, dim=-1)
            next_tok = int(torch.multinomial(probs, num_samples=1).item())
        else:
            next_tok = int(last_logits.argmax(dim=-1).item())

        # Log-prob of the sampled token under the *true* (untempered) policy.
        # Using the untempered log-prob aligns rollout and recomputed log-probs
        # so the importance ratio is well-defined; temperature only affects
        # exploration breadth.
        token_lp = F.log_softmax(last_logits, dim=-1)[next_tok].detach()

        if next_tok == num_token_id:
            sigma = last_log_sigma.exp()
            eps = torch.randn((), device=device)
            sampled_val = last_mu + sigma * eps
            value_lp = gaussian_log_prob(
                sampled_val, last_mu, last_log_sigma,
            ).detach()
            next_val = sampled_val.detach()
        else:
            next_val = torch.tensor(1.0, device=device)
            value_lp = torch.zeros((), device=device)

        tok_lps.append(token_lp.cpu())
        val_lps.append(value_lp.cpu())

        # Append to the running sequence.
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_tok]], device=device)], dim=1,
        )
        values = torch.cat(
            [values, next_val.view(1, 1)], dim=1,
        )

        if eos_id is not None and next_tok == eos_id:
            ended = True
            break

    full_ids = input_ids[0].cpu()
    full_vals = values[0].cpu()
    return {
        "input_ids":         full_ids,
        "values":            full_vals,
        "prompt_len":        prompt_len,
        "completion_tokens": full_ids[prompt_len:],
        "completion_values": full_vals[prompt_len:],
        "token_log_probs":   torch.stack(tok_lps) if tok_lps else torch.zeros(0),
        "value_log_probs":   torch.stack(val_lps) if val_lps else torch.zeros(0),
        "ended_with_eos":    ended,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test():
    print(f"Loading {MODEL_NAME} for xVal smoke test...")
    tok = setup_tokenizer()
    print(f"  vocab size after adding <num>: {len(tok)}")

    # 1. Round-trip encode/decode on a few representative strings.
    samples = [
        "Two points P0 and P1 are 5 units apart, with P0 at (0, 0).",
        "circle(γ12, C, 1.8571)",                              # γ12 must NOT be split
        "point(P, -0.5278, 0.9626)",                                  # negative coords
        "point(P0, 0, 0); circle(C0, P0, 2.77)",                     # mixed ints + floats
    ]
    for s in samples:
        ids, vals = encode_with_values(s, tok)
        roundtrip = decode_with_values(ids, vals, tok)
        nums = [float(v) for tid, v in zip(ids.tolist(), vals.tolist())
                if tid == tok.convert_tokens_to_ids(NUM_TOKEN)]
        print(f"\n  in : {s!r}")
        print(f"  num: {nums}")
        print(f"  out: {roundtrip!r}")

    # 2. One forward pass + loss on the simplest sample.
    print("\nBuilding XValModel (this may take a moment)...")
    model = XValModel(MODEL_NAME, tok)
    model.eval()

    ids, vals = encode_with_values(samples[0], tok)
    ids_b = ids.unsqueeze(0)
    vals_b = vals.unsqueeze(0)
    token_logits, num_pred = model(ids_b, vals_b)
    print(f"  token_logits: {tuple(token_logits.shape)}")
    print(f"  num_pred    : {tuple(num_pred.shape)}")

    loss, t_loss, n_loss = compute_loss(
        token_logits, num_pred, ids_b, vals_b, model.num_token_id,
    )
    print(f"  total loss : {loss.item():.4f}")
    print(f"  token loss : {t_loss.item():.4f}")
    print(f"  num loss   : {n_loss.item():.4f}  (zero is expected if there's "
          "no <num> in the next-token labels of this short sample)")

    # 3. One generate call. Untrained num_head returns ~0 for all numbers.
    print("\nUntrained generate (numbers will all be near zero):")
    out = generate(model, tok, samples[0], max_new_tokens=8, do_sample=False)
    print(f"  {out!r}")

    # 4. Gaussian variant: forward + SFT loss + a tiny rollout.
    print("\nBuilding GaussianXValModel...")
    g_model = GaussianXValModel(MODEL_NAME, tok)
    g_model.eval()
    token_logits, mu, log_sigma = g_model(ids_b, vals_b)
    print(f"  token_logits: {tuple(token_logits.shape)}")
    print(f"  mu          : {tuple(mu.shape)}")
    print(f"  log_sigma   : {tuple(log_sigma.shape)}  range=[{log_sigma.min().item():.3f}, "
          f"{log_sigma.max().item():.3f}]")

    g_loss, gt_loss, gn_loss = gaussian_sft_loss(
        token_logits, mu, log_sigma, ids_b, vals_b, g_model.num_token_id,
    )
    print(f"  gaussian total: {g_loss.item():.4f}   token: {gt_loss.item():.4f}   "
          f"num NLL: {gn_loss.item():.4f}")

    print("\nGaussian rollout (4 steps, T=1.0):")
    rollout = gaussian_rollout(g_model, tok, samples[0], max_new_tokens=4, temperature=1.0)
    print(f"  prompt_len     : {rollout['prompt_len']}")
    print(f"  completion ids : {rollout['completion_tokens'].tolist()}")
    print(f"  token log_probs: {[f'{x.item():.3f}' for x in rollout['token_log_probs']]}")
    print(f"  value log_probs: {[f'{x.item():.3f}' for x in rollout['value_log_probs']]}")

    print("\nOK")


if __name__ == "__main__":
    _smoke_test()
