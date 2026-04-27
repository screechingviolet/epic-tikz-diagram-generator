"""Run inference with a fine-tuned LoRA adapter on a single NL description.

Usage:
    python finetuning/infer.py --adapter Qwen2-0.5B-GRPO-geometry \
        --input "Two points P0 and P1 are 5 units apart and connected by a line L0."

    # or pipe the input on stdin:
    echo "..." | python finetuning/infer.py --adapter Qwen2-0.5B-GRPO-geometry

Loads MODEL_NAME (the base model) and stacks the trained LoRA adapter from
the directory passed via --adapter (typically the SAVE_DIR from train_grpo.py
or a checkpoint subdir of it).
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Shared with train_grpo.py — single source of truth for the base model
# string and the system prompt the adapter was tuned against. Using a
# different system prompt at inference time silently degrades quality.
from prompts import MODEL_NAME, SYSTEM_PROMPT  # noqa: E402


def load_model_with_adapter(adapter_path: str):
    """Load the base model and stack the saved LoRA adapter on top."""
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def generate(model, tokenizer, nl_input: str, max_new_tokens: int, temperature: float) -> str:
    """Run a single forward generation and return only the newly-generated text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": nl_input},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # temperature=0 → greedy decoding; >0 → sampling.
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt tokens; decode only the new portion.
    new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate geometry-language output for a single NL description."
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the saved LoRA adapter directory (e.g. the SAVE_DIR "
             "from train_grpo.py, or a checkpoint-N subdir).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Natural-language description. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="Cap on generated tokens (matches max_completion_length used in training).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0 means greedy decoding (default).",
    )
    args = parser.parse_args()

    nl = args.input if args.input is not None else sys.stdin.read().strip()
    if not nl:
        parser.error("no input provided (use --input or pipe to stdin)")

    print(f"[infer] base model: {MODEL_NAME}", file=sys.stderr)
    print(f"[infer] adapter:    {args.adapter}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model_with_adapter(args.adapter)

    print(f"[infer] input: {nl}", file=sys.stderr)
    print("---", file=sys.stderr)
    completion = generate(model, tokenizer, nl, args.max_new_tokens, args.temperature)
    print(completion)


if __name__ == "__main__":
    main()
