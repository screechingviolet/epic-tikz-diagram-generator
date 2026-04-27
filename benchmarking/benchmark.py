"""
benchmark.py

Tests 50 NL descriptions against 3 models (GPT, Claude, Gemini) in 2 modes:
  Mode 1: NL -> TikZ directly (baseline)
  Mode 2: NL -> Formal -> TikZ (our pipeline)

Results saved to benchmark_results.jsonl
Compilation scores saved to benchmark_scores.json
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

N_SAMPLES = 50
DATASET_PATH = "demo_dataset.jsonl"   # your generated dataset
OUTPUT_PATH  = "benchmark_results.jsonl"
SCORES_PATH  = "benchmark_scores.json"
DELAY        = 1.0                     # seconds between API calls

MODELS = {
    "gpt-4o-mini":        "openai",
    "claude-haiku-4-5-20251001":  "anthropic",
    "gemini-1.5-flash":   "gemini",
}

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

MODE1_SYSTEM = """You convert natural language geometry descriptions into TikZ code.
Output ONLY the tikzpicture environment, nothing else. No explanation, no markdown, no backticks.

Example output:
\\begin{tikzpicture}
  \\draw (0,0) circle (2cm);
  \\draw (-3,2) -- (3,2);
\\end{tikzpicture}"""

MODE2_FORMAL_SYSTEM = """You convert natural language geometry descriptions into formal geometry language.
Output ONLY the formal language, nothing else. No explanation, no markdown, no backticks.

Format:
point(NAME, x, y)
line(NAME, P1, P2)
circle(NAME, CENTER, RADIUS)
tangent(LINE, CIRCLE)
parallel(L1, L2)
perpendicular(L1, L2)
length(LINE, VALUE)
radius(CIRCLE, VALUE)
angle(L1, L2, DEGREES)
circle_tangent(C1, C2)
on_circle(POINT, CIRCLE)

Example:
Input: "A circle of radius 2 with a tangent line of length 3"
Output:
point(P0, 0.0, 0.0)
point(P1, -1.5, 2.0)
point(P2, 1.5, 2.0)
circle(C0, P0, 2.0)
line(L0, P1, P2)
tangent(L0, C0)
length(L0, 3.0)
radius(C0, 2.0)"""

MODE2_TIKZ_SYSTEM = """You convert formal geometry language into TikZ code.
Output ONLY the tikzpicture environment, nothing else. No explanation, no markdown, no backticks.

Formal language format:
point(NAME, x, y)         - a point at coordinates x,y
line(NAME, P1, P2)        - a line segment between two points
circle(NAME, CENTER, R)   - a circle with given center point and radius
tangent(LINE, CIRCLE)     - a line is tangent to a circle
parallel(L1, L2)          - two lines are parallel
perpendicular(L1, L2)     - two lines are perpendicular
length(LINE, VALUE)       - length of a line
radius(CIRCLE, VALUE)     - radius of a circle

Example output:
\\begin{tikzpicture}
  \\draw (0,0) circle (2cm);
  \\draw (-1.5,2) -- (1.5,2);
\\end{tikzpicture}"""


# ---------------------------------------------------------------------------
# MODEL CLIENTS
# ---------------------------------------------------------------------------

def init_clients():
    clients = {}
    clients["openai"]    = OpenAI()
    clients["anthropic"] = anthropic.Anthropic()
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    clients["gemini"]    = genai.GenerativeModel("gemini-1.5-flash")
    return clients


def query(clients, model_name, provider, system, user_message) -> str:
    try:
        if provider == "openai":
            response = clients["openai"].chat.completions.create(
                model=model_name,
                max_tokens=1500,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_message},
                ],
            )
            return response.choices[0].message.content.strip()

        elif provider == "anthropic":
            response = clients["anthropic"].messages.create(
                model=model_name,
                max_tokens=1500,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text.strip()

        elif provider == "gemini":
            response = clients["gemini"].generate_content(f"{system}\n\n{user_message}")
            return response.text.strip()

    except Exception as e:
        print(f"    API error ({model_name}): {e}")
        return ""


# ---------------------------------------------------------------------------
# TIKZ COMPILATION
# ---------------------------------------------------------------------------

def compile_tikz(tikz_code: str) -> bool:
    """Returns True if the TikZ compiles without errors."""
    tex = r"""
\documentclass{standalone}
\usepackage{tikz}
\begin{document}
""" + tikz_code + r"""
\end{document}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "test.tex"
        tex_path.write_text(tex)
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "test.tex"],
            capture_output=True,
            cwd=tmpdir,
        )
        return result.returncode == 0


def clean_tikz(raw: str) -> str:
    """Strip markdown fences if the model added them anyway."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return raw.strip()


# ---------------------------------------------------------------------------
# BENCHMARK MODES
# ---------------------------------------------------------------------------

def run_mode1(clients, nl: str, model_name: str, provider: str) -> dict:
    """NL -> TikZ directly."""
    tikz = clean_tikz(query(clients, model_name, provider, MODE1_SYSTEM, nl))
    compiles = compile_tikz(tikz) if tikz else False
    return {
        "mode":     "nl_to_tikz",
        "model":    model_name,
        "nl":       nl,
        "tikz":     tikz,
        "compiles": compiles,
    }


def run_mode2(clients, nl: str, model_name: str, provider: str) -> dict:
    """NL -> Formal -> TikZ (our pipeline)."""
    # Step 1: NL -> formal
    formal = query(clients, model_name, provider, MODE2_FORMAL_SYSTEM, nl).strip()

    # Step 2: formal -> TikZ
    tikz = clean_tikz(query(clients, model_name, provider, MODE2_TIKZ_SYSTEM, formal))
    compiles = compile_tikz(tikz) if tikz else False

    return {
        "mode":     "nl_to_formal_to_tikz",
        "model":    model_name,
        "nl":       nl,
        "formal":   formal,
        "tikz":     tikz,
        "compiles": compiles,
    }


# ---------------------------------------------------------------------------
# MAIN BENCHMARK LOOP
# ---------------------------------------------------------------------------

def load_nl_samples(dataset_path: str, n: int) -> list[str]:
    """Load n NL variant samples from the dataset."""
    samples = []
    with open(dataset_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("nl_variants"):
                samples.append(record["nl_variants"][0])  # take first variant
            if len(samples) >= n:
                break
    return samples


def run_benchmark():
    print("Initializing clients...")
    clients = init_clients()

    print(f"Loading {N_SAMPLES} samples from {DATASET_PATH}...")
    samples = load_nl_samples(DATASET_PATH, N_SAMPLES)
    if not samples:
        print("No NL variants found in dataset — make sure you've generated with NL variants enabled.")
        return

    print(f"Loaded {len(samples)} samples. Starting benchmark...\n")

    results = []
    scores  = {model: {"mode1": {"compiles": 0, "total": 0},
                        "mode2": {"compiles": 0, "total": 0}}
               for model in MODELS}

    for i, nl in enumerate(samples):
        print(f"Sample {i+1}/{len(samples)}: {nl[:60]}...")

        for model_name, provider in MODELS.items():
            print(f"  [{model_name}] mode1 ...", end=" ", flush=True)
            r1 = run_mode1(clients, nl, model_name, provider)
            results.append(r1)
            scores[model_name]["mode1"]["total"]    += 1
            scores[model_name]["mode1"]["compiles"] += int(r1["compiles"])
            print("✓" if r1["compiles"] else "✗")
            time.sleep(DELAY)

            print(f"  [{model_name}] mode2 ...", end=" ", flush=True)
            r2 = run_mode2(clients, nl, model_name, provider)
            results.append(r2)
            scores[model_name]["mode2"]["total"]    += 1
            scores[model_name]["mode2"]["compiles"] += int(r2["compiles"])
            print("✓" if r2["compiles"] else "✗")
            time.sleep(DELAY)

    # Save all raw results
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Compute and save scores
    summary = {}
    for model_name in MODELS:
        m1 = scores[model_name]["mode1"]
        m2 = scores[model_name]["mode2"]
        summary[model_name] = {
            "mode1_compile_rate": m1["compiles"] / max(m1["total"], 1),
            "mode2_compile_rate": m2["compiles"] / max(m2["total"], 1),
        }

    with open(SCORES_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 55)
    print(f"{'Model':<25} {'NL->TikZ':>12} {'NL->F->TikZ':>12}")
    print("=" * 55)
    for model_name, s in summary.items():
        print(f"{model_name:<25} {s['mode1_compile_rate']:>11.1%} {s['mode2_compile_rate']:>11.1%}")
    print("=" * 55)
    print(f"\nRaw results → {OUTPUT_PATH}")
    print(f"Scores      → {SCORES_PATH}")


if __name__ == "__main__":
    run_benchmark()