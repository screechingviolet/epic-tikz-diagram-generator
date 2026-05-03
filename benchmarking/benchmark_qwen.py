"""
benchmark.py

Tests 50 NL descriptions against models in 3 modes:
  Mode 1: NL -> TikZ directly (compile check)
  Mode 2: NL -> Epic Geometry Language -> score via loss function
  Mode 3: Ground truth geometry -> Epic Geometry Language -> score via loss function

All 3 modes use the same 50 NL inputs from the dataset.
Batched + cached — safe to interrupt and resume.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "loss-fn"))
from loss import check_constraints

import hashlib
import json
import subprocess
import tempfile
import time

import anthropic
import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv(Path(__file__).parent.parent / "data" / ".env")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

RUN_NAME     = "run3"
N_SAMPLES    = 50
DATASET_PATH = "curriculum_data/dataset_merged.jsonl"
OUTPUT_PATH  = f"benchmark_results_{RUN_NAME}.jsonl"
SCORES_PATH  = f"benchmark_scores_{RUN_NAME}.json"
CACHE_PATH   = f"benchmark_cache_{RUN_NAME}.json"
COMPILED_DIR = Path(f"compiled_outputs_{RUN_NAME}")
BATCH_SIZE   = 10
DELAY        = 1.0

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = str(Path(__file__).parent.parent / "Qwen2-0.5B-GRPO-geometry" / "checkpoint-300")

MODELS = {
    "gpt-4o-mini":                 "openai",
    "claude-haiku-4-5-20251001":   "anthropic",
    "Qwen/Qwen2.5-0.5B-Instruct": "hf_local",
    "qwen-geometry-adapter":       "hf_adapter",
}

# Modes to run per provider — local models only do mode2 (NL->Geo)
MODEL_MODES = {
    "openai":     [2, 3],
    "anthropic":  [2, 3],
    "hf_local":   [2, 3],
    "hf_adapter": [2, 3],
}

# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def to_binary(score) -> float:
    if not isinstance(score, (int, float)):
        return 0.0
    return 1.0 if score >= 3.0 else 0.0

def to_continuous(score) -> float:
    if not isinstance(score, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(score) / 3.0))

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

MODE1_SYSTEM = """You convert natural language geometry descriptions into TikZ code.
Output ONLY the tikzpicture environment — no explanation, no markdown, no backticks.

Rules:
- Include EVERY object mentioned: all points, all lines, all circles — nothing may be omitted
- Label EVERY object with its name exactly as given in the description
- Points: draw as filled circles (\\filldraw ... circle (2pt)) with the name label offset above-right
- Lines: draw the full segment between its two named endpoints; label at midpoint
- Circles: draw the full circle; label near the top of the circle
- Coordinates are typically in the range [-5, 5]; scale the diagram to be readable

Example input:
Point A and point B exist. Line la connects A and B. Circle ω0 is centered at A with radius 1.5.

Example output:
\\begin{tikzpicture}[scale=1.0]
  \\filldraw (0.0, 0.0) circle (2pt) node[above right] {$A$};
  \\filldraw (3.0, 0.0) circle (2pt) node[above right] {$B$};
  \\draw (0.0, 0.0) -- (3.0, 0.0) node[midway, below] {$l_a$};
  \\draw (0.0, 0.0) circle (1.5) node[above=1.5cm] {$\\omega_0$};
\\end{tikzpicture}"""

MODE2_SYSTEM = (
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
    "Description: A line segment L0 of length 3.4823 has endpoints P0 and P1. "
    "A circle C0 is centered at P0 and passes through P1.\n"
    "Output:\n"
    "point(P0, 0, 0)\n"
    "point(P1, 3.4823, 0)\n"
    "circle(C0, P0, 3.4823)\n"
    "line(L0, P0, P1)"
)

MODE3_SYSTEM = (
    "You are given a list of geometric primitives (points, lines, circles) in a formal language. "
    "Re-express them as a clean list of the same primitives. "
    "Output one primitive per line and nothing else.\n"
    "\n"
    "Primitives:\n"
    "  point(name: str, x: float, y: float)\n"
    "  line(name: str, p1: str, p2: str)\n"
    "  circle(name: str, center: str, radius: float)\n"
    "\n"
    "Keep the same names, coordinates, and structure. Output only the primitives, no constraints."
)

BATCH_WRAPPER = """You will process {n} inputs numbered 1 to {n}.
For each one, output the result prefixed with ===1===, ===2===, etc.

Example format for 2 inputs:
===1===
point(P0, 0, 0)
circle(C0, P0, 2.0)
===2===
point(P0, 0, 0)
point(P1, 5, 0)
line(L0, P0, P1)

Here are the {n} inputs:
{inputs}"""

MODE1_BATCH_WRAPPER = """You will process {n} geometry descriptions numbered 1 to {n}.
For each one, output the result prefixed with ===1===, ===2===, etc.

Example format for 2 inputs:
===1===
\\begin{{tikzpicture}}
  ...
\\end{{tikzpicture}}
===2===
\\begin{{tikzpicture}}
  ...
\\end{{tikzpicture}}

Here are the {n} inputs:
{inputs}"""


# ---------------------------------------------------------------------------
# LOCAL HF INFERENCE
# ---------------------------------------------------------------------------

_hf_models = {}

def _get_hf_model(model_name: str, provider: str):
    if model_name in _hf_models:
        return _hf_models[model_name]

    print(f"  Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    if provider == "hf_adapter":
        from peft import PeftModel
        adapter_path = str(Path(__file__).parent.parent / "Qwen2-0.5B-GRPO-geometry" / "checkpoint-300")
        base  = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32)

    model.eval()
    _hf_models[model_name] = (tokenizer, model)
    return tokenizer, model


def query_hf(model_name: str, provider: str, system: str, user_message: str) -> str:
    tokenizer, model = _get_hf_model(model_name, provider)

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_message},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------------

def load_cache(path: str) -> dict:
    if Path(path).exists():
        raw = json.loads(Path(path).read_text())
        rescored = 0
        for v in raw.values():
            if v.get("mode") in ("mode2_nl_to_geo", "mode3_geo_to_geo"):
                new_score = to_continuous(check_constraints(v["pred_geo"], v["truth_constraints"]))
                if v["score"] != new_score:
                    v["score"] = new_score
                    rescored += 1
        if rescored:
            print(f"  Rescored {rescored} stale mode2/3 cache entries")
        return raw
    return {}
def save_cache(cache: dict, path: str):
    Path(path).write_text(json.dumps(cache, indent=2))

def cache_key(model: str, mode: str, inp: str) -> str:
    return hashlib.md5(f"{model}:{mode}:{inp}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# API CLIENTS
# ---------------------------------------------------------------------------

def init_clients():
    return {
        "openai":    OpenAI(),
        "anthropic": anthropic.Anthropic(),
    }

def query_api(clients, model_name, provider, system, user_message) -> str:
    try:
        if provider == "openai":
            response = clients["openai"].chat.completions.create(
                model=model_name,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_message},
                ],
            )
            return response.choices[0].message.content.strip()
        elif provider == "anthropic":
            response = clients["anthropic"].messages.create(
                model=model_name,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text.strip()
    except Exception as e:
        print(f"    API error ({model_name}): {e}")
        return ""

def query(clients, model_name, provider, system, user_message) -> str:
    if provider in ("hf_local", "hf_adapter"):
        return query_hf(model_name, provider, system, user_message)
    return query_api(clients, model_name, provider, system, user_message)


# ---------------------------------------------------------------------------
# BATCHING
# ---------------------------------------------------------------------------

def build_batch_prompt(inputs: list[str], wrapper: str) -> str:
    numbered = "\n\n".join(f"{i+1}. {inp}" for i, inp in enumerate(inputs))
    return wrapper.format(n=len(inputs), inputs=numbered)

def parse_batch_response(raw: str, n: int) -> list[str]:
    results = [""] * n
    for i in range(n):
        marker      = f"==={i+1}==="
        next_marker = f"==={i+2}==="
        start = raw.find(marker)
        if start == -1:
            continue
        start += len(marker)
        end = raw.find(next_marker) if i < n - 1 else len(raw)
        results[i] = raw[start:end].strip()
    return results

def clean_tikz(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return raw.strip()


# ---------------------------------------------------------------------------
# TIKZ COMPILATION
# ---------------------------------------------------------------------------

def compile_tikz(tikz_code: str) -> bool:
    tex = (
        r"\documentclass{standalone}" + "\n"
        r"\usepackage{tikz}" + "\n"
        r"\begin{document}" + "\n"
        + tikz_code + "\n"
        + r"\end{document}"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "test.tex"
        tex_path.write_text(tex)
        result = subprocess.run(
            ["/Library/TeX/texbin/pdflatex", "-interaction=nonstopmode", "test.tex"],
            capture_output=True, cwd=tmpdir,
        )
        return result.returncode == 0

def save_compiled_tex(r: dict):
    mode    = r["mode"]
    model   = r["model"].replace("/", "-")
    inp     = r.get("input") or r.get("nl") or ""
    nl_hash = hashlib.md5(inp.encode()).hexdigest()[:8]
    fname   = COMPILED_DIR / f"{model}__{mode}__{nl_hash}.tex"
    tex = (
        r"\documentclass{standalone}" + "\n"
        r"\usepackage{tikz}" + "\n"
        r"\begin{document}" + "\n"
        + r["tikz"] + "\n"
        + r"\end{document}"
    )
    fname.write_text(tex)
    (COMPILED_DIR / f"{model}__{mode}__{nl_hash}.txt").write_text(inp)


# ---------------------------------------------------------------------------
# MODE RUNNERS
# ---------------------------------------------------------------------------

def run_mode1_batch(clients, nls, model_name, provider, cache) -> list[dict]:
    to_query, indices, results = [], [], [None] * len(nls)
    for i, nl in enumerate(nls):
        key = cache_key(model_name, "mode1", nl)
        if key in cache:
            results[i] = cache[key]
        else:
            to_query.append(nl); indices.append(i)

    if to_query:
        prompt = build_batch_prompt(to_query, MODE1_BATCH_WRAPPER)
        raw    = query(clients, model_name, provider, MODE1_SYSTEM, prompt)
        parsed = parse_batch_response(raw, len(to_query))
        for j, (i, nl) in enumerate(zip(indices, to_query)):
            tikz     = clean_tikz(parsed[j])
            compiles = compile_tikz(tikz) if tikz else False
            record   = {
                "mode": "mode1_nl_to_tikz", "model": model_name,
                "input": nl, "tikz": tikz, "compiles": compiles,
            }
            cache[cache_key(model_name, "mode1", nl)] = record
            results[i] = record
    return results


def run_mode2_batch(clients, nls, truth_constraints_list, model_name, provider, cache) -> list[dict]:
    to_query, indices, results = [], [], [None] * len(nls)
    for i, nl in enumerate(nls):
        key = cache_key(model_name, "mode2", nl)
        if key in cache:
            results[i] = cache[key]
        else:
            to_query.append(nl); indices.append(i)

    if to_query:
        if provider in ("hf_local", "hf_adapter"):
            for i, nl in zip(indices, to_query):
                raw      = query(clients, model_name, provider, MODE2_SYSTEM, nl)
                print(f"\n  RAW OUTPUT:\n{raw}\n")
                # strip markdown code fences
                cleaned  = raw.strip()
                if cleaned.startswith("```"):
                    lines   = cleaned.splitlines()
                    cleaned = "\n".join(l for l in lines if not l.strip().startswith("```"))
                # drop lines with nested calls (contain a second open paren inside args)
                pred_geo = []
                for line in cleaned.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # reject lines like circle(ω1, point(L, 0, 0), 0.5747)
                    after_first_paren = line[line.find("(")+1:] if "(" in line else ""
                    if "(" in after_first_paren:
                        continue
                    pred_geo.append(line)
                score  = check_constraints(pred_geo, truth_constraints_list[i])
                record = {
                    "mode": "mode2_nl_to_geo", "model": model_name,
                    "input": nl, "pred_geo": pred_geo,
                    "truth_constraints": truth_constraints_list[i], "score": score,
                }
                cache[cache_key(model_name, "mode2", nl)] = record
                results[i] = record
        else:
            prompt = build_batch_prompt(to_query, BATCH_WRAPPER)
            raw    = query(clients, model_name, provider, MODE2_SYSTEM, prompt)
            parsed = parse_batch_response(raw, len(to_query))
            for j, (i, nl) in enumerate(zip(indices, to_query)):
                pred_geo = [line for line in parsed[j].splitlines() if line.strip()]
                score    = check_constraints(pred_geo, truth_constraints_list[i])
                record   = {
                    "mode": "mode2_nl_to_geo", "model": model_name,
                    "input": nl, "pred_geo": pred_geo,
                    "truth_constraints": truth_constraints_list[i], "score": score,
                }
                cache[cache_key(model_name, "mode2", nl)] = record
                results[i] = record
    return results


def run_mode3_batch(clients, geo_inputs, truth_constraints_list, model_name, provider, cache) -> list[dict]:
    to_query, indices, results = [], [], [None] * len(geo_inputs)
    for i, geo in enumerate(geo_inputs):
        key = cache_key(model_name, "mode3", geo)
        if key in cache:
            results[i] = cache[key]
        else:
            to_query.append(geo); indices.append(i)

    if to_query:
        prompt = build_batch_prompt(to_query, BATCH_WRAPPER)
        raw    = query(clients, model_name, provider, MODE3_SYSTEM, prompt)
        parsed = parse_batch_response(raw, len(to_query))
        for j, (i, geo) in enumerate(zip(indices, to_query)):
            pred_geo = [line for line in parsed[j].splitlines() if line.strip()]
            score    = check_constraints(pred_geo, truth_constraints_list[i])
            record   = {
                "mode": "mode3_geo_to_geo", "model": model_name,
                "input": geo, "pred_geo": pred_geo,
                "truth_constraints": truth_constraints_list[i], "score": score,
            }
            cache[cache_key(model_name, "mode3", geo)] = record
            results[i] = record
    return results


# ---------------------------------------------------------------------------
# LOAD INPUTS
# ---------------------------------------------------------------------------

def load_samples() -> tuple[list[str], list[str], list[list[str]]]:
    nls, geos, constraints_list = [], [], []
    try:
        records = []
        with open(DATASET_PATH) as f:
            for line in f:
                records.append(json.loads(line))

        records = records[-N_SAMPLES:]

        for record in records:
            if record.get("nl_variants") and record.get("geometry") and record.get("constraints"):
                nls.append(record["nl_variants"][0])
                geos.append("\n".join(record["geometry"]))
                constraints_list.append(record["constraints"])

    except FileNotFoundError:
        print(f"Dataset not found at {DATASET_PATH}")

    print(f"Loaded {len(nls)} samples from dataset (last {N_SAMPLES} records)")
    return nls, geos, constraints_list


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_benchmark():
    print(f"Run: {RUN_NAME}")
    print("Initializing clients...")
    clients = init_clients()

    print("Loading samples...")
    nls, geos, constrs = load_samples()

    if not nls:
        print("No samples found — check dataset path.")
        return

    cache = load_cache(CACHE_PATH)
    print(f"Cache loaded ({len(cache)} entries)\n")
    COMPILED_DIR.mkdir(exist_ok=True)

    all_results = []
    scores = {
        model: {
            "mode1": {"compiles": 0, "total": 0},
            "mode2": {"score_sum": 0.0, "total": 0},
            "mode3": {"score_sum": 0.0, "total": 0},
        }
        for model in MODELS
    }

    batches = [list(range(i, min(i+BATCH_SIZE, len(nls)))) for i in range(0, len(nls), BATCH_SIZE)]

    for model_name, provider in MODELS.items():
        modes = MODEL_MODES[provider]
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  (modes: {modes})")
        print(f"{'='*60}")

        if 1 in modes:
            print(f"\n  [Mode 1: NL -> TikZ] {len(nls)} samples")
            for b, idx_batch in enumerate(batches):
                nls_batch = [nls[i] for i in idx_batch]
                print(f"    Batch {b+1}/{len(batches)} ...", end=" ", flush=True)
                r1s = run_mode1_batch(clients, nls_batch, model_name, provider, cache)
                n_ok = sum(r["compiles"] for r in r1s)
                print(f"{n_ok}/{len(r1s)} compiled")
                for r in r1s:
                    all_results.append(r)
                    scores[model_name]["mode1"]["total"]    += 1
                    scores[model_name]["mode1"]["compiles"] += int(r["compiles"])
                    if r["compiles"]:
                        save_compiled_tex(r)
                save_cache(cache, CACHE_PATH)
                time.sleep(DELAY)

        if 2 in modes:
            print(f"\n  [Mode 2: NL -> Geo -> Score] {len(nls)} samples")
            for b, idx_batch in enumerate(batches):
                nls_batch     = [nls[i]     for i in idx_batch]
                constrs_batch = [constrs[i] for i in idx_batch]
                print(f"    Batch {b+1}/{len(batches)} ...", end=" ", flush=True)
                r2s = run_mode2_batch(clients, nls_batch, constrs_batch, model_name, provider, cache)
                avg = sum(to_continuous(r["score"]) for r in r2s) / len(r2s)
                print(f"avg score {avg:.2f}")
                for r in r2s:
                    all_results.append(r)
                    scores[model_name]["mode2"]["score_sum"] += to_binary(r["score"])
                    scores[model_name]["mode2"]["total"]     += 1
                save_cache(cache, CACHE_PATH)
                if provider not in ("hf_local", "hf_adapter"):
                    time.sleep(DELAY)

        if 3 in modes:
            print(f"\n  [Mode 3: GT Geo -> Geo -> Score] {len(geos)} samples")
            for b, idx_batch in enumerate(batches):
                geos_batch    = [geos[i]    for i in idx_batch]
                constrs_batch = [constrs[i] for i in idx_batch]
                print(f"    Batch {b+1}/{len(batches)} ...", end=" ", flush=True)
                r3s = run_mode3_batch(clients, geos_batch, constrs_batch, model_name, provider, cache)
                avg = sum(to_continuous(r["score"]) for r in r3s) / len(r3s)
                print(f"avg score {avg:.2f}")
                for r in r3s:
                    all_results.append(r)
                    scores[model_name]["mode3"]["score_sum"] += to_binary(r["score"])
                    scores[model_name]["mode3"]["total"]     += 1
                save_cache(cache, CACHE_PATH)
                time.sleep(DELAY)

    with open(OUTPUT_PATH, "w") as f:
        for r in all_results:
            rec = {**r, "score": r.get("score") if isinstance(r.get("score"), (int, float)) else None}
            f.write(json.dumps(rec) + "\n")

    summary = {}
    for model_name in MODELS:
        m1 = scores[model_name]["mode1"]
        m2 = scores[model_name]["mode2"]
        m3 = scores[model_name]["mode3"]
        summary[model_name] = {
            "mode1_compile_rate": m1["compiles"]  / max(m1["total"], 1),
            "mode2_avg_score":    m2["score_sum"] / max(m2["total"], 1),
            "mode3_avg_score":    m3["score_sum"] / max(m3["total"], 1),
        }

    with open(SCORES_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print(f"{'Model':<35} {'NL->TikZ':>10} {'NL->Geo':>10} {'GT->Geo':>10}")
    print("=" * 65)
    for model_name, s in summary.items():
        m1 = f"{s['mode1_compile_rate']:.1%}" if scores[model_name]["mode1"]["total"] > 0 else "—"
        m2 = f"{s['mode2_avg_score']:.2f}"    if scores[model_name]["mode2"]["total"] > 0 else "—"
        m3 = f"{s['mode3_avg_score']:.2f}"    if scores[model_name]["mode3"]["total"] > 0 else "—"
        print(f"{model_name:<35} {m1:>10} {m2:>10} {m3:>10}")
    print("=" * 65)
    print(f"\nRaw results  → {OUTPUT_PATH}")
    print(f"Scores       → {SCORES_PATH}")
    print(f"Cache        → {CACHE_PATH}")
    print(f"Compiled tex → {COMPILED_DIR}/")


if __name__ == "__main__":
    run_benchmark()