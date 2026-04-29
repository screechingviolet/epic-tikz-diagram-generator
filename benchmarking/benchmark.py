"""
benchmark.py

Tests 50 NL descriptions against 2 models (GPT, Claude) in 3 modes:
  Mode 1: NL -> TikZ directly (compile check)
  Mode 2: NL -> Epic Geometry Language -> score via loss function
  Mode 3: Ground truth geometry -> Epic Geometry Language -> score via loss function

All 3 modes use the same 50 NL inputs from the dataset.
Batched + cached — safe to interrupt and resume.
"""

import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from openai import OpenAI
from shapely.geometry import LineString

load_dotenv(Path(__file__).parent.parent / "data" / ".env")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

N_SAMPLES    = 50
DATASET_PATH = "../curriculum_data/dataset_merged.jsonl"
OUTPUT_PATH  = "benchmark_results.jsonl"
SCORES_PATH  = "benchmark_scores.json"
CACHE_PATH   = "benchmark_cache.json"
COMPILED_DIR = Path("compiled_outputs")
BATCH_SIZE   = 10
DELAY        = 1.0

MODELS = {
    "gpt-4o-mini":               "openai",
    "claude-haiku-4-5-20251001": "anthropic",
}

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

MODE1_SYSTEM = """You convert natural language geometry descriptions into TikZ code.
For each input, output ONLY the tikzpicture environment. No explanation, no markdown, no backticks.

Example output:
\\begin{tikzpicture}
  \\draw (0,0) circle (2cm);
  \\draw (-3,2) -- (3,2);
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
# LOSS FUNCTION
# ---------------------------------------------------------------------------

FLOAT_CMP = 0.05

class _Circle:
    def __init__(self, name, center, radius):
        self.name = name; self.center = center; self.radius = radius

class _Point:
    def __init__(self, name, x, y):
        self.name = name; self.x = x; self.y = y

class _Line:
    def __init__(self, name, point_1, point_2):
        self.name = name; self.point_1 = point_1; self.point_2 = point_2

def _radius(circle): return circle.radius

def _length(line):
    return ((line.point_1.x - line.point_2.x)**2 + (line.point_1.y - line.point_2.y)**2)**0.5

def _line_intersect(line1, line2):
    return LineString([(line1.point_1.x, line1.point_1.y), (line1.point_2.x, line1.point_2.y)]).intersects(
           LineString([(line2.point_1.x, line2.point_1.y), (line2.point_2.x, line2.point_2.y)]))

def _line_circle_intersect(line, circle):
    p1, p2 = line.point_1, line.point_2
    cx, cy = circle.center.x, circle.center.y
    dx, dy = p2.x - p1.x, p2.y - p1.y
    if dx == 0 and dy == 0:
        return math.hypot(p1.x - cx, p1.y - cy) <= circle.radius
    t = max(0, min(1, ((cx - p1.x)*dx + (cy - p1.y)*dy) / (dx*dx + dy*dy)))
    return math.hypot(cx - p1.x - t*dx, cy - p1.y - t*dy) <= circle.radius

def _intersect(a, b):
    if isinstance(a, _Line) and isinstance(b, _Line):   return _line_intersect(a, b)
    if isinstance(a, _Circle) and isinstance(b, _Circle):
        d = math.hypot(a.center.x - b.center.x, a.center.y - b.center.y)
        return d <= a.radius + b.radius and d >= abs(a.radius - b.radius)
    if isinstance(a, _Line) and isinstance(b, _Circle): return _line_circle_intersect(a, b)
    if isinstance(a, _Circle) and isinstance(b, _Line): return _line_circle_intersect(b, a)
    raise ValueError

def _tangent(line, circle):
    p1, p2 = line.point_1, line.point_2
    cx, cy = circle.center.x, circle.center.y
    dx, dy = p2.x - p1.x, p2.y - p1.y
    if dx == 0 and dy == 0: return False
    t = ((cx - p1.x)*dx + (cy - p1.y)*dy) / (dx*dx + dy*dy)
    if t < 0 or t > 1: return False
    dist = math.hypot(cx - p1.x - t*dx, cy - p1.y - t*dy)
    return math.isclose(dist, circle.radius, rel_tol=FLOAT_CMP)

def _angle(l1, l2):
    dx1, dy1 = l1.point_2.x - l1.point_1.x, l1.point_2.y - l1.point_1.y
    dx2, dy2 = l2.point_2.x - l2.point_1.x, l2.point_2.y - l2.point_1.y
    dot  = dx1*dx2 + dy1*dy2
    mag1 = math.hypot(dx1, dy1)
    mag2 = math.hypot(dx2, dy2)
    if mag1 == 0 or mag2 == 0: return 0.0
    a = math.degrees(math.acos(max(-1, min(1, dot / (mag1 * mag2)))))
    return min(a, 360 - a)

def _slope(line):
    dx = line.point_2.x - line.point_1.x
    dy = line.point_2.y - line.point_1.y
    return None if dx == 0 else dy / dx

def _parallel(l1, l2):      return _slope(l1) == _slope(l2)

def _perpendicular(l1, l2):
    m1, m2 = _slope(l1), _slope(l2)
    if m1 is None: return m2 == 0
    if m2 is None: return m1 == 0
    return math.isclose(m1 * m2, -1, rel_tol=FLOAT_CMP)

def _circle_tangent(c1, c2):
    d = math.hypot(c1.center.x - c2.center.x, c1.center.y - c2.center.y)
    return (math.isclose(d, c1.radius + c2.radius, rel_tol=FLOAT_CMP) or
            math.isclose(d, abs(c1.radius - c2.radius), rel_tol=FLOAT_CMP))

def _on_circle(point, circle):
    d = math.hypot(point.x - circle.center.x, point.y - circle.center.y)
    return math.isclose(d, circle.radius, rel_tol=FLOAT_CMP)

def _parse_fn(s):
    parts = s.split("(")
    if len(parts) != 2: raise ValueError(f"bad parse: {s}")
    fn = parts[0].strip()
    if parts[1][-1] != ")": raise ValueError(f"missing paren: {s}")
    params = [p.strip() for p in parts[1][:-1].split(",")]
    return fn, params

def check_constraints(pred_geo: list[str], truth_constr: list[str]) -> float:
    """Returns fraction of truth constraints satisfied. 1.0 = perfect."""
    if not truth_constr:
        return 1.0
    try:
        shape_dict = {}
        for pred in pred_geo:
            try:
                fn, params = _parse_fn(pred)
            except ValueError:
                continue
            if fn == "point" and len(params) == 3:
                shape_dict[params[0]] = _Point(params[0], float(params[1]), float(params[2]))
            elif fn == "circle" and len(params) == 3:
                shape_dict[params[0]] = _Circle(params[0], params[1], float(params[2]))
            elif fn == "line" and len(params) == 3:
                shape_dict[params[0]] = _Line(params[0], params[1], params[2])

        for shape in shape_dict.values():
            if isinstance(shape, _Line):
                if shape.point_1 in shape_dict and shape.point_2 in shape_dict:
                    shape.point_1 = shape_dict[shape.point_1]
                    shape.point_2 = shape_dict[shape.point_2]
            if isinstance(shape, _Circle):
                if shape.center in shape_dict:
                    shape.center = shape_dict[shape.center]

        correct = 0
        total   = 0
        for constraint in truth_constr:
            try:
                fn, params = _parse_fn(constraint)
                # skip bare declarations
                if fn == "point"  and len(params) == 1: continue
                if fn == "circle" and len(params) == 2: continue
                total += 1
                match fn:
                    case "radius":
                        if math.isclose(_radius(shape_dict[params[0]]), float(params[1]), rel_tol=FLOAT_CMP):
                            correct += 1
                    case "length":
                        if math.isclose(_length(shape_dict[params[0]]), float(params[1]), rel_tol=FLOAT_CMP):
                            correct += 1
                    case "intersect":
                        if _intersect(shape_dict[params[0]], shape_dict[params[1]]):
                            correct += 1
                    case "tangent":
                        if _tangent(shape_dict[params[0]], shape_dict[params[1]]):
                            correct += 1
                    case "parallel":
                        if _parallel(shape_dict[params[0]], shape_dict[params[1]]):
                            correct += 1
                    case "perpendicular":
                        if _perpendicular(shape_dict[params[0]], shape_dict[params[1]]):
                            correct += 1
                    case "circle_tangent":
                        if _circle_tangent(shape_dict[params[0]], shape_dict[params[1]]):
                            correct += 1
                    case "on_circle":
                        if _on_circle(shape_dict[params[0]], shape_dict[params[1]]):
                            correct += 1
                    case "angle":
                        if math.isclose(_angle(shape_dict[params[0]], shape_dict[params[1]]),
                                        float(params[2]), rel_tol=FLOAT_CMP):
                            correct += 1
                    case _:
                        total -= 1
            except (KeyError, ValueError, TypeError):
                pass

        return correct / total if total > 0 else 1.0

    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------------

def load_cache(path: str) -> dict:
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}

def save_cache(cache: dict, path: str):
    Path(path).write_text(json.dumps(cache, indent=2))

def cache_key(model: str, mode: str, inp: str) -> str:
    return hashlib.md5(f"{model}:{mode}:{inp}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# MODEL CLIENTS
# ---------------------------------------------------------------------------

def init_clients():
    return {
        "openai":    OpenAI(),
        "anthropic": anthropic.Anthropic(),
    }

def query(clients, model_name, provider, system, user_message) -> str:
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
# LOAD INPUTS  (all 3 modes use the same 50 dataset records)
# ---------------------------------------------------------------------------

def load_samples() -> tuple[list[str], list[str], list[list[str]]]:
    """Returns (nl_variants, geo_strings, constraints_list) from last 50 dataset records."""
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
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # --- Mode 1: NL -> TikZ ---
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

        # --- Mode 2: NL -> Geo -> Score ---
        print(f"\n  [Mode 2: NL -> Geo -> Score] {len(nls)} samples")
        for b, idx_batch in enumerate(batches):
            nls_batch    = [nls[i]    for i in idx_batch]
            constrs_batch = [constrs[i] for i in idx_batch]
            print(f"    Batch {b+1}/{len(batches)} ...", end=" ", flush=True)
            r2s = run_mode2_batch(clients, nls_batch, constrs_batch, model_name, provider, cache)
            avg = sum(r["score"] for r in r2s) / len(r2s)
            print(f"avg score {avg:.2f}")
            for r in r2s:
                all_results.append(r)
                scores[model_name]["mode2"]["score_sum"] += r["score"]
                scores[model_name]["mode2"]["total"]     += 1
            save_cache(cache, CACHE_PATH)
            time.sleep(DELAY)

        # --- Mode 3: GT Geo -> Geo -> Score ---
        print(f"\n  [Mode 3: GT Geo -> Geo -> Score] {len(geos)} samples")
        for b, idx_batch in enumerate(batches):
            geos_batch    = [geos[i]    for i in idx_batch]
            constrs_batch = [constrs[i] for i in idx_batch]
            print(f"    Batch {b+1}/{len(batches)} ...", end=" ", flush=True)
            r3s = run_mode3_batch(clients, geos_batch, constrs_batch, model_name, provider, cache)
            avg = sum(r["score"] for r in r3s) / len(r3s)
            print(f"avg score {avg:.2f}")
            for r in r3s:
                all_results.append(r)
                scores[model_name]["mode3"]["score_sum"] += r["score"]
                scores[model_name]["mode3"]["total"]     += 1
            save_cache(cache, CACHE_PATH)
            time.sleep(DELAY)

    # Save raw results
    with open(OUTPUT_PATH, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Compute summary
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

    # Print table
    print("\n" + "=" * 65)
    print(f"{'Model':<25} {'NL->TikZ':>12} {'NL->Geo':>12} {'GT->Geo':>12}")
    print("=" * 65)
    for model_name, s in summary.items():
        print(f"{model_name:<25} "
              f"{s['mode1_compile_rate']:>11.1%} "
              f"{s['mode2_avg_score']:>11.2f} "
              f"{s['mode3_avg_score']:>11.2f}")
    print("=" * 65)
    print(f"\nRaw results  → {OUTPUT_PATH}")
    print(f"Scores       → {SCORES_PATH}")
    print(f"Cache        → {CACHE_PATH}")
    print(f"Compiled tex → {COMPILED_DIR}/")
    print(f"\nTo compile all PDFs:")
    print(f"  cd {COMPILED_DIR} && for f in *.tex; do /Library/TeX/texbin/pdflatex -interaction=nonstopmode \"$f\"; done")


if __name__ == "__main__":
    run_benchmark()