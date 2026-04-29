import json
import subprocess
from pathlib import Path

RUN_NAME     = "run1"   # match benchmark.py
RESULTS_PATH = f"benchmark_results_{RUN_NAME}.jsonl"
OUTPUT_DIR   = Path(f"clean_outputs_{RUN_NAME}")
PDFLATEX     = "/Library/TeX/texbin/pdflatex"

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load results
# ---------------------------------------------------------------------------

with open(RESULTS_PATH) as f:
    records = [json.loads(line) for line in f]

mode1 = [r for r in records if r["mode"] == "mode1_nl_to_tikz" and r["compiles"]]

# ---------------------------------------------------------------------------
# 2. Clean NL + TikZ text files
# ---------------------------------------------------------------------------

for model in ["gpt-4o-mini", "claude-haiku-4-5-20251001"]:
    pairs = [r for r in mode1 if r["model"] == model]
    out   = OUTPUT_DIR / f"{model.replace('-','_')}_nl_tikz.txt"
    with open(out, "w") as f:
        for i, r in enumerate(pairs, 1):
            f.write(f"{'='*60}\n")
            f.write(f"#{i}\n")
            f.write(f"NL:\n{r['input']}\n\n")
            f.write(f"TikZ:\n{r['tikz']}\n\n")
    print(f"Saved {len(pairs)} pairs → {out}")

# ---------------------------------------------------------------------------
# 3. Compile PDFs
# ---------------------------------------------------------------------------

pdf_dir = OUTPUT_DIR / "pdfs"
pdf_dir.mkdir(exist_ok=True)

compiled = 0
failed   = 0

for r in mode1:
    model    = r["model"].replace("-", "_")
    tikz     = r["tikz"]
    idx      = mode1.index(r) + 1
    name     = f"{model}_{idx:03d}"

    tex = (
        r"\documentclass[border=5pt]{standalone}" + "\n"
        r"\usepackage{tikz}" + "\n"
        r"\begin{document}" + "\n"
        + tikz + "\n"
        + r"\end{document}"
    )

    tex_path = pdf_dir / f"{name}.tex"
    tex_path.write_text(tex)

    result = subprocess.run(
        [PDFLATEX, "-interaction=nonstopmode", f"{name}.tex"],
        capture_output=True,
        cwd=pdf_dir,
    )

    if result.returncode == 0:
        compiled += 1
        print(f"  ✓ {name}.pdf")
    else:
        failed += 1
        print(f"  ✗ {name} failed")
        tex_path.unlink(missing_ok=True)

for ext in ["*.aux", "*.log"]:
    for f in pdf_dir.glob(ext):
        f.unlink()

print(f"\nDone: {compiled} PDFs, {failed} failed → {pdf_dir}/")

# ---------------------------------------------------------------------------
# 4. Index HTML
# ---------------------------------------------------------------------------

by_nl     = {}
seen_nls  = []
for r in mode1:
    nl = r["input"]
    if nl not in by_nl:
        by_nl[nl] = {}
        seen_nls.append(nl)
    by_nl[nl][r["model"]] = r

html = [
    "<!DOCTYPE html><html><head>",
    "<style>",
    "body { font-family: monospace; max-width: 960px; margin: auto; padding: 20px; }",
    "h2 { border-bottom: 2px solid #333; }",
    ".entry { margin-bottom: 40px; }",
    ".nl { background: #f0f0f0; padding: 10px; border-radius: 4px; margin-bottom: 8px; }",
    ".models { display: flex; gap: 20px; }",
    ".model { flex: 1; }",
    "iframe { width: 100%; height: 300px; border: 1px solid #ccc; }",
    "</style></head><body>",
    f"<h1>Benchmark {RUN_NAME} — NL → TikZ</h1>",
]

for i, nl in enumerate(seen_nls, 1):
    html.append(f'<div class="entry"><h2>#{i}</h2>')
    html.append(f'<div class="nl">{nl}</div>')
    html.append('<div class="models">')
    for model in ["gpt-4o-mini", "claude-haiku-4-5-20251001"]:
        r = by_nl[nl].get(model)
        model_safe = model.replace("-", "_")
        html.append(f'<div class="model"><b>{model}</b><br>')
        if r and r["compiles"]:
            idx      = mode1.index(r) + 1
            pdf_name = f"{model_safe}_{idx:03d}.pdf"
            html.append(f'<iframe src="pdfs/{pdf_name}"></iframe>')
        else:
            html.append('<p style="color:red">Did not compile</p>')
        html.append('</div>')
    html.append('</div></div>')

html.append("</body></html>")

index_path = OUTPUT_DIR / "index.html"
index_path.write_text("\n".join(html))
print(f"Index → {index_path}")
print("Open in browser to browse results side by side.")