import json
import subprocess
import tempfile
from pathlib import Path

RESULTS_PATH = "benchmark_results.jsonl"
OUTPUT_DIR   = Path("clean_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Extract clean NL + TikZ text files
# ---------------------------------------------------------------------------

with open(RESULTS_PATH) as f:
    records = [json.loads(line) for line in f]

mode1 = [r for r in records if r["mode"] == "mode1_nl_to_tikz" and r["compiles"]]

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
# 2. Compile each TikZ to PDF
# ---------------------------------------------------------------------------

PDFLATEX = "/Library/TeX/texbin/pdflatex"

pdf_dir = OUTPUT_DIR / "pdfs"
pdf_dir.mkdir(exist_ok=True)

compiled = 0
failed   = 0

for r in mode1:
    model   = r["model"].replace("-", "_")
    nl      = r["input"]
    tikz    = r["tikz"]
    idx     = mode1.index(r) + 1
    name    = f"{model}_{idx:03d}"

    tex = (
        r"\documentclass[border=5pt]{standalone}" + "\n"
        r"\usepackage{tikz}" + "\n"
        r"\begin{document}" + "\n"
        + tikz + "\n"
        + r"\end{document}"
    )

    # write .tex next to output pdf so pdflatex aux files land there
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

# clean up aux files
for ext in ["*.aux", "*.log"]:
    for f in pdf_dir.glob(ext):
        f.unlink()

print(f"\nDone: {compiled} PDFs compiled, {failed} failed → {pdf_dir}/")

# ---------------------------------------------------------------------------
# 3. Also write a companion index HTML so you can browse NL + PDF side by side
# ---------------------------------------------------------------------------

html_lines = [
    "<!DOCTYPE html><html><head>",
    "<style>",
    "body { font-family: monospace; max-width: 900px; margin: auto; padding: 20px; }",
    "h2 { border-bottom: 2px solid #333; }",
    ".entry { margin-bottom: 40px; }",
    ".nl { background: #f0f0f0; padding: 10px; border-radius: 4px; margin-bottom: 8px; }",
    ".models { display: flex; gap: 20px; }",
    ".model { flex: 1; }",
    "iframe { width: 100%; height: 300px; border: 1px solid #ccc; }",
    "</style></head><body>",
    "<h1>Benchmark Results — NL → TikZ</h1>",
]

# group by input NL
seen_nls = []
by_nl = {}
for r in mode1:
    nl = r["input"]
    if nl not in by_nl:
        by_nl[nl] = {}
        seen_nls.append(nl)
    by_nl[nl][r["model"]] = r

for i, nl in enumerate(seen_nls, 1):
    html_lines.append(f'<div class="entry">')
    html_lines.append(f'<h2>#{i}</h2>')
    html_lines.append(f'<div class="nl">{nl}</div>')
    html_lines.append('<div class="models">')
    for model in ["gpt-4o-mini", "claude-haiku-4-5-20251001"]:
        r = by_nl[nl].get(model)
        model_safe = model.replace("-", "_")
        html_lines.append(f'<div class="model"><b>{model}</b><br>')
        if r and r["compiles"]:
            idx      = mode1.index(r) + 1
            pdf_name = f"{model_safe}_{idx:03d}.pdf"
            html_lines.append(f'<iframe src="pdfs/{pdf_name}"></iframe>')
        else:
            html_lines.append('<p style="color:red">Did not compile</p>')
        html_lines.append('</div>')
    html_lines.append('</div></div>')

html_lines.append("</body></html>")

index_path = OUTPUT_DIR / "index.html"
index_path.write_text("\n".join(html_lines))
print(f"Index HTML → {index_path}")
print("Open clean_outputs/index.html in your browser to browse results side by side.")