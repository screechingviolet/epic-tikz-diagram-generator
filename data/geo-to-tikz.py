"""
geo_to_tikz.py

Converts EpicGeometryLanguage primitives directly to TikZ code.

Input format:
    point(name, x, y)
    line(name, p1, p2)
    circle(name, center, radius)

Output: a compilable tikzpicture environment
"""

import re
import math
from pathlib import Path


# ---------------------------------------------------------------------------
# PARSER
# ---------------------------------------------------------------------------

def parse_geometry(lines: list[str]) -> tuple[dict, dict, dict]:
    """Parse geometry primitives into three dicts keyed by name."""
    points, geo_lines, circles = {}, {}, {}

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        m = re.match(r'(\w+)\((.+)\)', raw)
        if not m:
            continue
        fn   = m.group(1)
        args = [a.strip() for a in m.group(2).split(",")]

        if fn == "point":
            name, x, y = args[0], float(args[1]), float(args[2])
            points[name] = (x, y)
        elif fn == "line":
            name, p1, p2 = args[0], args[1], args[2]
            geo_lines[name] = (p1, p2)
        elif fn == "circle":
            name, center, radius = args[0], args[1], float(args[2])
            circles[name] = (center, radius)

    return points, geo_lines, circles


# ---------------------------------------------------------------------------
# LAYOUT HELPERS
# ---------------------------------------------------------------------------

def _compute_scale(points: dict, circles: dict) -> float:
    """Choose a TikZ scale so the diagram fits in roughly 8x8 cm."""
    if not points:
        return 1.0

    xs = [x for x, y in points.values()]
    ys = [y for x, y in points.values()]

    # extend bounding box by circle radii
    for center, radius in circles.values():
        cx, cy = points[center]
        xs += [cx - radius, cx + radius]
        ys += [cy - radius, cy + radius]

    width  = max(xs) - min(xs) if len(xs) > 1 else 1.0
    height = max(ys) - min(ys) if len(ys) > 1 else 1.0
    span   = max(width, height, 0.1)

    # target ~8cm span
    return round(8.0 / span, 3)


def _label_name(name: str) -> str:
    """Convert internal name to LaTeX label: ω18 → $\\omega_{18}$, la → $l_a$."""
    # greek circle names
    m = re.match(r'ω(\d+)', name)
    if m:
        return f"$\\omega_{{{m.group(1)}}}$"
    m = re.match(r'γ(\d+)', name)
    if m:
        return f"$\\gamma_{{{m.group(1)}}}$"
    # line names like la, lb, lz
    m = re.match(r'l([a-z]+)', name)
    if m:
        return f"$l_{{{m.group(1)}}}$"
    # plain uppercase point names
    if re.match(r'^[A-Z]$', name):
        return f"${name}$"
    # fallback
    return f"${name}$"


def _label_offset(x: float, y: float, cx: float, cy: float) -> str:
    """Choose node placement based on position relative to diagram center."""
    dx = x - cx
    dy = y - cy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "above right"
    angle = math.degrees(math.atan2(dy, dx))
    if -45 <= angle < 45:
        return "right"
    elif 45 <= angle < 135:
        return "above"
    elif angle >= 135 or angle < -135:
        return "left"
    else:
        return "below"


# ---------------------------------------------------------------------------
# GENERATOR
# ---------------------------------------------------------------------------

def geo_to_tikz(geometry_lines: list[str]) -> str:
    """Convert a list of geometry primitives to a TikZ picture string."""
    points, geo_lines, circles = parse_geometry(geometry_lines)

    scale = _compute_scale(points, circles)

    # diagram center for label placement heuristic
    if points:
        cx = sum(x for x, y in points.values()) / len(points)
        cy = sum(y for x, y in points.values()) / len(points)
    else:
        cx, cy = 0.0, 0.0

    tikz_lines = [f"\\begin{{tikzpicture}}[scale={scale}]"]

    # --- circles (draw first so points appear on top) ---
    if circles:
        tikz_lines.append("  % circles")
    for name, (center, radius) in circles.items():
        if center not in points:
            continue
        px, py = points[center]
        label  = _label_name(name)
        tikz_lines.append(
            f"  \\draw ({px}, {py}) circle ({radius}) "
            f"node[above={radius}cm] {{{label}}};"
        )

    # --- lines ---
    if geo_lines:
        tikz_lines.append("  % lines")
    for name, (p1, p2) in geo_lines.items():
        if p1 not in points or p2 not in points:
            continue
        x1, y1 = points[p1]
        x2, y2 = points[p2]
        label  = _label_name(name)
        tikz_lines.append(
            f"  \\draw ({x1}, {y1}) -- ({x2}, {y2}) "
            f"node[midway, above] {{{label}}};"
        )

    # --- points (draw last so they appear on top of lines/circles) ---
    if points:
        tikz_lines.append("  % points")
    for name, (x, y) in points.items():
        label    = _label_name(name)
        placement = _label_offset(x, y, cx, cy)
        tikz_lines.append(
            f"  \\filldraw ({x}, {y}) circle (2pt) "
            f"node[{placement}] {{{label}}};"
        )

    tikz_lines.append("\\end{tikzpicture}")
    return "\n".join(tikz_lines)


def geo_to_tex(geometry_lines: list[str]) -> str:
    """Wrap the tikzpicture in a full compilable .tex document."""
    return (
        "\\documentclass{standalone}\n"
        "\\usepackage{tikz}\n"
        "\\begin{document}\n"
        + geo_to_tikz(geometry_lines) + "\n"
        "\\end{document}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # quick demo
        demo = [
            "point(A, 0.0, 0.0)",
            "point(B, 3.0, 0.0)",
            "point(C, 1.5, 2.5)",
            "line(la, A, B)",
            "line(lb, B, C)",
            "line(lc, A, C)",
            "circle(ω0, A, 1.5)",
        ]
        demo = ["point(Y, -2.1434, 7.5626)", "point(H, 0.5963, 5.6207)", "point(C, 0.5903, 1.3567)", "line(lj, H, Y)", "line(lb, C, Y)", "line(lp, H, C)", "circle(\u03b312, Y, 1.4987)", "circle(\u03c918, H, 1.8594)"]
        print(geo_to_tex(demo))

    elif len(sys.argv) == 2:
        # read from file
        path  = Path(sys.argv[1])
        lines = path.read_text().splitlines()
        print(geo_to_tex(lines))

    elif len(sys.argv) == 3:
        # read from file, write to output
        path     = Path(sys.argv[1])
        out_path = Path(sys.argv[2])
        lines    = path.read_text().splitlines()
        out_path.write_text(geo_to_tex(lines))
        print(f"Written to {out_path}")