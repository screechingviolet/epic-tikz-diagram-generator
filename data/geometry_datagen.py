"""
geometry_datagen.py

Full pipeline for generating geometry training data:
  1. Randomly generate scenes (points, lines, circles)
  2. Nudge for interesting constraints (tangency, perpendicularity, etc.)
  3. Extract all true constraints from the scene
  4. Serialize to formal language
  5. Generate natural language variants via Claude API
  6. Save dataset to JSONL
"""

import itertools
import json
import random
import time
from pathlib import Path

import anthropic
import numpy as np

# ---------------------------------------------------------------------------
# 1. GEOMETRY PRIMITIVES
# ---------------------------------------------------------------------------

def point_to_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Perpendicular distance from point p to the infinite line through a and b."""
    ab = b - a
    norm = np.linalg.norm(ab)
    if norm < 1e-10:
        return np.linalg.norm(p - a)
    ap = p - a
    return abs(np.cross(ab, ap)) / norm


def line_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a))


def angle_between(d1: np.ndarray, d2: np.ndarray) -> float:
    """Angle in degrees between two direction vectors."""
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    cos_a = np.clip(np.dot(d1, d2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


# ---------------------------------------------------------------------------
# 2. NUDGES  (each nudge modifies points in-place to force a constraint)
# ---------------------------------------------------------------------------

def nudge_line_tangent_to_circle(points, line_name, line, circle_name, circle):
    """
    Move both endpoints of the line so that it becomes tangent to the circle.
    The line is shifted to be at exactly radius distance from the center,
    while keeping its direction random.
    """
    p1n, p2n = line
    cn, r = circle
    center = points[cn]

    # Current line direction
    direction = points[p2n] - points[p1n]
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return
    direction /= norm
    perp = np.array([-direction[1], direction[0]])

    # Choose which side of the circle to place the line
    side = random.choice([-1, 1])
    offset = center + side * perp * r

    # Place both points along the tangent line at random positions
    t1 = random.uniform(-3, 0)
    t2 = random.uniform(0, 3)
    points[p1n] = offset + t1 * direction
    points[p2n] = offset + t2 * direction


def nudge_lines_parallel(points, l1, l2):
    """Make l2 parallel to l1 by rotating l2's second point."""
    p1n, p2n = l1
    p3n, p4n = l2
    direction = points[p2n] - points[p1n]
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return
    direction /= norm
    length2 = line_length(points[p3n], points[p4n])
    points[p4n] = points[p3n] + direction * max(length2, 0.5)


def nudge_lines_perpendicular(points, l1, l2):
    """Make l2 perpendicular to l1."""
    p1n, p2n = l1
    p3n, p4n = l2
    direction = points[p2n] - points[p1n]
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return
    direction /= norm
    perp = np.array([-direction[1], direction[0]])
    length2 = line_length(points[p3n], points[p4n])
    points[p4n] = points[p3n] + perp * max(length2, 0.5)


def nudge_circles_externally_tangent(points, c1, c2):
    """Move c2's center so the two circles are externally tangent."""
    cn1, r1 = c1
    cn2, r2 = c2
    direction = points[cn2] - points[cn1]
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        direction = np.array([1.0, 0.0])
    else:
        direction /= norm
    points[cn2] = points[cn1] + direction * (r1 + r2)


def nudge_point_on_circle(points, point_name, circle):
    """Move a point so it lies exactly on the circle."""
    cn, r = circle
    center = points[cn]
    direction = points[point_name] - center
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        direction = np.array([1.0, 0.0])
    else:
        direction /= norm
    points[point_name] = center + direction * r


# ---------------------------------------------------------------------------
# 3. SCENE GENERATION
# ---------------------------------------------------------------------------

NUDGE_TYPES = ["tangent", "parallel", "perpendicular", "ext_tangent", "none"]


def random_scene(
    n_points: int = None,
    n_lines: int = None,
    n_circles: int = None,
    nudge_probability: float = 0.6,
):
    """
    Generate a random geometric scene and return:
      points   : dict[str, np.ndarray]
      lines    : dict[str, tuple[str, str]]         (point_name, point_name)
      circles  : dict[str, tuple[str, float]]       (center_point_name, radius)
      constraints: list of constraint tuples
    """
    # Randomise complexity
    n_points  = n_points  or random.randint(3, 6)
    n_lines   = n_lines   or random.randint(1, 3)
    n_circles = n_circles or random.randint(1, 2)

    # ---- place points ----
    points = {f"P{i}": np.random.uniform(-5, 5, 2).astype(float)
              for i in range(n_points)}
    point_names = list(points.keys())

    # ---- build lines (distinct endpoints) ----
    lines = {}
    used_pairs = set()
    for i in range(n_lines):
        for _ in range(20):   # retry if we get a duplicate pair
            p1, p2 = random.sample(point_names, 2)
            pair = tuple(sorted([p1, p2]))
            if pair not in used_pairs:
                used_pairs.add(pair)
                lines[f"L{i}"] = (p1, p2)
                break

    # ---- build circles (center = existing point, random radius) ----
    circles = {}
    for i in range(n_circles):
        center = random.choice(point_names)
        radius = round(random.uniform(0.5, 3.0), 4)
        circles[f"C{i}"] = (center, radius)

    # ---- apply random nudges ----
    line_names   = list(lines.keys())
    circle_names = list(circles.keys())

    if line_names and circle_names and random.random() < nudge_probability:
        lname = random.choice(line_names)
        cname = random.choice(circle_names)
        cn = circles[cname][0]
        # If the center is an endpoint, reassign that endpoint to a free point
        p1n, p2n = lines[lname]
        if cn in (p1n, p2n):
            free = [p for p in point_names if p != cn]
            if free:
                other = random.choice(free)
                lines[lname] = (other, p2n) if p1n == cn else (p1n, other)
        nudge_line_tangent_to_circle(points, lname, lines[lname], cname, circles[cname])

    if len(line_names) >= 2 and random.random() < nudge_probability:
        l1, l2 = random.sample(line_names, 2)
        if random.random() < 0.5:
            nudge_lines_parallel(points, lines[l1], lines[l2])
        else:
            nudge_lines_perpendicular(points, lines[l1], lines[l2])

    if len(circle_names) >= 2 and random.random() < nudge_probability:
        c1, c2 = random.sample(circle_names, 2)
        nudge_circles_externally_tangent(points, circles[c1], circles[c2])

    constraints = extract_constraints(points, lines, circles)
    return points, lines, circles, constraints


# ---------------------------------------------------------------------------
# 4. CONSTRAINT EXTRACTION
# ---------------------------------------------------------------------------

def extract_constraints(points, lines, circles, eps=1e-6):
    constraints = []

    # Always true: lengths and radii
    for lname, (p1n, p2n) in lines.items():
        length = line_length(points[p1n], points[p2n])
        if length > 1e-6:
            constraints.append(("length", lname, round(length, 4)))

    for cname, (cn, r) in circles.items():
        constraints.append(("radius", cname, round(r, 4)))

    # Tangency: line to circle
    for lname, (p1n, p2n) in lines.items():
        for cname, (cn, r) in circles.items():
            dist = point_to_line_distance(points[cn], points[p1n], points[p2n])
            if abs(dist - r) < eps:
                constraints.append(("tangent", lname, cname))

    # Line-line relationships
    for (l1, (a1, b1)), (l2, (a2, b2)) in itertools.combinations(lines.items(), 2):
        d1 = points[b1] - points[a1]
        d2 = points[b2] - points[a2]
        cross = float(np.cross(d1, d2))
        dot   = float(np.dot(d1, d2))
        if abs(cross) < eps:
            constraints.append(("parallel", l1, l2))
        elif abs(dot) < eps:
            constraints.append(("perpendicular", l1, l2))

    # Circle-circle relationships
    for (c1, (cn1, r1)), (c2, (cn2, r2)) in itertools.combinations(circles.items(), 2):
        dist = float(np.linalg.norm(points[cn1] - points[cn2]))
        if abs(dist - (r1 + r2)) < eps:
            constraints.append(("externally_tangent", c1, c2))
        elif abs(dist - abs(r1 - r2)) < eps:
            constraints.append(("internally_tangent", c1, c2))

    # Point on circle
    for pname, pcoord in points.items():
        for cname, (cn, r) in circles.items():
            if pname == cn:
                continue
            dist = float(np.linalg.norm(pcoord - points[cn]))
            if abs(dist - r) < eps:
                constraints.append(("on_circle", pname, cname))

    return constraints


# ---------------------------------------------------------------------------
# 5. SERIALISER  →  formal language string
# ---------------------------------------------------------------------------

def serialize_scene(points, lines, circles, constraints) -> str:
    """
    Produces a clean s-expression style formal description, e.g.:

      (point P0 1.23 -4.56)
      (line L0 P0 P1)
      (circle C0 P0 2.0)
      (tangent L0 C0)
      (length L0 3.14)
    """
    lines_out = []

    for name, coord in points.items():
        lines_out.append(f"(point {name} {coord[0]:.4f} {coord[1]:.4f})")

    for name, (p1, p2) in lines.items():
        lines_out.append(f"(line {name} {p1} {p2})")

    for name, (center, radius) in circles.items():
        lines_out.append(f"(circle {name} {center} {radius:.4f})")

    for c in constraints:
        if c[0] in ("length", "radius"):
            lines_out.append(f"({c[0]} {c[1]} {c[2]:.4f})")
        elif len(c) == 3:
            lines_out.append(f"({c[0]} {c[1]} {c[2]})")
        else:
            lines_out.append(f"({' '.join(str(x) for x in c)})")

    return "\n".join(lines_out)


def constraints_only_str(constraints) -> str:
    """Human-readable constraint summary (used in the LLM prompt)."""
    parts = []
    for c in constraints:
        if c[0] == "tangent":
            parts.append(f"line {c[1]} is tangent to circle {c[2]}")
        elif c[0] == "parallel":
            parts.append(f"lines {c[1]} and {c[2]} are parallel")
        elif c[0] == "perpendicular":
            parts.append(f"lines {c[1]} and {c[2]} are perpendicular")
        elif c[0] == "externally_tangent":
            parts.append(f"circles {c[1]} and {c[2]} are externally tangent")
        elif c[0] == "internally_tangent":
            parts.append(f"circles {c[1]} and {c[2]} are internally tangent")
        elif c[0] == "on_circle":
            parts.append(f"point {c[1]} lies on circle {c[2]}")
        elif c[0] == "length":
            parts.append(f"line {c[1]} has length {c[2]}")
        elif c[0] == "radius":
            parts.append(f"circle {c[1]} has radius {c[2]}")
    return "\n".join(f"- {p}" for p in parts)


# ---------------------------------------------------------------------------
# 6. NL VARIANT GENERATION  (calls Claude)
# ---------------------------------------------------------------------------

# def generate_nl_variants(
#     formal: str,
#     constraint_summary: str,
#     n_variants: int = 8,
#     client: anthropic.Anthropic = None,
# ) -> list[str]:
#     """
#     Ask Claude to produce n_variants natural language descriptions
#     of the geometric scene described by `formal`.
#     Returns a list of strings.
#     """
#     if client is None:
#         client = anthropic.Anthropic()

#     prompt = f"""You are generating training data for a geometry diagram system that converts natural language into formal geometric descriptions.

# Here is a formal geometric description:
# {formal}

# Key relationships in this scene:
# {constraint_summary}

# Generate {n_variants} different natural language descriptions that a student, teacher, or textbook might use to describe this diagram. Rules:
# - Each description must be semantically equivalent (same objects and constraints)
# - Vary vocabulary: use synonyms (tangent / just touches / grazes, perpendicular / at right angles / 90 degrees, etc.)
# - Vary structure: some terse, some verbose, some conversational, some formal
# - Do NOT mention coordinate values like (1.23, -4.56) — describe relationships only
# - Do NOT number the descriptions

# Respond with ONLY a JSON array of strings, no other text. Example format:
# ["description one", "description two"]"""

#     response = client.messages.create(
#         model="claude-opus-4-5",
#         max_tokens=1500,
#         messages=[{"role": "user", "content": prompt}],
#     )

#     raw = response.content[0].text.strip()
#     # Strip markdown code fences if present
#     if raw.startswith("```"):
#         raw = raw.split("```")[1]
#         if raw.startswith("json"):
#             raw = raw[4:]
#     raw = raw.strip()

#     try:
#         variants = json.loads(raw)
#         return [v for v in variants if isinstance(v, str)]
#     except json.JSONDecodeError:
#         # Fallback: split on newlines
#         return [line.strip().strip('"') for line in raw.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 7. DATASET LOOP
# ---------------------------------------------------------------------------

def generate_dataset(
    n_scenes: int = 100,
    n_variants_per_scene: int = 8,
    output_path: str = "geometry_dataset.jsonl",
    delay: float = 0.5,   # seconds between API calls to avoid rate limits
):
    """
    Generate n_scenes random scenes, produce NL variants for each,
    and write to a JSONL file where each line is:
      {
        "formal": "<s-expression string>",
        "constraints": [list of constraint tuples],
        "nl_variants": ["variant 1", "variant 2", ...]
      }
    """
    # client = anthropic.Anthropic()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped   = 0

    with open(out, "w") as f:
        for i in range(n_scenes):
            print(f"Scene {i+1}/{n_scenes} ...", end=" ", flush=True)

            points, lines, circles, constraints = random_scene()

            # Skip degenerate scenes (no interesting constraints beyond lengths/radii)
            # interesting = [c for c in constraints
            #                if c[0] not in ("length", "radius")]
            # if not interesting:
            #     print("skipped (no interesting constraints)")
            #     skipped += 1
            #     continue

            formal   = serialize_scene(points, lines, circles, constraints)
            summary  = constraints_only_str(constraints)

            # try:
            #     variants = generate_nl_variants(
            #         formal, summary, n_variants_per_scene, client
            #     )
            # except Exception as e:
            #     print(f"API error: {e} — skipping")
            #     skipped += 1
            #     continue

            record = {
                "formal":      formal,
                "constraints": [list(c) for c in constraints],
                "nl_variants": [],  # change to "variants" later
            }
            f.write(json.dumps(record) + "\n")
            generated += 1
            # print(f"ok ({len(variants)} variants, {len(interesting)} interesting constraints)")s

            time.sleep(delay)

    print(f"\nDone. {generated} scenes written, {skipped} skipped → {out}")
    return out


# ---------------------------------------------------------------------------
# 8. QUICK DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Single scene demo ===\n")

    points, lines, circles, constraints = random_scene()
    formal  = serialize_scene(points, lines, circles, constraints)
    summary = constraints_only_str(constraints)

    print("Formal:\n" + formal)
    print("\nConstraint summary:\n" + summary)

    # print("\nGenerating NL variants...")
    # variants = generate_nl_variants(formal, summary, n_variants=5)
    # for i, v in enumerate(variants, 1):
    #     print(f"  {i}. {v}")

    print("\n=== Generating small dataset (10 scenes) ===\n")
    generate_dataset(n_scenes=10, n_variants_per_scene=5, output_path="demo_dataset.jsonl")