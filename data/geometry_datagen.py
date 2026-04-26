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

import numpy as np

from data_types import (
    Point, Line, Circle,
    Radius, Length, Intersect, Tangent, Angle, Parallel, Perpendicular,
    CircleTangent, OnCircle
)

# ---------------------------------------------------------------------------
# 0. HELPER  (keep math clean while using dataclasses)
# ---------------------------------------------------------------------------

def pt(p: Point) -> np.ndarray:
    """Convert a Point dataclass to a numpy array for math operations."""
    return np.array([p.x, p.y])

def set_pt(p: Point, arr: np.ndarray):
    """Write a numpy array back into a Point dataclass in-place."""
    p.x = float(arr[0])
    p.y = float(arr[1])


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
    return abs(ab[0] * ap[1] - ab[1] * ap[0]) / norm


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
# 2. NUDGES  (now take dataclass objects directly)
# ---------------------------------------------------------------------------

def nudge_line_tangent_to_circle(line: Line, circle: Circle):
    center = pt(circle.center)
    direction = pt(line.p2) - pt(line.p1)
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return
    direction /= norm
    perp = np.array([-direction[1], direction[0]])

    side = random.choice([-1, 1])
    offset = center + side * perp * circle.radius

    t1 = random.uniform(-3, 0)
    t2 = random.uniform(0, 3)
    set_pt(line.p1, offset + t1 * direction)
    set_pt(line.p2, offset + t2 * direction)


def nudge_lines_parallel(l1: Line, l2: Line):
    direction = pt(l1.p2) - pt(l1.p1)
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return
    direction /= norm
    length2 = line_length(pt(l2.p1), pt(l2.p2))
    set_pt(l2.p2, pt(l2.p1) + direction * max(length2, 0.5))


def nudge_lines_perpendicular(l1: Line, l2: Line):
    direction = pt(l1.p2) - pt(l1.p1)
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return
    direction /= norm
    perp = np.array([-direction[1], direction[0]])
    length2 = line_length(pt(l2.p1), pt(l2.p2))
    set_pt(l2.p2, pt(l2.p1) + perp * max(length2, 0.5))


def nudge_circles_externally_tangent(c1: Circle, c2: Circle):
    direction = pt(c2.center) - pt(c1.center)
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        direction = np.array([1.0, 0.0])
    else:
        direction /= norm
    set_pt(c2.center, pt(c1.center) + direction * (c1.radius + c2.radius))


def nudge_point_on_circle(point: Point, circle: Circle):
    direction = pt(point) - pt(circle.center)
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        direction = np.array([1.0, 0.0])
    else:
        direction /= norm
    set_pt(point, pt(circle.center) + direction * circle.radius)


# ---------------------------------------------------------------------------
# 3. SCENE GENERATION
# ---------------------------------------------------------------------------

def random_scene(
    n_points: int = None,
    n_lines: int = None,
    n_circles: int = None,
    nudge_probability: float = 0.6,
):
    n_points  = n_points  or random.randint(3, 6)
    n_lines   = n_lines   or random.randint(1, 3)
    n_circles = n_circles or random.randint(1, 2)

    points = {
        f"P{i}": Point(name=f"P{i}", x=float(x), y=float(y))
        for i, (x, y) in enumerate(np.random.uniform(-5, 5, (n_points, 2)))
    }
    point_list = list(points.values())

    lines = {}
    used_pairs = set()
    for i in range(n_lines):
        for _ in range(20):
            p1, p2 = random.sample(point_list, 2)
            pair = tuple(sorted([p1.name, p2.name]))
            if pair not in used_pairs:
                used_pairs.add(pair)
                lines[f"L{i}"] = Line(name=f"L{i}", p1=p1, p2=p2)
                break

    circles = {}
    for i in range(n_circles):
        center = random.choice(point_list)
        radius = round(random.uniform(0.5, 3.0), 4)
        circles[f"C{i}"] = Circle(name=f"C{i}", center=center, radius=radius)

    line_list   = list(lines.values())
    circle_list = list(circles.values())

    # nudges — pass dataclass objects directly now
    if line_list and circle_list and random.random() < nudge_probability:
        line   = random.choice(line_list)
        circle = random.choice(circle_list)
        # make sure the circle's center isn't an endpoint of the line
        if circle.center in (line.p1, line.p2):
            free = [p for p in point_list if p is not circle.center]
            if free:
                line.p1 = random.choice(free)
        nudge_line_tangent_to_circle(line, circle)

    if len(line_list) >= 2 and random.random() < nudge_probability:
        l1, l2 = random.sample(line_list, 2)
        if random.random() < 0.5:
            nudge_lines_parallel(l1, l2)
        else:
            nudge_lines_perpendicular(l1, l2)

    if len(circle_list) >= 2 and random.random() < nudge_probability:
        c1, c2 = random.sample(circle_list, 2)
        nudge_circles_externally_tangent(c1, c2)

    constraints = extract_constraints(points, lines, circles)
    return points, lines, circles, constraints


# ---------------------------------------------------------------------------
# 4. CONSTRAINT CHECKERS  (now take dataclass objects directly)
# ---------------------------------------------------------------------------

def check_tangent(line: Line, circle: Circle, eps=1e-6) -> Tangent | None:
    dist = point_to_line_distance(pt(circle.center), pt(line.p1), pt(line.p2))
    if abs(dist - circle.radius) < eps:
        return Tangent(circle_name=circle.name, line_name=line.name)
    return None


def check_parallel(l1: Line, l2: Line, eps=1e-6) -> Parallel | None:
    d1 = pt(l1.p2) - pt(l1.p1)
    d2 = pt(l2.p2) - pt(l2.p1)
    if abs(float(d1[0] * d2[1] - d1[1] * d2[0])) < eps:
        return Parallel(line_1_name=l1.name, line_2_name=l2.name)
    return None

def check_perpendicular(l1: Line, l2: Line, eps=1e-6) -> Perpendicular | None:
    d1 = pt(l1.p2) - pt(l1.p1)
    d2 = pt(l2.p2) - pt(l2.p1)
    if abs(float(np.dot(d1, d2))) < eps:
        return Perpendicular(line_1_name=l1.name, line_2_name=l2.name)
    return None


def check_angle(l1: Line, l2: Line, eps=1e-6) -> Angle | None:
    d1 = pt(l1.p2) - pt(l1.p1)
    d2 = pt(l2.p2) - pt(l2.p1)
    cross = float(d1[0] * d2[1] - d1[1] * d2[0])
    dot   = float(np.dot(d1, d2))
    if abs(cross) < eps or abs(dot) < eps:
        return None
    ang = angle_between(d1, d2)
    small = ang if ang <= 180 else 360 - ang
    return Angle(line_1_name=l1.name, line_2_name=l2.name, angle=round(small, 2))


def check_circle_tangent(c1: Circle, c2: Circle, eps=1e-6) -> CircleTangent | None:
    dist = float(np.linalg.norm(pt(c1.center) - pt(c2.center)))
    if abs(dist - (c1.radius + c2.radius)) < eps:
        return CircleTangent(circle_1_name=c1.name, circle_2_name=c2.name, kind="external")
    if abs(dist - abs(c1.radius - c2.radius)) < eps:
        return CircleTangent(circle_1_name=c1.name, circle_2_name=c2.name, kind="internal")
    return None


def check_on_circle(point: Point, circle: Circle, eps=1e-6) -> OnCircle | None:
    if point is circle.center:
        return None
    dist = float(np.linalg.norm(pt(point) - pt(circle.center)))
    if abs(dist - circle.radius) < eps:
        return OnCircle(point_name=point.name, circle_name=circle.name)
    return None


# ---------------------------------------------------------------------------
# EXTRACT CONSTRAINTS
# ---------------------------------------------------------------------------

def extract_constraints(points, lines, circles) -> list:
    constraints = []

    for line in lines.values():
        length = line_length(pt(line.p1), pt(line.p2))
        if length > 1e-6:
            constraints.append(Length(line_name=line.name, dist=round(length, 4)))

    for circle in circles.values():
        constraints.append(Radius(circle_name=circle.name, rad=round(circle.radius, 4)))

    for line in lines.values():
        for circle in circles.values():
            result = check_tangent(line, circle)
            if result:
                constraints.append(result)

    for l1, l2 in itertools.combinations(lines.values(), 2):
        result = check_parallel(l1, l2) or check_perpendicular(l1, l2) or check_angle(l1, l2)
        if result:
            constraints.append(result)

    for c1, c2 in itertools.combinations(circles.values(), 2):
        result = check_circle_tangent(c1, c2)
        if result:
            constraints.append(result)

    for point in points.values():
        for circle in circles.values():
            result = check_on_circle(point, circle)
            if result:
                constraints.append(result)

    return constraints


# ---------------------------------------------------------------------------
# 5. SERIALISER
# ---------------------------------------------------------------------------

def serialize_geometry(points, lines, circles) -> list[str]:
    """Just the objects — fed to the model as pred_geo."""
    out = []
    for p in points.values():
        out.append(f"point({p.name}, {p.x:.4f}, {p.y:.4f})")
    for l in lines.values():
        out.append(f"line({l.name}, {l.p1.name}, {l.p2.name})")
    for c in circles.values():
        out.append(f"circle({c.name}, {c.center.name}, {c.radius:.4f})")
    return out

def serialize_constraints(constraints) -> list[str]:
    """Just the constraints — used as truth_constr in the loss function."""
    out = []
    for c in constraints:
        if isinstance(c, Length):
            out.append(f"length({c.line_name}, {c.dist:.4f})")
        elif isinstance(c, Radius):
            out.append(f"radius({c.circle_name}, {c.rad:.4f})")
        elif isinstance(c, Tangent):
            out.append(f"tangent({c.line_name}, {c.circle_name})")
        elif isinstance(c, Parallel):
            out.append(f"parallel({c.line_1_name}, {c.line_2_name})")
        elif isinstance(c, Perpendicular):
            out.append(f"perpendicular({c.line_1_name}, {c.line_2_name})")
        elif isinstance(c, Angle):
            out.append(f"angle({c.line_1_name}, {c.line_2_name}, {c.angle:.2f})")
        elif isinstance(c, CircleTangent):
            out.append(f"circle_tangent({c.circle_1_name}, {c.circle_2_name})")
        elif isinstance(c, OnCircle):
            out.append(f"on_circle({c.point_name}, {c.circle_name})")
    return out

def serialize_scene(points, lines, circles, constraints) -> str:
    """Combined string — still useful for the NL generation prompt."""
    return "\n".join(serialize_geometry(points, lines, circles) + serialize_constraints(constraints))
def constraints_only_str(constraints) -> str:
    parts = []
    for c in constraints:
        if isinstance(c, Tangent):
            parts.append(f"line {c.line_name} is tangent to circle {c.circle_name}")
        elif isinstance(c, Parallel):
            parts.append(f"lines {c.line_1_name} and {c.line_2_name} are parallel")
        elif isinstance(c, Perpendicular):
            parts.append(f"lines {c.line_1_name} and {c.line_2_name} are perpendicular")
        elif isinstance(c, Angle):
            parts.append(f"lines {c.line_1_name} and {c.line_2_name} meet at {c.angle}°")
        elif isinstance(c, CircleTangent):
            parts.append(f"circles {c.circle_1_name} and {c.circle_2_name} are {c.kind}ly tangent")
        elif isinstance(c, OnCircle):
            parts.append(f"point {c.point_name} lies on circle {c.circle_name}")
        elif isinstance(c, Length):
            parts.append(f"line {c.line_name} has length {c.dist}")
        elif isinstance(c, Radius):
            parts.append(f"circle {c.circle_name} has radius {c.rad}")
    return "\n".join(f"- {p}" for p in parts)

# ---------------------------------------------------------------------------
# 6. NL VARIANT GENERATION  (calls Claude)
# ---------------------------------------------------------------------------

# def generate_nl_variants(
#     formal: str,
#     constraint_summary: str,
#     n_variants: int = 8,
#     client = None,
# ) -> list[str]:
#     prompt = f"""You are generating training data for a geometry diagram system that converts natural language into formal geometric descriptions.
#
# Here is a formal geometric description:
# {formal}
#
# Key relationships in this scene:
# {constraint_summary}
#
# Generate {n_variants} different natural language descriptions that a student, teacher, or textbook might use to describe this diagram. Rules:
# - Each description must be semantically equivalent (same objects and constraints)
# - Vary vocabulary: use synonyms (tangent / just touches / grazes, perpendicular / at right angles / 90 degrees, etc.)
# - Vary structure: some terse, some verbose, some conversational, some formal
# - Do NOT mention coordinate values like (1.23, -4.56) — describe relationships only
# - Do NOT number the descriptions
#
# Respond with ONLY a JSON array of strings, no other text. Example format:
# ["description one", "description two"]"""
#
#     response = client.messages.create(
#         model="claude-opus-4-5",
#         max_tokens=1500,
#         messages=[{"role": "user", "content": prompt}],
#     )
#
#     raw = response.content[0].text.strip()
#     if raw.startswith("```"):
#         raw = raw.split("```")[1]
#         if raw.startswith("json"):
#             raw = raw[4:]
#     raw = raw.strip()
#
#     try:
#         variants = json.loads(raw)
#         return [v for v in variants if isinstance(v, str)]
#     except json.JSONDecodeError:
#         return [line.strip().strip('"') for line in raw.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 7. DATASET LOOP
# ---------------------------------------------------------------------------

def generate_dataset(
    n_scenes: int = 100,
    n_variants_per_scene: int = 8,
    output_path: str = "geometry_dataset.jsonl",
    delay: float = 0.5,
):
    # client = anthropic.Anthropic()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped   = 0

    with open(out, "w") as f:
        for i in range(n_scenes):
            print(f"Scene {i+1}/{n_scenes} ...", end=" ", flush=True)

            points, lines, circles, constraints = random_scene()

            # interesting = [c for c in constraints
            #                if not isinstance(c, (Length, Radius))]
            # if not interesting:
            #     print("skipped (no interesting constraints)")
            #     skipped += 1
            #     continue

            formal  = serialize_scene(points, lines, circles, constraints)
            summary = constraints_only_str(constraints)

            # try:
            #     variants = generate_nl_variants(
            #         formal, summary, n_variants_per_scene, client
            #     )
            # except Exception as e:
            #     print(f"API error: {e} — skipping")
            #     skipped += 1
            #     continue

            record = {
                "geometry":    serialize_geometry(points, lines, circles),  # list[str] → pred_geo
                "constraints": serialize_constraints(constraints),           # list[str] → truth_constr
                "nl_variants": [],
            }
            f.write(json.dumps(record) + "\n")
            generated += 1
            print(f"ok ({len(constraints)} constraints)")

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

    print("\n=== Generating small dataset (10 scenes) ===\n")
    generate_dataset(n_scenes=10, n_variants_per_scene=5, output_path="demo_dataset.jsonl")