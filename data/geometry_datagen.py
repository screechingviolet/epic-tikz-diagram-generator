"""
geometry_datagen.py

Full pipeline for generating geometry training data:
  1. Randomly generate scenes (points, lines, circles)
  2. Nudge for interesting constraints (tangency, perpendicularity, etc.)
  3. Extract all true constraints from the scene
  4. Serialize to formal language
  5. Generate natural language variants via OpenAI API
  6. Save dataset to JSONL
"""

import itertools
import json
import random
import time
import string
from pathlib import Path

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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
# 0b. RANDOM NAME GENERATION
# ---------------------------------------------------------------------------

def _name_generator(prefix_pool: list[str], used: set) -> str:
    """Pick a random unused name from the pool, falling back to numbered names."""
    remaining = [n for n in prefix_pool if n not in used]
    if remaining:
        name = random.choice(remaining)
        used.add(name)
        return name
    # fallback if pool exhausted
    i = 0
    while True:
        name = f"{prefix_pool[0]}{i}"
        if name not in used:
            used.add(name)
            return name
        i += 1

POINT_NAMES  = list(string.ascii_uppercase)                        # A-Z
LINE_NAMES   = [f"l{c}" for c in string.ascii_lowercase]          # la, lb, ...
CIRCLE_NAMES = [f"ω{i}" for i in range(26)] + \
               [f"γ{i}" for i in range(26)]                        # ω0, γ0, ...

def fresh_name_pools():
    used = set()
    def point_name():  return _name_generator(POINT_NAMES,  used)
    def line_name():   return _name_generator(LINE_NAMES,   used)
    def circle_name(): return _name_generator(CIRCLE_NAMES, used)
    return point_name, line_name, circle_name


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

    point_name, line_name, circle_name = fresh_name_pools()

    points = {
        (pn := point_name()): Point(name=pn, x=float(x), y=float(y))
        for x, y in np.random.uniform(-5, 5, (n_points, 2))
    }
    point_list = list(points.values())

    lines = {}
    used_pairs = set()
    for _ in range(n_lines):
        for _ in range(20):
            p1, p2 = random.sample(point_list, 2)
            pair = tuple(sorted([p1.name, p2.name]))
            if pair not in used_pairs:
                used_pairs.add(pair)
                ln = line_name()
                lines[ln] = Line(name=ln, p1=p1, p2=p2)
                break

    circles = {}
    for _ in range(n_circles):
        center = random.choice(point_list)
        cn = circle_name()
        circles[cn] = Circle(name=cn, center=center,
                             radius=round(random.uniform(0.5, 3.0), 4))


    line_list   = list(lines.values())
    circle_list = list(circles.values())

    if line_list and circle_list and random.random() < nudge_probability:
        line   = random.choice(line_list)
        circle = random.choice(circle_list)
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
# 3b. CURRICULUM SCENE GENERATORS
# ---------------------------------------------------------------------------

def simple_scene():
    point_name, line_name, circle_name = fresh_name_pools()
    choice = random.randint(0, 2)

    if choice == 0:
        pn = point_name()
        points = {pn: Point(name=pn, x=float(np.random.uniform(-5, 5)),
                                     y=float(np.random.uniform(-5, 5)))}
        return points, {}, {}, extract_constraints(points, {}, {})

    elif choice == 1:
        coords = np.random.uniform(-5, 5, (2, 2))
        p1n, p2n = point_name(), point_name()
        points = {
            p1n: Point(name=p1n, x=float(coords[0][0]), y=float(coords[0][1])),
            p2n: Point(name=p2n, x=float(coords[1][0]), y=float(coords[1][1])),
        }
        ln = line_name()
        lines = {ln: Line(name=ln, p1=points[p1n], p2=points[p2n])}
        return points, lines, {}, extract_constraints(points, lines, {})

    else:
        pn = point_name()
        center = Point(name=pn, x=float(np.random.uniform(-5, 5)),
                                y=float(np.random.uniform(-5, 5)))
        points = {pn: center}
        cn = circle_name()
        circles = {cn: Circle(name=cn, center=center,
                              radius=round(random.uniform(0.5, 3.0), 4))}
        return points, {}, circles, extract_constraints(points, {}, circles)


def medium_scene():
    point_name, line_name, circle_name = fresh_name_pools()
    n_points  = random.randint(2, 4)
    n_lines   = random.randint(1, 2)
    n_circles = random.randint(0, 1)

    points = {
        (pn := point_name()): Point(name=pn, x=float(x), y=float(y))
        for x, y in np.random.uniform(-5, 5, (n_points, 2))
    }
    point_list = list(points.values())

    lines = {}
    used_pairs = set()
    for _ in range(n_lines):
        for _ in range(20):
            p1, p2 = random.sample(point_list, 2)
            pair = tuple(sorted([p1.name, p2.name]))
            if pair not in used_pairs:
                used_pairs.add(pair)
                ln = line_name()
                lines[ln] = Line(name=ln, p1=p1, p2=p2)
                break

    circles = {}
    for _ in range(n_circles):
        center = random.choice(point_list)
        cn = circle_name()
        circles[cn] = Circle(name=cn, center=center,
                             radius=round(random.uniform(0.5, 3.0), 4))

    return points, lines, circles, extract_constraints(points, lines, circles)

# ---------------------------------------------------------------------------
# 3c. CONSTRAINT-SPECIFIC SCENE GENERATORS
# ---------------------------------------------------------------------------

def scene_two_parallel_lines():
    point_name, line_name, _ = fresh_name_pools()

    angle = np.random.uniform(0, np.pi)
    dx, dy = np.cos(angle), np.sin(angle)
    offset = np.random.uniform(1.0, 4.0)
    perp = np.array([-dy, dx])
    base = np.random.uniform(-2, 2, 2)

    p1n, p2n = point_name(), point_name()
    p3n, p4n = point_name(), point_name()

    l1_len = random.uniform(1.0, 5.0)   # randomize each line length
    l2_len = random.uniform(1.0, 5.0)

    points = {
        p1n: Point(name=p1n, x=float(base[0]),              y=float(base[1])),
        p2n: Point(name=p2n, x=float(base[0] + dx*l1_len),  y=float(base[1] + dy*l1_len)),
        p3n: Point(name=p3n, x=float(base[0] + perp[0]*offset),
                             y=float(base[1] + perp[1]*offset)),
        p4n: Point(name=p4n, x=float(base[0] + perp[0]*offset + dx*l2_len),
                             y=float(base[1] + perp[1]*offset + dy*l2_len)),
    }
    l1n, l2n = line_name(), line_name()
    lines = {
        l1n: Line(name=l1n, p1=points[p1n], p2=points[p2n]),
        l2n: Line(name=l2n, p1=points[p3n], p2=points[p4n]),
    }
    return points, lines, {}, extract_constraints(points, lines, {})


def scene_two_perpendicular_lines():
    point_name, line_name, _ = fresh_name_pools()

    angle = np.random.uniform(0, np.pi)
    dx, dy = np.cos(angle), np.sin(angle)
    perp = np.array([-dy, dx])
    ix, iy = np.random.uniform(-2, 2, 2)

    l1_half = random.uniform(1.0, 4.0)   # each arm length randomized independently
    l2_half = random.uniform(1.0, 4.0)

    p1n, p2n = point_name(), point_name()
    p3n, p4n = point_name(), point_name()
    points = {
        p1n: Point(name=p1n, x=float(ix + dx*l1_half),      y=float(iy + dy*l1_half)),
        p2n: Point(name=p2n, x=float(ix - dx*l1_half),      y=float(iy - dy*l1_half)),
        p3n: Point(name=p3n, x=float(ix + perp[0]*l2_half), y=float(iy + perp[1]*l2_half)),
        p4n: Point(name=p4n, x=float(ix - perp[0]*l2_half), y=float(iy - perp[1]*l2_half)),
    }
    l1n, l2n = line_name(), line_name()
    lines = {
        l1n: Line(name=l1n, p1=points[p1n], p2=points[p2n]),
        l2n: Line(name=l2n, p1=points[p3n], p2=points[p4n]),
    }
    return points, lines, {}, extract_constraints(points, lines, {})


def scene_two_lines_at_angle():
    point_name, line_name, _ = fresh_name_pools()

    angle1 = np.random.uniform(0, np.pi)
    delta  = np.radians(np.random.uniform(15, 75))
    angle2 = angle1 + delta
    ix, iy = np.random.uniform(-2, 2, 2)

    def make_endpoints(angle, ix, iy, pn1, pn2):
        dx, dy = np.cos(angle), np.sin(angle)
        half = random.uniform(1.0, 4.0)   # randomized per line
        return {
            pn1: Point(name=pn1, x=float(ix + dx*half), y=float(iy + dy*half)),
            pn2: Point(name=pn2, x=float(ix - dx*half), y=float(iy - dy*half)),
        }

    p1n, p2n = point_name(), point_name()
    p3n, p4n = point_name(), point_name()
    points = {
        **make_endpoints(angle1, ix, iy, p1n, p2n),
        **make_endpoints(angle2, ix, iy, p3n, p4n),
    }
    l1n, l2n = line_name(), line_name()
    lines = {
        l1n: Line(name=l1n, p1=points[p1n], p2=points[p2n]),
        l2n: Line(name=l2n, p1=points[p3n], p2=points[p4n]),
    }
    return points, lines, {}, extract_constraints(points, lines, {})


def scene_line_tangent_to_circle():
    point_name, line_name, circle_name = fresh_name_pools()

    cn = point_name()
    center = Point(name=cn, x=float(np.random.uniform(-2, 2)),
                            y=float(np.random.uniform(-2, 2)))
    radius = round(random.uniform(0.5, 2.0), 4)
    circ_n = circle_name()
    circles = {circ_n: Circle(name=circ_n, center=center, radius=radius)}
    points  = {cn: center}

    angle = np.random.uniform(0, 2 * np.pi)
    tx = center.x + radius * np.cos(angle)
    ty = center.y + radius * np.sin(angle)
    tdx, tdy = -np.sin(angle), np.cos(angle)

    p1n, p2n = point_name(), point_name()
    t_len = random.uniform(1.0, 4.0)   # randomized
    points[p1n] = Point(name=p1n, x=float(tx + tdx*t_len), y=float(ty + tdy*t_len))
    points[p2n] = Point(name=p2n, x=float(tx - tdx*t_len), y=float(ty - tdy*t_len))
    ln = line_name()
    lines = {ln: Line(name=ln, p1=points[p1n], p2=points[p2n])}

    return points, lines, circles, extract_constraints(points, lines, circles)


def scene_two_circles_tangent():
    point_name, _, circle_name = fresh_name_pools()

    c1n = point_name()
    center1 = Point(name=c1n, x=float(np.random.uniform(-3, 0)),
                              y=float(np.random.uniform(-2, 2)))
    r1 = round(random.uniform(0.5, 1.5), 4)

    angle = np.random.uniform(0, 2 * np.pi)
    r2    = round(random.uniform(0.5, 1.5), 4)
    c2n   = point_name()
    center2 = Point(
        name=c2n,
        x=float(center1.x + (r1 + r2) * np.cos(angle)),
        y=float(center1.y + (r1 + r2) * np.sin(angle)),
    )

    points  = {c1n: center1, c2n: center2}
    circ1n, circ2n = circle_name(), circle_name()
    circles = {
        circ1n: Circle(name=circ1n, center=center1, radius=r1),
        circ2n: Circle(name=circ2n, center=center2, radius=r2),
    }
    return points, {}, circles, extract_constraints(points, {}, circles)


def scene_point_on_circle():
    point_name, _, circle_name = fresh_name_pools()

    cn = point_name()
    center = Point(name=cn, x=float(np.random.uniform(-2, 2)),
                            y=float(np.random.uniform(-2, 2)))
    radius = round(random.uniform(0.5, 2.0), 4)
    circ_n = circle_name()
    circles = {circ_n: Circle(name=circ_n, center=center, radius=radius)}

    angle = np.random.uniform(0, 2 * np.pi)
    pn = point_name()
    on_pt = Point(name=pn,
                  x=float(center.x + radius * np.cos(angle)),
                  y=float(center.y + radius * np.sin(angle)))
    points = {cn: center, pn: on_pt}

    return points, {}, circles, extract_constraints(points, {}, circles)


CONSTRAINT_SPECIFIC_SCENES = {
    "parallel":          scene_two_parallel_lines,
    "perpendicular":     scene_two_perpendicular_lines,
    "angle":             scene_two_lines_at_angle,
    "line_tangent":      scene_line_tangent_to_circle,
    "circle_tangent":    scene_two_circles_tangent,
    "point_on_circle":   scene_point_on_circle,
}
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

    # object declarations — reuse existing dataclasses
    for p in points.values():
        constraints.append(p)

    for l in lines.values():
        constraints.append(l)

    for c in circles.values():
        constraints.append(c)

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
    out = []
    for p in points.values():
        out.append(f"point({p.name}, {p.x:.4f}, {p.y:.4f})")
    for l in lines.values():
        out.append(f"line({l.name}, {l.p1.name}, {l.p2.name})")
    for c in circles.values():
        out.append(f"circle({c.name}, {c.center.name}, {c.radius:.4f})")
    return out

def serialize_constraints(constraints) -> list[str]:
    out = []
    for c in constraints:
        if isinstance(c, Point):
            out.append(f"point({c.name})")
        elif isinstance(c, Line):
            out.append(f"line({c.name}, {c.p1.name}, {c.p2.name})")
        elif isinstance(c, Circle):
            out.append(f"circle({c.name}, {c.center.name})")
        elif isinstance(c, Length):
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
    return "\n".join(serialize_geometry(points, lines, circles) + serialize_constraints(constraints))

def constraints_only_str(constraints) -> str:
    parts = []
    for c in constraints:
        if isinstance(c, Point):
            parts.append(f"point {c.name} exists")
        elif isinstance(c, Line):
            parts.append(f"line {c.name} connects points {c.p1.name} and {c.p2.name}")
        elif isinstance(c, Circle):
            parts.append(f"circle {c.name} is centered at point {c.center.name}")
        elif isinstance(c, Tangent):
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
# 6. NL VARIANT GENERATION  (OpenAI — batched + cached)
# ---------------------------------------------------------------------------

def build_prompt(formal: str, summary: str, n_variants: int) -> list[dict]:
    # static instructions in system message (gets cached by OpenAI automatically)
    # variable scene content in user message (not cached)
    return [
        {
            "role": "system",
            "content": """You are generating training data for a geometry diagram system
that converts natural language into formal geometric descriptions.

Generate natural language descriptions following these rules:
- Each description must be semantically equivalent (same objects and constraints)
- Vary vocabulary: tangent / just touches / perpendicular / at right angles, etc.
- Mention the name of every object
- Vary structure: some terse, some verbose, some conversational, some formal
- Do NOT mention coordinate values — describe relationships only
- Do NOT round the exact numbers involved in the question nor convert it into words
- Do NOT number the descriptions
- COORDINATE VALUES ARE STRICTLY FORBIDDEN
- Respond with ONLY a JSON array of strings, no other text. Example format:
["description one", "description two"]"""
        },
        {
            "role": "user",
            "content": f"""Formal description:
{formal}

Key relationships:
{summary}

Generate {{n_variants}} different natural language descriptions.""".format(n_variants=n_variants)
        }
    ]


def parse_variants(raw: str) -> list[str]:
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return [v for v in json.loads(raw) if isinstance(v, str)]
    except json.JSONDecodeError:
        return [line.strip().strip('"') for line in raw.splitlines() if line.strip()]


def generate_nl_variants_batched(
    scenes: list[dict],
    n_variants: int = 8,
    client: OpenAI = None,
) -> list[list[str]]:
    """
    Takes a list of scene dicts (each with points/lines/circles/constraints),
    submits all NL generation as a single batch job, and returns a list of
    variant lists in the same order as the input scenes.
    """
    if client is None:
        client = OpenAI()

    # build batch request file
    batch_requests = []
    for i, scene in enumerate(scenes):
        formal  = serialize_scene(scene["points"], scene["lines"], scene["circles"], scene["constraints"])
        summary = constraints_only_str(scene["constraints"])
        batch_requests.append({
            "custom_id": f"scene-{i}",
            "method":    "POST",
            "url":       "/v1/chat/completions",
            "body": {
                "model":      "gpt-4o-mini",
                "max_tokens": 1500,
                "messages":   build_prompt(formal, summary, n_variants),
            }
        })

    # write to temp file and upload
    batch_input_path = Path("_batch_input.jsonl")
    with open(batch_input_path, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")

    with open(batch_input_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch submitted: {batch.id}")

    # poll until done
    while True:
        batch = client.batches.retrieve(batch.id)
        print(f"  status: {batch.status} "
              f"({batch.request_counts.completed}/{batch.request_counts.total} completed)")
        if batch.status == "completed":
            break
        elif batch.status in ("failed", "cancelled"):
            raise RuntimeError(f"Batch {batch.id} failed with status: {batch.status}")
        time.sleep(30)

    # download results and reassemble in original order
    result_content = client.files.content(batch.output_file_id).text
    results = {
        json.loads(line)["custom_id"]: json.loads(line)
        for line in result_content.splitlines()
        if line.strip()
    }

    all_variants = []
    for i in range(len(scenes)):
        result = results.get(f"scene-{i}")
        if result is None or result.get("error"):
            print(f"  scene-{i} failed: {result.get('error') if result else 'missing'}")
            all_variants.append([])
            continue
        raw = result["response"]["body"]["choices"][0]["message"]["content"]
        all_variants.append(parse_variants(raw))

    # cleanup temp file
    batch_input_path.unlink(missing_ok=True)

    return all_variants

# ---------------------------------------------------------------------------
# 7. DATASET LOOP
# ---------------------------------------------------------------------------

def generate_dataset(
    n_scenes: int = 100,
    n_variants_per_scene: int = 8,
    output_path: str = "geometry_dataset.jsonl",
):
    client = OpenAI()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_scenes} scenes...")
    scenes = []
    for _ in range(n_scenes):
        points, lines, circles, constraints = random_scene()
        interesting = [c for c in constraints if not isinstance(c, (Length, Radius))]
        if not interesting:
            continue
        scenes.append({
            "points": points, "lines": lines,
            "circles": circles, "constraints": constraints,
        })
    print(f"{len(scenes)} scenes generated ({n_scenes - len(scenes)} skipped)")

    all_variants = generate_nl_variants_batched(scenes, n_variants_per_scene, client)

    generated = 0
    skipped   = 0
    with open(out, "w") as f:
        for scene, variants in zip(scenes, all_variants):
            if not variants:
                skipped += 1
                continue
            record = {
                "geometry":    serialize_geometry(scene["points"], scene["lines"], scene["circles"]),
                "constraints": serialize_constraints(scene["constraints"]),
                "nl_variants": variants,
            }
            f.write(json.dumps(record) + "\n")
            generated += 1

    print(f"\nDone. {generated} scenes written, {skipped} skipped → {out}")
    return out


def generate_curriculum_datasets(
    n_simple: int = 100,
    n_medium: int = 100,
    n_complex: int = 100,
    n_variants_per_scene: int = 8,
    output_dir: str = ".",
):
    client = OpenAI()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("simple",  n_simple,  simple_scene),
        ("medium",  n_medium,  medium_scene),
        ("complex", n_complex, lambda: random_scene(nudge_probability=0.8)),
    ]

    # step 1: generate all scenes locally for all levels
    all_scenes = {}
    for level, n_scenes, scene_fn in configs:
        print(f"Generating {level} scenes...")
        scenes = []
        attempts = 0
        while len(scenes) < n_scenes:
            attempts += 1
            points, lines, circles, constraints = scene_fn()
            if level == "complex":
                interesting = [c for c in constraints if not isinstance(c, (Length, Radius, Point, Line, Circle))]
                if not interesting:
                    continue
            scenes.append(dict(zip(("points", "lines", "circles", "constraints"),
                                   (points, lines, circles, constraints))))
        print(f"  {len(scenes)} scenes ({attempts - len(scenes)} skipped)")
        all_scenes[level] = scenes

    # step 2: submit all three batches simultaneously
    batch_ids = {}
    for level, scenes in all_scenes.items():
        print(f"Submitting {level} batch...")
        batch_requests = []
        for i, scene in enumerate(scenes):
            formal  = serialize_scene(scene["points"], scene["lines"], scene["circles"], scene["constraints"])
            summary = constraints_only_str(scene["constraints"])
            batch_requests.append({
                "custom_id": f"scene-{i}",
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model":      "gpt-4o-mini",
                    "max_tokens": 1500,
                    "messages":   build_prompt(formal, summary, n_variants_per_scene),
                }
            })

        batch_input_path = Path(f"_batch_input_{level}.jsonl")
        with open(batch_input_path, "w") as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")

        with open(batch_input_path, "rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")

        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_ids[level] = batch.id
        batch_input_path.unlink(missing_ok=True)
        print(f"  {level} batch submitted: {batch.id}")

    # step 3: poll all batches together until all done
    print("\nWaiting for all batches...")
    pending = set(batch_ids.values())
    results = {}
    while pending:
        for level, batch_id in batch_ids.items():
            if batch_id not in pending:
                continue
            batch = client.batches.retrieve(batch_id)
            print(f"  {level}: {batch.status} "
                  f"({batch.request_counts.completed}/{batch.request_counts.total})")
            if batch.status == "completed":
                results[level] = client.files.content(batch.output_file_id).text
                pending.remove(batch_id)
            elif batch.status in ("failed", "cancelled"):
                raise RuntimeError(f"{level} batch failed: {batch.status}")
        if pending:
            time.sleep(30)

    # step 4: write all output files
    for level, scenes in all_scenes.items():
        out_path = output_dir / f"dataset_{level}.jsonl"
        raw_results = {
            json.loads(line)["custom_id"]: json.loads(line)
            for line in results[level].splitlines() if line.strip()
        }

        generated = skipped = 0
        with open(out_path, "a") as f:
            for i, scene in enumerate(scenes):
                result = raw_results.get(f"scene-{i}")
                if result is None or result.get("error"):
                    skipped += 1
                    continue
                raw = result["response"]["body"]["choices"][0]["message"]["content"]
                variants = parse_variants(raw)
                if not variants:
                    skipped += 1
                    continue
                record = {
                    "geometry":    serialize_geometry(scene["points"], scene["lines"], scene["circles"]),
                    "constraints": serialize_constraints(scene["constraints"]),
                    "nl_variants": variants,
                }
                f.write(json.dumps(record) + "\n")
                generated += 1

        print(f"{level}: {generated} written, {skipped} skipped → {out_path}")
        
def generate_constraint_specific_datasets(
    n_scenes_per_type: int = 100,
    n_variants_per_scene: int = 5,
    output_dir: str = ".",
):
    client = OpenAI()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # step 1: generate all scenes locally for all constraint types
    all_scenes = {}
    for constraint_type, scene_fn in CONSTRAINT_SPECIFIC_SCENES.items():
        print(f"Generating {constraint_type} scenes...")
        scenes = [
            dict(zip(("points", "lines", "circles", "constraints"), scene_fn()))
            for _ in range(n_scenes_per_type)
        ]
        all_scenes[constraint_type] = scenes
        print(f"  {len(scenes)} scenes generated")

    # step 2: submit all batches simultaneously
    batch_ids = {}
    for constraint_type, scenes in all_scenes.items():
        print(f"Submitting {constraint_type} batch...")
        batch_requests = []
        for i, scene in enumerate(scenes):
            formal  = serialize_scene(scene["points"], scene["lines"], scene["circles"], scene["constraints"])
            summary = constraints_only_str(scene["constraints"])
            batch_requests.append({
                "custom_id": f"scene-{i}",
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model":      "gpt-4o-mini",
                    "max_tokens": 1500,
                    "messages":   build_prompt(formal, summary, n_variants_per_scene),
                }
            })

        batch_input_path = Path(f"_batch_input_{constraint_type}.jsonl")
        with open(batch_input_path, "w") as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")

        with open(batch_input_path, "rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")

        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_ids[constraint_type] = batch.id
        batch_input_path.unlink(missing_ok=True)
        print(f"  {constraint_type} batch submitted: {batch.id}")

    # step 3: poll all batches together until all done
    print("\nWaiting for all batches...")
    pending = set(batch_ids.values())
    results = {}
    while pending:
        for constraint_type, batch_id in batch_ids.items():
            if batch_id not in pending:
                continue
            batch = client.batches.retrieve(batch_id)
            print(f"  {constraint_type}: {batch.status} "
                  f"({batch.request_counts.completed}/{batch.request_counts.total})")
            if batch.status == "completed":
                results[constraint_type] = client.files.content(batch.output_file_id).text
                pending.remove(batch_id)
            elif batch.status in ("failed", "cancelled"):
                raise RuntimeError(f"{constraint_type} batch failed: {batch.status}")
        if pending:
            time.sleep(30)

    # step 4: write all output files
    for constraint_type, scenes in all_scenes.items():
        out_path = output_dir / f"dataset_{constraint_type}.jsonl"
        raw_results = {
            json.loads(line)["custom_id"]: json.loads(line)
            for line in results[constraint_type].splitlines() if line.strip()
        }

        generated = skipped = 0
        with open(out_path, "a") as f:
            for i, scene in enumerate(scenes):
                result = raw_results.get(f"scene-{i}")
                if result is None or result.get("error"):
                    skipped += 1
                    continue
                raw = result["response"]["body"]["choices"][0]["message"]["content"]
                variants = parse_variants(raw)
                if not variants:
                    skipped += 1
                    continue
                record = {
                    "geometry":    serialize_geometry(scene["points"], scene["lines"], scene["circles"]),
                    "constraints": serialize_constraints(scene["constraints"]),
                    "nl_variants": variants,
                }
                f.write(json.dumps(record) + "\n")
                generated += 1

        print(f"{constraint_type}: {generated} written, {skipped} skipped → {out_path}")
# ---------------------------------------------------------------------------
# 8. QUICK DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Generating curriculum datasets ===\n")
    generate_curriculum_datasets(
        n_simple=100, n_medium=200, n_complex=300,
        n_variants_per_scene=5, output_dir="curriculum_data",
    )

    print("\n=== Generating constraint-specific datasets ===\n")
    generate_constraint_specific_datasets(
        n_scenes_per_type=50,
        n_variants_per_scene=5,
        output_dir="constraint_data",
    )