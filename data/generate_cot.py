"""
generate_cot.py

Takes an existing .jsonl dataset and adds a <think> block to each example
by generating templated CoT from the constraints we already know.
"""

import json
import random
from pathlib import Path

from data_types import (
    Point, Line, Circle,
    Radius, Length, Tangent, Parallel, Perpendicular,
    Angle, CircleTangent, OnCircle
)


# ---------------------------------------------------------------------------
# TEMPLATED CoT GENERATOR
# ---------------------------------------------------------------------------

def generate_cot_from_constraints(constraints_str: list[str]) -> str:
    """
    Given a list of serialized constraint strings (e.g. ["tangent(la, ω0)", ...]),
    generate a templated <think> block that walks through the construction.
    """
    lines = []

    # parse constraint strings back into categories
    points, geom_lines, circles = [], [], []
    lengths, radii = {}, {}
    tangents, parallels, perps, angles = [], [], [], []
    circle_tangents, on_circles = [], []

    for c in constraints_str:
        name = c.split("(")[0]
        args = c.split("(")[1].rstrip(")").split(",")
        args = [a.strip() for a in args]

        if name == "point":       points.append(args[0])
        elif name == "line":      geom_lines.append((args[0], args[1], args[2]))
        elif name == "circle":    circles.append((args[0], args[1]))
        elif name == "length":    lengths[args[0]] = args[1]
        elif name == "radius":    radii[args[0]] = args[1]
        elif name == "tangent":   tangents.append((args[0], args[1]))
        elif name == "parallel":  parallels.append(args)
        elif name == "perpendicular": perps.append((args[0], args[1]))
        elif name == "angle":     angles.append((args[0], args[1], args[2]))
        elif name == "circle_tangent": circle_tangents.append((args[0], args[1]))
        elif name == "on_circle": on_circles.append((args[0], args[1]))

    # step 1: identify what objects are needed
    lines.append("Let me work through what I need to construct:")

    for p in points:
        lines.append(f"- Point {p}: I'll place this freely in the plane.")

    for ln, p1, p2 in geom_lines:
        length = lengths.get(ln)
        if length:
            lines.append(f"- Line {ln} connecting {p1} and {p2} "
                        f"with length {length}: I'll place {p1} first, "
                        f"then offset {p2} by {length} units.")
        else:
            lines.append(f"- Line {ln} connecting {p1} and {p2}: "
                        f"I'll place both endpoints freely.")

    for cn, center in circles:
        r = radii.get(cn, "unknown")
        lines.append(f"- Circle {cn} centered at {center} with radius {r}: "
                    f"I'll place {center} first, then the circle follows.")

    # step 2: reason through each interesting constraint
    if tangents or parallels or perps or angles or circle_tangents or on_circles:
        lines.append("\nNow I'll handle the geometric relationships:")

    for ln, cn in tangents:
        lines.append(
            f"- {ln} must be tangent to {cn}: the distance from {cn}'s center "
            f"to {ln} must equal {cn}'s radius exactly. I'll pick a tangent point "
            f"on the circle and place the line perpendicular to the radius there."
        )

    for args in parallels:
        l1, l2 = args[0], args[1]
        dist = args[2] if len(args) > 2 else None
        if dist:
            lines.append(
                f"- {l1} and {l2} must be parallel with distance {dist}: "
                f"I'll give them the same direction vector, then offset {l2} "
                f"by {dist} units perpendicularly."
            )
        else:
            lines.append(
                f"- {l1} and {l2} must be parallel: "
                f"I'll give them the same direction vector."
            )

    for l1, l2 in perps:
        lines.append(
            f"- {l1} and {l2} must be perpendicular: "
            f"I'll make {l2}'s direction the 90-degree rotation of {l1}'s."
        )

    for l1, l2, deg in angles:
        lines.append(
            f"- {l1} and {l2} meet at {deg} degrees: "
            f"I'll set {l1}'s direction, then rotate by {deg} degrees to get {l2}'s."
        )

    for c1, c2 in circle_tangents:
        lines.append(
            f"- {c1} and {c2} are tangent: the distance between their centers "
            f"must equal the sum of their radii (external) or the absolute "
            f"difference (internal). I'll place them accordingly."
        )

    for p, cn in on_circles:
        lines.append(
            f"- Point {p} lies on {cn}: I'll place {p} at distance r from "
            f"{cn}'s center, picking a random angle."
        )

    lines.append("\nNow I'll output the coordinates:")

    return "<think>\n" + "\n".join(lines) + "\n</think>"


def add_cot_to_dataset(
    input_path: str,
    output_path: str,
):
    """
    Reads a .jsonl dataset and writes a new one where each nl_variant
    is paired with a CoT-prefixed expected output.
    The new format adds a 'cot' field to each record.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)

            cot = generate_cot_from_constraints(record["constraints"])
            geometry_str = "\n".join(record["geometry"])

            # full expected output = <think>...</think> + geometry
            full_output = cot + "\n" + geometry_str

            record["cot"] = cot
            record["full_output"] = full_output
            fout.write(json.dumps(record) + "\n")
            written += 1

    print(f"Done. {written} records written → {output_path}")


if __name__ == "__main__":
    datasets = [
        ("curriculum_data/dataset_simple.jsonl",      "cot_data/cot_simple.jsonl"),
        ("curriculum_data/dataset_medium.jsonl",      "cot_data/cot_medium.jsonl"),
        ("curriculum_data/dataset_complex.jsonl",     "cot_data/cot_complex.jsonl"),
        ("curriculum_data/dataset_merged.jsonl",      "cot_data/cot_merged.jsonl"),
    ]
    for inp, out in datasets:
        if Path(inp).exists():
            print(f"Processing {inp}...")
            add_cot_to_dataset(inp, out)