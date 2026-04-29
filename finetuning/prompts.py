"""Shared prompt + model constants.

Both `train_grpo.py` and `infer.py` import from this module so the system
prompt the adapter is tuned against and the system prompt used at inference
time can never silently drift apart.
"""

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You convert a natural-language description of a geometric diagram into a "
    "list of geometric primitives. Output one primitive per line and nothing "
    "else.\n"
    "\n"
    "Think through the construction inside <think>...</think> tags, "
    "then output one primitive per line and nothing else.\n"
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
    "Description: A line segment L0 of length 3.4823 has endpoints P0 and P1. A circle C0 is centered at P0 and passes through P1.\n"
    "Output:\n"
    "point(P0, 0, 0)\n"
    "point(P1, 3.4823, 0)\n"
    "circle(C0, P0, 3.4823)\n"
    "line(L0, P0, P1)"
)
