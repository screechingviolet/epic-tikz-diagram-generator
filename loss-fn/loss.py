import math
from shapely.geometry import LineString


# given geometrylanguage thing and constraint language label, check constraints against geometry
'''

EPIC CONSTRAINT LANGUAGE
radius(circle_name) = float
length(line1) = float
intersect(line1/circle, line2/circle) = bool
tangent(line, circle) = bool
parallel(line, line) = bool
perpendicular(line, line) = bool
circle_tangent(circle1, circle2) = bool
on_circle(point, circle) = bool
acute_angle(line, line) = float

EPIC GEOMETRY LANGUAGE
point(name, x, y)
line(name, point_1_name, point_2_name)
circle(name, center, radius)

'''

BIG_BAD_LOSS = 100
FLOAT_CMP = 0.05

class Circle:
    def __init__(self, name, center, radius):
        self.name = name
        self.center = center
        self.radius = radius

class Point:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

class Line:
    def __init__(self, name, point_1, point_2):
        self.name = name
        self.point_1 = point_1
        self.point_2 = point_2

def radius(circle) -> float:
    return circle.radius # lol

def length(line1) -> float:
    return ((line1.point_1.x - line1.point_2.x)**2 + (line1.point_1.y - line1.point_2.y)**2)**0.5

def line_intersect_is_pain(line1, line2):
    return LineString([(line1.point_1.x, line1.point_1.y), (line1.point_2.x, line1.point_2.y)]).intersects(LineString([(line2.point_1.x, line2.point_1.y), (line2.point_2.x, line2.point_2.y)]))

def line_circle_intersect(line, circle) -> bool:
    p1 = line.point_1
    p2 = line.point_2
    cx, cy = circle.center.x, circle.center.y
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    if dx == 0 and dy == 0:
        return math.hypot(p1.x - circle.center.x, p1.y - circle.center.y) <= circle.radius
    t = ((cx - p1.x)*dx + (cy - p1.y)*dy) / (dx*dx + dy*dy) # closest to center
    t = max(0, min(1, t))
    closest_x = p1.x + t*dx
    closest_y = p1.y + t*dy

    return math.hypot(cx - closest_x, cy - closest_y) <= circle.radius

def intersect(circle_or_line1, circle_or_line2) -> bool:
    if isinstance(circle_or_line1, Line) and isinstance(circle_or_line2, Line):
        return line_intersect_is_pain(circle_or_line1, circle_or_line2)

    if isinstance(circle_or_line1, Circle) and isinstance(circle_or_line2, Circle):
        d = distance(circle_or_line1.center, circle_or_line2.center)
        return d <= circle_or_line1.radius + circle_or_line2.radius and d >= abs(circle_or_line1.radius - circle_or_line2.radius)

    if isinstance(circle_or_line1, Line) and isinstance(circle_or_line2, Circle):
        return line_circle_intersect(circle_or_line1, circle_or_line2)

    if isinstance(circle_or_line1, Circle) and isinstance(circle_or_line2, Line):
        return line_circle_intersect(circle_or_line2, circle_or_line1)

    raise ValueError

def tangent(line, circle) -> bool:
    p1 = line.point_1
    p2 = line.point_2
    cx, cy = circle.center.x, circle.center.y
    dx = p2.x - p1.x
    dy = p2.y - p1.y

    if dx == 0 and dy == 0: # just a point :<
        return False
    t = ((cx - p1.x)*dx + (cy - p1.y)*dy) / (dx*dx + dy*dy)

    if t < 0 or t > 1: # not within bounds of line
        return False

    closest_x = p1.x + t*dx
    closest_y = p1.y + t*dy
    dist = math.hypot(cx - closest_x, cy - closest_y)
    return math.isclose(dist, circle.radius, rel_tol=FLOAT_CMP)

def acute_angle(line1, line2) -> float:
    dx1 = line1.point_2.x - line1.point_1.x
    dy1 = line1.point_2.y - line1.point_1.y
    dx2 = line2.point_2.x - line2.point_1.x
    dy2 = line2.point_2.y - line2.point_1.y
    dot = dx1 * dx2 + dy1 * dy2
    mag1 = math.hypot(dx1, dy1)
    mag2 = math.hypot(dx2, dy2)
    cos_theta = dot / (mag1 * mag2)
    angle = math.acos(max(-1, min(1, cos_theta)))

    return min(angle, math.pi - angle)

def slope(line):
    dx = line.point_2.x - line.point_1.x
    dy = line.point_2.y - line.point_1.y
    if dx == 0:
        return None
    return dy / dx

def parallel(line1, line2) -> bool:
    return slope(line1) == slope(line2)

def perpendicular(line1, line2) -> bool:
    m1 = slope(line1)
    m2 = slope(line2)
    if m1 is None:
        return m2 == 0
    if m2 is None:
        return m1 == 0

    return math.isclose(m1 * m2, -1, rel_tol=FLOAT_CMP)

def circle_tangent(circle1, circle2) -> bool:
    d = math.hypot(circle1.center.x - circle2.center.x, circle1.center.y - circle2.center.y)
    return (
        math.isclose(d, circle1.radius + circle2.radius, rel_tol=FLOAT_CMP) or
        math.isclose(d, abs(circle1.radius - circle2.radius), rel_tol=FLOAT_CMP)
    )

def on_circle(point, circle) -> bool:
    d = math.hypot(point.x - circle.center.x, point.y - circle.center.y)
    return math.isclose(d, circle.radius, rel_tol=FLOAT_CMP)

def parse_fn(str1):
    # returns string name and list of arguments
    name_params_split = str1.split("(")
    if len(name_params_split) != 2:
        raise ValueError
    fn_name = name_params_split[0]
    if name_params_split[1][-1] != ")":
        raise ValueError

    params = name_params_split[1][:-1]
    params = [item.strip() for item in params.split(",")]
    return fn_name, params

# discrete: if within 0.5% accuracy, give full marks for constraint and weight equally (1 point per constraint satisfied)
# continuous: 
def check_constraints(pred_geo, truth_constr):
    try:
        shape_dict = {}
        for pred in pred_geo:
            parsed = parse_fn(pred)
            match parsed[0]:
                case "point":
                    shape_dict[parsed[1][0]] = Point(parsed[1][0], float(parsed[1][1]), float(parsed[1][2]))
                case "circle":
                    shape_dict[parsed[1][0]] = Circle(parsed[1][0], parsed[1][1], float(parsed[1][2]))
                case "line":
                    shape_dict[parsed[1][0]] = Line(parsed[1][0], parsed[1][1], parsed[1][2])

        for name, shape in shape_dict.items():
            if isinstance(shape, Line):
                assert isinstance(shape_dict[shape.point_1], Point) and isinstance(shape_dict[shape.point_2], Point)
                shape.point_1 = shape_dict[shape.point_1]
                shape.point_2 = shape_dict[shape.point_2]
            if isinstance(shape, Circle):
                assert isinstance(shape_dict[shape.center], Point)
                shape.center = shape_dict[shape.center]
                
        correct_constraints = 0
        for constraint in truth_constr:
            parsed = parse_fn(constraint)
            match parsed[0]:
                case "radius":
                    if math.isclose(radius(shape_dict[parsed[1][0]]), float(parsed[1][1]), rel_tol=FLOAT_CMP):
                        correct_constraints += 1
                case "length":
                    if math.isclose(length(shape_dict[parsed[1][0]]), float(parsed[1][1]), rel_tol=FLOAT_CMP):
                        correct_constraints += 1
                case "intersect":
                    if intersect(shape_dict[parsed[1][0]], shape_dict[parsed[1][1]]):
                        correct_constraints += 1
                case "tangent":
                    if tangent(shape_dict[parsed[1][0]], shape_dict[parsed[1][1]]):
                        correct_constraints += 1
                case "parallel":
                    if parallel(shape_dict[parsed[1][0]], shape_dict[parsed[1][1]]):
                        correct_constraints += 1
                case "perpendicular":
                    if perpendicular(shape_dict[parsed[1][0]], shape_dict[parsed[1][1]]):
                        correct_constraints += 1
                case "circle_tangent":
                    if circle_tangent(shape_dict[parsed[1][0]], shape_dict[parsed[1][1]]):
                        correct_constraints += 1
                case "on_circle":
                    if on_circle(shape_dict[parsed[1][0]], shape_dict[parsed[1][1]]):
                        correct_constraints += 1
                case "acute_angle":
                    if math.isclose(acute_angle(shape_dict[parsed[1][0]], shape_dict[parsed[1][1]]), float(parsed[1][2]), rel_tol=FLOAT_CMP):
                        correct_constraints += 1
                case _:
                    raise ValueError
                # etc
        return correct_constraints
    except Exception as e:
        print(e)
        return BIG_BAD_LOSS

# testing oof
print(check_constraints(["point(p1, 10, 10.45)", "line(l1, p1, p2)", "point(p2, 15, 10.45)"], ["length(l1,5)"]))
print(check_constraints(
    ["point(c, 0, 0)", "circle(c1, c, 5)", "point(p, 3, 4)"],
    ["on_circle(p, c1)"]
))
print(check_constraints(
    [
        "point(a, 0, 0)", "point(b, 2, 2)",
        "point(c, 1, 0)", "point(d, 3, 2)",
        "line(l1, a, b)", "line(l2, c, d)"
    ],
    ["parallel(l1, l2)"]
))
print(check_constraints(
    [
        "point(a, 0, 0)", "point(b, 1, 0)",
        "point(c, 0, 0)", "point(d, 1, 1)",
        "line(l1, a, b)", "line(l2, c, d)"
    ],
    ["acute_angle(l1,l2,0.7854)"]
))
print(check_constraints(
    [
        "point(a, 0, 0)", "point(b, 4, 4)",
        "point(c, 0, 4)", "point(d, 4, 0)",
        "line(l1, a, b)", "line(l2, c, d)"
    ],
    ["intersect(l1, l2)"]
))
print(check_constraints(
    [
        "point(c1c, 0, 0)", "circle(c1, c1c, 5)",
        "point(c2c, 10, 0)", "circle(c2, c2c, 5)"
    ],
    ["circle_tangent(c1, c2)"]
))
