from dataclasses import dataclass, field

# Epic Geometry Language
@dataclass
class Point:
    name: str
    x: float
    y: float

@dataclass
class Line:
    name: str
    p1: Point
    p2: Point

@dataclass
class Circle:
    name: str
    center: Point
    radius: float
    
# Epic Constraint Language
@dataclass
class Radius:
    circle_name: str
    rad: float

@dataclass
class Length:
    line_name: str
    dist: float

@dataclass
class Intersect:
    object_1_name: str
    object_2_name: str
    
@dataclass
class Tangent:
    circle_name: str
    line_name: str

@dataclass
class Angle:
    line_1_name: str
    line_2_name: str
    angle: float
    
@dataclass
class Parallel:
    line_1_name: str
    line_2_name: str
    distance: float

@dataclass
class Perpendicular:
    line_1_name: str
    line_2_name: str

@dataclass
class CircleTangent:
    circle_1_name: str
    circle_2_name: str
    kind: str

@dataclass
class OnCircle:
    point_name: str
    circle_name: str