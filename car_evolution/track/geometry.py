"""
Pure geometry for the racetrack: checkpoints, corridor containment, segment intersection.

Used by :class:`car_evolution.core.car.Car` for valid gate crossings and wall checks.
"""

from __future__ import annotations

import math
from typing import Sequence

# Type aliases: 2D point, segment as (a, b), checkpoint segment adds forward tangent
Point = tuple[float, float]
Segment = tuple[Point, Point]
CheckpointSegment = tuple[Point, Point, tuple[float, float]]


def checkpoint_segments(
    waypoint_centers: Sequence[Point],
    half_len: float = 100,
) -> list[CheckpointSegment]:
    """
    Build perpendicular gate segments through each waypoint, aligned with track direction.

    Args:
        waypoint_centers: Closed-loop ordered centers (indices wrap).
        half_len: Half-length of each gate segment along the perpendicular axis.

    Returns:
        List of ``(left, right, forward_unit)`` where crossing the segment in the
        forward direction counts as clearing the gate.
    """
    n = len(waypoint_centers)
    segs: list[CheckpointSegment] = []
    for i in range(n):
        prev_c = waypoint_centers[i - 1]
        nxt_c = waypoint_centers[(i + 1) % n]
        tx = nxt_c[0] - prev_c[0]
        ty = nxt_c[1] - prev_c[1]
        mag = math.hypot(tx, ty) or 1.0
        tx, ty = tx / mag, ty / mag
        px, py = -ty, tx
        cx, cy = waypoint_centers[i]
        segs.append(
            (
                (cx - px * half_len, cy - py * half_len),
                (cx + px * half_len, cy + py * half_len),
                (tx, ty),
            )
        )
    return segs


def segment_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    """
    Return True if open segments ``ab`` and ``cd`` intersect (proper intersection).

    Args:
        a, b: Endpoints of first segment.
        c, d: Endpoints of second segment.
    """
    def orient(p: Point, q: Point, r: Point) -> float:
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return False


def side_of_line(p: Point, line_a: Point, line_b: Point) -> float:
    """
    Signed 2D cross product indicating which side of line ``line_a``->``line_b`` point ``p`` lies on.
    """
    return (line_b[0] - line_a[0]) * (p[1] - line_a[1]) - (line_b[1] - line_a[1]) * (p[0] - line_a[0])


def point_in_polygon(px: float, py: float, vertices: Sequence[Point]) -> bool:
    """Ray-casting test: True if ``(px, py)`` is inside a simple polygon."""
    n = len(vertices)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def point_in_corridor(px: float, py: float, outer_poly: Sequence[Point], inner_poly: Sequence[Point]) -> bool:
    """True if the point lies in the track ring (inside outer boundary, outside inner hole)."""
    return point_in_polygon(px, py, outer_poly) and not point_in_polygon(px, py, inner_poly)


def segment_cuts_any_wall(a: Point, b: Point, wall_segments: Sequence[Segment]) -> bool:
    """True if segment ``ab`` intersects any wall segment."""
    for c, d in wall_segments:
        if segment_cross(a, b, c, d):
            return True
    return False


def movement_corridor_valid(
    prev: Point,
    curr: Point,
    outer_poly: Sequence[Point],
    inner_poly: Sequence[Point],
    wall_segments: Sequence[Segment],
    samples: int = 14,
) -> bool:
    """
    True if the straight move from ``prev`` to ``cur`` stays in the corridor and does not cut walls.

    Samples intermediate points along the segment for containment checks.
    """
    if segment_cuts_any_wall(prev, curr, wall_segments):
        return False
    for i in range(samples + 1):
        t = i / samples
        px = prev[0] + t * (curr[0] - prev[0])
        py = prev[1] + t * (curr[1] - prev[1])
        if not point_in_corridor(px, py, outer_poly, inner_poly):
            return False
    return True


def progress_lex_better(cp_a: int, dist_a: float, cp_b: int, dist_b: float) -> bool:
    """
    Lexicographic ordering: more gates cleared wins; tie-break by smaller distance to next gate.
    """
    if cp_a != cp_b:
        return cp_a > cp_b
    return dist_a < dist_b
