"""
Static track backdrop: striped grass, asphalt corridor, zebra curbs, start grid, decorations.

Uses :class:`~car_evolution.track.layout.RaceTrack` polygons; physics and collision are unchanged.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import pygame

from car_evolution.config import Colors, DisplayConfig
from car_evolution.rendering.ui import draw_track_arrow
from car_evolution.track import geometry as tg
from car_evolution.track.layout import RaceTrack


def _poly_points(pts: Sequence[tuple[float, float]]) -> list[tuple[int, int]]:
    """Convert float vertices to integer pixel tuples for pygame drawing APIs."""
    return [(int(p[0]), int(p[1])) for p in pts]


def _draw_grass_stripes(surf: pygame.Surface, width: int, height: int) -> None:
    """Fill ``surf`` with horizontal alternating light/dark green stripes (off-track look)."""
    stripe = 40
    for y in range(0, height, stripe):
        color = Colors.GRASS_LIGHT if (y // stripe) % 2 == 0 else Colors.GRASS_DARK
        pygame.draw.rect(surf, color, (0, y, width, stripe))


def _draw_zebra_polyline(
    surf: pygame.Surface,
    pts: Sequence[tuple[float, float]],
    line_width: int,
) -> None:
    """
    Stroke a closed polyline with alternating red/white thick segments (curb effect).

    Edge ``i`` connects vertex ``i`` to ``(i+1) % n``.
    """
    n = len(pts)
    if n < 2:
        return
    ip = _poly_points(pts)
    for i in range(n):
        a = ip[i]
        b = ip[(i + 1) % n]
        color = Colors.ZEBRA_WHITE if i % 2 == 0 else Colors.ZEBRA_RED
        pygame.draw.line(surf, color, a, b, line_width)


def _draw_poly_outline(surf: pygame.Surface, pts: Sequence[tuple[float, float]], width: int) -> None:
    """Draw only the polygon border in :attr:`~car_evolution.config.settings.Colors.TRACK_OUTLINE`."""
    ip = _poly_points(pts)
    if len(ip) < 2:
        return
    pygame.draw.polygon(surf, Colors.TRACK_OUTLINE, ip, width=width)


def _draw_grandstand(
    surf: pygame.Surface,
    x: int,
    y: int,
    w: int,
    h: int,
    rng: random.Random,
) -> None:
    """
    Draw a simple grandstand rectangle and scatter deterministic crowd dots (``rng``).

    ``rng`` should be fixed per bake so the image does not flicker between frames.
    """
    pygame.draw.rect(surf, Colors.GRANDSTAND_FACADE, (x, y, w, h), border_radius=3)
    pygame.draw.rect(surf, Colors.GRANDSTAND_FRAME, (x, y, w, h), 2, border_radius=3)
    crowd = [(220, 50, 50), (50, 100, 220), (255, 255, 255), (255, 180, 0)]
    for _ in range(20):
        c = rng.choice(crowd)
        rx = rng.randint(x + 5, x + w - 5)
        ry = rng.randint(y + 5, y + h - 5)
        pygame.draw.circle(surf, c, (rx, ry), 2)


def _travel_angle_spawn_to_gate0(track: RaceTrack) -> float:
    """
    Direction from spawn to the first checkpoint so the start grid matches the racing line.
    Falls back to ``start_angle`` if the points coincide.
    """
    sx, sy = track.start_position
    wx, wy = track.waypoints[0]
    dx, dy = wx - sx, wy - sy
    if dx * dx + dy * dy < 1.0:
        return track.start_angle
    return math.atan2(dy, dx)


def _draw_start_grid(
    surf: pygame.Surface,
    pos: tuple[float, float],
    travel_angle: float,
    half_width: float = 56.0,
) -> None:
    """
    Draw a small checkered pattern at the spawn (start/finish style).

    ``travel_angle`` points along the racing line; the pattern extends perpendicular across
    ``half_width`` pixels on each side. ``pos`` is the grid origin (spawn point).
    """
    px, py = pos
    perp = travel_angle + math.pi / 2
    for step in range(int(-half_width), int(half_width), 10):
        for col in range(3):
            cx = px + math.cos(perp) * step + math.cos(travel_angle) * (col * 10)
            cy = py + math.sin(perp) * step + math.sin(travel_angle) * (col * 10)
            color = Colors.WHITE if (step // 10 + col) % 2 == 0 else Colors.BLACK
            pygame.draw.rect(surf, color, (int(cx - 5), int(cy - 5), 10, 10))


def _closest_point_on_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float]:
    """Orthogonal projection of ``(px, py)`` onto segment ``ab``."""
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby + 1e-12
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    return ax + t * abx, ay + t * aby


def _closest_point_on_closed_polyline(
    px: float,
    py: float,
    vertices: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], int]:
    """
    Closest point on the closed polyline ``vertices`` to ``(px, py)``.

    Returns:
        ``((qx, qy), edge_index)`` where ``edge_index`` is the starting vertex of the segment.
    """
    n = len(vertices)
    best_q = (px, py)
    best_d2 = float("inf")
    best_i = 0
    for i in range(n):
        ax, ay = vertices[i]
        bx, by = vertices[(i + 1) % n]
        qx, qy = _closest_point_on_segment(px, py, ax, ay, bx, by)
        d2 = (qx - px) ** 2 + (qy - py) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_q = (qx, qy)
            best_i = i
    return best_q, best_i


def _outward_normal_from_outer_edge(
    vertices: Sequence[tuple[float, float]],
    edge_index: int,
) -> tuple[float, float]:
    """
    Unit normal perpendicular to the outer-boundary edge, pointing toward grass outside the island.

    Picks between the two perpendiculars by testing which side leaves the outer polygon first.
    """
    ax, ay = vertices[edge_index]
    bx, by = vertices[(edge_index + 1) % len(vertices)]
    ex, ey = bx - ax, by - ay
    el = math.hypot(ex, ey) or 1.0
    ex, ey = ex / el, ey / el
    mx, my = (ax + bx) * 0.5, (ay + by) * 0.5
    candidates = ((ey, -ex), (-ey, ex))
    for ox, oy in candidates:
        tx, ty = mx + ox * 40.0, my + oy * 40.0
        if not tg.point_in_polygon(tx, ty, vertices):
            return ox, oy
    return ey, -ex


def _checkpoint_flag_base_on_grass(
    waypoint: tuple[float, float],
    outer_points: Sequence[tuple[float, float]],
    margin: float = 16.0,
) -> tuple[float, float]:
    """Place the flag pole base just outside the outer asphalt edge, on the grass."""
    wx, wy = waypoint
    (qx, qy), ei = _closest_point_on_closed_polyline(wx, wy, outer_points)
    ox, oy = _outward_normal_from_outer_edge(outer_points, ei)
    return qx + ox * margin, qy + oy * margin


def _quad_bilinear(
    p00: tuple[float, float],
    p10: tuple[float, float],
    p11: tuple[float, float],
    p01: tuple[float, float],
    u: float,
    v: float,
) -> tuple[float, float]:
    """Bilinear interpolation on the unit square mapped to an arbitrary quad."""
    top_x = p00[0] * (1.0 - u) + p10[0] * u
    top_y = p00[1] * (1.0 - u) + p10[1] * u
    bot_x = p01[0] * (1.0 - u) + p11[0] * u
    bot_y = p01[1] * (1.0 - u) + p11[1] * u
    x = top_x * (1.0 - v) + bot_x * v
    y = top_y * (1.0 - v) + bot_y * v
    return x, y


def _draw_checkered_flag_quad(
    surf: pygame.Surface,
    p00: tuple[float, float],
    p10: tuple[float, float],
    p11: tuple[float, float],
    p01: tuple[float, float],
    cols: int,
    rows: int,
) -> None:
    """Fill a quad with a black/white checkerboard (classic racing flag, top-down)."""
    dark = (22, 22, 26)
    light = (238, 238, 242)
    for ci in range(cols):
        for cj in range(rows):
            u0, u1 = ci / cols, (ci + 1) / cols
            v0, v1 = cj / rows, (cj + 1) / rows
            cell = [
                _quad_bilinear(p00, p10, p11, p01, u0, v0),
                _quad_bilinear(p00, p10, p11, p01, u1, v0),
                _quad_bilinear(p00, p10, p11, p01, u1, v1),
                _quad_bilinear(p00, p10, p11, p01, u0, v1),
            ]
            color = dark if (ci + cj) % 2 == 0 else light
            pygame.draw.polygon(surf, color, [(int(p[0]), int(p[1])) for p in cell])


def _draw_checkpoint_flag(
    surf: pygame.Surface,
    base_x: float,
    base_y: float,
    along_angle: float,
    index: int,
) -> None:
    """
    Slim metal pole + checkered banner in the track plane, aligned with the racing line.

    First checkpoint uses a green border hint (start/finish style); others use a neutral rim.
    """
    ca, sa = math.cos(along_angle), math.sin(along_angle)
    perp_a = along_angle + math.pi / 2
    cp, sp = math.cos(perp_a), math.sin(perp_a)

    pole_len = 20.0
    tip_x = base_x + ca * pole_len
    tip_y = base_y + sa * pole_len

    bx, by = int(round(base_x)), int(round(base_y))
    tx, ty = int(round(tip_x)), int(round(tip_y))

    # Painted rings on grass (marshal post marking)
    pygame.draw.circle(surf, (255, 232, 170), (bx, by), 13, 2)
    pygame.draw.circle(surf, (255, 255, 255), (bx, by), 9, 1)
    pygame.draw.circle(surf, (180, 160, 90), (bx, by), 4, 1)

    # Pole: cast + body + highlight (read as brushed metal in top-down)
    pygame.draw.line(surf, (20, 22, 26), (bx + 2, by + 2), (tx + 2, ty + 2), 4)
    pygame.draw.line(surf, (88, 92, 102), (bx, by), (tx, ty), 3)
    hlx0 = base_x + cp * 0.55
    hly0 = base_y + sp * 0.55
    hlx1 = tip_x + cp * 0.55
    hly1 = tip_y + sp * 0.55
    pygame.draw.line(
        surf,
        (195, 200, 215),
        (int(round(hlx0)), int(round(hly0))),
        (int(round(hlx1)), int(round(hly1))),
        1,
    )

    # Banner quad: width across track, depth along forward
    fw, fd = 22.0, 15.0
    p00 = (tip_x, tip_y)
    p10 = (tip_x + cp * fw, tip_y + sp * fw)
    p11 = (tip_x + cp * fw + ca * fd, tip_y + sp * fw + sa * fd)
    p01 = (tip_x + ca * fd, tip_y + sa * fd)

    _draw_checkered_flag_quad(surf, p00, p10, p11, p01, cols=6, rows=4)

    rim = [(int(round(p00[0])), int(round(p00[1]))), (int(round(p10[0])), int(round(p10[1])))]
    rim += [(int(round(p11[0])), int(round(p11[1]))), (int(round(p01[0])), int(round(p01[1])))]
    border = (48, 175, 95) if index == 0 else (48, 48, 55)
    pygame.draw.lines(surf, border, True, rim, 2)

    # Ball finial
    pygame.draw.circle(surf, (70, 74, 82), (tx, ty), 3, 0)
    pygame.draw.circle(surf, (160, 165, 180), (tx - 1, ty - 1), 1, 0)


def _draw_checkpoint_flags(
    surf: pygame.Surface,
    track: RaceTrack,
) -> None:
    """One checkered marshal post per waypoint on the grass outside the outer curb."""
    outer = track.outer_points
    wps = list(track.waypoints)
    tangents = _waypoint_tangents(wps)
    for i, wp in enumerate(wps):
        bx, by = _checkpoint_flag_base_on_grass(wp, outer)
        _draw_checkpoint_flag(surf, bx, by, tangents[i], i)


def _waypoint_tangents(waypoints: Sequence[tuple[float, float]]) -> list[float]:
    """
    For each waypoint, heading from previous to next along the closed loop (for direction arrows).

    Returns:
        List of ``atan2`` angles parallel to the local track direction, length ``len(waypoints)``.
    """
    n = len(waypoints)
    angles: list[float] = []
    for i in range(n):
        prev_p = waypoints[(i - 1) % n]
        next_p = waypoints[(i + 1) % n]
        angles.append(math.atan2(next_p[1] - prev_p[1], next_p[0] - prev_p[0]))
    return angles


def build_track_background_surface(track: RaceTrack, display: DisplayConfig) -> pygame.Surface:
    """
    Bake the track visuals once for blitting each frame.

    Draws checkered checkpoint markers (metal pole + banner + grass rings) outside the outer curb.

    Args:
        track: Outer/inner polygons, waypoints, spawn pose.
        display: ``track_width`` and ``height`` for the left pane.

    Returns:
        Surface of size ``(track_width, height)``.
    """
    w, h = display.track_width, display.height
    surf = pygame.Surface((w, h))

    _draw_grass_stripes(surf, w, h)

    outer = list(track.outer_points)
    inner = list(track.inner_points)

    pygame.draw.polygon(surf, Colors.ASPHALT, _poly_points(outer))
    pygame.draw.polygon(surf, Colors.GRASS_DARK, _poly_points(inner))

    _draw_zebra_polyline(surf, outer, line_width=12)
    _draw_zebra_polyline(surf, inner, line_width=8)

    _draw_poly_outline(surf, outer, width=2)
    _draw_poly_outline(surf, inner, width=2)

    wps = list(track.waypoints)
    tangents = _waypoint_tangents(wps)
    for i in range(0, len(wps), 4):
        draw_track_arrow(surf, Colors.ARROW_MUTED, wps[i], tangents[i], size=14)

    _draw_checkpoint_flags(surf, track)

    _draw_start_grid(surf, track.start_position, _travel_angle_spawn_to_gate0(track))

    rng = random.Random(42)
    _draw_grandstand(surf, 450, 15, 300, 25, rng)
    _draw_grandstand(surf, 450, h - 40, 300, 25, rng)

    pygame.draw.line(surf, Colors.PIT_LINE, (800, 100), (800, 250), 2)

    return surf
