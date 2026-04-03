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
from car_evolution.track.layout import RaceTrack


def _poly_points(pts: Sequence[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(int(p[0]), int(p[1])) for p in pts]


def _draw_grass_stripes(surf: pygame.Surface, width: int, height: int) -> None:
    stripe = 40
    for y in range(0, height, stripe):
        color = Colors.GRASS_LIGHT if (y // stripe) % 2 == 0 else Colors.GRASS_DARK
        pygame.draw.rect(surf, color, (0, y, width, stripe))


def _draw_zebra_polyline(
    surf: pygame.Surface,
    pts: Sequence[tuple[float, float]],
    line_width: int,
) -> None:
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
    """Checkered boxes: ``travel_angle`` is along the track; width spans the corridor."""
    px, py = pos
    perp = travel_angle + math.pi / 2
    for step in range(int(-half_width), int(half_width), 10):
        for col in range(3):
            cx = px + math.cos(perp) * step + math.cos(travel_angle) * (col * 10)
            cy = py + math.sin(perp) * step + math.sin(travel_angle) * (col * 10)
            color = Colors.WHITE if (step // 10 + col) % 2 == 0 else Colors.BLACK
            pygame.draw.rect(surf, color, (int(cx - 5), int(cy - 5), 10, 10))


def _waypoint_tangents(waypoints: Sequence[tuple[float, float]]) -> list[float]:
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

    _draw_start_grid(surf, track.start_position, _travel_angle_spawn_to_gate0(track))

    rng = random.Random(42)
    _draw_grandstand(surf, 450, 15, 300, 25, rng)
    _draw_grandstand(surf, 450, h - 40, 300, 25, rng)

    pygame.draw.line(surf, Colors.PIT_LINE, (800, 100), (800, 250), 2)

    return surf
