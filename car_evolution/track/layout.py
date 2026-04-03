"""
Racetrack definition: outer/inner boundaries, waypoints, and derived geometry for the simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from . import geometry as tg


@dataclass(frozen=True)
class RaceTrack:
    """
    Immutable track layout: polygon boundaries, line segments, checkpoints, and spawn pose.

    ``collision_geometry`` matches the dict previously built inline in ``main`` for
    :class:`~car_evolution.core.car.Car`.
    """

    outer_points: tuple[tg.Point, ...]
    inner_points: tuple[tg.Point, ...]
    waypoints: tuple[tg.Point, ...]
    start_position: tg.Point
    start_angle: float
    checkpoint_half_len: float = 120.0

    @property
    def outer_segments(self) -> list[tg.Segment]:
        """Directed edges of the outer boundary."""
        pts = self.outer_points
        return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]

    @property
    def inner_segments(self) -> list[tg.Segment]:
        """Directed edges of the inner boundary."""
        pts = self.inner_points
        return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]

    @property
    def all_track_lines(self) -> list[tg.Segment]:
        """All wall segments (outer then inner) for collision and raycasts."""
        return self.outer_segments + self.inner_segments

    @property
    def gates(self) -> list[tg.CheckpointSegment]:
        """Perpendicular checkpoint segments through each waypoint."""
        return tg.checkpoint_segments(self.waypoints, half_len=self.checkpoint_half_len)

    @property
    def collision_geometry(self) -> dict[str, Any]:
        """
        Dict passed to :meth:`car_evolution.core.car.Car.update`:

        - ``outer_poly``, ``inner_poly``: vertex lists
        - ``wall_segments``: list of line segments
        """
        return {
            "outer_poly": list(self.outer_points),
            "inner_poly": list(self.inner_points),
            "wall_segments": self.all_track_lines,
        }

    @classmethod
    def default_hardcore(cls) -> RaceTrack:
        """Original bundled track: hardcore layout and 16 gates."""
        outer_points = (
            (1000, 100),
            (200, 100),
            (50, 250),
            (50, 500),
            (200, 650),
            (400, 650),
            (500, 450),
            (600, 450),
            (700, 650),
            (900, 700),
            (1100, 600),
            (1150, 350),
        )
        inner_points = (
            (900, 250),
            (300, 250),
            (200, 300),
            (200, 450),
            (300, 500),
            (350, 500),
            (450, 300),
            (650, 300),
            (800, 500),
            (950, 500),
            (1000, 400),
        )
        waypoints = (
            (600, 175),
            (400, 175),
            (250, 175),
            (125, 250),
            (125, 400),
            (125, 550),
            (250, 600),
            (350, 575),
            (475, 400),
            (600, 375),
            (700, 500),
            (800, 600),
            (950, 625),
            (1050, 500),
            (950, 350),
            (800, 175),
        )

        return cls(
            outer_points=outer_points,
            inner_points=inner_points,
            waypoints=waypoints,
            start_position=(800, 175),
            start_angle=math.pi + 0.35,
            checkpoint_half_len=120.0,
        )
