"""
Agent car: sensors, physics step, fitness, and rendering.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from car_evolution.config import Colors
from car_evolution.core.neural_network import NeuralNetwork
from car_evolution.track import geometry as tg


class Car:
    """
    A car controlled by a :class:`~car_evolution.core.neural_network.NeuralNetwork`.

    Reads five ray sensors, applies throttle/steer from the network, updates pose,
    clears checkpoints, and accumulates fitness until collision, stall, or finish.
    """

    TRACK_COMPLETE_BONUS = 500_000
    SEGMENT_BONUS = 1000
    MAX_FRAMES_WITHOUT_CHECKPOINT = 260

    def __init__(self, x: float, y: float, angle: float) -> None:
        """
        Args:
            x, y: Initial position in track coordinates.
            angle: Heading in radians.
        """
        self.x = x
        self.y = y
        self.angle = angle
        self.prev_x = x
        self.prev_y = y
        self.speed = 0.0
        self.max_speed = 8.0

        self.alive = True
        self.finished = False
        self.fitness = 0.0
        self.best_fitness = 0.0

        self.target_wp = 0
        self.frames_since_last_wp = 0
        self.checkpoints_cleared = 0

        self.age_frames = 0
        self.finish_frame: int | None = None
        self.just_finished_track = False

        self.brain = NeuralNetwork([5, 5, 2])
        self.sensor_lengths: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]

    def progress_gates(self, n_seg: int) -> int:
        """Effective gate count for progress display (full lap => ``n_seg``)."""
        if self.finished:
            return n_seg
        return self.checkpoints_cleared

    def dist_to_next_gate(self, waypoint_centers: Sequence[tg.Point]) -> float:
        """Euclidean distance to the current target waypoint; 0 if finished or past last."""
        n = len(waypoint_centers)
        if self.finished or self.target_wp >= n:
            return 0.0
        cx, cy = waypoint_centers[self.target_wp]
        return math.hypot(self.x - cx, self.y - cy)

    def _compute_fitness(self, waypoint_centers: Sequence[tg.Point]) -> float:
        """Fitness from checkpoints plus a small tie-breaker from distance to next gate."""
        n_seg = len(waypoint_centers)
        if self.finished:
            return float(n_seg * self.SEGMENT_BONUS + self.TRACK_COMPLETE_BONUS)

        base = self.checkpoints_cleared * self.SEGMENT_BONUS

        cx, cy = waypoint_centers[self.target_wp]
        dist_now = math.hypot(self.x - cx, self.y - cy)
        tie_breaker = max(0.0, 500 - dist_now) / 500.0

        return float(base + tie_breaker)

    def update(
        self,
        track_lines: Sequence[tg.Segment],
        waypoint_centers: Sequence[tg.Point],
        checkpoint_segments: Sequence[tg.CheckpointSegment],
        track_geom: dict[str, Any],
    ) -> None:
        """
        One simulation step: sense, act, integrate motion, gates, fitness, death rules.

        Args:
            track_lines: Wall segments for sensors and collision.
            waypoint_centers: Gate centers in order.
            checkpoint_segments: From :meth:`car_evolution.track.layout.RaceTrack.gates`.
            track_geom: ``collision_geometry`` from the track (outer/inner poly, walls).
        """
        if not self.alive or self.finished:
            return

        self.prev_x, self.prev_y = self.x, self.y
        self.age_frames += 1
        self.just_finished_track = False

        self.read_sensors(track_lines)

        action = self.brain.predict(self.sensor_lengths)
        throttle = float(action[0])
        steering = float(action[1])

        if throttle > 0:
            self.speed += throttle * 0.3
        else:
            self.speed += throttle * 0.5

        self.speed = max(-1.0, min(self.speed, self.max_speed))

        if abs(self.speed) > 0.1:
            direction = 1 if self.speed > 0 else -1
            self.angle += steering * 0.05 * direction

        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        outer_poly = track_geom["outer_poly"]
        inner_poly = track_geom["inner_poly"]
        wall_segments = track_geom["wall_segments"]
        nseg = len(checkpoint_segments)

        # Strong wrong-way penalty at start straight
        if self.target_wp == 0 and self.x > 850:
            self.alive = False
            self.fitness -= 1000
        else:
            if self.target_wp < nseg:
                la, lb, forward = checkpoint_segments[self.target_wp]
                crossed = tg.segment_cross(
                    (self.prev_x, self.prev_y), (self.x, self.y), la, lb
                )
                geom_ok = crossed and tg.movement_corridor_valid(
                    (self.prev_x, self.prev_y),
                    (self.x, self.y),
                    outer_poly,
                    inner_poly,
                    wall_segments,
                )
                if geom_ok:
                    vx = self.x - self.prev_x
                    vy = self.y - self.prev_y
                    motion_dot = vx * forward[0] + vy * forward[1]
                    side_prev = tg.side_of_line((self.prev_x, self.prev_y), la, lb)
                    side_now = tg.side_of_line((self.x, self.y), la, lb)

                    if motion_dot > 0.02 and side_prev * side_now < 0:
                        old_wp = self.target_wp
                        self.checkpoints_cleared += 1
                        self.frames_since_last_wp = 0

                        if old_wp == nseg - 1:
                            self.finished = True
                            self.speed = 0.0
                            self.finish_frame = self.age_frames
                            self.just_finished_track = True
                            self.target_wp = nseg
                        else:
                            self.target_wp += 1

            self.fitness = self._compute_fitness(waypoint_centers)

        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness

        self.frames_since_last_wp += 1
        if self.frames_since_last_wp > self.MAX_FRAMES_WITHOUT_CHECKPOINT:
            self.alive = False

        if not self.finished:
            self.check_collision(track_lines)

    def read_sensors(self, track_lines: Sequence[tg.Segment]) -> None:
        """Cast five rays and store normalized distances in ``self.sensor_lengths``."""
        angles = [-math.pi / 2, -math.pi / 4, 0, math.pi / 4, math.pi / 2]
        self.sensor_lengths = []
        max_dist = 200

        for a in angles:
            ray_angle = self.angle + a
            ray_end_x = self.x + math.cos(ray_angle) * max_dist
            ray_end_y = self.y + math.sin(ray_angle) * max_dist

            closest_dist = max_dist
            for line in track_lines:
                dist = self.point_line_distance(
                    self.x, self.y, line[0], line[1], True, ray_end_x, ray_end_y
                )
                if dist is not None and dist < closest_dist:
                    closest_dist = dist
            self.sensor_lengths.append(closest_dist / max_dist)

    def check_collision(self, track_lines: Sequence[tg.Segment]) -> None:
        """Kill the car if the body is too close to any wall segment."""
        for line in track_lines:
            if self.point_line_distance(self.x, self.y, line[0], line[1]) < 10:
                self.alive = False
                break

    def point_line_distance(
        self,
        px: float,
        py: float,
        p1: tg.Point,
        p2: tg.Point,
        is_raycast: bool = False,
        ray_x: float = 0.0,
        ray_y: float = 0.0,
    ) -> float | None:
        """
        Distance from ``(px, py)`` to segment ``p1``-``p2``, or ray-segment hit distance.

        Args:
            is_raycast: If True, measure along the ray from ``(px, py)`` to ``(ray_x, ray_y)``.
        """
        if is_raycast:
            x1, y1 = px, py
            x2, y2 = ray_x, ray_y
            x3, y3 = p1[0], p1[1]
            x4, y4 = p2[0], p2[1]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if den == 0:
                return None
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
            if 0 < t < 1 and 0 < u < 1:
                ix = x1 + t * (x2 - x1)
                iy = y1 + t * (y2 - y1)
                return math.hypot(ix - px, iy - py)
            return None

        line_mag = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if line_mag == 0:
            return math.hypot(px - p1[0], py - p1[1])
        u = ((px - p1[0]) * (p2[0] - p1[0]) + (py - p1[1]) * (p2[1] - p1[1])) / (line_mag**2)
        if u < 0:
            return math.hypot(px - p1[0], py - p1[1])
        if u > 1:
            return math.hypot(px - p2[0], py - p2[1])
        ix = p1[0] + u * (p2[0] - p1[0])
        iy = p1[1] + u * (p2[1] - p1[1])
        return math.hypot(px - ix, py - iy)

    def draw(self, screen: Any) -> None:
        """Small kart: green while driving, red when dead, cyan when lap complete."""
        import pygame

        ix, iy = int(self.x), int(self.y)
        if self.finished:
            body = Colors.CYAN
        elif self.alive:
            body = Colors.GREEN
        else:
            body = Colors.RED

        pygame.draw.circle(screen, Colors.CAR_SHADOW, (ix + 3, iy + 3), 10)
        pygame.draw.circle(screen, body, (ix, iy), 9)
        fx = self.x + math.cos(self.angle) * 7
        fy = self.y + math.sin(self.angle) * 7
        pygame.draw.circle(screen, Colors.CAR_NOSE, (int(fx), int(fy)), 5)
        pygame.draw.circle(screen, Colors.WHITE, (ix, iy), 4)
