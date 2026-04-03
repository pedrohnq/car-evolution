"""Polygon corridor math, checkpoint segments, and :class:`~car_evolution.track.layout.RaceTrack` layouts."""

from car_evolution.track import geometry
from car_evolution.track.geometry import (
    CheckpointSegment,
    Point,
    Segment,
    checkpoint_segments,
    movement_corridor_valid,
    point_in_corridor,
    point_in_polygon,
    progress_lex_better,
    segment_cross,
    segment_cuts_any_wall,
    side_of_line,
)
from car_evolution.track.layout import RaceTrack

__all__ = [
    "RaceTrack",
    "geometry",
    "CheckpointSegment",
    "Point",
    "Segment",
    "checkpoint_segments",
    "movement_corridor_valid",
    "point_in_corridor",
    "point_in_polygon",
    "progress_lex_better",
    "segment_cross",
    "segment_cuts_any_wall",
    "side_of_line",
]
