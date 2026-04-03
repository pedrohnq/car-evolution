"""
Neural-network-controlled cars evolved with a genetic algorithm in a 2D top-down track.

Package layout:

- ``car_evolution.config`` - Display/simulation defaults and color palette.
- ``car_evolution.core`` - Car agent, feed-forward network, population, RNG seeding.
- ``car_evolution.track`` - Polygon geometry, checkpoints, and :class:`~car_evolution.track.layout.RaceTrack`.
- ``car_evolution.rendering`` - Pygame HUD, static track backdrop, text helpers.
- ``car_evolution.evolution`` - Per-generation CSV logging and milestone schedules.
- ``car_evolution.io`` - Resolved project root and log paths under ``logs/``.
- ``car_evolution.app`` - :class:`~car_evolution.app.game.EvolutionGame` main loop.

Entry points: :func:`car_evolution.run` or :class:`car_evolution.EvolutionGame`.
"""

from car_evolution.app.game import EvolutionGame, run

__all__ = ["EvolutionGame", "run"]
