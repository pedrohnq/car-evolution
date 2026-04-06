"""
Central configuration: window layout, timing, debug flags, and color palette.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DisplayConfig:
    """
    Pygame window geometry.

    Attributes:
        track_width: Pixels for the racing view (left pane).
        ui_width: Pixels for the GA dashboard (right pane).
        height: Window height in pixels.
        fps: Target frames per second for :meth:`pygame.time.Clock.tick`.
    """

    track_width: int = 1200
    ui_width: int = 420
    height: int = 800
    fps: int = 60

    @property
    def width(self) -> int:
        """Total window width (track + UI panel)."""
        return self.track_width + self.ui_width


@dataclass(frozen=True)
class SimulationConfig:
    """
    Core simulation and GA sizing.

    Attributes:
        default_seed: Initial RNG seed for reproducible first runs (``R`` key restores it).
        population_size: Number of :class:`~car_evolution.core.car.Car` instances per generation.
        max_frames_per_generation: Force evolution after this many frames if cars still run.
        convergence_plateau_generations: End a parameter run after this many generations without
            improving session peak fitness.
        max_generations_per_run: Hard cap on generations per parameter run (whichever comes first).
    """

    default_seed: int = 42
    population_size: int = 40
    max_frames_per_generation: int = 600
    convergence_plateau_generations: int = 50
    max_generations_per_run: int = 500


@dataclass(frozen=True)
class DebugConfig:
    """
    Optional developer switches (reserved for future visualization hooks).

    Attributes:
        show_checkpoint_debug: Placeholder for drawing gate geometry in-game.
        debug_evolution: Placeholder for verbose GA logging.
        debug_progress_every_n_frames: Placeholder for throttled debug prints.
    """

    show_checkpoint_debug: bool = False
    debug_evolution: bool = True
    debug_progress_every_n_frames: int = 60


class Colors:
    """
    Shared RGB ``(R, G, B)`` constants for pygame drawing.

    Grouped conceptually: dashboard text, track backdrop, and car accents (body uses ``GREEN`` /
    ``RED`` / ``CYAN`` from car state).
    """

    # --- UI panel ---
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (150, 150, 150)
    DARK_GRAY = (30, 30, 30)
    PANEL_BG = (15, 15, 15)
    GREEN = (0, 255, 0)
    RED = (255, 50, 50)
    CYAN = (0, 255, 255)
    ORANGE = (255, 165, 0)
    YELLOW = (255, 200, 0)
    PURPLE = (200, 0, 255)
    TEXT = (220, 220, 220)
    HIGHLIGHT = (70, 130, 180)
    UI_SECTION_DIVIDER = (50, 50, 50)

    # --- Track (grass, asphalt, curbs) ---
    GRASS_LIGHT = (34, 139, 34)
    GRASS_DARK = (28, 115, 28)
    ASPHALT = (60, 60, 65)
    TRACK_OUTLINE = (25, 25, 25)
    ZEBRA_RED = (200, 30, 30)
    ZEBRA_WHITE = (240, 240, 240)
    PIT_LINE = (255, 200, 0)
    ARROW_MUTED = (100, 100, 105)
    GRANDSTAND_FACADE = (140, 140, 145)
    GRANDSTAND_FRAME = (60, 60, 60)

    # --- Car sprite (body uses GREEN / RED / CYAN from states above) ---
    CAR_SHADOW = (15, 50, 15)
    CAR_NOSE = (25, 25, 25)


# Default singletons used by the game loop
DISPLAY = DisplayConfig()
SIMULATION = SimulationConfig()
DEBUG = DebugConfig()
