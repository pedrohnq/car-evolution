"""
Pygame drawing helpers: text, track arrows, and the right-hand evolution dashboard.
"""

from __future__ import annotations

import math
from typing import Sequence

import pygame

from car_evolution.config import Colors, DisplayConfig
from car_evolution.core.population import Population


def draw_text(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = Colors.TEXT,
) -> int:
    """
    Render and blit a single line of text.

    Args:
        screen: Destination surface (usually the full window).
        font: Pygame font used for rendering.
        text: UTF-8 string (no automatic wrapping).
        x, y: Top-left pixel position.
        color: RGB tuple.

    Returns:
        The y-coordinate just below this line, including a 5px gap for stacking.
    """
    img = font.render(text, True, color)
    screen.blit(img, (x, y))
    return y + img.get_height() + 5


def _text_width(font: pygame.font.Font, text: str) -> int:
    """Pixel width of ``text`` when rendered with ``font`` (used for wrap decisions)."""
    return font.render(text, True, Colors.TEXT).get_width()


def draw_text_wrapped(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    x: int,
    y: int,
    max_width: int,
    color: tuple[int, int, int] = Colors.TEXT,
) -> int:
    """
    Draw text on one or two lines so each fits within ``max_width``.

    If the full string is too wide, splits once at the nearest space before the midpoint.

    Returns:
        The y-coordinate below the last rendered line (including spacing).
    """
    if _text_width(font, text) <= max_width:
        return draw_text(screen, font, text, x, y, color)
    mid = len(text) // 2
    split = text.rfind(" ", 0, min(len(text), mid + 15))
    if split <= 0:
        split = text.find(" ")
    if split <= 0:
        return draw_text(screen, font, text, x, y, color)
    y = draw_text(screen, font, text[:split], x, y, color)
    return draw_text(screen, font, text[split + 1 :].strip(), x, y, color)


def draw_track_arrow(
    screen: pygame.Surface,
    color: tuple[int, int, int],
    center: tuple[float, float],
    angle_rad: float,
    size: float = 25,
) -> None:
    """
    Draw a filled triangle pointing in direction ``angle_rad`` (radians, standard math axes).

    Args:
        screen: Destination surface.
        color: Fill RGB.
        center: Arrow base center in screen coordinates.
        angle_rad: Heading of the tip.
        size: Distance from center to tip.
    """
    p1 = (center[0] + math.cos(angle_rad) * size, center[1] + math.sin(angle_rad) * size)
    p2 = (center[0] + math.cos(angle_rad + 2.6) * size, center[1] + math.sin(angle_rad + 2.6) * size)
    p3 = (center[0] + math.cos(angle_rad - 2.6) * size, center[1] + math.sin(angle_rad - 2.6) * size)
    pygame.draw.polygon(screen, color, [p1, p2, p3])


class EvolutionDashboard:
    """
    Stats at the top, history in the middle (uses only space above the footer),
    parameters and controls drawn last and anchored to the bottom of the panel.
    """

    def __init__(
        self,
        display: DisplayConfig,
        font_large: pygame.font.Font,
        font_normal: pygame.font.Font,
        font_small: pygame.font.Font,
    ) -> None:
        """
        Args:
            display: Window dimensions and track/UI split.
            font_large: Section titles (e.g. GA DASHBOARD).
            font_normal: Stats and parameter labels.
            font_small: History lines and hotkey hints.
        """
        self._display = display
        self._font_large = font_large
        self._font_normal = font_normal
        self._font_small = font_small

    def _footer_height(self) -> int:
        """
        Estimated vertical pixels reserved for PARAMETERS + CONTROLS at the panel bottom.

        Includes a small safety margin so the last line is not clipped by the window.
        """
        fl = self._font_large
        fn = self._font_normal
        fs = self._font_small
        gap = 5
        h = 0
        h += fl.get_height() + gap + 8 + gap
        h += (fn.get_height() + gap) * 3
        h += 12
        h += fl.get_height() + gap + 8 + gap
        h += (fs.get_height() + gap) * 6
        return h + 18  # safety margin so the last control line is not clipped

    def draw(
        self,
        screen: pygame.Surface,
        pop: Population,
        frame_counter: int,
        max_frames_per_generation: int,
        global_best_cp: int,
        global_best_dist: float,
        global_max_fitness: float,
        victory_history: Sequence[tuple[int, int]],
        num_waypoints: int,
    ) -> None:
        """
        Paint the full sidebar: stats, history (middle, height-limited), footer parameters/controls.

        Clips drawing to the panel interior so text cannot spill into the track. History rows are
        truncated with a ``+N older`` line if they would overlap the footer band.

        Args:
            screen: Full window surface.
            pop: Current genetic population (generation, rates, selection labels).
            frame_counter: Frames elapsed in the current generation.
            max_frames_per_generation: Used to show remaining time.
            global_best_cp: Best checkpoint count achieved this session.
            global_best_dist: Distance to next gate for that progress leader.
            global_max_fitness: Peak fitness seen since session start.
            victory_history: ``(generation_index, num_cars_finished)`` chronologically.
            num_waypoints: Total gates on the track (for progress strings).
        """
        d = self._display
        pad_l = 16
        pad_r = 12
        text_max_w = d.ui_width - pad_l - pad_r
        bottom_margin = 10
        top_margin = 6

        ui_rect = pygame.Rect(d.track_width, 0, d.ui_width, d.height)
        pygame.draw.rect(screen, Colors.PANEL_BG, ui_rect)
        pygame.draw.line(screen, Colors.HIGHLIGHT, (d.track_width, 0), (d.track_width, d.height), 3)

        inner = pygame.Rect(
            d.track_width + pad_l,
            top_margin,
            d.ui_width - pad_l - pad_r,
            d.height - top_margin - bottom_margin,
        )
        prev_clip = screen.get_clip()
        screen.set_clip(inner)

        px = d.track_width + pad_l
        right_x = d.track_width + d.ui_width - pad_r
        fl, fn, fs = self._font_large, self._font_normal, self._font_small
        line_fs = fs.get_height() + 5

        footer_h = self._footer_height()
        y_footer = d.height - bottom_margin - footer_h
        gap_above_footer = 12
        y_hist_hard_stop = y_footer - gap_above_footer

        # --- Top: GA DASHBOARD ---
        y = top_margin
        y = draw_text(screen, fl, "GA DASHBOARD", px, y, Colors.YELLOW)
        pygame.draw.line(screen, Colors.UI_SECTION_DIVIDER, (px, y), (right_x, y), 1)
        y += 10

        time_left = max(0, (max_frames_per_generation - frame_counter) // 60)
        driving = sum(1 for c in pop.cars if c.alive and not c.finished)
        finished = sum(1 for c in pop.cars if c.finished)

        y = draw_text(screen, fn, f"Generation: {pop.generation}", px, y)
        y = draw_text(screen, fn, f"Driving: {driving} / {pop.size}", px, y)
        y = draw_text(screen, fn, f"Finished: {finished} / {pop.size}", px, y)
        y = draw_text(screen, fn, f"Time left: {time_left}s", px, y)

        y += 6
        dist_lbl = f"{global_best_dist:.0f}px" if global_best_dist < float("inf") else "--"
        y = draw_text(screen, fn, f"Best gates: {global_best_cp}/{num_waypoints}", px, y, Colors.PURPLE)
        y = draw_text(screen, fn, f"Next gate: {dist_lbl}", px, y, Colors.PURPLE)
        y = draw_text(screen, fs, f"Max fitness: {global_max_fitness:.2f}", px, y, Colors.GRAY)

        leader = max(pop.cars, key=lambda c: c.best_fitness) if pop.cars else None
        if leader:
            prog = leader.progress_gates(num_waypoints)
            y = draw_text_wrapped(
                screen,
                fs,
                f"Leader {prog}/{num_waypoints} done={leader.finished}",
                px,
                y,
                text_max_w,
                Colors.CYAN,
            )

        # --- Middle: HISTORY (stops above footer; footer draw clears overlap) ---
        y += 12
        if y < y_hist_hard_stop - fl.get_height() - line_fs:
            y = draw_text(screen, fl, "HISTORY", px, y, Colors.YELLOW)
            pygame.draw.line(screen, Colors.UI_SECTION_DIVIDER, (px, y), (right_x, y), 1)
            y += 8

            if not victory_history:
                if y < y_hist_hard_stop:
                    draw_text(screen, fs, "No full lap yet.", px, y, Colors.GRAY)
            else:
                n = len(victory_history)
                y_line = y
                max_lines = max(0, (y_hist_hard_stop - y_line) // line_fs)
                if max_lines > 0:
                    if n > max_lines:
                        y_line = draw_text(screen, fs, f"+{n - max_lines} older", px, y_line, Colors.GRAY)
                        max_lines = max(0, (y_hist_hard_stop - y_line) // line_fs)
                    start = max(0, n - max_lines)
                    for gen, count in victory_history[start:]:
                        if y_line >= y_hist_hard_stop:
                            break
                        y_line = draw_text_wrapped(
                            screen,
                            fs,
                            f"G{gen}: {count} finished",
                            px,
                            y_line,
                            text_max_w,
                            Colors.GREEN,
                        )

        # Clear band behind footer (history may have extended into this vertical band).
        footer_clear = pygame.Rect(
            d.track_width + 2,
            y_footer - 6,
            d.ui_width - 4,
            d.height - (y_footer - 6),
        )
        pygame.draw.rect(screen, Colors.PANEL_BG, footer_clear)

        # --- Bottom: PARAMETERS + CONTROLS ---
        yf = y_footer
        yf = draw_text(screen, fl, "PARAMETERS", px, yf, Colors.HIGHLIGHT)
        pygame.draw.line(screen, Colors.UI_SECTION_DIVIDER, (px, yf), (right_x, yf), 1)
        yf += 8
        yf = draw_text(screen, fn, f"Mutation:  {pop.mutation_rate:.3f}", px, yf, Colors.CYAN)
        yf = draw_text(screen, fn, f"Crossover: {pop.crossover_rate:.2f}", px, yf, Colors.CYAN)
        yf = draw_text_wrapped(
            screen,
            fn,
            f"Selection: {pop.selection_method}",
            px,
            yf,
            text_max_w,
            Colors.CYAN,
        )
        yf += 12
        yf = draw_text(screen, fl, "CONTROLS", px, yf, Colors.GREEN)
        pygame.draw.line(screen, Colors.UI_SECTION_DIVIDER, (px, yf), (right_x, yf), 1)
        yf += 8
        for key in (
            "[UP/DN] Mutation",
            "[L/R] Crossover",
            "[S] Selection",
            "[C] Crossover type",
            "[R] Restart (same seed)",
            "[N] New seed",
        ):
            yf = draw_text_wrapped(screen, fs, key, px, yf, text_max_w, Colors.TEXT)

        screen.set_clip(prev_clip)
