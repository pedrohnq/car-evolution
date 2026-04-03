"""
Pygame application: main loop, event input, track backdrop, car simulation, and GA steps.

Creates one CSV log per session under ``logs/`` (see :mod:`car_evolution.io.paths`).
"""

from __future__ import annotations

import random

import pygame

from car_evolution.config import Colors, DISPLAY, SIMULATION
from car_evolution.core.population import Population
from car_evolution.core.rng import set_global_seed
from car_evolution.evolution.logger import EvolutionCSVLogger
from car_evolution.evolution.schedule import EvolutionParameterSchedule
from car_evolution.io.paths import evolution_log_path
from car_evolution.rendering.track_background import build_track_background_surface
from car_evolution.rendering.ui import EvolutionDashboard
from car_evolution.track import geometry as tg
from car_evolution.track.layout import RaceTrack


class EvolutionGame:
    """
    Runs the full-screen simulation: track rendering, car updates, GA steps, and CSV logs.

    Uses :data:`~car_evolution.config.settings.DISPLAY` and :data:`~car_evolution.config.settings.SIMULATION`
    unless you subclass and override wiring.
    """

    def __init__(self, track: RaceTrack | None = None) -> None:
        """
        Args:
            track: Layout to simulate; ``None`` uses :meth:`~car_evolution.track.layout.RaceTrack.default_hardcore`.

        Also stores :class:`~car_evolution.evolution.schedule.EvolutionParameterSchedule` for generation milestones.
        """
        self._track = track or RaceTrack.default_hardcore()
        self._display = DISPLAY
        self._sim = SIMULATION
        self._schedule = EvolutionParameterSchedule()

    def run(self) -> None:
        """
        Create the window, load fonts and static track art, then run until quit.

        Handles pygame events, car updates, generation boundaries, CSV logging via
        :class:`~car_evolution.evolution.logger.EvolutionCSVLogger`, and ``pygame.quit`` on exit.
        """
        d = self._display
        sim = self._sim
        track = self._track

        pygame.init()
        screen = pygame.display.set_mode((d.width, d.height))
        pygame.display.set_caption("AI Car Evolution - Genetic Algorithm")
        clock = pygame.time.Clock()

        font_large = pygame.font.SysFont("Courier New", 22, bold=True)
        font_normal = pygame.font.SysFont("Courier New", 18, bold=True)
        font_small = pygame.font.SysFont("Courier New", 15)
        dashboard = EvolutionDashboard(d, font_large, font_normal, font_small)

        track_background = build_track_background_surface(track, d)

        all_track_lines = track.all_track_lines
        track_geom = track.collision_geometry
        waypoints = track.waypoints
        checkpoint_segments = track.gates
        start_pos = track.start_position
        start_angle = track.start_angle

        current_seed = sim.default_seed
        set_global_seed(current_seed)
        pop = Population(size=sim.population_size, start_pos=start_pos, start_angle=start_angle)

        max_frames = sim.max_frames_per_generation
        frame_counter = 0
        global_best_cp = 0
        global_best_dist = float("inf")
        global_progress_pos: tuple[float, float] = start_pos
        global_max_fitness = 0.0
        victory_history: list[tuple[int, int]] = []

        logger = EvolutionCSVLogger(evolution_log_path())
        n_wp = len(waypoints)

        running = True
        while running:
            screen.blit(track_background, (0, 0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_UP:
                        pop.mutation_rate = min(0.1, pop.mutation_rate + 0.01)
                    elif event.key == pygame.K_DOWN:
                        pop.mutation_rate = max(0.001, pop.mutation_rate - 0.01)
                    elif event.key == pygame.K_RIGHT:
                        pop.crossover_rate = min(1.0, pop.crossover_rate + 0.05)
                    elif event.key == pygame.K_LEFT:
                        pop.crossover_rate = max(0.6, pop.crossover_rate - 0.05)
                    elif event.key == pygame.K_s:
                        pop.selection_method = (
                            "Roulette" if pop.selection_method == "Tournament" else "Tournament"
                        )
                    elif event.key == pygame.K_c:
                        pop.crossover_method = (
                            "One-Point" if pop.crossover_method == "Uniform" else "Uniform"
                        )
                    elif event.key == pygame.K_r:
                        set_global_seed(current_seed)
                        pop = Population(size=sim.population_size, start_pos=start_pos, start_angle=start_angle)
                        frame_counter = 0
                        global_best_cp = 0
                        global_best_dist = float("inf")
                        global_progress_pos = start_pos
                        global_max_fitness = 0.0
                        victory_history.clear()
                    elif event.key == pygame.K_n:
                        current_seed = random.randint(1, 999_999)
                        set_global_seed(current_seed)
                        pop = Population(size=sim.population_size, start_pos=start_pos, start_angle=start_angle)
                        frame_counter = 0
                        global_best_cp = 0
                        global_best_dist = float("inf")
                        global_progress_pos = start_pos
                        global_max_fitness = 0.0
                        victory_history.clear()

            for car in pop.cars:
                car.update(all_track_lines, waypoints, checkpoint_segments, track_geom)
                car.draw(screen)

                cp = car.progress_gates(n_wp)
                dnext = car.dist_to_next_gate(waypoints)
                if tg.progress_lex_better(cp, dnext, global_best_cp, global_best_dist):
                    global_best_cp = cp
                    global_best_dist = dnext
                    global_progress_pos = (car.x, car.y)

                if car.best_fitness > global_max_fitness:
                    global_max_fitness = car.best_fitness

            mx, my = int(global_progress_pos[0]), int(global_progress_pos[1])
            pygame.draw.line(screen, Colors.PURPLE, (mx - 28, my), (mx + 28, my), 3)
            pygame.draw.line(screen, Colors.PURPLE, (mx, my - 28), (mx, my + 28), 3)
            pygame.draw.circle(screen, Colors.PURPLE, (mx, my), 22, 3)
            pygame.draw.circle(screen, Colors.YELLOW, (mx, my), 6, 2)
            pygame.draw.circle(screen, Colors.BLACK, (mx, my), 2)

            frame_counter += 1

            if pop.all_inactive() or frame_counter >= max_frames:
                finished_count = sum(1 for c in pop.cars if c.finished)
                if finished_count > 0:
                    victory_history.append((pop.generation, finished_count))

                self._schedule.apply(pop)

                leader = max(pop.cars, key=lambda c: c.best_fitness) if pop.cars else None
                leader_gates = leader.progress_gates(n_wp) if leader else 0

                logger.append_generation(pop, global_max_fitness, finished_count, leader_gates)

                pop.evolve()
                frame_counter = 0

            dashboard.draw(
                screen,
                pop,
                frame_counter,
                max_frames,
                global_best_cp,
                global_best_dist,
                global_max_fitness,
                victory_history,
                n_wp,
            )

            pygame.display.flip()
            clock.tick(d.fps)

        pygame.quit()


def run() -> None:
    """Convenience wrapper: instantiate :class:`EvolutionGame` with defaults and call :meth:`~EvolutionGame.run`."""
    EvolutionGame().run()
