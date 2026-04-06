"""
Pygame application: main loop, event input, track backdrop, car simulation, and GA steps.

Creates one CSV log per **parameter run** under ``logs/`` (see :mod:`car_evolution.io.paths`).
Each run keeps fixed GA settings until convergence (fitness plateau or generation cap).
"""

from __future__ import annotations

import random
from datetime import datetime

import pygame

from car_evolution.config import Colors, DISPLAY, SIMULATION
from car_evolution.core.population import Population
from car_evolution.core.rng import set_global_seed
from car_evolution.evolution.logger import EvolutionCSVLogger
from car_evolution.evolution.run_params import EvolutionRunParams, default_run_presets
from car_evolution.io.paths import evolution_run_log_path
from car_evolution.rendering.track_background import build_track_background_surface
from car_evolution.rendering.ui import EvolutionDashboard
from car_evolution.track import geometry as tg
from car_evolution.track.layout import RaceTrack


class EvolutionGame:
    """
    Runs the full-screen simulation: track rendering, car updates, GA steps, and CSV logs.

    Cycles through fixed hyperparameter presets (:func:`~car_evolution.evolution.run_params.default_run_presets`);
    each preset runs until session fitness plateaus or ``max_generations_per_run`` is reached.
    """

    def __init__(
        self,
        track: RaceTrack | None = None,
        run_presets: list[EvolutionRunParams] | None = None,
    ) -> None:
        """
        Args:
            track: Layout to simulate; ``None`` uses :meth:`~car_evolution.track.layout.RaceTrack.default_hardcore`.
            run_presets: Ordered GA configurations; ``None`` uses :func:`default_run_presets`.
        """
        self._track = track or RaceTrack.default_hardcore()
        self._display = DISPLAY
        self._sim = SIMULATION
        self._run_presets = tuple(run_presets or default_run_presets())

    def _make_population(
        self,
        params: EvolutionRunParams,
        start_pos: tuple[float, float],
        start_angle: float,
    ) -> Population:
        return Population(
            size=params.population_size,
            start_pos=start_pos,
            start_angle=start_angle,
            mutation_rate=params.mutation_rate,
            crossover_rate=params.crossover_rate,
            elitism=params.elitism,
            selection_method=params.selection_method,
            crossover_method=params.crossover_method,
        )

    def run(self) -> None:
        """
        Create the window, load fonts and static track art, then run until quit.

        Handles pygame events, car updates, generation boundaries, one CSV per parameter run, and
        ``pygame.quit`` on exit.
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

        session_ts = datetime.now()
        run_index = 0
        params = self._run_presets[run_index]
        current_seed = sim.default_seed
        set_global_seed(current_seed)
        pop = self._make_population(params, start_pos, start_angle)

        max_frames = sim.max_frames_per_generation
        frame_counter = 0
        global_best_cp = 0
        global_best_dist = float("inf")
        global_progress_pos: tuple[float, float] = start_pos
        global_max_fitness = 0.0
        victory_history: list[tuple[int, int]] = []

        session_peak = -1.0
        gens_without_improvement = 0
        all_runs_complete = False

        logger = EvolutionCSVLogger(evolution_run_log_path(session_ts, run_index))
        n_wp = len(waypoints)

        def reset_run_tracking() -> None:
            nonlocal frame_counter, global_best_cp, global_best_dist, global_progress_pos
            nonlocal global_max_fitness, victory_history, session_peak, gens_without_improvement
            frame_counter = 0
            global_best_cp = 0
            global_best_dist = float("inf")
            global_progress_pos = start_pos
            global_max_fitness = 0.0
            victory_history.clear()
            session_peak = -1.0
            gens_without_improvement = 0

        running = True
        while running:
            screen.blit(track_background, (0, 0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif not all_runs_complete and event.key == pygame.K_r:
                        set_global_seed(current_seed)
                        pop = self._make_population(params, start_pos, start_angle)
                        reset_run_tracking()
                    elif not all_runs_complete and event.key == pygame.K_n:
                        current_seed = random.randint(1, 999_999)
                        set_global_seed(current_seed)
                        pop = self._make_population(params, start_pos, start_angle)
                        reset_run_tracking()

            if not all_runs_complete:
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
            else:
                for car in pop.cars:
                    car.draw(screen)

            mx, my = int(global_progress_pos[0]), int(global_progress_pos[1])
            pygame.draw.line(screen, Colors.PURPLE, (mx - 28, my), (mx + 28, my), 3)
            pygame.draw.line(screen, Colors.PURPLE, (mx, my - 28), (mx, my + 28), 3)
            pygame.draw.circle(screen, Colors.PURPLE, (mx, my), 22, 3)
            pygame.draw.circle(screen, Colors.YELLOW, (mx, my), 6, 2)
            pygame.draw.circle(screen, Colors.BLACK, (mx, my), 2)

            if not all_runs_complete:
                frame_counter += 1

            if (
                not all_runs_complete
                and (pop.all_inactive() or frame_counter >= max_frames)
            ):
                finished_count = sum(1 for c in pop.cars if c.finished)
                if finished_count > 0:
                    victory_history.append((pop.generation, finished_count))

                leader = max(pop.cars, key=lambda c: c.best_fitness) if pop.cars else None
                leader_gates = leader.progress_gates(n_wp) if leader else 0

                logger.append_generation(pop, global_max_fitness, finished_count, leader_gates)

                if global_max_fitness > session_peak:
                    session_peak = global_max_fitness
                    gens_without_improvement = 0
                else:
                    gens_without_improvement += 1

                plateau = gens_without_improvement >= sim.convergence_plateau_generations
                gen_cap = pop.generation >= sim.max_generations_per_run
                converged = plateau or gen_cap

                if converged:
                    if run_index + 1 >= len(self._run_presets):
                        all_runs_complete = True
                    else:
                        run_index += 1
                        params = self._run_presets[run_index]
                        set_global_seed(current_seed)
                        pop = self._make_population(params, start_pos, start_angle)
                        logger = EvolutionCSVLogger(evolution_run_log_path(session_ts, run_index))
                        reset_run_tracking()
                else:
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
                run_index=run_index,
                total_runs=len(self._run_presets),
                run_label=params.label,
                all_runs_complete=all_runs_complete,
                gens_without_improvement=gens_without_improvement,
                plateau_generations=sim.convergence_plateau_generations,
                max_generations_per_run=sim.max_generations_per_run,
            )

            pygame.display.flip()
            clock.tick(d.fps)

        pygame.quit()


def run() -> None:
    """Convenience wrapper: instantiate :class:`EvolutionGame` with defaults and call :meth:`~EvolutionGame.run`."""
    EvolutionGame().run()
