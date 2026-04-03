"""
Append-only CSV logging for each generation (metrics and hyperparameters).
"""

from __future__ import annotations

import csv
from pathlib import Path

from car_evolution.core.population import Population


class EvolutionCSVLogger:
    """
    Writes a header row on construction, then one row per call for each finished generation.
    """

    def __init__(self, log_path: Path) -> None:
        """
        Args:
            log_path: Destination CSV file (parent directory should exist - use
                :func:`car_evolution.io.paths.ensure_logs_dir`).
        """
        self._path = log_path
        with self._path.open(mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Generation",
                    "Mutation_Rate",
                    "Crossover_Rate",
                    "Selection_Method",
                    "Max_Fitness_Session",
                    "Finished_Cars",
                    "Leader_Gates",
                ]
            )

    @property
    def path(self) -> Path:
        """Filesystem path of the log file."""
        return self._path

    def append_generation(
        self,
        pop: Population,
        global_max_fitness: float,
        finished_count: int,
        leader_gates: int,
    ) -> None:
        """
        Append one record for the generation that just completed.

        Args:
            pop: Population (uses current hyperparameters and ``generation`` index).
            global_max_fitness: Best fitness seen so far in the session.
            finished_count: Cars that completed the lap this generation.
            leader_gates: Gate progress of the single best car by fitness.
        """
        with self._path.open(mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    pop.generation,
                    f"{pop.mutation_rate:.3f}",
                    f"{pop.crossover_rate:.2f}",
                    pop.selection_method,
                    f"{global_max_fitness:.2f}",
                    finished_count,
                    leader_gates,
                ]
            )
