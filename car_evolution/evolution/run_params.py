"""
Fixed hyperparameters for one full evolutionary run (no mid-run schedule).

Each :class:`EvolutionRunParams` instance describes a single execution from the first generation
until convergence; the game cycles through :func:`default_run_presets` in order.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvolutionRunParams:
    """
    GA settings held constant for one run (until plateau or generation cap).

    Attributes:
        population_size: Number of cars per generation for this run.
        mutation_rate: Per-gene mutation probability in :meth:`~car_evolution.core.population.Population.mutate`.
        crossover_rate: Probability of two-parent crossover vs cloning one parent.
        elitism: Number of top individuals copied unchanged each generation.
        selection_method: ``\"Tournament\"`` or ``\"Roulette\"``.
        crossover_method: ``\"Uniform\"`` or ``\"One-Point\"``.
        label: Short name for the dashboard (not logged to CSV).
    """

    population_size: int
    mutation_rate: float
    crossover_rate: float
    elitism: int
    selection_method: str
    crossover_method: str
    label: str


def default_run_presets() -> list[EvolutionRunParams]:
    """
    Ordered list of distinct parameter sets for successive full runs.

    Mirrors the variety that previously appeared across generation milestones in the old schedule,
    but each configuration now runs alone until convergence.
    """
    return [
        EvolutionRunParams(
        population_size=40,
        mutation_rate=0.05,
        crossover_rate=0.80,
        elitism=5,
        selection_method="Tournament",
        crossover_method="Uniform",
        label="Pop40 | Elitism 5",
    ),
    EvolutionRunParams(
        population_size=40,
        mutation_rate=0.05,
        crossover_rate=0.80,
        elitism=10,
        selection_method="Tournament",
        crossover_method="Uniform",
        label="Pop40 | Elitism 10",
    ),

    # =========================
    # POPULAÇÃO 80 (grande)
    # =========================
    EvolutionRunParams(
        population_size=80,
        mutation_rate=0.05,
        crossover_rate=0.80,
        elitism=0,
        selection_method="Tournament",
        crossover_method="Uniform",
        label="Pop80 | Elitism 0",
    ),
    EvolutionRunParams(
        population_size=80,
        mutation_rate=0.05,
        crossover_rate=0.80,
        elitism=2,
        selection_method="Tournament",
        crossover_method="Uniform",
        label="Pop80 | Elitism 2",
    ),
    EvolutionRunParams(
        population_size=80,
        mutation_rate=0.05,
        crossover_rate=0.80,
        elitism=5,
        selection_method="Tournament",
        crossover_method="Uniform",
        label="Pop80 | Elitism 5",
    ),
    EvolutionRunParams(
        population_size=80,
        mutation_rate=0.05,
        crossover_rate=0.80,
        elitism=10,
        selection_method="Tournament",
        crossover_method="Uniform",
        label="Pop80 | Elitism 10",
    ),
    ]
