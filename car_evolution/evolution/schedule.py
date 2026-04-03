"""
Generation-indexed tweaks to mutation, crossover, and selection (presentation / curriculum).
"""

from __future__ import annotations

from car_evolution.core.population import Population


class EvolutionParameterSchedule:
    """
    Mutates :class:`~car_evolution.core.population.Population` hyperparameters at fixed generation milestones.

    Call :meth:`apply` once per generation **before** :meth:`~car_evolution.core.population.Population.evolve`
    increments the counter (i.e. while ``pop.generation`` still labels the run that just ended).
    """

    def apply(self, pop: Population) -> None:
        """
        Mutate ``pop.mutation_rate``, ``crossover_rate``, ``selection_method``, and/or
        ``crossover_method`` when ``pop.generation`` hits configured milestones.

        Safe to call every generation end; no-op on generations without rules.

        Args:
            pop: Active population (modified in place).

        Note:
            Call **before** :meth:`~car_evolution.core.population.Population.evolve` so the logged row
            matches the generation that just finished.
        """
        g = pop.generation

        if g == 15:
            pop.mutation_rate = 0.12
            pop.selection_method = "Roulette"

        elif g == 30:
            pop.mutation_rate = 0.05
            pop.crossover_rate = 0.90
            pop.selection_method = "Tournament"

        elif g == 45:
            pop.mutation_rate = 0.02

        if g == 50:
            pop.mutation_rate = 0.08
            pop.selection_method = "Tournament"

        elif g == 120:
            pop.mutation_rate = 0.03
            pop.crossover_rate = 0.85
            pop.crossover_method = "One-Point"

        elif g == 200:
            pop.mutation_rate = 0.15
            pop.selection_method = "Roulette"

        elif g == 250:
            pop.mutation_rate = 0.05
            pop.selection_method = "Tournament"
            pop.crossover_method = "Uniform"

        elif g == 350:
            pop.mutation_rate = 0.01
            pop.crossover_rate = 0.95

        elif g == 450:
            pop.crossover_rate = 1.0
