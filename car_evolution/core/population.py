"""
Genetic algorithm over a population of cars (neural-network DNA).
"""

from __future__ import annotations

import random

import numpy as np

from car_evolution.core.car import Car


class Population:
    """
    Fixed-size population of :class:`~car_evolution.core.car.Car` agents.

    Each generation: sort by ``best_fitness``, elitism copy, tournament/roulette parents,
    crossover and mutation on flat DNA, then replace the population.
    """

    def __init__(
        self,
        size: int,
        start_pos: tuple[float, float],
        start_angle: float,
        *,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.80,
        elitism: int = 2,
        selection_method: str = "Tournament",
        crossover_method: str = "Uniform",
    ) -> None:
        """
        Args:
            size: Number of cars.
            start_pos: ``(x, y)`` spawn position.
            start_angle: Spawn heading in radians.
            mutation_rate: Fixed for this population instance unless replaced by a new run.
            crossover_rate: Fixed for this population instance unless replaced by a new run.
            elitism: Number of elite clones per generation.
            selection_method: ``\"Tournament\"`` or ``\"Roulette\"``.
            crossover_method: ``\"Uniform\"`` or ``\"One-Point\"``.
        """
        self.size = size
        self.start_pos = start_pos
        self.start_angle = start_angle
        self.generation = 1

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.selection_method = selection_method
        self.crossover_method = crossover_method

        self.cars: list[Car] = [
            Car(start_pos[0], start_pos[1], start_angle) for _ in range(size)
        ]

    def all_inactive(self) -> bool:
        """
        Returns:
            ``True`` if no car is both alive and still racing (dead or ``finished``).
        """
        return all((not c.alive) or c.finished for c in self.cars)

    def evolve(self) -> None:
        """
        Sort by ``best_fitness``, keep ``elitism`` clones, fill the rest with offspring.

        Offspring DNA comes from ``select_parent`` pairs, ``crossover`` / clone, and ``mutate``.
        Replaces ``self.cars`` and increments ``self.generation``.
        """
        self.cars.sort(key=lambda x: x.best_fitness, reverse=True)
        new_cars: list[Car] = []

        for i in range(self.elitism):
            elite_car = Car(self.start_pos[0], self.start_pos[1], self.start_angle)
            elite_car.brain.set_dna(self.cars[i].brain.get_dna())
            new_cars.append(elite_car)

        while len(new_cars) < self.size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()

            if random.random() < self.crossover_rate:
                child_dna = self.crossover(parent1.brain.get_dna(), parent2.brain.get_dna())
            else:
                child_dna = parent1.brain.get_dna()

            child_dna = self.mutate(child_dna)
            child = Car(self.start_pos[0], self.start_pos[1], self.start_angle)
            child.brain.set_dna(child_dna)
            new_cars.append(child)

        self.cars = new_cars
        self.generation += 1

    def select_parent(self) -> Car:
        """
        Choose one car for reproduction.

        Tournament: sample three cars, return the best by ``best_fitness``.
        Roulette: probability proportional to ``max(0, best_fitness)`` (with epsilon).

        Returns:
            A reference to a car already in ``self.cars`` (not a copy).
        """
        if self.selection_method == "Tournament":
            tournament = random.sample(self.cars, 3)
            return max(tournament, key=lambda x: x.best_fitness)

        total_fitness = sum(max(0.0, c.best_fitness) + 1e-9 for c in self.cars)
        if total_fitness <= 0:
            return random.choice(self.cars)
        pick = random.uniform(0, total_fitness)
        current = 0.0
        for car in self.cars:
            current += max(0.0, car.best_fitness) + 1e-9
            if current > pick:
                return car
        return self.cars[-1]

    def crossover(self, dna1: np.ndarray, dna2: np.ndarray) -> np.ndarray:
        """
        Combine two flat weight vectors into one child vector.

        ``Uniform``: per-gene random choice between parents.
        ``One-Point``: prefix from ``dna1`` and suffix from ``dna2`` at a random cut.
        """
        if self.crossover_method == "Uniform":
            mask = np.random.rand(len(dna1)) > 0.5
            return np.where(mask, dna1, dna2)
        pt = random.randint(0, len(dna1) - 1)
        return np.concatenate((dna1[:pt], dna2[pt:]))

    def mutate(self, dna: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to a random subset of genes.

        Each index is mutated independently with probability ``mutation_rate``; noise scale is fixed.

        Returns:
            A new float64 array (input is not modified in place).
        """
        dna = np.asarray(dna, dtype=np.float64).copy()
        rate = self.mutation_rate
        scale = 0.55

        mask = np.random.rand(len(dna)) < rate
        mutations = np.random.randn(len(dna)) * scale
        dna[mask] += mutations[mask]
        return dna
