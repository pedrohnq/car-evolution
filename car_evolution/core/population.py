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

    def __init__(self, size: int, start_pos: tuple[float, float], start_angle: float) -> None:
        """
        Args:
            size: Number of cars.
            start_pos: ``(x, y)`` spawn position.
            start_angle: Spawn heading in radians.
        """
        self.size = size
        self.start_pos = start_pos
        self.start_angle = start_angle
        self.generation = 1

        self.mutation_rate = 0.05
        self.crossover_rate = 0.80
        self.elitism = 2
        self.selection_method = "Tournament"
        self.crossover_method = "Uniform"

        self.cars: list[Car] = [
            Car(start_pos[0], start_pos[1], start_angle) for _ in range(size)
        ]

    def all_inactive(self) -> bool:
        """True when every car is either dead or finished the lap."""
        return all((not c.alive) or c.finished for c in self.cars)

    def evolve(self) -> None:
        """Produce the next generation in place and increment ``generation``."""
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
        """Pick one parent using tournament (k=3) or fitness-proportionate roulette."""
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
        """Uniform bitwise mask or one-point splice, depending on ``crossover_method``."""
        if self.crossover_method == "Uniform":
            mask = np.random.rand(len(dna1)) > 0.5
            return np.where(mask, dna1, dna2)
        pt = random.randint(0, len(dna1) - 1)
        return np.concatenate((dna1[:pt], dna2[pt:]))

    def mutate(self, dna: np.ndarray) -> np.ndarray:
        """Gaussian perturbation on a fraction of genes controlled by ``mutation_rate``."""
        dna = np.asarray(dna, dtype=np.float64).copy()
        rate = self.mutation_rate
        scale = 0.55

        mask = np.random.rand(len(dna)) < rate
        mutations = np.random.randn(len(dna)) * scale
        dna[mask] += mutations[mask]
        return dna
