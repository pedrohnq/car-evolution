"""
Feed-forward neural network with tanh activations for car control (throttle, steering).

Weights and biases are flattened into a single "DNA" vector for the genetic algorithm.
"""

from __future__ import annotations

import numpy as np


class NeuralNetwork:
    """
    Fully connected network: ``layer_sizes`` defines neurons per layer.

    Forward pass applies ``tanh`` after each affine layer. The last layer outputs
    continuous actions (no extra activation beyond tanh).
    """

    def __init__(self, layer_sizes: list[int]) -> None:
        """
        Initialize random small weights and biases.

        Args:
            layer_sizes: e.g. ``[5, 5, 2]`` for 5 inputs, one hidden layer of 5, 2 outputs.
        """
        self.layer_sizes = layer_sizes
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1)
            self.biases.append(np.random.randn(1, layer_sizes[i + 1]) * 0.1)

    def predict(self, inputs: list[float] | np.ndarray) -> np.ndarray:
        """
        Run one forward pass.

        Args:
            inputs: Feature vector (length must match first layer size).

        Returns:
            1D array of outputs (e.g. throttle and steering).
        """
        a = np.array(inputs).reshape(1, -1)
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = np.tanh(z)
        return a[0]

    def get_dna(self) -> np.ndarray:
        """
        Serialize all weights and biases into one 1D float vector.

        Returns:
            Concatenated flat parameters in fixed order (weights per layer, then biases).
        """
        parts: list[np.ndarray] = []
        for w in self.weights:
            parts.append(np.asarray(w, dtype=np.float64).reshape(-1))
        for b in self.biases:
            parts.append(np.asarray(b, dtype=np.float64).reshape(-1))
        return np.concatenate(parts)

    def set_dna(self, dna: np.ndarray) -> None:
        """
        Restore weights and biases from a flat vector produced by :meth:`get_dna`.

        Args:
            dna: 1D array with length matching total parameter count.
        """
        idx = 0
        for i in range(len(self.weights)):
            shape = self.weights[i].shape
            size = shape[0] * shape[1]
            self.weights[i] = dna[idx : idx + size].reshape(shape)
            idx += size

            shape_b = self.biases[i].shape
            size_b = shape_b[0] * shape_b[1]
            self.biases[i] = dna[idx : idx + size_b].reshape(shape_b)
            idx += size_b
