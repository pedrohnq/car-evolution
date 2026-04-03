"""
Global RNG seeding for ``random`` and ``numpy`` (reproducible runs).
"""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed_value: int) -> None:
    """
    Seed Python's ``random`` and NumPy's global RNG.

    Args:
        seed_value: Integer seed shared by both libraries.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
