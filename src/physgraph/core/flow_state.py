"""Numerical history required to continue a fixed-step implicit flow solve."""
from dataclasses import dataclass
import numpy as np


@dataclass
class FlowRestartState:
    contract: dict
    step: int
    previous: np.ndarray
    older: np.ndarray
    guess: np.ndarray
    initial_diagnostics: dict
