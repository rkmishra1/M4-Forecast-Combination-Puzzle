"""Unified forecaster interface.

Every model in the pool implements `forecast`: given a batch of univariate
series (mapping id -> 1-D array) and a horizon/seasonality, return a mapping
id -> point forecast of shape (h,). Probabilistic models contribute their
median as the point forecast.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    name: str = "base"
    kind: str = "classical"  # "classical" | "foundation"

    @abstractmethod
    def forecast(
        self, series: dict[str, np.ndarray], h: int, m: int
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError
