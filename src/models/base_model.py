from abc import ABC, abstractmethod

import numpy as np
from numpy import typing as npt


class BaseModel(ABC):
    @abstractmethod
    def predict(self,
                raw_protocol : npt.NDArray[np.float64]) -> float:
        """Predicts number of surviving cells after given treatment protocol
        Args:
        ---------------
            raw_protocol (np.array): treatment protocol in format [[time, dose, time_gap], ...] of length 20"""
