

from abc import ABC, abstractmethod
from numpy import typing as npt
import numpy as np
from src.models.base_model import BaseModel
from src.schemas.boundaries import Boundaries

class BaseOptimizer(ABC):
    @abstractmethod
    def minimize(self) -> float:
        """Minimizes the number of surviving cells after given treatment protocol
        Args:
        ---------------
            model (BaseModel): model to be optimized
            initial_protocol (np.ndarray): initial treatment protocol in format [[time, dose, time_gap], ...] of length 20
        Returns:
        ---------------
            float: minimum number of surviving cells after given treatment protocol"""