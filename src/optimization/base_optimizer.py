from abc import ABC, abstractmethod
import numpy as np
from src.schemas.optimization import OptResult


class BaseOptimizer(ABC):
    @abstractmethod
    def minimize(self) -> OptResult:
        """Minimizes the number of surviving cells after given treatment protocol
        Args:
        ---------------
            model (BaseModel): model to be optimized
            initial_protocol (np.ndarray): initial treatment protocol in format [[time, dose, time_gap], ...] of length 20
        Returns:
        ---------------
            float: minimum number of surviving cells after given treatment protocol"""