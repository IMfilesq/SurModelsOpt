from abc import ABC, abstractmethod
import scipy as sp


class BaseOptimizer(ABC):
    @abstractmethod
    def minimize(self) -> sp.optimize.OptimizeResult:
        """Minimizes the number of surviving cells after given treatment protocol
        Args:
        ---------------
            model (BaseModel): model to be optimized
            initial_protocol (np.ndarray): initial treatment protocol in format [[time, dose, time_gap], ...] of length 20
        Returns:
        ---------------
            float: minimum number of surviving cells after given treatment protocol"""