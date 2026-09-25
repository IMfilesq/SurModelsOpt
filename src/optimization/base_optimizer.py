from abc import ABC, abstractmethod

from src.schemas.optimization import OptResult


class BaseOptimizer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the optimizer."""
        pass

    @abstractmethod
    def minimize(self) -> OptResult:
        """
        Should implement the minimization logic and return OptResult dataclass.
        """