from abc import ABC, abstractmethod

from src.schemas.protocols import MatrixProtocol
from jaxtyping import jaxtyped
from beartype import beartype


class BaseModel(ABC):

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def predict(self,
                protocol : MatrixProtocol) -> float:
        """Predicts number of surviving cells after given treatment protocol
        Args:
        ---------------
            raw_protocol (np.array): treatment protocol in format [[time, dose, time_gap], ...] of length 20"""
