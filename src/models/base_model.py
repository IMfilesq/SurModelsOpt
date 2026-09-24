from abc import ABC, abstractmethod

from beartype import beartype
from jaxtyping import jaxtyped

from src.schemas.protocols import MatrixProtocol


class BaseModel(ABC):
    """
    Blueprint for implementation of models used in the optimization pipeline.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def predict(self,
                protocol : MatrixProtocol) -> float:
        """Predicts number of surviving cells after given treatment protocol
        Args:
        ---------------
            raw_protocol (np.array): treatment protocol in format [[time, dose, time_gap], ...] of fixed length 20"""
