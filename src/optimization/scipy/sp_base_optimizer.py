import time
from abc import ABC, abstractmethod
import numpy as np
from numpy import typing as npt

from src.models.base_model import BaseModel
from src.optimization.base_optimizer import BaseOptimizer
from src.schemas.boundaries import Boundaries
from src.schemas.protocols import FlatProtocol, MatrixProtocol, TupleProtocol
from src.utils.converter import Converter


class ScipyBaseOptimizer(BaseOptimizer, ABC):
    def __init__(
        self,
        start: TupleProtocol,
        model: BaseModel,
        boundaries: Boundaries,
    ):
        self.model = model
        self.boundaries = boundaries
        self.start = start

    def unnormalize(self, opt_protocol: npt.NDArray[np.float64]) -> FlatProtocol:
        """Przekształca znormalizowany wektor [0, 1] na fizyczny FlatProtocol."""
        n = len(opt_protocol) // 2
        time_gap_norm, dose_norm = opt_protocol[:n], opt_protocol[n:]

        time_gap = self.boundaries.min_interval + time_gap_norm * (
            self.boundaries.max_interval - self.boundaries.min_interval
        )
        dose = self.boundaries.min_single_dose + dose_norm * (
            self.boundaries.max_single_dose - self.boundaries.min_single_dose
        )

        return np.concatenate([time_gap, dose])

    def normalize(self, protocol: FlatProtocol) -> FlatProtocol:
        """Przekształca fizyczny FlatProtocol na znormalizowany wektor [0, 1]."""
        n = len(protocol) // 2
        time_gap, dose = protocol[:n], protocol[n:]

        time_gap_norm = (time_gap - self.boundaries.min_interval) / (
            self.boundaries.max_interval - self.boundaries.min_interval
        )
        dose_norm = (dose - self.boundaries.min_single_dose) / (
            self.boundaries.max_single_dose - self.boundaries.min_single_dose
        )

        return np.concatenate([time_gap_norm, dose_norm])

    def opt_to_model(self, opt_protocol: npt.NDArray[np.float64]) -> MatrixProtocol:
        physical_flat = self.unnormalize(opt_protocol)
        return Converter.flat_to_matrix(physical_flat)

    def fun(self, opt_protocol: npt.NDArray[np.float64]) -> float:
        model_protocol = self.opt_to_model(opt_protocol)
        return float(self.model.predict(model_protocol))

    def total_dose_fun(self, opt_protocol: npt.NDArray[np.float64]) -> float:
        physical_flat = self.unnormalize(opt_protocol)
        n = len(physical_flat) // 2
        doses = physical_flat[n:]
        return float(self.boundaries.max_total_dose - np.sum(doses))

    def total_time_fun(self, opt_protocol: npt.NDArray[np.float64]) -> float:
        physical_flat = self.unnormalize(opt_protocol)
        n = len(physical_flat) // 2
        intervals = physical_flat[:n]
        return float(self.boundaries.max_interval - np.sum(intervals))

    def get_bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0) for _ in range(self.boundaries.max_n_doses * 2)]

    def get_constraints(self) -> list[dict]:
        return [
            {"type": "ineq", "fun": self.total_time_fun},
            {"type": "ineq", "fun": self.total_dose_fun},
        ]

    def get_x0(self) -> npt.NDArray[np.float64]:
        return self.normalize(
            Converter.tuples_to_flat(self.start, self.boundaries.max_n_doses)
        )