from abc import ABC

import numpy as np
from numpy import typing as npt
from scipy.optimize import NonlinearConstraint

from src.models.base_model import BaseModel
from src.optimization.base_optimizer import BaseOptimizer
from src.schemas.boundaries import Boundaries
from src.schemas.protocols import FlatProtocol, MatrixProtocol, TupleProtocol
from src.utils.converter import Converter


class ScipyBaseOptimizer(BaseOptimizer, ABC):
    """"
    Handles normalization and constrains of optimization for scipy alghoritms.
    """
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
        """Transforms normalized protocol to physical values"""
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
        """Normalizes physical protocol to [0, 1] vals"""
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
        """"
        A definition of target function.
        """
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
        return float(self.boundaries.max_total_time - np.sum(intervals))

    def get_bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0) for _ in range(self.boundaries.max_n_doses * 2)]

    def get_de_constraints(self) -> list[NonlinearConstraint]:
        """Differential evolution constrains"""
        return [
            NonlinearConstraint(self.total_time_fun, 0.0, np.inf),
            NonlinearConstraint(self.total_dose_fun, 0.0, np.inf),
        ]

    def get_constraints(self) -> list[dict]:
        """SLSQP constrains format"""
        return [
            {"type": "ineq", "fun": self.total_time_fun},
            {"type": "ineq", "fun": self.total_dose_fun},
        ]

    def get_x0(self) -> npt.NDArray[np.float64]:
        return self.normalize(
            Converter.tuples_to_flat(self.start, self.boundaries.max_n_doses)
        )