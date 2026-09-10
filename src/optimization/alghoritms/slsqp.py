import numpy as np
import scipy as sp
import time
from numpy import typing as npt

from src.models.base_model import BaseModel
from src.optimization.base_optimizer import BaseOptimizer
from src.utils.converter import Converter
from src.schemas.boundaries import Boundaries
from src.schemas.protocols import FlatProtocol, MatrixProtocol, TupleProtocol
from src.schemas.optimization import OptResult


class SLSQP(BaseOptimizer):
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
        """Przekształca znormalizowany wektor [0, 1] na fizyczny FlatProtocol (interwały + dawki)."""
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
        """Przekształca fizyczny FlatProtocol (interwały + dawki) na znormalizowany wektor [0, 1]."""
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
        """Odskalowuje wektor optymalizatora i konwertuje go na macierz wymaganą przez model ML."""
        physical_flat = self.unnormalize(opt_protocol)
        return Converter.flat_to_matrix(physical_flat)

    def fun(self, opt_protocol: npt.NDArray[np.float64]) -> float:
        model_protocol = self.opt_to_model(opt_protocol)
        return float(self.model.predict(model_protocol))

    def total_dose_fun(self, opt_protocol: npt.NDArray[np.float64]) -> float:
        """Ograniczenie nierównościowe: max_total_dose - total_dose >= 0"""
        physical_flat = self.unnormalize(opt_protocol)
        n = len(physical_flat) // 2
        doses = physical_flat[n:]
        total_dose = np.sum(doses)
        return float(self.boundaries.max_total_dose - total_dose)

    def total_time_fun(self, opt_protocol: npt.NDArray[np.float64]) -> float:
        physical_flat = self.unnormalize(opt_protocol)
        n = len(physical_flat) // 2
        intervals = physical_flat[:n]
        return float(self.boundaries.max_interval - np.sum(intervals))

    def minimize(self) -> OptResult:
        n_doses = self.boundaries.max_n_doses
        bounds = [(0.0, 1.0) for _ in range(n_doses * 2)]

        constraints = [{"type": "ineq", "fun": self.total_time_fun},
                       {"type": "ineq", "fun": self.total_dose_fun}]

        x0 = self.normalize(Converter.tuples_to_flat(self.start, n_doses))
        start_time = time.perf_counter()
        result = sp.optimize.minimize(
            self.fun,
            x0=x0,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 1000, "eps": 1e-4},
        )
        end_time = time.perf_counter()

        result = OptResult(min_protocol= Converter.flat_to_tuples(result.x),
                           min_val = result.fun,
                           search_time = end_time - start_time,
                           n_iter = result.nit,
                           n_calls = result.nfev)

        return result