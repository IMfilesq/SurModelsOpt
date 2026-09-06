import numpy as np
import scipy as sp
from numpy import typing as npt

from src.models.base_model import BaseModel
from optimization.base_optimizer import BaseOptimizer
from src.schemas.boundaries import Boundaries


class SLSQP(BaseOptimizer):
    def __init__(
        self,
        model: BaseModel,
        boundaries: Boundaries,
        model_input_len: int = 20,  # Stały wymiar oczekiwany przez sieć PyTorch
    ):
        self.model = model
        self.boundaries = boundaries
        self.model_input_len = model_input_len

    def fun(self, opt_protocol: npt.NDArray[np.float64]) -> float:
        model_protocol = self.opt_to_model(opt_protocol)
        return float(self.model.predict(model_protocol))

    def total_dose_fun(self, opt_protocol: npt.NDArray[np.float64]) -> float:
        """Ograniczenie nierównościowe: max_total_dose - total_dose >= 0"""
        n = len(opt_protocol) // 2
        scaled_dose = opt_protocol[n:]
        total_dose = np.sum(
            self.boundaries.min_single_dose
            + scaled_dose * (self.boundaries.max_single_dose - self.boundaries.min_single_dose)
        )
        return float(self.boundaries.max_total_dose - total_dose)

    def minimize(self) -> sp.optimize.OptimizeResult:
        n_doses = self.boundaries.max_n_doses
        bounds = [(0.0, 1.0) for _ in range(n_doses * 2)]

        constraints = [{"type": "ineq", "fun": self.total_dose_fun}]

        x0 = np.full(n_doses * 2, 0.5)

        result = sp.optimize.minimize(
            self.fun,
            x0=x0,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 1000, "eps": 1e-4},
        )

        return result

    def opt_to_model(self, opt_protocol: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        n = len(opt_protocol) // 2

        time_gap_norm = opt_protocol[:n]
        time_gap = self.boundaries.min_interval + time_gap_norm * (
            self.boundaries.max_interval - self.boundaries.min_interval
        )

        dose_norm = opt_protocol[n:]
        dose = self.boundaries.min_single_dose + dose_norm * (
            self.boundaries.max_single_dose - self.boundaries.min_single_dose
        )

        time = time_gap.cumsum()
        model_protocol = np.stack([time, dose, time_gap], axis=1)

        # ZAWSZE dopełniamy zerami z przodu do stałej długości wejściowej sieci (np. 20)
        if model_protocol.shape[0] < self.model_input_len:
            pad_size = self.model_input_len - model_protocol.shape[0]
            model_protocol = np.pad(
                model_protocol,
                pad_width=((pad_size, 0), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )

        return model_protocol