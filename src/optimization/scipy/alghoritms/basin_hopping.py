import time

import numpy as np
import scipy as sp

from src.models.base_model import BaseModel
from src.optimization.scipy.sp_base_optimizer import ScipyBaseOptimizer
from src.schemas.boundaries import Boundaries
from src.schemas.optimization import OptResult
from src.schemas.protocols import TupleProtocol
from src.utils.converter import Converter


class BoundsChecker:
    """Validates if the stochastic jump landed in the constrains"""

    def __init__(self, xmin: float = 0.0, xmax: float = 1.0):
        self.xmin = xmin
        self.xmax = xmax

    def __call__(self, f_new: float, x_new: np.ndarray, f_old: float, x_old: np.ndarray) -> bool:
        return bool(np.all(x_new >= self.xmin) and np.all(x_new <= self.xmax))


class BasinHopping(ScipyBaseOptimizer):
    """
    BasinHopping alghortitm used for the sake of optimization,  with SLSQP opimizer as default.
    """
    def __init__(
        self,
        start: TupleProtocol,
        model: BaseModel,
        boundaries: Boundaries,
        niter: int = 15,
        T: float = 1.0,
        stepsize: float = 0.2,
        local_method: str = "SLSQP",
        use_bounds_checker: bool = True,
        tol: float = 1e-3,  # Tolerancja "z grubsza" dla ograniczeń
    ):
        super().__init__(start=start, model=model, boundaries=boundaries)
        self.niter = niter
        self.T = T
        self.stepsize = stepsize
        self.local_method = local_method
        self.use_bounds_checker = use_bounds_checker
        self.tol = tol

        self._best_x = None
        self._best_f = float("inf")

    @property
    def name(self) -> str:
        return f"BasinHopping_{self.local_method}"

    def get_constraints(self) -> list[dict]:
        """SLSQP constrains format"""
        return [
            {"type": "ineq", "fun": self.total_time_fun},
            {"type": "ineq", "fun": self.total_dose_fun},
        ]

    def _is_feasible(self, x: np.ndarray) -> bool:
        """Sprawdza, czy punkt x spełnia granice oraz ograniczenia z tolerancją `self.tol`."""

        if not (np.all(x >= -self.tol) and np.all(x <= 1.0 + self.tol)):
            return False

        for constraint in self.get_constraints():
            val = constraint["fun"](x)
            if constraint["type"] == "ineq" and val < -self.tol or constraint["type"] == "eq" and abs(val) > self.tol:
                return False

        return True

    def _tracked_fun(self, x: np.ndarray) -> float:
        val = self.fun(x)
        if val < self._best_f and self._is_feasible(x):
            self._best_f = val
            self._best_x = x.copy()
        return val

    def minimize(self) -> OptResult:
        x0 = self.get_x0()
        bounds = self.get_bounds()
        constraints = self.get_constraints()

        # Inicjalizacja punktem startowym (jeśli jest dopuszczalny)
        if self._is_feasible(x0):
            self._best_f = self.fun(x0)
            self._best_x = x0.copy()
        else:
            self._best_f = float("inf")
            self._best_x = x0.copy()

        minimizer_kwargs = {
            "method": self.local_method,
            "bounds": bounds,
            "constraints": constraints,
            "options": {"eps": 1e-3, "maxiter": 500},
        }

        accept_test = BoundsChecker(xmin=0.0, xmax=1.0) if self.use_bounds_checker else None

        start_time = time.perf_counter()
        result = sp.optimize.basinhopping(
            self._tracked_fun,
            x0=x0,
            niter=self.niter,
            T=self.T,
            stepsize=self.stepsize,
            minimizer_kwargs=minimizer_kwargs,
            accept_test=accept_test,
            disp=True,
        )
        end_time = time.perf_counter()

        return OptResult(
            min_protocol=Converter.flat_to_tuples(self.unnormalize(self._best_x)),
            min_val=float(self._best_f),
            search_time=end_time - start_time,
            n_iter=result.nit,
            n_calls=result.nfev,
            opt_name=self.name,
            start=self.start,
        )