import time

import scipy as sp

from src.models.base_model import BaseModel
from src.optimization.scipy.sp_base_optimizer import ScipyBaseOptimizer
from src.schemas.boundaries import Boundaries
from src.schemas.optimization import OptResult
from src.schemas.protocols import TupleProtocol
from src.utils.converter import Converter


class SLSQP(ScipyBaseOptimizer):
    """
    SLSQP algorithm implementation for the sake of optimization
    """

    def __init__(
        self,
        start: TupleProtocol,
        model: BaseModel,
        boundaries: Boundaries,
        maxiter: int = 100,
        eps: float = 1e-5,
    ):
        super().__init__(start=start, model=model, boundaries=boundaries)
        self.maxiter = maxiter
        self.eps = eps

    @property
    def name(self) -> str:
        return "SLSQP"

    def get_constraints(self) -> list[dict]:
        """SLSQP constrains format"""
        return [
            {"type": "ineq", "fun": self.total_time_fun},
            {"type": "ineq", "fun": self.total_dose_fun},
        ]

    def minimize(self) -> OptResult:
        x0 = self.get_x0()
        bounds = self.get_bounds()
        constraints = self.get_constraints()

        start_time = time.perf_counter()
        result = sp.optimize.minimize(
            self.fun,
            x0=x0,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": self.maxiter, "eps": self.eps},
        )
        end_time = time.perf_counter()

        return OptResult(
            min_protocol=Converter.flat_to_tuples(self.unnormalize(result.x)),
            min_val=float(result.fun),
            search_time=end_time - start_time,
            n_iter=result.nit,
            n_calls=result.nfev,
            opt_name=self.name,
            start=self.start,
        )

    