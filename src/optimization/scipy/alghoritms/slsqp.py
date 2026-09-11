import scipy as sp
import time


from src.models.base_model import BaseModel
from src.utils.converter import Converter
from src.schemas.boundaries import Boundaries
from src.schemas.protocols import TupleProtocol
from src.schemas.optimization import OptResult
from src.optimization.scipy.sp_base_optimizer import ScipyBaseOptimizer


class SLSQP(ScipyBaseOptimizer):
    def __init__(
        self,
        start: TupleProtocol,
        model: BaseModel,
        boundaries: Boundaries,
        maxiter: int = 1000,
        eps: float = 1e-5,
    ):
        super().__init__(start=start, model=model, boundaries=boundaries)
        self.name = "SLSQP"
        self.maxiter = maxiter
        self.eps = eps

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