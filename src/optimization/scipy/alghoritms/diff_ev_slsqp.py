import time

import scipy as sp

from src.models.base_model import BaseModel
from src.optimization.scipy.sp_base_optimizer import ScipyBaseOptimizer
from src.schemas.boundaries import Boundaries
from src.schemas.optimization import OptResult
from src.schemas.protocols import TupleProtocol
from src.utils.converter import Converter


class DiffEvSLSQP(ScipyBaseOptimizer):
    """"
    Combination of differential evolution and SLSQP as a final polish.
    """
    def __init__(
        self,
        start: TupleProtocol,
        model: BaseModel,
        boundaries: Boundaries,
        strategy : str = "best1bin",
        popsize : int = 20,
        mutation : tuple[float, float] = (0.5, 1.0),
        recombination : float = 0.8,
        maxiter : int = 100,
        seed : int = 42,
        disp : bool = False,
        workers : int = 1
    ):
        super().__init__(start=start,
                         model=model,
                         boundaries=boundaries)
        
        self.name = "Differential Evolution + SLSQP"
        self.strategy = strategy
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.maxiter = maxiter
        self.seed = seed
        self.disp = disp
        self.workers = workers

    def minimize(self) -> OptResult:
        x0 = self.get_x0()
        bounds = self.get_bounds()

        start_time = time.perf_counter()
        de_result = sp.optimize.differential_evolution(
            self.fun,
            x0=x0,
            bounds=bounds,
            constraints = self.get_de_constraints(),
            strategy = self.strategy,
            popsize = self.popsize,
            mutation = self.mutation,
            recombination = self.recombination,
            maxiter = self.maxiter,
            seed = self.seed,
            disp = self.disp,
            workers = self.workers,
        ) 

        result = sp.optimize.minimize(
            self.fun,
            de_result.x,
            method="SLSQP",
            bounds=bounds,
            constraints = self.get_constraints(),
            options={"maxiter": 100, "eps": 1e-4, "disp": True},
        )
        print("slsqp min at:", result.x)

        end_time = time.perf_counter()

        return OptResult(
            min_protocol=Converter.flat_to_tuples(self.unnormalize(result.x)),
            min_val=float(result.fun),
            search_time=end_time - start_time,
            n_iter=result.nit + de_result.nit,
            n_calls=result.nfev + de_result.nfev,
            opt_name=self.name,
            start=self.start,
        )