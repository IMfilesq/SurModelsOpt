import time

import scipy as sp
import numpy as np

from src.models.base_model import BaseModel
from src.optimization.scipy.sp_base_optimizer import ScipyBaseOptimizer
from src.schemas.boundaries import Boundaries
from src.schemas.optimization import OptResult
from src.schemas.protocols import TupleProtocol
from src.utils.converter import Converter
from scipy.optimize import NonlinearConstraint


class DiffEv(ScipyBaseOptimizer):
    """
    Differential evoluton alghoritm for the sake of optimization.
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
        
        self.strategy = strategy
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.maxiter = maxiter
        self.seed = seed
        self.disp = disp
        self.workers = workers

    @property
    def name(self) -> str:
        return "Differential Evolution"

    def get_de_constraints(self) -> list[NonlinearConstraint]:
        """Differential evolution constrains"""
        return [
            NonlinearConstraint(self.total_time_fun, 0.0, np.inf),
            NonlinearConstraint(self.total_dose_fun, 0.0, np.inf),
        ]

    def minimize(self) -> OptResult:
        x0 = self.get_x0()
        bounds = self.get_bounds()

        start_time = time.perf_counter()
        result = sp.optimize.differential_evolution(
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