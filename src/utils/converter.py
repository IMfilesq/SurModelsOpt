import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from src.schemas.boundaries import Boundaries
from src.schemas.protocols import FlatProtocol, MatrixProtocol, TupleProtocol


class Converter:
    """
    Contains methods used for conversion between protocol and constraint formats
    """

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def flat_to_matrix(flat: FlatProtocol) -> MatrixProtocol:
        n_doses = len(flat) // 2
        intervals, doses = flat[:n_doses], flat[n_doses:]
        times = np.cumsum(intervals)

        matrix = np.zeros((20, 3), dtype=np.float64)
        limit = min(n_doses, 20)

        if limit > 0:
            start_idx = 20 - limit  # Indeks, od którego zaczynają się dane

            matrix[start_idx:, 0] = times[:limit]      # Col 0: Cumulative Time
            matrix[start_idx:, 1] = doses[:limit]      # Col 1: Dose
            matrix[start_idx:, 2] = intervals[:limit]  # Col 2: Time Gap

        return matrix

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def flat_to_tuples(flat: FlatProtocol) -> TupleProtocol:
        n_protocols = flat.size//2
        intervals, doses = flat[:n_protocols], flat[n_protocols:]
        times = np.cumsum(intervals)
        valid_mask = doses > 0
        return list(zip(times[valid_mask].tolist(), doses[valid_mask].tolist()))

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def tuples_to_flat(tuples: TupleProtocol, max_doses: int) -> FlatProtocol:
        intervals = np.zeros(max_doses, dtype=np.float64)
        doses = np.zeros(max_doses, dtype=np.float64)

        sorted_tuples = sorted(tuples, key=lambda x: x[0])
        prev_t = 0.0

        for i, (t, d) in enumerate(sorted_tuples[:max_doses]):
            intervals[i] = t - prev_t
            doses[i] = d
            prev_t = t

        return np.concatenate([intervals, doses])

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def matrix_to_tuples(matrix: MatrixProtocol) -> TupleProtocol:
        doses = matrix[:, 1]
        times = matrix[:, 0]
        valid_mask = doses > 0
        return list(zip(times[valid_mask].tolist(), doses[valid_mask].tolist()))

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def tuples_to_matrix(tuples: TupleProtocol, max_doses : int) -> MatrixProtocol:
        flat = Converter.tuples_to_flat(tuples, max_doses)
        return Converter.flat_to_matrix(flat)

    
    @staticmethod
    @jaxtyped(typechecker=beartype)
    def make_stricter(bounds : Boundaries, safety_eps : float) -> Boundaries:
        tol = safety_eps
        stricter = Boundaries(bounds.min_interval + tol,
                              bounds.max_interval - tol,
                              bounds.min_single_dose + tol,
                              bounds.max_single_dose - tol,
                              bounds.max_total_dose - tol,
                              bounds.max_total_time - tol,
                              bounds.max_n_doses)
        return stricter

    
        