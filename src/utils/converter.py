import numpy as np
from jaxtyping import jaxtyped
from beartype import beartype

from src.schemas.protocols import TupleProtocol, FlatProtocol, MatrixProtocol


class Converter:
    """
    Contains methods used for conversion between protocol formats
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

    # @staticmethod
    # @jaxtyped(typechecker=beartype)
    # def matrix_to_flat(matrix: MatrixProtocol) -> FlatProtocol:
    #     doses = matrix[:, 0]
    #     intervals = matrix[:, 2]
    #     return np.concatenate([intervals, doses])


    @staticmethod
    @jaxtyped(typechecker=beartype)
    def flat_to_tuples(flat: FlatProtocol) -> TupleProtocol:
        intervals, doses = flat[:20], flat[20:]
        times = np.cumsum(intervals)
        valid_mask = doses > 0
        return list(zip(times[valid_mask].tolist(), doses[valid_mask].tolist()))

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def tuples_to_flat(tuples: TupleProtocol, max_doses: int = 20) -> FlatProtocol:
        intervals = np.zeros(max_doses, dtype=np.float64)
        doses = np.zeros(max_doses, dtype=np.float64)

        sorted_tuples = sorted(tuples, key=lambda x: x[0])
        prev_t = 0.0

        for i, (t, d) in enumerate(sorted_tuples[:max_doses]):
            intervals[i] = t - prev_t
            doses[i] = d
            prev_t = t

        return np.concatenate([intervals, doses])

    # --- MATRIX <-> TUPLES ---

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def matrix_to_tuples(matrix: MatrixProtocol) -> TupleProtocol:
        doses = matrix[:, 1]
        times = matrix[:, 0]
        valid_mask = doses > 0
        return list(zip(times[valid_mask].tolist(), doses[valid_mask].tolist()))

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def tuples_to_matrix(tuples: TupleProtocol) -> MatrixProtocol:
        flat = Converter.tuples_to_flat(tuples)
        return Converter.flat_to_matrix(flat)