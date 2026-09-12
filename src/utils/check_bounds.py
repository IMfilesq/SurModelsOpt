from src.schemas.boundaries import Boundaries
from src.schemas.protocols import TupleProtocol
import numpy as np

def check_bounds(protocol : TupleProtocol) -> Boundaries:
    doses = [dose for time, dose in protocol]
    times = [time for time, dose in protocol]
    intervals = list(np.diff(times, prepend = 0))
    realized = Boundaries(min_interval = min(intervals),
                          max_interval= max(intervals),
                          min_single_dose = min(doses),
                          max_single_dose = max(doses),
                          max_total_dose = sum(doses),
                          max_n_doses = len(protocol),
                          max_total_time = times[-1],)
    return realized

