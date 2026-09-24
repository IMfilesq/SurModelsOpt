from dataclasses import dataclass


@dataclass
class Boundaries:
    """Stores the constrains used in optimization"""
    min_interval: float
    max_interval: float
    min_single_dose: float
    max_single_dose: float
    max_total_dose: float
    max_total_time: float
    max_n_doses: int
