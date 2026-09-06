from dataclasses import dataclass

@dataclass
class Boundaries:
    min_interval: float
    max_interval: float
    min_single_dose: float
    max_single_dose: float
    max_total_dose: float
    max_n_doses: int