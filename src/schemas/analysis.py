from dataclasses import dataclass
from typing import Optional
import matplotlib.figure


@dataclass
class DatasetAnalysis:
    """Data structure storing descriptive statistics for the filtered dataset."""

    # Single Doses
    min_single_dose: float
    max_single_dose: float
    mean_single_dose: float
    median_single_dose: float
    q1_single_dose: float
    q3_single_dose: float
    std_single_dose: float

    # Total Doses per Series
    min_total_dose: float
    max_total_dose: float
    mean_total_dose: float
    median_total_dose: float

    # Time Gaps
    min_interval: float
    max_interval: float
    mean_interval: float
    median_interval: float

    # General Statistics
    total_series_count: int
    total_rows_count: int
    mean_doses_per_series: float
    max_doses_per_series: int
    min_doses_per_series: int

    # Cancer Cells / Tumor Count Statistics
    min_cancer_cells: float
    max_cancer_cells: float
    mean_cancer_cells: float
    median_cancer_cells: float

    # Histogram Figure Objects
    cancer_cells_histogram: Optional[matplotlib.figure.Figure] = None
    dose_histogram: Optional[matplotlib.figure.Figure] = None
    total_dose_histogram: Optional[matplotlib.figure.Figure] = None

