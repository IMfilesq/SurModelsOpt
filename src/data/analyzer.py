
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.schemas.analysis import DatasetAnalysis


def _build_histogram_figure(
    series: pd.Series, title: str = "Histogram", num_bins: int = 10
) -> Figure | None:
    """Generate a standalone Matplotlib Figure object for a data series histogram.

    Args:
        series: Pandas Series containing numerical data to plot.
        title: Title of the generated histogram plot.
        num_bins: Number of bins to use in the histogram.

    Returns:
        A Matplotlib Figure object, or None if the series is empty after dropping NaNs.
    """
    clean_series = series.dropna()
    if clean_series.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(clean_series, bins=num_bins, edgecolor="black", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig


def analyze_data(
    df: pd.DataFrame,
    num_bins: int = 10,
) -> DatasetAnalysis:
    """Calculate descriptive statistics and generate histogram figures for the dataset.

    Performs comprehensive analysis on doses, time intervals, series counts,
    and cancer cell targets.

    Args:
        df: Input pandas DataFrame containing experimental runs. Expected columns:
            'dose', 'series', 'time_gap', 'is_target', 'target'.
        num_bins: Number of bins for the generated histogram figures.

    Returns:
        A DatasetAnalysis schema populated with statistical metrics and Matplotlib figures.

    Raises:
        ValueError: If the provided DataFrame is empty.
    """
    if df.empty:
        raise ValueError("DataFrame is empty. Cannot perform analysis.")

    # 1. Single positive doses
    positive_doses = df[df["dose"] > 0]["dose"]

    min_single = float(positive_doses.min()) if not positive_doses.empty else 0.0
    max_single = float(positive_doses.max()) if not positive_doses.empty else 0.0
    mean_single = (
        float(positive_doses.mean()) if not positive_doses.empty else 0.0
    )
    median_single = (
        float(positive_doses.median()) if not positive_doses.empty else 0.0
    )
    q1_single = (
        float(positive_doses.quantile(0.25))
        if not positive_doses.empty
        else 0.0
    )
    q3_single = (
        float(positive_doses.quantile(0.75))
        if not positive_doses.empty
        else 0.0
    )
    std_single = float(positive_doses.std()) if not positive_doses.empty else 0.0

    dose_fig = _build_histogram_figure(
        positive_doses, title="Dose Distribution", num_bins=num_bins
    )

    # 2. Total doses per series
    total_doses_per_series = df.groupby("series")["dose"].sum()
    min_total = float(total_doses_per_series.min())
    max_total = float(total_doses_per_series.max())
    mean_total = float(total_doses_per_series.mean())
    median_total = float(total_doses_per_series.median())

    total_dose_fig = _build_histogram_figure(
        total_doses_per_series, 
        title="Total Dose Per Series Distribution", 
        num_bins=num_bins
    )


    # 3. Time intervals
    has_positive = df["dose"] > 0
    dose_number = has_positive.groupby(df["series"]).cumsum()
    subsequent_gaps = df[has_positive & dose_number.gt(1)]["time_gap"]

    min_gap = float(subsequent_gaps.min()) if not subsequent_gaps.empty else 0.0
    max_gap = float(subsequent_gaps.max()) if not subsequent_gaps.empty else 0.0
    mean_gap = (
        float(subsequent_gaps.mean()) if not subsequent_gaps.empty else 0.0
    )
    median_gap = (
        float(subsequent_gaps.median()) if not subsequent_gaps.empty else 0.0
    )

    # 4. General statistics
    doses_count_per_series = df[df["dose"] > 0].groupby("series")["dose"].count()

    total_series = int(df["series"].nunique())
    total_rows = len(df)
    mean_doses_series = (
        float(doses_count_per_series.mean())
        if not doses_count_per_series.empty
        else 0.0
    )
    max_doses_series = (
        int(doses_count_per_series.max())
        if not doses_count_per_series.empty
        else 0
    )
    min_doses_series = (
        int(doses_count_per_series.min())
        if not doses_count_per_series.empty
        else 0
    )

    # 5. Cancer cells statistics
    cancer_data = df.loc[df["is_target"] == 1, "target"]

    min_cancer = float(cancer_data.min()) if not cancer_data.empty else 0.0
    max_cancer = float(cancer_data.max()) if not cancer_data.empty else 0.0
    mean_cancer = float(cancer_data.mean()) if not cancer_data.empty else 0.0
    median_cancer = (
        float(cancer_data.median()) if not cancer_data.empty else 0.0
    )

    cancer_fig = _build_histogram_figure(
        cancer_data, title="Remaining Cancer Cells Distribution", num_bins=num_bins
    )

    return DatasetAnalysis(
        min_single_dose=min_single,
        max_single_dose=max_single,
        mean_single_dose=mean_single,
        median_single_dose=median_single,
        q1_single_dose=q1_single,
        q3_single_dose=q3_single,
        std_single_dose=std_single,
        min_total_dose=min_total,
        max_total_dose=max_total,
        mean_total_dose=mean_total,
        median_total_dose=median_total,
        min_interval=min_gap,
        max_interval=max_gap,
        mean_interval=mean_gap,
        median_interval=median_gap,
        total_series_count=total_series,
        total_rows_count=total_rows,
        mean_doses_per_series=mean_doses_series,
        max_doses_per_series=max_doses_series,
        min_doses_per_series=min_doses_series,
        min_cancer_cells=min_cancer,
        max_cancer_cells=max_cancer,
        mean_cancer_cells=mean_cancer,
        median_cancer_cells=median_cancer,
        cancer_cells_histogram=cancer_fig,
        dose_histogram=dose_fig,
        total_dose_histogram=total_dose_fig
    )