import pandas as pd


def filter_data(
    df: pd.DataFrame,
    min_single_dose: float = 0.0,
    max_single_dose: float = 2.5,
    max_total_dose: float = 10.0,
    min_interval: int = 2400,
    max_interval: int = 7200,
) -> pd.DataFrame:
    """Filtruje całe serie na podstawie zasad klinicznych."""
    filtered_df = df.copy()

    valid_single_dose = (
        (filtered_df["dose"] >= min_single_dose)
        & (filtered_df["dose"] <= max_single_dose)
    ).groupby(filtered_df["series"]).transform("all")

    valid_total_dose = (
        filtered_df.groupby("series")["dose"].transform("sum") <= max_total_dose
    )

    has_positive_dose = filtered_df["dose"] > 0
    valid_gap_range = (filtered_df["time_gap"] >= min_interval) & (
        filtered_df["time_gap"] <= max_interval
    )

    valid_gap_for_positive = (~has_positive_dose) | (
        has_positive_dose & valid_gap_range
    )
    valid_intervals = valid_gap_for_positive.groupby(filtered_df["series"]).transform(
        "all"
    )

    protocol_mask = valid_single_dose & valid_total_dose & valid_intervals
    return filtered_df[protocol_mask]