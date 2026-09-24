from typing import cast

import pandas as pd

from src.schemas.boundaries import Boundaries


def filter_data(
    df: pd.DataFrame,
    boundaries: Boundaries,
) -> pd.DataFrame:
    """Filteres out protocols that do not satisfy boundaries"""

    filtered_df = df.copy()

    valid_single_dose = (
        (filtered_df["dose"] >= boundaries.min_single_dose)
        & (filtered_df["dose"] <= boundaries.max_single_dose)
    ).groupby(filtered_df["series"]).transform("all")

    total_dose_series = cast(
        pd.Series,
        filtered_df.groupby("series")["dose"].transform("sum")
    )
    valid_total_dose = total_dose_series <= boundaries.max_total_dose

    has_positive_dose = filtered_df["dose"] > 0
    valid_gap_range = (filtered_df["time_gap"] >= boundaries.min_interval) & (
        filtered_df["time_gap"] <= boundaries.max_interval
    )

    valid_gap_for_positive = (~has_positive_dose) | (
        has_positive_dose & valid_gap_range
    )
    valid_intervals = valid_gap_for_positive.groupby(filtered_df["series"]).transform(
        "all"
    )

    protocol_mask = valid_single_dose & valid_total_dose & valid_intervals

    return cast(pd.DataFrame, filtered_df[protocol_mask])