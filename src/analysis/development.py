from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_individual_age_to_age_factors(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula factores edad-a-edad para cada periodo de ocurrencia.

    Para cada accident_period y edad j:

        factor_{i,j} = LI_{i,j+1} / LI_{i,j}

    Returns
    -------
    pd.DataFrame
        Base con:
        - scenario
        - accident_period
        - dev_month
        - loss_incurred_current
        - loss_incurred_next
        - age_to_age_factor
    """

    required = {
        "scenario",
        "accident_period",
        "dev_month",
        "loss_incurred",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing)}"
        )

    data = df[
        [
            "scenario",
            "accident_period",
            "dev_month",
            "loss_incurred",
        ]
    ].copy()

    data = data.sort_values(
        ["scenario", "accident_period", "dev_month"]
    )

    data["loss_incurred_next"] = (
        data
        .groupby(
            ["scenario", "accident_period"]
        )["loss_incurred"]
        .shift(-1)
    )

    data = data.dropna(
        subset=["loss_incurred_next"]
    ).copy()

    data = data.rename(
        columns={
            "loss_incurred":
                "loss_incurred_current"
        }
    )

    data["age_to_age_factor"] = (
        data["loss_incurred_next"]
        / data["loss_incurred_current"]
    )

    return data.reset_index(drop=True)


def calculate_volume_weighted_factors(
    individual_factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula factores edad-a-edad agregados mediante
    ponderación por volumen:

        f_j =
        sum_i LI_{i,j+1} / sum_i LI_{i,j}
    """

    required = {
        "scenario",
        "dev_month",
        "loss_incurred_current",
        "loss_incurred_next",
    }

    missing = required - set(individual_factors.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing)}"
        )

    grouped = (
        individual_factors
        .groupby(
            ["scenario", "dev_month"],
            as_index=False,
        )
        .agg(
            current_sum=(
                "loss_incurred_current",
                "sum",
            ),
            next_sum=(
                "loss_incurred_next",
                "sum",
            ),
            n_observations=(
                "age_to_age_factor",
                "size",
            ),
            median_individual_factor=(
                "age_to_age_factor",
                "median",
            ),
            mean_individual_factor=(
                "age_to_age_factor",
                "mean",
            ),
        )
    )

    grouped["volume_weighted_factor"] = (
        grouped["next_sum"]
        / grouped["current_sum"]
    )

    return grouped