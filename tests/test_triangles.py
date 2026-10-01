import numpy as np
import pandas as pd

from src.triangles import (
    build_triangle_pair,
    incremental_to_cumulative,
)

from src.triangles import extract_latest_diagonal


VALUATION_PERIOD = "2025-12"


def test_latest_accident_has_only_development_zero(
    simulations,
):

    for df in simulations.values():

        cumulative, _ = build_triangle_pair(
            df=df,
            valuation_period=VALUATION_PERIOD,
        )

        latest_accident = cumulative.index.max()

        observed_development = (
            cumulative
            .loc[latest_accident]
            .dropna()
            .index
            .tolist()
        )

        assert observed_development == [0]


def test_2025_january_has_expected_development(
    simulations,
):

    accident_period = pd.Period(
        "2025-01",
        freq="M",
    )

    for df in simulations.values():

        cumulative, _ = build_triangle_pair(
            df=df,
            valuation_period=VALUATION_PERIOD,
        )

        observed_development = (
            cumulative
            .loc[accident_period]
            .dropna()
            .index
            .tolist()
        )

        assert observed_development == list(
            range(12)
        )


def test_cumulative_incremental_roundtrip(
    simulations,
):

    for df in simulations.values():

        cumulative, incremental = (
            build_triangle_pair(
                df=df,
                valuation_period=VALUATION_PERIOD,
            )
        )

        reconstructed = (
            incremental_to_cumulative(
                incremental
            )
        )

        mask = cumulative.notna().to_numpy()

        original_values = (
            cumulative.to_numpy()[mask]
        )

        reconstructed_values = (
            reconstructed.to_numpy()[mask]
        )

        assert np.allclose(
            original_values,
            reconstructed_values,
        )

def test_latest_diagonal_has_one_row_per_accident_period(
    simulations,
):

    for df in simulations.values():

        cumulative, _ = build_triangle_pair(
            df=df,
            valuation_period=VALUATION_PERIOD,
        )

        diagonal = extract_latest_diagonal(
            cumulative,
            valuation_period=VALUATION_PERIOD,
        )

        assert len(diagonal) == (
            cumulative.shape[0]
        )        


def test_latest_diagonal_recent_period_has_age_zero(
    simulations,
):

    for df in simulations.values():

        cumulative, _ = build_triangle_pair(
            df=df,
            valuation_period=VALUATION_PERIOD,
        )

        diagonal = extract_latest_diagonal(
            cumulative,
            valuation_period=VALUATION_PERIOD,
        )

        latest = diagonal.loc[
            diagonal["accident_period"]
            == diagonal["accident_period"].max()
        ]

        assert (
            latest["latest_dev_month"].iloc[0]
            == 0
        )

def test_latest_diagonal_mature_periods_use_max_dev(
    simulations,
):

    for df in simulations.values():

        cumulative, _ = build_triangle_pair(
            df=df,
            valuation_period=VALUATION_PERIOD,
        )

        diagonal = extract_latest_diagonal(
            cumulative,
            valuation_period=VALUATION_PERIOD,
        )

        max_dev = max(cumulative.columns)

        mature = diagonal.loc[
            diagonal["is_mature"]
        ]

        assert (
            mature["latest_dev_month"]
            == max_dev
        ).all()