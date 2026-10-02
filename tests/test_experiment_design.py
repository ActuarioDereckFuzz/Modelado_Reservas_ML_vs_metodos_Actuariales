import pandas as pd

from src.experiment import (
    DEV_M,
    FINAL_PERIOD,
    VALUATION_PERIODS,
    build_eligibility_grid,
    first_possible_ml_valuation,
    validate_eligibility_frame,
    validate_experiment_config,
)


def test_experiment_config():
    validate_experiment_config()


def test_first_possible_ml_valuation():

    assert (
        first_possible_ml_valuation("creciente")
        == pd.Period("2020-12", freq="M")
    )

    assert (
        first_possible_ml_valuation("decreciente")
        == pd.Period("2019-12", freq="M")
    )

    assert (
        first_possible_ml_valuation("mixto")
        == pd.Period("2019-12", freq="M")
    )


def test_latest_evaluable_accident_periods():

    assert (
        FINAL_PERIOD - DEV_M["creciente"]
        == pd.Period("2020-12", freq="M")
    )

    assert (
        FINAL_PERIOD - DEV_M["decreciente"]
        == pd.Period("2021-12", freq="M")
    )

    assert (
        FINAL_PERIOD - DEV_M["mixto"]
        == pd.Period("2021-12", freq="M")
    )


def test_all_development_ages_are_evaluable(
    accident_periods,
):

    for scenario, dev_m in DEV_M.items():

        grid = build_eligibility_grid(
            accident_periods=accident_periods,
            scenario=scenario,
            valuation_periods=VALUATION_PERIODS,
        )

        validate_eligibility_frame(grid)

        evaluable = grid.loc[
            grid["is_backtest_evaluable"]
        ]

        observed_ages = set(
            evaluable["latest_dev_month"].unique()
        )

        assert observed_ages == set(range(dev_m))


def test_backtest_targets_are_future_at_valuation(
    accident_periods,
):

    for scenario in DEV_M:

        grid = build_eligibility_grid(
            accident_periods=accident_periods,
            scenario=scenario,
        )

        test = grid.loc[
            grid["is_backtest_evaluable"]
        ]

        assert (
            test["target_period"]
            > test["valuation_period"]
        ).all()

        assert (
            test["target_period"]
            <= FINAL_PERIOD
        ).all()