import pandas as pd
import pytest

from src.experiment import (
    BACKTEST_START,
    DEV_M,
    FINAL_PERIOD,
    build_target_table,
    build_test_snapshots,
    build_training_snapshots,
)


SCENARIOS = [
    "creciente",
    "decreciente",
    "mixto",
]


@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
)
def test_real_targets_match_loss_incurred_at_dev_m(
    simulations_base,
    scenario,
):
    """
    El target debe provenir exactamente de loss_incurred
    observado en dev_M.
    """
    df = simulations_base[scenario].copy()

    targets = build_target_table(
        df=df,
        scenario=scenario,
    )

    dev_m = DEV_M[scenario]

    mature = (
        df.loc[
            df["dev_month"].eq(dev_m),
            [
                "accident_period",
                "loss_incurred",
            ],
        ]
        .copy()
    )

    mature["accident_period"] = pd.PeriodIndex(
        mature["accident_period"],
        freq="M",
    )

    check = targets.merge(
        mature,
        on="accident_period",
        how="inner",
        validate="one_to_one",
    )

    assert len(check) == len(targets)

    assert (
        check["target_amount"]
        == check["loss_incurred"]
    ).all()


@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
)
def test_first_backtest_valuation_contains_all_ages(
    simulations_base,
    scenario,
):
    """
    En la primera valuación deben aparecer todas las edades
    0,...,dev_M-1 entre las observaciones evaluables.

    Esta prueba protege directamente la corrección realizada
    al diseño original.
    """
    df = simulations_base[scenario]

    test = build_test_snapshots(
        df=df,
        scenario=scenario,
        valuation_period=BACKTEST_START,
        final_period=FINAL_PERIOD,
    )

    dev_m = DEV_M[scenario]

    observed_ages = set(
        test["snapshot_dev_month"].unique()
    )

    assert observed_ages == set(
        range(dev_m)
    )


def test_creciente_has_12_mature_cohorts_at_ml_start(
    simulations_base,
):
    """
    Para creciente, el primer momento con 12 cohortes maduras
    debe ser 2020-12.
    """
    train = build_training_snapshots(
        df=simulations_base["creciente"],
        scenario="creciente",
        valuation_period="2020-12",
    )

    assert (
        train["accident_period"].nunique()
        == 12
    )

    counts = (
        train
        .groupby("accident_period")
        .size()
    )

    assert (counts == 60).all()

    assert len(train) == 12 * 60


@pytest.mark.parametrize(
    "scenario",
    [
        "decreciente",
        "mixto",
    ],
)
def test_48m_scenarios_already_have_enough_history_at_start(
    simulations_base,
    scenario,
):
    """
    Con dev_M=48, en 2020-01 ya existen 13 cohortes maduras:
    2015-01,...,2016-01.
    """
    train = build_training_snapshots(
        df=simulations_base[scenario],
        scenario=scenario,
        valuation_period="2020-01",
    )

    assert (
        train["accident_period"].nunique()
        == 13
    )

    counts = (
        train
        .groupby("accident_period")
        .size()
    )

    assert (counts == 48).all()

    assert len(train) == 13 * 48


@pytest.mark.parametrize(
    "scenario,valuation_period",
    [
        ("creciente", "2021-06"),
        ("decreciente", "2021-06"),
        ("mixto", "2021-06"),
    ],
)
def test_real_train_has_no_future_targets(
    simulations_base,
    scenario,
    valuation_period,
):
    train = build_training_snapshots(
        df=simulations_base[scenario],
        scenario=scenario,
        valuation_period=valuation_period,
    )

    valuation_period = pd.Period(
        valuation_period,
        freq="M",
    )

    assert (
        train["target_period"]
        <= valuation_period
    ).all()

    assert (
        train["snapshot_period"]
        < train["target_period"]
    ).all()

    assert (
        train["max_feature_dev_month"]
        <= train["snapshot_dev_month"]
    ).all()


@pytest.mark.parametrize(
    "scenario,valuation_period",
    [
        ("creciente", "2020-01"),
        ("creciente", "2022-06"),
        ("decreciente", "2020-01"),
        ("decreciente", "2022-06"),
        ("mixto", "2020-01"),
        ("mixto", "2022-06"),
    ],
)
def test_real_test_snapshots_are_temporally_valid(
    simulations_base,
    scenario,
    valuation_period,
):
    test = build_test_snapshots(
        df=simulations_base[scenario],
        scenario=scenario,
        valuation_period=valuation_period,
        final_period=FINAL_PERIOD,
    )

    valuation_period = pd.Period(
        valuation_period,
        freq="M",
    )

    assert not test.empty

    assert (
        test["snapshot_period"]
        == valuation_period
    ).all()

    assert (
        test["target_period"]
        > valuation_period
    ).all()

    assert (
        test["target_period"]
        <= FINAL_PERIOD
    ).all()

    assert (
        test["snapshot_dev_month"]
        < test["dev_M"]
    ).all()

    assert not test[
        "accident_period"
    ].duplicated().any()