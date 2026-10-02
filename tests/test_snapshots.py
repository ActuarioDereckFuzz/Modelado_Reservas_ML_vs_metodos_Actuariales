import pandas as pd

from src.experiment import (
    build_target_table,
    build_test_snapshots,
    build_training_snapshots,
)


def make_toy_creciente():
    """
    Base mínima con la misma estructura relevante
    que las simulaciones reales.

    Se utiliza un ultimate_loss deliberadamente absurdo
    para comprobar que snapshots.py NO lo utiliza.
    """
    accident_periods = [
        pd.Period("2015-01", freq="M"),
        pd.Period("2016-01", freq="M"),
        pd.Period("2020-12", freq="M"),
    ]

    rows = []

    for cohort_idx, accident_period in enumerate(
        accident_periods
    ):
        base = 10_000 + cohort_idx * 1_000

        for dev_month in range(61):

            rows.append(
                {
                    "scenario": "creciente",
                    "accident_period": accident_period,
                    "dev_month": dev_month,
                    "calendar_period": (
                        accident_period + dev_month
                    ),
                    "dev_M": 60,
                    "loss_incurred": (
                        base + 100 * dev_month
                    ),

                    # No debe utilizarse como target.
                    "ultimate_loss": 999_999_999,
                }
            )

    return pd.DataFrame(rows)


def test_target_comes_from_mature_loss_incurred():

    df = make_toy_creciente()

    targets = build_target_table(
        df=df,
        scenario="creciente",
    )

    first = targets.loc[
        targets["accident_period"]
        == pd.Period("2015-01", freq="M")
    ].iloc[0]

    expected = (
        df.loc[
            (df["accident_period"]
             == pd.Period("2015-01", freq="M"))
            & (df["dev_month"] == 60),
            "loss_incurred",
        ]
        .iloc[0]
    )

    assert first["target_amount"] == expected

    # Prueba explícita de que NO usamos ultimate_loss.
    assert first["target_amount"] != 999_999_999


def test_training_uses_only_mature_cohorts():

    df = make_toy_creciente()

    train = build_training_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    # 2015-01 madura en 2020-01
    # 2016-01 madura en 2021-01
    # 2020-12 todavía no madura.
    expected_accident_periods = {
        pd.Period("2015-01", freq="M"),
        pd.Period("2016-01", freq="M"),
    }

    assert set(
        train["accident_period"]
    ) == expected_accident_periods

    assert (
        train["target_period"]
        <= pd.Period("2021-06", freq="M")
    ).all()


def test_each_mature_cohort_generates_60_training_snapshots():

    df = make_toy_creciente()

    train = build_training_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    counts = (
        train
        .groupby("accident_period")
        .size()
    )

    assert (counts == 60).all()

    assert (
        train["snapshot_dev_month"].min()
        == 0
    )

    assert (
        train["snapshot_dev_month"].max()
        == 59
    )


def test_training_never_contains_dev_M():

    df = make_toy_creciente()

    train = build_training_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    assert (
        train["snapshot_dev_month"]
        < train["dev_M"]
    ).all()


def test_test_snapshot_uses_current_valuation_age():

    df = make_toy_creciente()

    test = build_test_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
        final_period="2025-12",
    )

    # 2020-12 tiene 6 meses de desarrollo en 2021-06.
    row = test.loc[
        test["accident_period"]
        == pd.Period("2020-12", freq="M")
    ].iloc[0]

    assert row["snapshot_dev_month"] == 6

    assert (
        row["snapshot_period"]
        == pd.Period("2021-06", freq="M")
    )


def test_test_contains_only_immature_but_evaluable_cohorts():

    df = make_toy_creciente()

    test = build_test_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
        final_period="2025-12",
    )

    assert (
        test["target_period"]
        > pd.Period("2021-06", freq="M")
    ).all()

    assert (
        test["target_period"]
        <= pd.Period("2025-12", freq="M")
    ).all()

    assert (
        test["snapshot_dev_month"]
        < test["dev_M"]
    ).all()


def test_test_has_one_row_per_accident_period():

    df = make_toy_creciente()

    test = build_test_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
        final_period="2025-12",
    )

    assert not test[
        "accident_period"
    ].duplicated().any()


def test_max_feature_dev_month_never_exceeds_snapshot():

    df = make_toy_creciente()

    train = build_training_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    test = build_test_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
        final_period="2025-12",
    )

    assert (
        train["max_feature_dev_month"]
        <= train["snapshot_dev_month"]
    ).all()

    assert (
        test["max_feature_dev_month"]
        <= test["snapshot_dev_month"]
    ).all()