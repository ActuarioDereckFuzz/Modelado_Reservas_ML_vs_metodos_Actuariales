import pandas as pd

from src.experiment import (
    build_prediction_snapshots,
    build_target_table,
    build_training_snapshots,
    reveal_targets,
)


# ============================================================
# BASE TOY
# ============================================================

def make_toy_creciente():
    """
    Base mínima con la misma estructura relevante
    que las simulaciones reales.

    Incluye:

    - dos cohortes maduras en 2021-06;
    - una cohorte inmadura con edad 6;
    - una cohorte inmadura con edad 0 cuyo target ocurre
      después de 2025-12.

    Se utiliza un `ultimate_loss` deliberadamente absurdo
    para comprobar que snapshots.py NO lo utiliza.
    """
    accident_periods = [
        pd.Period("2015-01", freq="M"),
        pd.Period("2016-01", freq="M"),
        pd.Period("2020-12", freq="M"),
        pd.Period("2021-06", freq="M"),
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


# ============================================================
# TARGET
# ============================================================

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
            (
                df["accident_period"]
                == pd.Period("2015-01", freq="M")
            )
            & (
                df["dev_month"] == 60
            ),
            "loss_incurred",
        ]
        .iloc[0]
    )

    assert first["target_amount"] == expected

    # Prueba explícita de que NO usamos ultimate_loss.
    assert first["target_amount"] != 999_999_999


# ============================================================
# TRAINING
# ============================================================

def test_training_uses_only_mature_cohorts():

    df = make_toy_creciente()

    train = build_training_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    # 2015-01 madura en 2020-01.
    # 2016-01 madura en 2021-01.
    # 2020-12 y 2021-06 todavía son inmaduras.
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


def test_training_contains_known_target():

    df = make_toy_creciente()

    train = build_training_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    assert "target_amount" in train.columns

    assert train[
        "target_amount"
    ].notna().all()

    assert train[
        "target_amount"
    ].gt(0).all()


# ============================================================
# PREDICTION SNAPSHOTS
# ============================================================

def test_prediction_snapshot_uses_current_valuation_age():

    df = make_toy_creciente()

    prediction = build_prediction_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    # 2020-12 tiene 6 meses de desarrollo en 2021-06.
    row = prediction.loc[
        prediction["accident_period"]
        == pd.Period("2020-12", freq="M")
    ].iloc[0]

    assert row["snapshot_dev_month"] == 6

    assert (
        row["snapshot_period"]
        == pd.Period("2021-06", freq="M")
    )


def test_prediction_includes_age_zero():

    df = make_toy_creciente()

    prediction = build_prediction_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    # La cohorte 2021-06 acaba de ocurrir.
    row = prediction.loc[
        prediction["accident_period"]
        == pd.Period("2021-06", freq="M")
    ].iloc[0]

    assert row["snapshot_dev_month"] == 0

    assert (
        row["snapshot_period"]
        == pd.Period("2021-06", freq="M")
    )


def test_prediction_contains_only_immature_cohorts():

    df = make_toy_creciente()

    prediction = build_prediction_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    assert (
        prediction["target_period"]
        > pd.Period("2021-06", freq="M")
    ).all()

    assert (
        prediction["snapshot_dev_month"] >= 0
    ).all()

    assert (
        prediction["snapshot_dev_month"]
        < prediction["dev_M"]
    ).all()


def test_prediction_can_include_target_after_final_period():

    df = make_toy_creciente()

    prediction = build_prediction_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
        final_period="2025-12",
    )

    # 2021-06 + 60 meses = 2026-06.
    row = prediction.loc[
        prediction["accident_period"]
        == pd.Period("2021-06", freq="M")
    ].iloc[0]

    assert (
        row["target_period"]
        == pd.Period("2026-06", freq="M")
    )

    assert (
        row["target_period"]
        > pd.Period("2025-12", freq="M")
    )


def test_prediction_has_one_row_per_accident_period():

    df = make_toy_creciente()

    prediction = build_prediction_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    assert not prediction[
        "accident_period"
    ].duplicated().any()


# ============================================================
# ANTI-LEAKAGE
# ============================================================

def test_prediction_does_not_contain_target_amount():

    df = make_toy_creciente()

    prediction = build_prediction_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    assert "target_amount" not in prediction.columns


def test_max_feature_dev_month_never_exceeds_snapshot():

    df = make_toy_creciente()

    train = build_training_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    prediction = build_prediction_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    assert (
        train["max_feature_dev_month"]
        <= train["snapshot_dev_month"]
    ).all()

    assert (
        prediction["max_feature_dev_month"]
        <= prediction["snapshot_dev_month"]
    ).all()


# ============================================================
# REVELACIÓN DEL TARGET
# ============================================================

def test_reveal_targets_adds_target_only_after_prediction():

    df = make_toy_creciente()

    prediction = build_prediction_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    assert "target_amount" not in prediction.columns

    revealed = reveal_targets(
        predictions=prediction,
        df=df,
        scenario="creciente",
    )

    assert "target_amount" in revealed.columns

    assert revealed[
        "target_amount"
    ].notna().all()

    assert revealed[
        "target_amount"
    ].gt(0).all()


def test_revealed_target_matches_loss_incurred_at_dev_M():

    df = make_toy_creciente()

    prediction = build_prediction_snapshots(
        df=df,
        scenario="creciente",
        valuation_period="2021-06",
    )

    revealed = reveal_targets(
        predictions=prediction,
        df=df,
        scenario="creciente",
    )

    expected = (
        df.loc[
            df["dev_month"].eq(60),
            [
                "accident_period",
                "loss_incurred",
            ],
        ]
        .rename(
            columns={
                "loss_incurred": "expected_target"
            }
        )
    )

    check = revealed.merge(
        expected,
        on="accident_period",
        how="left",
        validate="many_to_one",
    )

    assert (
        check["target_amount"]
        == check["expected_target"]
    ).all()