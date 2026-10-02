# tests/test_development.py

import numpy as np
import pandas as pd
import pytest

from src.models.development import (
    development_alternative_name,
    calculate_individual_factors,
    apply_history_window,
    apply_exclusion_rule,
    aggregate_development_factors,
    fit_development_factors,
    build_development_cdf,
    predict_development_ultimate,
    predict_development_snapshot,
    run_development_alternative,
    run_development_alternatives,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def train_snapshot():
    """
    Snapshot de entrenamiento artificial.

    Se utilizan 4 cohortes completamente desarrolladas hasta
    dev_M = 3.

    Cada cohorte contiene:

        L_0, L_1, L_2, target = L_3

    Esto permite verificar manualmente los factores:

        f_0 = L_1 / L_0
        f_1 = L_2 / L_1
        f_2 = L_3 / L_2
    """

    data = []

    cohorts = {
        "2020-01": {
            "amounts": [100.0, 120.0, 150.0],
            "target": 180.0,
        },
        "2020-02": {
            "amounts": [200.0, 220.0, 264.0],
            "target": 316.8,
        },
        "2020-03": {
            "amounts": [300.0, 450.0, 495.0],
            "target": 544.5,
        },
        "2020-04": {
            "amounts": [400.0, 440.0, 484.0],
            "target": 580.8,
        },
    }

    for accident_period, values in cohorts.items():

        for dev_month, amount in enumerate(
            values["amounts"]
        ):
            data.append(
                {
                    "scenario": "test",
                    "accident_period": pd.Period(
                        accident_period,
                        freq="M",
                    ),
                    "snapshot_dev_month": dev_month,
                    "observed_amount": amount,
                    "target_amount": values["target"],
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def prediction_snapshot():
    """
    Tres cohortes inmaduras que serán utilizadas para probar
    la etapa de predicción.

    target_amount se omite deliberadamente.
    """

    return pd.DataFrame(
        {
            "scenario": ["test"] * 3,
            "valuation_period": [
                pd.Period("2021-03", freq="M")
            ] * 3,
            "accident_period": [
                pd.Period("2021-01", freq="M"),
                pd.Period("2021-02", freq="M"),
                pd.Period("2021-03", freq="M"),
            ],
            "snapshot_dev_month": [2, 1, 0],
            "observed_amount": [
                500.0,
                600.0,
                700.0,
            ],
        }
    )


# ============================================================
# 1. Nombre de alternativa
# ============================================================


@pytest.mark.parametrize(
    "alternative, expected",
    [
        (
            ("VW", 24, "none"),
            "VW_24_none",
        ),
        (
            ("S", "all", "std2"),
            "S_all_std2",
        ),
        (
            ("M", 24, "minmax"),
            "M_24_minmax",
        ),
    ],
)
def test_development_alternative_name(
    alternative,
    expected,
):
    assert (
        development_alternative_name(alternative)
        == expected
    )


# ============================================================
# 2. Factores individuales
# ============================================================


def test_calculate_individual_factors_shape(
    train_snapshot,
):
    factors = calculate_individual_factors(
        train_snapshot,
        dev_M=3,
    )

    # 4 cohortes × 3 transiciones
    assert len(factors) == 12

    assert set(
        factors["dev_month"].unique()
    ) == {0, 1, 2}


def test_calculate_individual_factors_values(
    train_snapshot,
):
    factors = calculate_individual_factors(
        train_snapshot,
        dev_M=3,
    )

    cohort = factors.loc[
        factors["accident_period"]
        == pd.Period("2020-01", freq="M")
    ].sort_values("dev_month")

    expected = np.array(
        [
            120 / 100,
            150 / 120,
            180 / 150,
        ]
    )

    np.testing.assert_allclose(
        cohort["individual_factor"]
        .to_numpy(),
        expected,
    )


def test_final_transition_uses_target(
    train_snapshot,
):
    factors = calculate_individual_factors(
        train_snapshot,
        dev_M=3,
    )

    row = factors.loc[
        (
            factors["accident_period"]
            == pd.Period(
                "2020-01",
                freq="M",
            )
        )
        & (factors["dev_month"] == 2)
    ].iloc[0]

    assert row["current_amount"] == 150.0
    assert row["next_amount"] == 180.0

    assert row["individual_factor"] == pytest.approx(
        1.2
    )


def test_incomplete_training_trajectory_rejected(
    train_snapshot,
):
    incomplete = train_snapshot.drop(
        train_snapshot.index[0]
    )

    with pytest.raises(
        ValueError,
        match="trayectoria incompleta",
    ):
        calculate_individual_factors(
            incomplete,
            dev_M=3,
        )


# ============================================================
# 3. Ventanas históricas
# ============================================================


def test_history_window_all(
    train_snapshot,
):
    factors = calculate_individual_factors(
        train_snapshot,
        dev_M=3,
    )

    result = apply_history_window(
        factors,
        "all",
    )

    assert len(result) == len(factors)


@pytest.mark.parametrize(
    "window",
    [1, 2, 6, 12, 24, 36],
)
def test_history_window_keeps_at_most_n_per_age(
    train_snapshot,
    window,
):
    factors = calculate_individual_factors(
        train_snapshot,
        dev_M=3,
    )

    result = apply_history_window(
        factors,
        window,
    )

    counts = (
        result
        .groupby("dev_month")
        .size()
    )

    assert (counts <= window).all()


def test_history_window_uses_most_recent_cohorts(
    train_snapshot,
):
    factors = calculate_individual_factors(
        train_snapshot,
        dev_M=3,
    )

    result = apply_history_window(
        factors,
        2,
    )

    expected_periods = {
        pd.Period("2020-03", freq="M"),
        pd.Period("2020-04", freq="M"),
    }

    for _, group in result.groupby(
        "dev_month"
    ):
        assert set(
            group["accident_period"]
        ) == expected_periods


# ============================================================
# 4. Reglas de exclusión
# ============================================================


def test_exclusion_none_changes_nothing(
    train_snapshot,
):
    factors = calculate_individual_factors(
        train_snapshot,
        dev_M=3,
    )

    result = apply_exclusion_rule(
        factors,
        "none",
    )

    assert len(result) == len(factors)


def test_exclusion_minmax_removes_two_per_age(
    train_snapshot,
):
    factors = calculate_individual_factors(
        train_snapshot,
        dev_M=3,
    )

    result = apply_exclusion_rule(
        factors,
        "minmax",
    )

    counts = (
        result
        .groupby("dev_month")
        .size()
    )

    # 4 observaciones iniciales:
    # mínimo + máximo eliminados
    assert (counts == 2).all()


def test_exclusion_std2_removes_extreme_outlier():
    factors = pd.DataFrame(
        {
            "accident_period": pd.period_range(
                "2020-01",
                periods=10,
                freq="M",
            ),
            "dev_month": [0] * 10,
            "current_amount": [100.0] * 10,
            "individual_factor": (
                [1.0] * 9
                + [10.0]
            ),
        }
    )

    factors["next_amount"] = (
        factors["current_amount"]
        * factors["individual_factor"]
    )

    result = apply_exclusion_rule(
        factors,
        "std2",
    )

    assert len(result) == 9

    assert result[
        "individual_factor"
    ].max() == pytest.approx(1.0)


# ============================================================
# 5. Métodos de agregación
# ============================================================


@pytest.fixture
def aggregation_example():
    return pd.DataFrame(
        {
            "dev_month": [0, 0],
            "accident_period": pd.period_range(
                "2020-01",
                periods=2,
                freq="M",
            ),
            "current_amount": [
                100.0,
                300.0,
            ],
            "next_amount": [
                120.0,
                330.0,
            ],
            "individual_factor": [
                1.2,
                1.1,
            ],
        }
    )


def test_simple_average(
    aggregation_example,
):
    result = aggregate_development_factors(
        aggregation_example,
        "S",
    )

    assert result.loc[
        0,
        "development_factor",
    ] == pytest.approx(1.15)


def test_median(
    aggregation_example,
):
    result = aggregate_development_factors(
        aggregation_example,
        "M",
    )

    assert result.loc[
        0,
        "development_factor",
    ] == pytest.approx(1.15)


def test_volume_weighted_average(
    aggregation_example,
):
    result = aggregate_development_factors(
        aggregation_example,
        "VW",
    )

    expected = (
        120.0 + 330.0
    ) / (
        100.0 + 300.0
    )

    assert result.loc[
        0,
        "development_factor",
    ] == pytest.approx(expected)


# ============================================================
# 6. Ajuste completo
# ============================================================


def test_fit_development_factors(
    train_snapshot,
):
    factors = fit_development_factors(
        train_snapshot,
        dev_M=3,
        method="VW",
        window="all",
        exclusion="none",
    )

    assert len(factors) == 3

    assert list(
        factors["dev_month"]
    ) == [0, 1, 2]

    assert (
        factors["development_factor"] > 0
    ).all()

    assert np.isfinite(
        factors["development_factor"]
    ).all()


# ============================================================
# 7. CDF
# ============================================================


def test_build_development_cdf():
    factors = pd.DataFrame(
        {
            "dev_month": [0, 1, 2],
            "development_factor": [
                1.20,
                1.10,
                1.05,
            ],
        }
    )

    cdf = build_development_cdf(
        factors,
        dev_M=3,
    )

    expected = pd.Series(
        {
            0: 1.20 * 1.10 * 1.05,
            1: 1.10 * 1.05,
            2: 1.05,
        },
        name="development_cdf",
    )

    np.testing.assert_allclose(
        cdf.to_numpy(),
        expected.to_numpy(),
    )


# ============================================================
# 8. Predicción individual
# ============================================================


def test_predict_development_ultimate():
    cdf = pd.Series(
        {
            0: 1.386,
            1: 1.155,
            2: 1.05,
        },
        name="development_cdf",
    )

    prediction = (
        predict_development_ultimate(
            observed_amount=100.0,
            current_dev_month=1,
            development_cdf=cdf,
            dev_M=3,
        )
    )

    assert prediction == pytest.approx(
        115.5
    )


# ============================================================
# 9. Prediction snapshot
# ============================================================


def test_prediction_snapshot_has_one_prediction_per_cohort(
    train_snapshot,
    prediction_snapshot,
):
    factors = fit_development_factors(
        train_snapshot,
        dev_M=3,
        method="VW",
        window="all",
        exclusion="none",
    )

    predictions = (
        predict_development_snapshot(
            prediction_snapshot,
            factors,
            dev_M=3,
            method_name="VW_all_none",
        )
    )

    assert len(predictions) == len(
        prediction_snapshot
    )

    assert predictions[
        "prediction"
    ].notna().all()

    assert (
        predictions["prediction"] > 0
    ).all()

    assert np.isfinite(
        predictions["prediction"]
    ).all()


def test_prediction_snapshot_rejects_target_amount(
    train_snapshot,
    prediction_snapshot,
):
    factors = fit_development_factors(
        train_snapshot,
        dev_M=3,
        method="VW",
        window="all",
        exclusion="none",
    )

    leaked = prediction_snapshot.copy()

    leaked["target_amount"] = 999.0

    with pytest.raises(
        ValueError,
        match="target_amount",
    ):
        predict_development_snapshot(
            leaked,
            factors,
            dev_M=3,
            method_name="VW_all_none",
        )


# ============================================================
# 10. Alternativa completa
# ============================================================


def test_run_single_development_alternative(
    train_snapshot,
    prediction_snapshot,
):
    factors, predictions = (
        run_development_alternative(
            train_snapshot,
            prediction_snapshot,
            alternative=(
                "VW",
                "all",
                "none",
            ),
            dev_M=3,
        )
    )

    assert len(factors) == 3

    assert len(predictions) == len(
        prediction_snapshot
    )

    assert (
        predictions["method_name"]
        == "VW_all_none"
    ).all()

    assert (
        predictions["method_family"]
        == "development"
    ).all()


# ============================================================
# 11. Catálogo completo de 21 alternativas
# ============================================================


DEVELOPMENT_ALTERNATIVES_TEST = [
    # Simple
    ("S", "all", "none"),
    ("S", "all", "minmax"),
    ("S", "all", "std2"),
    ("S", 24, "none"),
    ("S", 24, "minmax"),
    ("S", 24, "std2"),

    # Volume weighted
    ("VW", "all", "none"),
    ("VW", "all", "minmax"),
    ("VW", "all", "std2"),
    ("VW", 24, "none"),
    ("VW", 24, "minmax"),
    ("VW", 24, "std2"),

    # Median
    ("M", "all", "none"),
    ("M", "all", "minmax"),
    ("M", "all", "std2"),
    ("M", 24, "none"),
    ("M", 24, "minmax"),
    ("M", 24, "std2"),

    # Ventanas VW adicionales
    ("VW", 6, "none"),
    ("VW", 12, "none"),
    ("VW", 36, "none"),
]


def test_catalog_contains_21_alternatives():
    assert len(
        DEVELOPMENT_ALTERNATIVES_TEST
    ) == 21

    names = [
        development_alternative_name(x)
        for x in DEVELOPMENT_ALTERNATIVES_TEST
    ]

    assert len(set(names)) == 21


def test_run_21_development_alternatives(
    train_snapshot,
    prediction_snapshot,
):
    factors, predictions = (
        run_development_alternatives(
            train_snapshot,
            prediction_snapshot,
            alternatives=(
                DEVELOPMENT_ALTERNATIVES_TEST
            ),
            dev_M=3,
        )
    )

    n_methods = len(
        DEVELOPMENT_ALTERNATIVES_TEST
    )

    n_predictions = len(
        prediction_snapshot
    )

    # 3 factores por alternativa
    assert len(factors) == (
        3 * n_methods
    )

    # Cada método debe predecir las
    # mismas 3 cohortes.
    assert len(predictions) == (
        n_predictions
        * n_methods
    )

    assert (
        predictions[
            "method_name"
        ].nunique()
        == 21
    )

    assert (
        predictions["prediction"] > 0
    ).all()

    assert np.isfinite(
        predictions["prediction"]
    ).all()

    assert (
        "target_amount"
        not in predictions.columns
    )


def test_all_methods_predict_same_cohorts(
    train_snapshot,
    prediction_snapshot,
):
    _, predictions = (
        run_development_alternatives(
            train_snapshot,
            prediction_snapshot,
            alternatives=(
                DEVELOPMENT_ALTERNATIVES_TEST
            ),
            dev_M=3,
        )
    )

    cohorts_by_method = (
        predictions
        .groupby("method_name")[
            "accident_period"
        ]
        .apply(
            lambda x: tuple(
                sorted(x.tolist())
            )
        )
    )

    reference = cohorts_by_method.iloc[0]

    assert cohorts_by_method.apply(
        lambda x: x == reference
    ).all()


# ============================================================
# 12. Controles adicionales
# ============================================================


def test_invalid_dev_M_rejected(
    train_snapshot,
):
    with pytest.raises(
        ValueError,
        match="mayor que cero",
    ):
        calculate_individual_factors(
            train_snapshot,
            dev_M=0,
        )


def test_invalid_method_rejected(
    aggregation_example,
):
    with pytest.raises(
        ValueError,
        match="method",
    ):
        aggregate_development_factors(
            aggregation_example,
            "INVALID",
        )


def test_invalid_exclusion_rejected(
    train_snapshot,
):
    factors = calculate_individual_factors(
        train_snapshot,
        dev_M=3,
    )

    with pytest.raises(
        ValueError,
        match="exclusion",
    ):
        apply_exclusion_rule(
            factors,
            "INVALID",
        )