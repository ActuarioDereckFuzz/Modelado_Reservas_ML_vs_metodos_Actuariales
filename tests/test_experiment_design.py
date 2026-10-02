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


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

def test_experiment_config():
    """
    La configuración congelada del experimento debe ser
    internamente consistente.
    """
    validate_experiment_config()


# ============================================================
# PRIMERA VALUACIÓN POSIBLE PARA ML
# ============================================================

def test_first_possible_ml_valuation():
    """
    Verifica la primera valuación en la que existen al menos
    las cohortes maduras mínimas requeridas para ML.
    """

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


# ============================================================
# FINAL_PERIOD ES SOLO DIAGNÓSTICO
# ============================================================

def test_final_period_does_not_limit_prediction_candidates(
    accident_periods,
):
    """
    FINAL_PERIOD ya no limita el universo de predicción.

    Deben existir observaciones candidatas cuyo target ocurra
    después de FINAL_PERIOD.
    """

    for scenario in DEV_M:

        grid = build_eligibility_grid(
            accident_periods=accident_periods,
            scenario=scenario,
            valuation_periods=VALUATION_PERIODS,
        )

        validate_eligibility_frame(grid)

        prediction = grid.loc[
            grid["is_prediction_candidate"]
        ]

        assert not prediction.empty

        # Deben existir predicciones cuyo target se revele
        # después del antiguo cierre del experimento.
        assert (
            prediction["target_period"]
            > FINAL_PERIOD
        ).any()

        # La variable de cierre se conserva únicamente
        # como diagnóstico.
        after_final = prediction.loc[
            prediction["target_period"]
            > FINAL_PERIOD
        ]

        assert not after_final[
            "is_target_revealed_by_end"
        ].any()


# ============================================================
# COBERTURA DE TODAS LAS EDADES
# ============================================================

def test_all_development_ages_are_prediction_candidates(
    accident_periods,
):
    """
    El universo de backtesting debe contener todas las edades:

        0, ..., dev_M - 1

    incluyendo las edades jóvenes que el diseño anterior
    excluía por exigir target_period <= FINAL_PERIOD.
    """

    for scenario, dev_m in DEV_M.items():

        grid = build_eligibility_grid(
            accident_periods=accident_periods,
            scenario=scenario,
            valuation_periods=VALUATION_PERIODS,
        )

        validate_eligibility_frame(grid)

        prediction = grid.loc[
            grid["is_prediction_candidate"]
        ]

        observed_ages = set(
            prediction[
                "latest_dev_month"
            ].unique()
        )

        assert observed_ages == set(
            range(dev_m)
        )


# ============================================================
# TARGET FUTURO EN LA FECHA DE PREDICCIÓN
# ============================================================

def test_prediction_targets_are_future_at_valuation(
    accident_periods,
):
    """
    Toda observación candidata a predicción debe permanecer
    inmadura en la fecha histórica de valuación.
    """

    for scenario in DEV_M:

        grid = build_eligibility_grid(
            accident_periods=accident_periods,
            scenario=scenario,
            valuation_periods=VALUATION_PERIODS,
        )

        validate_eligibility_frame(grid)

        prediction = grid.loc[
            grid["is_prediction_candidate"]
        ]

        assert not prediction.empty

        assert (
            prediction["target_period"]
            > prediction["valuation_period"]
        ).all()

        assert (
            prediction["latest_dev_month"] >= 0
        ).all()

        assert (
            prediction["latest_dev_month"]
            < prediction["dev_M"]
        ).all()

        assert not prediction[
            "is_mature_at_valuation"
        ].any()


# ============================================================
# COMPATIBILIDAD TEMPORAL DEL ALIAS
# ============================================================

def test_backtest_evaluable_alias_matches_prediction_candidate(
    accident_periods,
):
    """
    Mientras conservemos `is_backtest_evaluable` por
    compatibilidad con código anterior, debe ser exactamente
    equivalente a `is_prediction_candidate`.
    """

    for scenario in DEV_M:

        grid = build_eligibility_grid(
            accident_periods=accident_periods,
            scenario=scenario,
            valuation_periods=VALUATION_PERIODS,
        )

        assert (
            grid["is_backtest_evaluable"]
            == grid["is_prediction_candidate"]
        ).all()