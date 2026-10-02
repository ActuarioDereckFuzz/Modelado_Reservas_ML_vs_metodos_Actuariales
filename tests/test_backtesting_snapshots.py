from __future__ import annotations

import pandas as pd
import pytest

from src.experiment.config import (
    DEV_M,
    FINAL_PERIOD,
)

from src.experiment.snapshots import (
    build_prediction_snapshots,
    build_target_table,
    build_train_prediction_snapshots,
    build_training_snapshots,
    reveal_targets,
)


SCENARIOS = tuple(DEV_M)

# Fecha con historia suficiente para observar todas las edades
# 0, ..., dev_M - 1.
MID_VALUATION = pd.Period(
    "2022-06",
    freq="M",
)

# Fecha útil para comprobar que podemos predecir targets
# posteriores al antiguo cierre FINAL_PERIOD.
LATE_VALUATION = pd.Period(
    "2025-11",
    freq="M",
)


def _get_scenario_df(
    simulations_base,
    scenario: str,
) -> pd.DataFrame:
    """
    Obtiene la base correspondiente a un escenario.

    Soporta tanto:

        simulations_base = {
            "creciente": df,
            ...
        }

    como una única base concatenada con columna `scenario`.
    """
    if isinstance(simulations_base, dict):
        return simulations_base[scenario].copy()

    return (
        simulations_base.loc[
            simulations_base["scenario"].eq(scenario)
        ]
        .copy()
        .reset_index(drop=True)
    )


# ============================================================
# 1. COBERTURA DE EDADES JÓVENES
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
)
def test_prediction_covers_all_immature_ages(
    simulations_base,
    scenario,
):
    """
    El nuevo backtesting debe recuperar también las edades
    jóvenes que quedaban excluidas por la condición:

        target_period <= FINAL_PERIOD
    """
    df = _get_scenario_df(
        simulations_base,
        scenario,
    )

    prediction = build_prediction_snapshots(
        df=df,
        scenario=scenario,
        valuation_period=MID_VALUATION,
    )

    assert not prediction.empty

    observed_ages = set(
        prediction[
            "snapshot_dev_month"
        ].astype(int)
    )

    expected_ages = set(
        range(DEV_M[scenario])
    )

    assert observed_ages == expected_ages

    assert prediction[
        "snapshot_dev_month"
    ].min() == 0

    assert prediction[
        "snapshot_dev_month"
    ].max() == DEV_M[scenario] - 1


# ============================================================
# 2. TARGET BLOQUEADO DURANTE PREDICCIÓN
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
)
def test_prediction_snapshot_blocks_target_amount(
    simulations_base,
    scenario,
):
    """
    El monto objetivo no puede existir físicamente en el
    dataset entregado al modelo.
    """
    df = _get_scenario_df(
        simulations_base,
        scenario,
    )

    prediction = build_prediction_snapshots(
        df=df,
        scenario=scenario,
        valuation_period=MID_VALUATION,
    )

    assert "target_amount" not in prediction.columns

    assert (
        prediction["target_period"]
        > MID_VALUATION
    ).all()

    assert (
        prediction["snapshot_period"]
        == MID_VALUATION
    ).all()

    assert (
        prediction["snapshot_dev_month"] >= 0
    ).all()

    assert (
        prediction["snapshot_dev_month"]
        < prediction["dev_M"]
    ).all()

    assert not prediction[
        "accident_period"
    ].duplicated().any()


# ============================================================
# 3. FINAL_PERIOD YA NO LIMITA EL BACKTEST
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
)
def test_prediction_allows_targets_after_final_period(
    simulations_base,
    scenario,
):
    """
    Comprueba específicamente la corrección del problema
    detectado en el diseño anterior.

    Una cohorte puede formar parte del backtesting aunque
    alcance dev_M después de FINAL_PERIOD.
    """
    df = _get_scenario_df(
        simulations_base,
        scenario,
    )

    prediction = build_prediction_snapshots(
        df=df,
        scenario=scenario,
        valuation_period=LATE_VALUATION,
    )

    assert not prediction.empty

    after_final = (
        prediction["target_period"]
        > FINAL_PERIOD
    )

    assert after_final.any()

    # La cohorte recién ocurrida debe estar presente.
    age_zero = prediction.loc[
        prediction[
            "snapshot_dev_month"
        ].eq(0)
    ]

    assert not age_zero.empty

    assert (
        age_zero["target_period"]
        > FINAL_PERIOD
    ).all()


# ============================================================
# 4. REVELACIÓN = LOSS INCURRED REAL EN dev_M
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
)
def test_revealed_target_matches_loss_incurred_at_dev_m(
    simulations_base,
    scenario,
):
    """
    Después de congelar la predicción, reveal_targets()
    debe recuperar exactamente el Loss Incurred observado
    en dev_M.
    """
    df = _get_scenario_df(
        simulations_base,
        scenario,
    )

    prediction = build_prediction_snapshots(
        df=df,
        scenario=scenario,
        valuation_period=MID_VALUATION,
    )

    assert "target_amount" not in prediction.columns

    revealed = reveal_targets(
        predictions=prediction,
        df=df,
        scenario=scenario,
    )

    assert "target_amount" in revealed.columns

    targets = build_target_table(
        df=df,
        scenario=scenario,
    )[
        [
            "accident_period",
            "target_amount",
        ]
    ].rename(
        columns={
            "target_amount": "expected_target"
        }
    )

    check = revealed.merge(
        targets,
        on="accident_period",
        how="left",
        validate="many_to_one",
    )

    pd.testing.assert_series_equal(
        check["target_amount"].reset_index(
            drop=True
        ),
        check["expected_target"].reset_index(
            drop=True
        ),
        check_names=False,
    )


# ============================================================
# 5. TRAIN SOLO USA TARGETS YA CONOCIDOS
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
)
def test_training_uses_only_revealed_targets(
    simulations_base,
    scenario,
):
    """
    El entrenamiento mantiene la regla temporal:

        target_period <= valuation_period
    """
    df = _get_scenario_df(
        simulations_base,
        scenario,
    )

    train = build_training_snapshots(
        df=df,
        scenario=scenario,
        valuation_period=LATE_VALUATION,
    )

    assert not train.empty

    assert "target_amount" in train.columns

    assert train[
        "target_amount"
    ].notna().all()

    assert train[
        "target_amount"
    ].gt(0).all()

    assert (
        train["target_period"]
        <= LATE_VALUATION
    ).all()

    assert (
        train["snapshot_period"]
        <= LATE_VALUATION
    ).all()

    assert (
        train["snapshot_dev_month"]
        < train["dev_M"]
    ).all()


# ============================================================
# 6. WRAPPER = CONSTRUCCIÓN INDIVIDUAL
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
)
def test_train_prediction_wrapper_matches_individual_builds(
    simulations_base,
    scenario,
):
    """
    El wrapper no debe modificar el resultado de las
    funciones individuales.
    """
    df = _get_scenario_df(
        simulations_base,
        scenario,
    )

    expected_train = build_training_snapshots(
        df=df,
        scenario=scenario,
        valuation_period=MID_VALUATION,
    )

    expected_prediction = (
        build_prediction_snapshots(
            df=df,
            scenario=scenario,
            valuation_period=MID_VALUATION,
        )
    )

    train, prediction = (
        build_train_prediction_snapshots(
            df=df,
            scenario=scenario,
            valuation_period=MID_VALUATION,
        )
    )

    pd.testing.assert_frame_equal(
        train,
        expected_train,
    )

    pd.testing.assert_frame_equal(
        prediction,
        expected_prediction,
    )