from __future__ import annotations

import pandas as pd

from .config import (
    BACKTEST_END,
    BACKTEST_START,
    DEVELOPMENT_ALTERNATIVES,
    DEV_BANDS,
    DEV_M,
    FINAL_PERIOD,
    FORBIDDEN_PREDICTORS,
    GLM_FEATURES,
    MIN_MATURE_COHORTS_ML,
    RELATIVE_SELECTION_THRESHOLD,
    TREE_FEATURES,
    VALUATION_PERIODS,
)


def validate_experiment_config() -> None:
    """
    Valida que la configuración congelada del experimento
    sea internamente consistente.
    """
    assert DEV_M["creciente"] == 60
    assert DEV_M["decreciente"] == 48
    assert DEV_M["mixto"] == 48

    assert BACKTEST_START == pd.Period(
        "2020-01",
        freq="M",
    )

    assert BACKTEST_END == pd.Period(
        "2025-11",
        freq="M",
    )

    assert FINAL_PERIOD == pd.Period(
        "2025-12",
        freq="M",
    )

    assert BACKTEST_END < FINAL_PERIOD

    assert len(VALUATION_PERIODS) == 71

    assert len(DEVELOPMENT_ALTERNATIVES) == 21

    assert (
        len(set(DEVELOPMENT_ALTERNATIVES))
        == len(DEVELOPMENT_ALTERNATIVES)
    )

    assert MIN_MATURE_COHORTS_ML == 12

    assert RELATIVE_SELECTION_THRESHOLD == 0.05

    forbidden = set(FORBIDDEN_PREDICTORS)

    assert not forbidden.intersection(
        GLM_FEATURES
    )

    assert not forbidden.intersection(
        TREE_FEATURES
    )

    validate_development_bands()


def validate_development_bands() -> None:
    """
    Verifica que las bandas cubran exactamente
    0,...,dev_M-1 sin huecos ni solapamientos.
    """
    for dev_m, bands in DEV_BANDS.items():

        ages = []

        for lower, upper in bands:

            assert lower <= upper

            ages.extend(
                range(lower, upper + 1)
            )

        assert ages == list(range(dev_m))


def validate_eligibility_frame(
    df: pd.DataFrame,
) -> None:
    """
    Validaciones generales de la cuadrícula de elegibilidad.
    """
    required = {
        "scenario",
        "accident_period",
        "valuation_period",
        "latest_dev_month",
        "dev_M",
        "target_period",
        "is_mature_at_valuation",
        "is_target_revealed_by_end",
        "is_backtest_evaluable",
    }

    missing = required.difference(df.columns)

    assert not missing, (
        f"Columnas faltantes: {sorted(missing)}"
    )

    evaluable = df.loc[
        df["is_backtest_evaluable"]
    ]

    assert (
        evaluable["latest_dev_month"] >= 0
    ).all()

    assert (
        evaluable["latest_dev_month"]
        < evaluable["dev_M"]
    ).all()

    assert (
        evaluable["target_period"]
        <= FINAL_PERIOD
    ).all()

    assert (
        evaluable["target_period"]
        > evaluable["valuation_period"]
    ).all()


def validate_training_set(
    train: pd.DataFrame,
    valuation_period,
) -> None:
    """
    Verifica que el dataset de entrenamiento no utilice
    targets que todavía eran desconocidos en la valuación.
    """
    valuation_period = pd.Period(
        valuation_period,
        freq="M",
    )

    required = {
        "target_period",
        "target_amount",
        "snapshot_dev_month",
        "dev_M",
    }

    missing = required.difference(
        train.columns
    )

    assert not missing, (
        f"Columnas faltantes en train: "
        f"{sorted(missing)}"
    )

    assert (
        train["target_period"]
        <= valuation_period
    ).all()

    assert (
        train["snapshot_dev_month"]
        < train["dev_M"]
    ).all()

    assert (
        train["snapshot_dev_month"] >= 0
    ).all()

    assert train["target_amount"].gt(0).all()

    if "max_feature_dev_month" in train.columns:

        assert (
            train["max_feature_dev_month"]
            <= train["snapshot_dev_month"]
        ).all()


def validate_test_set(
    test: pd.DataFrame,
    valuation_period,
    final_period=FINAL_PERIOD,
) -> None:
    """
    Verifica que las observaciones de test sean inmaduras
    en la valuación, pero evaluables al cierre.
    """
    valuation_period = pd.Period(
        valuation_period,
        freq="M",
    )

    final_period = pd.Period(
        final_period,
        freq="M",
    )

    required = {
        "target_period",
        "target_amount",
        "snapshot_dev_month",
        "dev_M",
    }

    missing = required.difference(
        test.columns
    )

    assert not missing, (
        f"Columnas faltantes en test: "
        f"{sorted(missing)}"
    )

    assert (
        test["target_period"]
        > valuation_period
    ).all()

    assert (
        test["target_period"]
        <= final_period
    ).all()

    assert (
        test["snapshot_dev_month"]
        >= 0
    ).all()

    assert (
        test["snapshot_dev_month"]
        < test["dev_M"]
    ).all()

    assert test["target_amount"].gt(0).all()

    if "max_feature_dev_month" in test.columns:

        assert (
            test["max_feature_dev_month"]
            <= test["snapshot_dev_month"]
        ).all()


def validate_feature_timing(
    df: pd.DataFrame,
    snapshot_col: str = "snapshot_dev_month",
    feature_col: str = "max_feature_dev_month",
) -> None:
    """
    Comprueba explícitamente que ninguna feature utilice
    una edad posterior al snapshot.
    """
    assert snapshot_col in df.columns
    assert feature_col in df.columns

    assert (
        df[feature_col]
        <= df[snapshot_col]
    ).all()


def validate_prediction_keys_equal(
    *prediction_frames: pd.DataFrame,
    key_cols=(
        "scenario",
        "valuation_period",
        "accident_period",
    ),
) -> None:
    """
    Comprueba que varios modelos estén siendo evaluados
    exactamente sobre las mismas observaciones.
    """
    if len(prediction_frames) < 2:
        return

    key_sets = []

    for df in prediction_frames:

        missing = set(key_cols).difference(
            df.columns
        )

        assert not missing, (
            f"Columnas de clave faltantes: "
            f"{sorted(missing)}"
        )

        keys = set(
            df[list(key_cols)]
            .drop_duplicates()
            .itertuples(
                index=False,
                name=None,
            )
        )

        key_sets.append(keys)

    reference = key_sets[0]

    for keys in key_sets[1:]:
        assert keys == reference


def validate_no_prediction_duplicates(
    predictions: pd.DataFrame,
) -> None:
    """
    Una metodología solo puede generar una predicción
    por observación histórica.
    """
    key = [
        "scenario",
        "valuation_period",
        "accident_period",
        "method_name",
    ]

    assert not predictions.duplicated(
        key
    ).any()