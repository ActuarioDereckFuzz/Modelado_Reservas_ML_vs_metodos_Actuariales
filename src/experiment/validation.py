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
    Valida la cuadrícula temporal de elegibilidad.

    Una observación candidata a predicción debe:

    - corresponder a una cohorte ya observable;
    - tener edad de desarrollo no negativa;
    - permanecer inmadura en la fecha de valuación;
    - cumplir 0 <= latest_dev_month < dev_M.

    La disponibilidad del target antes de FINAL_PERIOD
    no determina la elegibilidad para predicción.
    """

    required = {
        "scenario",
        "accident_period",
        "valuation_period",
        "latest_dev_month",
        "dev_M",
        "target_period",
        "is_observed_at_valuation",
        "is_mature_at_valuation",
        "is_target_revealed_by_end",
        "is_prediction_candidate",
        "is_backtest_evaluable",
    }

    missing = required.difference(df.columns)

    assert not missing, (
        f"Columnas faltantes: {sorted(missing)}"
    )

    prediction = df.loc[
        df["is_prediction_candidate"]
    ]

    # La cohorte debe existir en la valuación.
    assert prediction[
        "is_observed_at_valuation"
    ].all()

    # Debe encontrarse en una edad válida.
    assert (
        prediction["latest_dev_month"] >= 0
    ).all()

    assert (
        prediction["latest_dev_month"]
        < prediction["dev_M"]
    ).all()

    # El target todavía no debía conocerse.
    assert (
        prediction["target_period"]
        > prediction["valuation_period"]
    ).all()

    assert not prediction[
        "is_mature_at_valuation"
    ].any()

    # Mientras mantengamos el alias antiguo,
    # ambas definiciones deben ser equivalentes.
    assert (
        df["is_prediction_candidate"]
        == df["is_backtest_evaluable"]
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


def validate_prediction_set(
    prediction: pd.DataFrame,
    valuation_period,
) -> None:
    """
    Valida el dataset utilizado para generar predicciones.

    El dataset debe representar cohortes inmaduras en la
    fecha histórica de valuación y no puede contener el
    target futuro.
    """

    valuation_period = pd.Period(
        valuation_period,
        freq="M",
    )

    required = {
        "scenario",
        "accident_period",
        "snapshot_period",
        "snapshot_dev_month",
        "dev_M",
        "observed_amount",
        "target_period",
    }

    missing = required.difference(
        prediction.columns
    )

    assert not missing, (
        f"Columnas faltantes en prediction: "
        f"{sorted(missing)}"
    )

    # --------------------------------------------------------
    # Target bloqueado
    # --------------------------------------------------------

    forbidden_target_columns = {
        "target_amount",
    }

    leaked = forbidden_target_columns.intersection(
        prediction.columns
    )

    assert not leaked, (
        "El dataset de predicción contiene información "
        f"futura: {sorted(leaked)}"
    )

    # --------------------------------------------------------
    # Temporalidad
    # --------------------------------------------------------

    assert (
        prediction["target_period"]
        > valuation_period
    ).all()

    assert (
        prediction["snapshot_dev_month"] >= 0
    ).all()

    assert (
        prediction["snapshot_dev_month"]
        < prediction["dev_M"]
    ).all()

    # El snapshot de test representa exactamente
    # la fecha histórica de valuación.
    assert (
        prediction["snapshot_period"]
        == valuation_period
    ).all()

    # Una cohorte produce una única observación
    # en una valuación histórica.
    assert not prediction[
        "accident_period"
    ].duplicated().any()

    if "max_feature_dev_month" in prediction.columns:

        assert (
            prediction["max_feature_dev_month"]
            <= prediction["snapshot_dev_month"]
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

def validate_revealed_test_set(
    revealed: pd.DataFrame,
) -> None:
    """
    Valida la base posterior a la revelación del target.

    Esta función debe ejecutarse únicamente después de que
    las predicciones hayan sido generadas y congeladas.
    """

    required = {
        "scenario",
        "accident_period",
        "valuation_period",
        "target_period",
        "target_amount",
    }

    missing = required.difference(
        revealed.columns
    )

    assert not missing, (
        f"Columnas faltantes después de revelar target: "
        f"{sorted(missing)}"
    )

    assert revealed["target_amount"].notna().all()

    assert revealed["target_amount"].gt(0).all()

    # La predicción se realizó antes de que el target
    # estuviera disponible.
    assert (
        revealed["target_period"]
        > revealed["valuation_period"]
    ).all()