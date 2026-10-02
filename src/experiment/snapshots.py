from __future__ import annotations

import pandas as pd

from .config import DEV_M, FINAL_PERIOD
from .eligibility import (
    as_month_period,
    build_eligibility_grid,
)
from .validation import (
    validate_prediction_set,
    validate_revealed_test_set,
    validate_training_set,
)


# ============================================================
# COLUMNAS REQUERIDAS DE LA BASE SIMULADA
# ============================================================

REQUIRED_SOURCE_COLUMNS = {
    "scenario",
    "accident_period",
    "dev_month",
    "calendar_period",
    "dev_M",
    "loss_incurred",
}


# ============================================================
# COLUMNAS DE SNAPSHOTS
# ============================================================

COMMON_SNAPSHOT_COLUMNS = [
    "scenario",
    "accident_period",
    "snapshot_period",
    "model_valuation_period",
    "snapshot_dev_month",
    "dev_M",
    "observed_amount",
    "target_period",
    "max_feature_dev_month",
]


TRAINING_SNAPSHOT_COLUMNS = [
    "scenario",
    "accident_period",
    "snapshot_period",
    "model_valuation_period",
    "snapshot_dev_month",
    "dev_M",
    "observed_amount",
    "target_period",
    "target_amount",
    "max_feature_dev_month",
]


PREDICTION_SNAPSHOT_COLUMNS = (
    COMMON_SNAPSHOT_COLUMNS.copy()
)


# ============================================================
# PREPARACIÓN Y VALIDACIÓN DE LA BASE
# ============================================================

def _prepare_scenario_frame(
    df: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    """
    Prepara y valida la base longitudinal de un escenario.

    La base debe contener una fila por:

        scenario × accident_period × dev_month

    y debe utilizar `loss_incurred` como monto acumulado.
    """
    if scenario not in DEV_M:
        raise ValueError(
            f"Escenario desconocido: {scenario!r}. "
            f"Opciones válidas: {list(DEV_M)}"
        )

    missing = REQUIRED_SOURCE_COLUMNS.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "La base no contiene todas las columnas "
            f"requeridas. Faltan: {sorted(missing)}"
        )

    data = (
        df.loc[
            df["scenario"].eq(scenario)
        ]
        .copy()
    )

    if data.empty:
        raise ValueError(
            f"No existen observaciones para "
            f"{scenario!r}."
        )

    # --------------------------------------------------------
    # Normalización temporal
    # --------------------------------------------------------

    data["accident_period"] = pd.PeriodIndex(
        data["accident_period"],
        freq="M",
    )

    data["calendar_period"] = pd.PeriodIndex(
        data["calendar_period"],
        freq="M",
    )

    data["dev_month"] = (
        data["dev_month"]
        .astype(int)
    )

    data["dev_M"] = (
        data["dev_M"]
        .astype(int)
    )

    expected_dev_m = DEV_M[scenario]

    if not data["dev_M"].eq(
        expected_dev_m
    ).all():
        raise ValueError(
            "`dev_M` no coincide con la "
            "configuración del escenario "
            f"{scenario!r}: "
            f"se esperaba {expected_dev_m}."
        )

    # --------------------------------------------------------
    # Clave única
    # --------------------------------------------------------

    key = [
        "scenario",
        "accident_period",
        "dev_month",
    ]

    if data.duplicated(key).any():
        raise ValueError(
            "Se detectaron duplicados en la clave "
            "(scenario, accident_period, dev_month)."
        )

    # --------------------------------------------------------
    # Positividad
    # --------------------------------------------------------

    if not data["loss_incurred"].gt(0).all():
        raise ValueError(
            "`loss_incurred` debe ser "
            "estrictamente positivo."
        )

    # --------------------------------------------------------
    # calendar_period = accident_period + dev_month
    # --------------------------------------------------------

    expected_calendar = pd.PeriodIndex(
        [
            accident_period + int(dev_month)
            for accident_period, dev_month
            in zip(
                data["accident_period"],
                data["dev_month"],
            )
        ],
        freq="M",
    )

    if not (
        data["calendar_period"].array
        == expected_calendar.array
    ).all():
        raise ValueError(
            "Se detectaron inconsistencias entre "
            "`accident_period`, `dev_month` y "
            "`calendar_period`."
        )

    return (
        data
        .sort_values(
            [
                "accident_period",
                "dev_month",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# TARGETS A MADUREZ
# ============================================================

def build_target_table(
    df: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    """
    Construye la tabla de targets a madurez.

    El target NO se toma de `ultimate_loss`.

    Se obtiene directamente como:

        target_amount =
            loss_incurred en dev_month == dev_M

    reproduciendo así la información que estaría
    disponible en una base real una vez alcanzada
    la madurez.
    """
    data = _prepare_scenario_frame(
        df=df,
        scenario=scenario,
    )

    dev_m = DEV_M[scenario]

    targets = (
        data.loc[
            data["dev_month"].eq(dev_m),
            [
                "scenario",
                "accident_period",
                "calendar_period",
                "loss_incurred",
                "dev_M",
            ],
        ]
        .rename(
            columns={
                "calendar_period": "target_period",
                "loss_incurred": "target_amount",
            }
        )
        .copy()
    )

    if targets.empty:
        raise ValueError(
            "No existen observaciones en "
            f"dev_M={dev_m} para el escenario "
            f"{scenario!r}."
        )

    if targets[
        "accident_period"
    ].duplicated().any():
        raise ValueError(
            "Cada accident_period debe tener "
            "exactamente un target a madurez."
        )

    if not targets[
        "target_amount"
    ].gt(0).all():
        raise ValueError(
            "Todos los targets deben ser positivos."
        )

    return (
        targets
        .sort_values("accident_period")
        .reset_index(drop=True)
    )


# ============================================================
# SNAPSHOTS DE ENTRENAMIENTO
# ============================================================

def build_training_snapshots(
    df: pd.DataFrame,
    scenario: str,
    valuation_period,
) -> pd.DataFrame:
    """
    Construye los snapshots disponibles para
    entrenamiento en una valuación histórica.

    Solamente utiliza cohortes cuyo target ya era
    conocido:

        target_period <= valuation_period

    Para cada cohorte madura genera snapshots:

        d = 0, ..., dev_M - 1

    El snapshot en dev_M se excluye porque en ese
    punto `loss_incurred` coincide con el target.

    El target sí puede estar presente en entrenamiento,
    pues pertenece únicamente a cohortes que ya habían
    alcanzado madurez en la valuación histórica.
    """
    valuation_period = as_month_period(
        valuation_period
    )

    data = _prepare_scenario_frame(
        df=df,
        scenario=scenario,
    )

    targets = build_target_table(
        df=data,
        scenario=scenario,
    )

    dev_m = DEV_M[scenario]

    # --------------------------------------------------------
    # Targets conocidos en la valuación
    # --------------------------------------------------------

    mature_targets = (
        targets.loc[
            targets["target_period"]
            <= valuation_period
        ]
        .copy()
    )

    if mature_targets.empty:
        return pd.DataFrame(
            columns=TRAINING_SNAPSHOT_COLUMNS
        )

    mature_accident_periods = set(
        mature_targets["accident_period"]
    )

    # --------------------------------------------------------
    # Snapshots históricos de cohortes maduras
    # --------------------------------------------------------

    snapshots = (
        data.loc[
            data["accident_period"].isin(
                mature_accident_periods
            )
            & data["dev_month"].between(
                0,
                dev_m - 1,
            ),
            [
                "scenario",
                "accident_period",
                "calendar_period",
                "dev_month",
                "dev_M",
                "loss_incurred",
            ],
        ]
        .rename(
            columns={
                "calendar_period": (
                    "snapshot_period"
                ),
                "dev_month": (
                    "snapshot_dev_month"
                ),
                "loss_incurred": (
                    "observed_amount"
                ),
            }
        )
        .copy()
    )

    # --------------------------------------------------------
    # Incorporación del target conocido
    # --------------------------------------------------------

    snapshots = snapshots.merge(
        mature_targets[
            [
                "accident_period",
                "target_period",
                "target_amount",
            ]
        ],
        on="accident_period",
        how="left",
        validate="many_to_one",
    )

    snapshots[
        "model_valuation_period"
    ] = valuation_period

    # Ninguna feature podrá utilizar información posterior
    # al propio snapshot.
    snapshots[
        "max_feature_dev_month"
    ] = snapshots[
        "snapshot_dev_month"
    ]

    snapshots = snapshots[
        TRAINING_SNAPSHOT_COLUMNS
    ]

    snapshots = (
        snapshots
        .sort_values(
            [
                "accident_period",
                "snapshot_dev_month",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validaciones
    # --------------------------------------------------------

    validate_training_set(
        train=snapshots,
        valuation_period=valuation_period,
    )

    # La información representada por el snapshot debe
    # corresponder a una edad anterior a dev_M.
    assert (
        snapshots["snapshot_period"]
        < snapshots["target_period"]
    ).all()

    # Como las cohortes utilizadas para entrenamiento ya
    # estaban maduras, todos sus snapshots históricos
    # ocurrieron como máximo en la fecha de valuación.
    assert (
        snapshots["snapshot_period"]
        <= valuation_period
    ).all()

    return snapshots


# ============================================================
# SNAPSHOTS DE PREDICCIÓN / BACKTEST
# ============================================================

def build_prediction_snapshots(
    df: pd.DataFrame,
    scenario: str,
    valuation_period,
    final_period=FINAL_PERIOD,
) -> pd.DataFrame:
    """
    Construye las observaciones disponibles para
    predicción en una valuación histórica.

    Cada accident_period aparece una sola vez y utiliza
    exactamente la edad observable que tenía en
    `valuation_period`.

    Una observación es candidata cuando:

        0 <= snapshot_dev_month < dev_M

    equivalentemente:

        target_period > valuation_period

    El hecho de que el target ocurra después de
    FINAL_PERIOD NO elimina la observación.

    `final_period` se conserva únicamente porque
    `build_eligibility_grid()` lo utiliza para construir
    la variable diagnóstica:

        is_target_revealed_by_end

    Importante
    ----------
    `target_amount` NO se incorpora en esta etapa.

    La función representa únicamente la información
    disponible en la fecha histórica de valuación.
    """
    valuation_period = as_month_period(
        valuation_period
    )

    final_period = as_month_period(
        final_period
    )

    data = _prepare_scenario_frame(
        df=df,
        scenario=scenario,
    )

    accident_periods = (
        data["accident_period"]
        .drop_duplicates()
        .sort_values()
    )

    # --------------------------------------------------------
    # Universo temporal de predicción
    # --------------------------------------------------------

    eligibility = build_eligibility_grid(
        accident_periods=accident_periods,
        scenario=scenario,
        valuation_periods=[
            valuation_period
        ],
        final_period=final_period,
    )

    eligibility = (
        eligibility.loc[
            eligibility[
                "is_prediction_candidate"
            ]
        ]
        .copy()
    )

    if eligibility.empty:
        return pd.DataFrame(
            columns=PREDICTION_SNAPSHOT_COLUMNS
        )

    # --------------------------------------------------------
    # Monto observable en la fecha de valuación
    # --------------------------------------------------------

    current_rows = data[
        [
            "scenario",
            "accident_period",
            "dev_month",
            "calendar_period",
            "loss_incurred",
        ]
    ].copy()

    prediction = eligibility.merge(
        current_rows,
        left_on=[
            "scenario",
            "accident_period",
            "latest_dev_month",
        ],
        right_on=[
            "scenario",
            "accident_period",
            "dev_month",
        ],
        how="left",
        validate="one_to_one",
    )

    if prediction[
        "loss_incurred"
    ].isna().any():
        raise ValueError(
            "No fue posible encontrar el monto "
            "observable para todas las observaciones "
            "de predicción."
        )

    # --------------------------------------------------------
    # Construcción del snapshot
    # --------------------------------------------------------

    prediction = prediction.rename(
        columns={
            "calendar_period": (
                "snapshot_period"
            ),
            "latest_dev_month": (
                "snapshot_dev_month"
            ),
            "loss_incurred": (
                "observed_amount"
            ),
        }
    )

    prediction[
        "model_valuation_period"
    ] = valuation_period

    # Ninguna feature podrá utilizar información posterior
    # a la edad realmente observable.
    prediction[
        "max_feature_dev_month"
    ] = prediction[
        "snapshot_dev_month"
    ]

    # --------------------------------------------------------
    # Selección explícita de columnas
    #
    # Aquí se elimina físicamente cualquier posibilidad de
    # que `target_amount` llegue al modelo.
    # --------------------------------------------------------

    prediction = prediction[
        PREDICTION_SNAPSHOT_COLUMNS
    ]

    prediction = (
        prediction
        .sort_values("accident_period")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validación anti-leakage
    # --------------------------------------------------------

    validate_prediction_set(
        prediction=prediction,
        valuation_period=valuation_period,
    )

    return prediction


# ============================================================
# REVELACIÓN POSTERIOR DEL TARGET
# ============================================================

def reveal_targets(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    """
    Incorpora el target real a predicciones ya generadas.

    Esta función representa la fase de revelación del
    backtesting y debe ejecutarse únicamente DESPUÉS de
    que las predicciones hayan sido generadas y congeladas.

    El target se obtiene directamente como:

        target_amount =
            loss_incurred en dev_month == dev_M

    Nunca se utiliza una variable auxiliar como
    `ultimate_loss`.

    Parameters
    ----------
    predictions:
        DataFrame que contiene predicciones ya congeladas.

        Debe contener al menos:

            scenario
            accident_period
            target_period

        y una fecha de valuación mediante:

            valuation_period

        o:

            model_valuation_period

        No puede contener `target_amount`.

    df:
        Base longitudinal simulada completa.

    scenario:
        Escenario cuya trayectoria se utilizará para revelar
        los targets.

    Returns
    -------
    pd.DataFrame
        Copia de las predicciones con `target_amount`
        incorporado.
    """
    if predictions.empty:
        return predictions.copy()

    if "target_amount" in predictions.columns:
        raise ValueError(
            "`target_amount` ya está presente. "
            "La revelación solo puede realizarse sobre "
            "predicciones cuyo target todavía esté "
            "bloqueado."
        )

    frozen = predictions.copy()

    # --------------------------------------------------------
    # Comprobaciones básicas
    # --------------------------------------------------------

    required_prediction_columns = {
        "scenario",
        "accident_period",
        "target_period",
    }

    missing = required_prediction_columns.difference(
        frozen.columns
    )

    if missing:
        raise ValueError(
            "Las predicciones no contienen todas las "
            "columnas requeridas para revelar targets. "
            f"Faltan: {sorted(missing)}"
        )

    if not frozen[
        "scenario"
    ].eq(scenario).all():
        raise ValueError(
            "Las predicciones contienen escenarios "
            "distintos al escenario solicitado: "
            f"{scenario!r}."
        )

    # --------------------------------------------------------
    # Normalización de la fecha de valuación
    # --------------------------------------------------------

    if "valuation_period" not in frozen.columns:

        if (
            "model_valuation_period"
            not in frozen.columns
        ):
            raise ValueError(
                "Las predicciones deben contener "
                "`valuation_period` o "
                "`model_valuation_period`."
            )

        frozen["valuation_period"] = (
            frozen["model_valuation_period"]
        )

    frozen["valuation_period"] = pd.PeriodIndex(
        frozen["valuation_period"],
        freq="M",
    )

    # --------------------------------------------------------
    # Target real observado en dev_M
    # --------------------------------------------------------

    targets = build_target_table(
        df=df,
        scenario=scenario,
    )[
        [
            "scenario",
            "accident_period",
            "target_period",
            "target_amount",
        ]
    ].copy()

    revealed = frozen.merge(
        targets,
        on=[
            "scenario",
            "accident_period",
        ],
        how="left",
        suffixes=(
            "_prediction",
            "_target",
        ),
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Comprobación de la fecha del target
    # --------------------------------------------------------

    if (
        "target_period_prediction"
        in revealed.columns
    ):
        if (
            revealed[
                "target_period_prediction"
            ].isna().any()
            or revealed[
                "target_period_target"
            ].isna().any()
        ):
            raise ValueError(
                "No fue posible determinar "
                "`target_period` para todas las "
                "predicciones."
            )

        if not (
            revealed[
                "target_period_prediction"
            ]
            == revealed[
                "target_period_target"
            ]
        ).all():
            raise ValueError(
                "La fecha del target calculada durante "
                "predicción no coincide con la observada "
                "en la trayectoria simulada."
            )

        revealed["target_period"] = (
            revealed[
                "target_period_target"
            ]
        )

        revealed = revealed.drop(
            columns=[
                "target_period_prediction",
                "target_period_target",
            ]
        )

    # --------------------------------------------------------
    # Comprobación de targets revelados
    # --------------------------------------------------------

    if revealed[
        "target_amount"
    ].isna().any():
        missing_keys = (
            revealed.loc[
                revealed[
                    "target_amount"
                ].isna(),
                [
                    "scenario",
                    "accident_period",
                    "valuation_period",
                ],
            ]
            .drop_duplicates()
        )

        raise ValueError(
            "No fue posible revelar el target para "
            "todas las predicciones.\n"
            f"Claves sin target:\n{missing_keys}"
        )

    # --------------------------------------------------------
    # Validación posterior a la revelación
    # --------------------------------------------------------

    validate_revealed_test_set(
        revealed=revealed
    )

    return revealed


# ============================================================
# WRAPPER TRAIN + PREDICTION
# ============================================================

def build_train_prediction_snapshots(
    df: pd.DataFrame,
    scenario: str,
    valuation_period,
    final_period=FINAL_PERIOD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye simultáneamente los datasets de entrenamiento
    y predicción correspondientes a una valuación histórica.

    El dataset de entrenamiento contiene únicamente targets
    que ya eran conocidos en la fecha de valuación.

    El dataset de predicción mantiene `target_amount`
    completamente bloqueado.

    Returns
    -------
    train:
        Snapshots históricos provenientes de cohortes
        maduras.

    prediction:
        Una observación por cohorte inmadura utilizando
        exclusivamente la información observable en
        `valuation_period`.
    """
    train = build_training_snapshots(
        df=df,
        scenario=scenario,
        valuation_period=valuation_period,
    )

    prediction = build_prediction_snapshots(
        df=df,
        scenario=scenario,
        valuation_period=valuation_period,
        final_period=final_period,
    )

    return train, prediction