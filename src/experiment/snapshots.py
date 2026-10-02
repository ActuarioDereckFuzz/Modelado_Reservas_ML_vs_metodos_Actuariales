from __future__ import annotations

import pandas as pd

from .config import DEV_M, FINAL_PERIOD
from .eligibility import (
    as_month_period,
    build_eligibility_grid,
)
from .validation import (
    validate_test_set,
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


SNAPSHOT_COLUMNS = [
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

    missing = REQUIRED_SOURCE_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "La base no contiene todas las columnas "
            f"requeridas. Faltan: {sorted(missing)}"
        )

    data = (
        df.loc[df["scenario"] == scenario]
        .copy()
    )

    if data.empty:
        raise ValueError(
            f"No existen observaciones para {scenario!r}."
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

    if not data["dev_M"].eq(expected_dev_m).all():
        raise ValueError(
            f"`dev_M` no coincide con la configuración "
            f"del escenario {scenario!r}: "
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
            "`loss_incurred` debe ser estrictamente positivo."
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
# TARGETS MADUROS
# ============================================================

def build_target_table(
    df: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    """
    Construye la tabla de targets a madurez.

    El target NO se toma de `ultimate_loss`.

    Se obtiene directamente como:

        target_amount = loss_incurred en dev_month == dev_M

    reproduciendo así la información que estaría disponible
    en una base real una vez alcanzada la madurez.
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
            f"No existen observaciones en dev_M={dev_m} "
            f"para el escenario {scenario!r}."
        )

    if targets["accident_period"].duplicated().any():
        raise ValueError(
            "Cada accident_period debe tener exactamente "
            "un target a madurez."
        )

    if not targets["target_amount"].gt(0).all():
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
    Construye los snapshots disponibles para entrenamiento
    en una valuación histórica.

    Solamente utiliza cohortes cuyo target ya era conocido:

        target_period <= valuation_period

    Para cada cohorte madura genera snapshots:

        d = 0, ..., dev_M - 1

    El snapshot en dev_M se excluye porque en ese punto
    `loss_incurred` coincide con el target.
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

    mature_targets = (
        targets.loc[
            targets["target_period"]
            <= valuation_period
        ]
        .copy()
    )

    if mature_targets.empty:
        return pd.DataFrame(
            columns=SNAPSHOT_COLUMNS
        )

    mature_accident_periods = set(
        mature_targets["accident_period"]
    )

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
                "calendar_period": "snapshot_period",
                "dev_month": "snapshot_dev_month",
                "loss_incurred": "observed_amount",
            }
        )
        .copy()
    )

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

    snapshots["model_valuation_period"] = (
        valuation_period
    )

    # En esta etapa ninguna feature puede utilizar
    # información posterior al propio snapshot.
    snapshots["max_feature_dev_month"] = (
        snapshots["snapshot_dev_month"]
    )

    snapshots = snapshots[
        SNAPSHOT_COLUMNS
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

    validate_training_set(
        train=snapshots,
        valuation_period=valuation_period,
    )

    # Toda la información representada por el snapshot
    # debe haber ocurrido antes del momento en que el
    # target se conoció.
    assert (
        snapshots["snapshot_period"]
        < snapshots["target_period"]
    ).all()

    # Y, como la cohorte ya está madura en la valuación,
    # todos estos snapshots deben ser históricos respecto
    # del momento de ajuste.
    assert (
        snapshots["snapshot_period"]
        <= valuation_period
    ).all()

    return snapshots


# ============================================================
# SNAPSHOTS DE TEST / BACKTEST
# ============================================================

def build_test_snapshots(
    df: pd.DataFrame,
    scenario: str,
    valuation_period,
    final_period=FINAL_PERIOD,
) -> pd.DataFrame:
    """
    Construye las observaciones de backtesting existentes
    en una valuación histórica.

    Cada accident_period aparece una sola vez y utiliza la
    edad que realmente tenía en `valuation_period`.

    Requisitos:

        0 <= d < dev_M

        target_period > valuation_period

        target_period <= final_period
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

    targets = build_target_table(
        df=data,
        scenario=scenario,
    )

    accident_periods = (
        data["accident_period"]
        .drop_duplicates()
        .sort_values()
    )

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
                "is_backtest_evaluable"
            ]
        ]
        .copy()
    )

    if eligibility.empty:
        return pd.DataFrame(
            columns=SNAPSHOT_COLUMNS
        )

    current_rows = data[
        [
            "scenario",
            "accident_period",
            "dev_month",
            "calendar_period",
            "loss_incurred",
        ]
    ].copy()

    test = eligibility.merge(
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

    if test["loss_incurred"].isna().any():
        raise ValueError(
            "No fue posible encontrar el monto observable "
            "para todas las observaciones de backtesting."
        )

    test = test.merge(
        targets[
            [
                "accident_period",
                "target_period",
                "target_amount",
            ]
        ],
        on="accident_period",
        how="left",
        suffixes=(
            "_eligibility",
            "_target",
        ),
        validate="many_to_one",
    )

    # La fecha de target calculada por eligibility.py
    # debe coincidir con la observada directamente en la
    # trayectoria simulada.
    assert (
        test["target_period_eligibility"]
        == test["target_period_target"]
    ).all()

    test = test.rename(
        columns={
            "calendar_period": "snapshot_period",
            "latest_dev_month": "snapshot_dev_month",
            "loss_incurred": "observed_amount",
            "target_period_target": "target_period",
        }
    )

    test["model_valuation_period"] = (
        valuation_period
    )

    test["max_feature_dev_month"] = (
        test["snapshot_dev_month"]
    )

    test = test[
        SNAPSHOT_COLUMNS
    ]

    test = (
        test
        .sort_values("accident_period")
        .reset_index(drop=True)
    )

    validate_test_set(
        test=test,
        valuation_period=valuation_period,
        final_period=final_period,
    )

    # En test el snapshot representa exactamente
    # la valuación histórica.
    assert (
        test["snapshot_period"]
        == valuation_period
    ).all()

    # Una cohorte produce una sola observación
    # en cada fecha histórica.
    assert not test[
        "accident_period"
    ].duplicated().any()

    return test


# ============================================================
# WRAPPER TRAIN + TEST
# ============================================================

def build_train_test_snapshots(
    df: pd.DataFrame,
    scenario: str,
    valuation_period,
    final_period=FINAL_PERIOD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye simultáneamente train y test para una
    valuación histórica.
    """
    train = build_training_snapshots(
        df=df,
        scenario=scenario,
        valuation_period=valuation_period,
    )

    test = build_test_snapshots(
        df=df,
        scenario=scenario,
        valuation_period=valuation_period,
        final_period=final_period,
    )

    return train, test