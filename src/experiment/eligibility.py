from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .config import (
    DEV_M,
    FINAL_PERIOD,
    FIRST_ACCIDENT_PERIOD,
    MIN_MATURE_COHORTS_ML,
    VALUATION_PERIODS,
)


def as_month_period(value) -> pd.Period:
    """
    Convierte un valor compatible a Period mensual.
    """
    if isinstance(value, pd.Period):
        return value.asfreq("M")

    return pd.Period(value, freq="M")


def months_between(
    start: pd.Period,
    end: pd.Period,
) -> int:
    """
    Número entero de meses entre dos períodos mensuales.

    Ejemplo
    -------
    2020-01 -> 2020-07 = 6 meses.
    """
    start = as_month_period(start)
    end = as_month_period(end)

    return end.ordinal - start.ordinal


def get_target_period(
    accident_period: pd.Period,
    dev_m: int,
) -> pd.Period:
    """
    Período en el cual una cohorte alcanza dev_M.
    """
    accident_period = as_month_period(accident_period)

    return accident_period + dev_m


def first_possible_ml_valuation(
    scenario: str,
    first_accident_period: pd.Period = FIRST_ACCIDENT_PERIOD,
    min_mature_cohorts: int = MIN_MATURE_COHORTS_ML,
) -> pd.Period:
    """
    Primera fecha en la que existen al menos
    `min_mature_cohorts` cohortes maduras.
    """
    if scenario not in DEV_M:
        raise ValueError(
            f"Escenario desconocido: {scenario!r}. "
            f"Opciones válidas: {list(DEV_M)}"
        )

    dev_m = DEV_M[scenario]

    last_required_accident_period = (
        as_month_period(first_accident_period)
        + min_mature_cohorts
        - 1
    )

    return last_required_accident_period + dev_m


def build_eligibility_grid(
    accident_periods: Iterable,
    scenario: str,
    valuation_periods: Iterable = VALUATION_PERIODS,
    final_period: pd.Period = FINAL_PERIOD,
) -> pd.DataFrame:
    """
    Construye la cuadrícula temporal de elegibilidad.

    Una fila representa una combinación:

        scenario
        + accident_period
        + valuation_period

    Para cada combinación se determina:

    - la edad observable en la valuación;
    - si la cohorte ya existía;
    - si ya había alcanzado dev_M;
    - si su target habría sido revelado antes de FINAL_PERIOD;
    - si constituye una observación candidata a predicción.

    Importante
    ----------
    `is_target_revealed_by_end` es únicamente una variable
    diagnóstica.

    No interviene en la selección de observaciones para
    backtesting.

    Una observación puede predecirse siempre que:

        0 <= latest_dev_month < dev_M

    independientemente de que su target ocurra después de
    FINAL_PERIOD.

    La disponibilidad real del target se verificará
    posteriormente contra la trayectoria completa simulada,
    durante la fase de revelación.
    """
    if scenario not in DEV_M:
        raise ValueError(
            f"Escenario desconocido: {scenario!r}"
        )

    dev_m = DEV_M[scenario]

    final_period = as_month_period(
        final_period
    )

    accident_periods = [
        as_month_period(p)
        for p in accident_periods
    ]

    valuation_periods = [
        as_month_period(p)
        for p in valuation_periods
    ]

    rows = []

    for valuation_period in valuation_periods:

        for accident_period in accident_periods:

            latest_dev_month = months_between(
                accident_period,
                valuation_period,
            )

            target_period = get_target_period(
                accident_period,
                dev_m,
            )

            # La cohorte ya existe en la fecha de valuación.
            is_observed_at_valuation = (
                latest_dev_month >= 0
            )

            # La cohorte ya alcanzó dev_M.
            is_mature_at_valuation = (
                target_period <= valuation_period
            )

            # Diagnóstico únicamente.
            # NO determina si una observación puede predecirse.
            is_target_revealed_by_end = (
                target_period <= final_period
            )

            # Universo que realmente queremos predecir:
            # cohortes existentes pero todavía inmaduras.
            is_prediction_candidate = (
                is_observed_at_valuation
                and not is_mature_at_valuation
            )

            # Alias temporal para mantener compatibilidad
            # con el resto del proyecto.
            is_backtest_evaluable = (
                is_prediction_candidate
            )

            rows.append(
                {
                    "scenario": scenario,
                    "accident_period": accident_period,
                    "valuation_period": valuation_period,
                    "latest_dev_month": latest_dev_month,
                    "dev_M": dev_m,
                    "target_period": target_period,
                    "is_observed_at_valuation": (
                        is_observed_at_valuation
                    ),
                    "is_mature_at_valuation": (
                        is_mature_at_valuation
                    ),
                    "is_target_revealed_by_end": (
                        is_target_revealed_by_end
                    ),
                    "is_prediction_candidate": (
                        is_prediction_candidate
                    ),
                    "is_backtest_evaluable": (
                        is_backtest_evaluable
                    ),
                }
            )

    return pd.DataFrame(rows)


def get_backtest_rows(
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    """
    Devuelve las observaciones candidatas a predicción
    histórica.

    Una observación es candidata cuando la cohorte ya existe
    en la valuación pero todavía no ha alcanzado dev_M.
    """
    return (
        eligibility
        .loc[
            eligibility["is_prediction_candidate"]
        ]
        .copy()
        .reset_index(drop=True)
    )

def get_mature_cohorts(
    accident_periods: Iterable,
    scenario: str,
    valuation_period,
) -> pd.DataFrame:
    """
    Cohortes cuyo target ya era conocido en una valuación.

    Se utiliza para determinar qué accident periods pueden
    generar snapshots para entrenamiento.
    """
    if scenario not in DEV_M:
        raise ValueError(
            f"Escenario desconocido: {scenario!r}"
        )

    valuation_period = as_month_period(
        valuation_period
    )

    dev_m = DEV_M[scenario]

    rows = []

    for accident_period in accident_periods:

        accident_period = as_month_period(
            accident_period
        )

        target_period = get_target_period(
            accident_period,
            dev_m,
        )

        if target_period <= valuation_period:
            rows.append(
                {
                    "scenario": scenario,
                    "accident_period": accident_period,
                    "target_period": target_period,
                    "dev_M": dev_m,
                }
            )

    return pd.DataFrame(rows)