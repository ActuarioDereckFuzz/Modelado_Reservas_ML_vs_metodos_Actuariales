from __future__ import annotations

import pandas as pd


def extract_latest_diagonal(
    cumulative_triangle: pd.DataFrame,
    valuation_period: str | pd.Period,
) -> pd.DataFrame:
    """
    Extrae la observación más reciente disponible para cada
    periodo de ocurrencia de un triángulo acumulado.

    Para periodos todavía inmaduros, la observación seleccionada
    pertenece a la fecha de valuación.

    Para periodos que ya alcanzaron la edad máxima del triángulo,
    se conserva su valor maduro en la última edad disponible.

    Parameters
    ----------
    cumulative_triangle:
        Triángulo acumulado con accident_period como índice
        y dev_month como columnas.

    valuation_period:
        Fecha mensual de valuación.

    Returns
    -------
    pd.DataFrame
        Tabla con:
        - accident_period
        - latest_dev_month
        - latest_calendar_period
        - latest_loss_incurred
        - is_mature
    """

    if cumulative_triangle.empty:
        raise ValueError(
            "El triángulo acumulado está vacío."
        )

    valuation_period = pd.Period(
        valuation_period,
        freq="M",
    )

    max_dev = int(max(cumulative_triangle.columns))

    records = []

    for accident_period, row in cumulative_triangle.iterrows():

        accident_period = pd.Period(
            accident_period,
            freq="M",
        )

        observed = row.dropna()

        if observed.empty:
            continue

        latest_dev = int(observed.index.max())
        latest_value = float(observed.loc[latest_dev])

        latest_calendar = (
            accident_period + latest_dev
        )

        records.append(
            {
                "accident_period": accident_period,
                "latest_dev_month": latest_dev,
                "latest_calendar_period": latest_calendar,
                "latest_loss_incurred": latest_value,
                "is_mature": latest_dev == max_dev,
            }
        )

    diagonal = pd.DataFrame(records)

    return diagonal