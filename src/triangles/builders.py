from __future__ import annotations

import numpy as np
import pandas as pd


def build_cumulative_triangle(
    df: pd.DataFrame,
    valuation_period: str | pd.Period,
    value_col: str = "loss_incurred",
) -> pd.DataFrame:
    """
    Construye un triángulo acumulado de Loss Incurred
    a una fecha de valuación determinada.

    Solo se utilizan observaciones cuyo calendar_period
    sea menor o igual a valuation_period.

    Parameters
    ----------
    df:
        Base simulada en formato largo.

    valuation_period:
        Periodo de valuación mensual. Por ejemplo: '2025-12'.

    value_col:
        Columna que contiene el valor acumulado.

    Returns
    -------
    pd.DataFrame
        Triángulo con:
        - filas: accident_period
        - columnas: dev_month
        - valores: value_col
    """

    required = {
        "accident_period",
        "dev_month",
        "calendar_period",
        value_col,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing)}"
        )

    valuation_period = pd.Period(
        valuation_period,
        freq="M",
    )

    data = df.copy()

    data["accident_period"] = pd.PeriodIndex(
        data["accident_period"],
        freq="M",
    )

    data["calendar_period"] = pd.PeriodIndex(
        data["calendar_period"],
        freq="M",
    )

    # ---------------------------------------------------------
    # Corte de información
    # ---------------------------------------------------------
    data = data.loc[
        data["calendar_period"] <= valuation_period
    ].copy()

    if data.empty:
        raise ValueError(
            "No existen observaciones disponibles "
            "para la fecha de valuación indicada."
        )

    # ---------------------------------------------------------
    # Construcción del triángulo acumulado
    # ---------------------------------------------------------
    triangle = data.pivot(
        index="accident_period",
        columns="dev_month",
        values=value_col,
    )

    triangle = triangle.sort_index()
    triangle = triangle.reindex(
        sorted(triangle.columns),
        axis=1,
    )

    triangle.columns.name = "dev_month"
    triangle.index.name = "accident_period"

    return triangle


def cumulative_to_incremental(
    cumulative_triangle: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convierte un triángulo acumulado en incremental.

    Para j = 0:

        I_{i,0} = C_{i,0}

    Para j > 0:

        I_{i,j} = C_{i,j} - C_{i,j-1}

    Los NaN estructurales del triángulo se conservan.
    """

    if cumulative_triangle.empty:
        raise ValueError(
            "El triángulo acumulado está vacío."
        )

    incremental = cumulative_triangle.diff(axis=1)

    first_col = cumulative_triangle.columns[0]

    incremental[first_col] = (
        cumulative_triangle[first_col]
    )

    # Mantener exactamente la misma forma y orden
    incremental = incremental[
        cumulative_triangle.columns
    ]

    incremental.index.name = (
        cumulative_triangle.index.name
    )

    incremental.columns.name = (
        cumulative_triangle.columns.name
    )

    return incremental


def incremental_to_cumulative(
    incremental_triangle: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruye el triángulo acumulado a partir
    del triángulo incremental.
    """

    if incremental_triangle.empty:
        raise ValueError(
            "El triángulo incremental está vacío."
        )

    cumulative = incremental_triangle.cumsum(
        axis=1
    )

    # pandas puede propagar valores después de un NaN
    # dependiendo de la operación. Restauramos la máscara
    # estructural del triángulo original.
    cumulative = cumulative.where(
        incremental_triangle.notna()
    )

    cumulative.index.name = (
        incremental_triangle.index.name
    )

    cumulative.columns.name = (
        incremental_triangle.columns.name
    )

    return cumulative


def build_triangle_pair(
    df: pd.DataFrame,
    valuation_period: str | pd.Period,
    value_col: str = "loss_incurred",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye conjuntamente los triángulos
    acumulado e incremental.
    """

    cumulative = build_cumulative_triangle(
        df=df,
        valuation_period=valuation_period,
        value_col=value_col,
    )

    incremental = cumulative_to_incremental(
        cumulative
    )

    return cumulative, incremental