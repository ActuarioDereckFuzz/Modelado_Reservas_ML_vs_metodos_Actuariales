from __future__ import annotations

import pandas as pd


OCCURRENCE_COLUMNS = [
    "scenario",
    "accident_period",
    "month_index",
    "trend_factor",
    "seasonality_factor",
    "occurrence_noise",
    "expected_ultimate",
    "ultimate_loss",
]


def build_occurrence_view(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye una vista con una observación por periodo
    de ocurrencia.

    Las variables del proceso de ocurrencia se repiten
    posteriormente a través de las distintas edades de
    desarrollo. Esta función elimina dicha repetición para
    permitir analizar tendencia, estacionalidad y ruido.
    """

    missing = (
        set(OCCURRENCE_COLUMNS)
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # Verificar que los componentes de ocurrencia sean
    # constantes dentro de cada accident_period.
    # ---------------------------------------------------------
    component_columns = [
        "month_index",
        "trend_factor",
        "seasonality_factor",
        "occurrence_noise",
        "expected_ultimate",
        "ultimate_loss",
    ]

    consistency = (
        df.groupby("accident_period")[
            component_columns
        ]
        .nunique(dropna=False)
    )

    if (consistency > 1).any().any():
        raise ValueError(
            "Los componentes de ocurrencia no son "
            "constantes dentro de accident_period."
        )

    # ---------------------------------------------------------
    # Una fila por periodo de ocurrencia
    # ---------------------------------------------------------
    occurrence = (
        df[OCCURRENCE_COLUMNS]
        .drop_duplicates(
            subset=[
                "scenario",
                "accident_period",
            ]
        )
        .copy()
    )

    occurrence["accident_period"] = (
        pd.PeriodIndex(
            occurrence["accident_period"],
            freq="M",
        )
    )

    occurrence = occurrence.sort_values(
        [
            "scenario",
            "accident_period",
        ]
    ).reset_index(drop=True)

    occurrence["year"] = (
        occurrence["accident_period"].dt.year
    )

    occurrence["month"] = (
        occurrence["accident_period"].dt.month
    )

    return occurrence