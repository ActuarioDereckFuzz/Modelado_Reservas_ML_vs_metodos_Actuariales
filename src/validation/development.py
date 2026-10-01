from __future__ import annotations

import pandas as pd


def validate_development_patterns(
    development_factors: pd.DataFrame,
    min_share: float = 0.80,
) -> pd.DataFrame:
    """
    Valida que los factores edad-a-edad agregados presenten
    la dirección esperada para cada escenario.

    Reglas
    ------
    creciente:
        Al menos `min_share` de los factores deben ser > 1.

    decreciente:
        Al menos `min_share` de los factores deben ser < 1.

    mixto:
        En el tercio inicial predominan factores > 1.
        En el tercio final predominan factores < 1.

    La zona central no se utiliza para validar el escenario mixto,
    ya que corresponde naturalmente a la transición entre ambos
    comportamientos.
    """

    required = {
        "scenario",
        "dev_month",
        "volume_weighted_factor",
    }

    missing = required - set(development_factors.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing)}"
        )

    results = []

    # ---------------------------------------------------------
    # Creciente
    # ---------------------------------------------------------
    creciente = (
        development_factors
        .loc[
            development_factors["scenario"] == "creciente"
        ]
        .sort_values("dev_month")
    )

    share_above = (
        creciente["volume_weighted_factor"] > 1
    ).mean()

    results.append(
        {
            "scenario": "creciente",
            "check": "predominantly_above_1",
            "observed_share": share_above,
            "required_share": min_share,
            "passed": share_above >= min_share,
        }
    )

    # ---------------------------------------------------------
    # Decreciente
    # ---------------------------------------------------------
    decreciente = (
        development_factors
        .loc[
            development_factors["scenario"] == "decreciente"
        ]
        .sort_values("dev_month")
    )

    share_below = (
        decreciente["volume_weighted_factor"] < 1
    ).mean()

    results.append(
        {
            "scenario": "decreciente",
            "check": "predominantly_below_1",
            "observed_share": share_below,
            "required_share": min_share,
            "passed": share_below >= min_share,
        }
    )

    # ---------------------------------------------------------
    # Mixto
    # ---------------------------------------------------------
    mixto = (
        development_factors
        .loc[
            development_factors["scenario"] == "mixto"
        ]
        .sort_values("dev_month")
        .reset_index(drop=True)
    )

    n = len(mixto)
    third = n // 3

    early = mixto.iloc[:third]
    late = mixto.iloc[-third:]

    early_share_above = (
        early["volume_weighted_factor"] > 1
    ).mean()

    late_share_below = (
        late["volume_weighted_factor"] < 1
    ).mean()

    results.append(
        {
            "scenario": "mixto",
            "check": "early_development_above_1",
            "observed_share": early_share_above,
            "required_share": min_share,
            "passed": early_share_above >= min_share,
        }
    )

    results.append(
        {
            "scenario": "mixto",
            "check": "late_development_below_1",
            "observed_share": late_share_below,
            "required_share": min_share,
            "passed": late_share_below >= min_share,
        }
    )

    return pd.DataFrame(results)