from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "scenario",
    "accident_period",
    "dev_month",
    "calendar_period",
    "dev_M",
    "month_index",
    "trend_factor",
    "seasonality_factor",
    "occurrence_noise",
    "expected_ultimate",
    "ultimate_loss",
    "development_factor",
    "development_noise",
    "expected_loss_incurred",
    "loss_incurred",
]


def _to_month_period(series: pd.Series) -> pd.PeriodIndex:
    """
    Convierte una serie de fechas o periodos a PeriodIndex mensual.
    """
    return pd.PeriodIndex(series, freq="M")


def _check_row(
    scenario: str,
    check: str,
    passed: bool,
    detail: str,
) -> dict:
    """
    Estandariza el formato de salida de cada validación.
    """
    return {
        "scenario": scenario,
        "check": check,
        "passed": bool(passed),
        "detail": detail,
    }


def validate_base_structure(
    df: pd.DataFrame,
    scenario: str,
    expected_dev_M: int,
    expected_start: str = "2015-01",
    expected_end: str = "2025-12",
    rtol: float = 1e-10,
    atol: float = 1e-8,
) -> pd.DataFrame:
    """
    Ejecuta validaciones estructurales sobre una base simulada
    de Loss Incurred.

    Parameters
    ----------
    df:
        Base larga simulada.
    scenario:
        Nombre esperado del escenario.
    expected_dev_M:
        Horizonte de madurez definido para el escenario.
    expected_start:
        Primer periodo de ocurrencia esperado.
    expected_end:
        Último periodo de ocurrencia esperado.
    rtol, atol:
        Tolerancias utilizadas en comparaciones numéricas.

    Returns
    -------
    pd.DataFrame
        Tabla con el resultado de cada validación.
    """

    results = []

    # ---------------------------------------------------------
    # 1. Columnas requeridas
    # ---------------------------------------------------------
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))

    results.append(
        _check_row(
            scenario,
            "required_columns",
            len(missing_columns) == 0,
            (
                "Todas las columnas requeridas están presentes."
                if not missing_columns
                else f"Columnas faltantes: {missing_columns}"
            ),
        )
    )

    # Si faltan columnas esenciales, las demás comprobaciones
    # podrían producir errores poco informativos.
    if missing_columns:
        return pd.DataFrame(results)

    # ---------------------------------------------------------
    # 2. Base no vacía
    # ---------------------------------------------------------
    results.append(
        _check_row(
            scenario,
            "non_empty",
            not df.empty,
            f"Número de observaciones: {len(df):,}",
        )
    )

    # ---------------------------------------------------------
    # 3. Escenario correcto
    # ---------------------------------------------------------
    observed_scenarios = sorted(df["scenario"].dropna().unique().tolist())

    results.append(
        _check_row(
            scenario,
            "scenario_consistency",
            observed_scenarios == [scenario],
            f"Escenarios observados: {observed_scenarios}",
        )
    )

    # ---------------------------------------------------------
    # 4. Valores faltantes
    # ---------------------------------------------------------
    null_count = int(df[REQUIRED_COLUMNS].isna().sum().sum())

    results.append(
        _check_row(
            scenario,
            "no_missing_values",
            null_count == 0,
            f"Valores faltantes encontrados: {null_count}",
        )
    )

    # ---------------------------------------------------------
    # 5. Clave única
    # scenario + accident_period + dev_month
    # ---------------------------------------------------------
    key = ["scenario", "accident_period", "dev_month"]

    duplicate_count = int(df.duplicated(key).sum())

    results.append(
        _check_row(
            scenario,
            "unique_key",
            duplicate_count == 0,
            f"Claves duplicadas: {duplicate_count}",
        )
    )

    # ---------------------------------------------------------
    # 6. dev_M constante y consistente con el diseño
    # ---------------------------------------------------------
    observed_dev_M = sorted(df["dev_M"].dropna().unique().tolist())

    results.append(
        _check_row(
            scenario,
            "dev_M_consistency",
            observed_dev_M == [expected_dev_M],
            (
                f"dev_M observado: {observed_dev_M}; "
                f"dev_M esperado: {expected_dev_M}"
            ),
        )
    )

    # ---------------------------------------------------------
    # 7. Rango de edades de desarrollo
    # ---------------------------------------------------------
    dev_min = int(df["dev_month"].min())
    dev_max = int(df["dev_month"].max())

    expected_development = set(range(expected_dev_M + 1))
    observed_development = set(df["dev_month"].astype(int).unique())

    complete_dev_range = observed_development == expected_development

    results.append(
        _check_row(
            scenario,
            "development_range",
            (
                dev_min == 0
                and dev_max == expected_dev_M
                and complete_dev_range
            ),
            (
                f"Rango observado: {dev_min}–{dev_max}; "
                f"rango esperado: 0–{expected_dev_M}"
            ),
        )
    )

    # ---------------------------------------------------------
    # 8. Periodos de ocurrencia
    # ---------------------------------------------------------
    accident_period = _to_month_period(df["accident_period"])

    observed_accident_periods = pd.PeriodIndex(
        accident_period.unique(),
        freq="M",
    ).sort_values()

    expected_accident_periods = pd.period_range(
        expected_start,
        expected_end,
        freq="M",
    )

    occurrence_ok = observed_accident_periods.equals(
        expected_accident_periods
    )

    results.append(
        _check_row(
            scenario,
            "accident_period_range",
            occurrence_ok,
            (
                f"Periodo observado: "
                f"{observed_accident_periods.min()}–"
                f"{observed_accident_periods.max()}; "
                f"{len(observed_accident_periods)} meses."
            ),
        )
    )

    # ---------------------------------------------------------
    # 9. Cada periodo de ocurrencia posee todo su desarrollo
    # ---------------------------------------------------------
    dev_counts = (
        df.groupby("accident_period")["dev_month"]
        .nunique()
    )

    expected_count = expected_dev_M + 1

    complete_by_accident = bool(
        (dev_counts == expected_count).all()
    )

    results.append(
        _check_row(
            scenario,
            "complete_development_by_accident_period",
            complete_by_accident,
            (
                f"Edades esperadas por periodo: {expected_count}; "
                f"mínimo observado: {dev_counts.min()}; "
                f"máximo observado: {dev_counts.max()}."
            ),
        )
    )

    # ---------------------------------------------------------
    # 10. Coherencia del periodo calendario
    #
    # calendar_period = accident_period + dev_month
    # ---------------------------------------------------------
    accident = _to_month_period(df["accident_period"])
    calendar = _to_month_period(df["calendar_period"])

    expected_calendar = pd.PeriodIndex(
        [
            period + int(dev)
            for period, dev in zip(
                accident,
                df["dev_month"],
            )
        ],
        freq="M",
    )

    calendar_ok = bool(
        np.all(calendar == expected_calendar)
    )

    results.append(
        _check_row(
            scenario,
            "calendar_period_consistency",
            calendar_ok,
            (
                "calendar_period coincide con "
                "accident_period + dev_month."
                if calendar_ok
                else "Se encontraron periodos calendario inconsistentes."
            ),
        )
    )

    # ---------------------------------------------------------
    # 11. Importes estrictamente positivos
    # ---------------------------------------------------------
    monetary_columns = [
        "expected_ultimate",
        "ultimate_loss",
        "expected_loss_incurred",
        "loss_incurred",
    ]

    non_positive = {
        col: int((df[col] <= 0).sum())
        for col in monetary_columns
    }

    positive_ok = all(value == 0 for value in non_positive.values())

    results.append(
        _check_row(
            scenario,
            "positive_amounts",
            positive_ok,
            f"Observaciones no positivas: {non_positive}",
        )
    )

    # ---------------------------------------------------------
    # 12. Factores de simulación positivos
    # ---------------------------------------------------------
    factor_columns = [
        "trend_factor",
        "seasonality_factor",
        "occurrence_noise",
        "development_factor",
        "development_noise",
    ]

    invalid_factors = {
        col: int((df[col] <= 0).sum())
        for col in factor_columns
    }

    factors_ok = all(
        value == 0
        for value in invalid_factors.values()
    )

    results.append(
        _check_row(
            scenario,
            "positive_simulation_factors",
            factors_ok,
            f"Factores no positivos: {invalid_factors}",
        )
    )

    # ---------------------------------------------------------
    # 13. Condición de madurez:
    # Loss Incurred en dev_M = Ultimate
    # ---------------------------------------------------------
    mature = df.loc[
        df["dev_month"] == expected_dev_M
    ]

    maturity_ok = np.allclose(
        mature["loss_incurred"],
        mature["ultimate_loss"],
        rtol=rtol,
        atol=atol,
    )

    max_maturity_error = float(
        np.abs(
            mature["loss_incurred"]
            - mature["ultimate_loss"]
        ).max()
    )

    results.append(
        _check_row(
            scenario,
            "maturity_equals_ultimate",
            maturity_ok,
            (
                "LI(dev_M) coincide con Ultimate. "
                f"Máxima diferencia absoluta: "
                f"{max_maturity_error:.6g}"
            ),
        )
    )

    return pd.DataFrame(results)


def validate_all_scenarios(
    simulations: dict[str, pd.DataFrame],
    expected_dev_M: dict[str, int],
    expected_start: str = "2015-01",
    expected_end: str = "2025-12",
) -> pd.DataFrame:
    """
    Ejecuta validate_base_structure para todos los escenarios.
    """

    results = []

    for scenario, df in simulations.items():

        if scenario not in expected_dev_M:
            raise KeyError(
                f"No existe dev_M esperado para '{scenario}'."
            )

        validation = validate_base_structure(
            df=df,
            scenario=scenario,
            expected_dev_M=expected_dev_M[scenario],
            expected_start=expected_start,
            expected_end=expected_end,
        )

        results.append(validation)

    return pd.concat(
        results,
        ignore_index=True,
    )