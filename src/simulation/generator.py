"""
Generador principal de bases simuladas de Loss Incurred.

Este módulo combina:

- períodos de ocurrencia;
- tendencia;
- estacionalidad;
- ruido por período;
- curvas de desarrollo;
- ruido por observación de desarrollo.

El resultado es una base en formato longitudinal (long format), con una
fila por combinación:

    accident_period x dev_month

Esta representación es deliberada porque posteriormente puede
transformarse fácilmente en:

- triángulos acumulados;
- triángulos incrementales;
- diagonales de valuación;
- muestras para modelos estadísticos y de machine learning;
- ventanas de backtesting temporal.
"""

import numpy as np
import pandas as pd

from .components import (
    lognormal_noise,
    seasonality_factor,
    trend_factor,
)

from .config import (
    SCENARIO_CONFIGS,
    ScenarioName,
    SimulationConfig,
)

from .development import development_curve


def create_occurrence_table(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Genera el nivel definitivo esperado para cada período de ocurrencia.

    Returns
    -------
    pd.DataFrame
        Tabla con una fila por mes de ocurrencia y las columnas:

        - accident_period
        - month_index
        - trend_factor
        - seasonality_factor
        - occurrence_noise
        - expected_ultimate
        - ultimate_loss
    """

    periods = pd.period_range(
        start=config.start_period,
        end=config.end_period,
        freq="M",
    )

    month_index = np.arange(len(periods))

    trend = trend_factor(
        month_index=month_index,
        annual_trend=config.annual_trend,
    )

    seasonality = seasonality_factor(
        periods=periods,
        amplitude=config.seasonality_amplitude,
    )

    occurrence_noise = lognormal_noise(
        size=len(periods),
        sigma=config.occurrence_noise_sigma,
        rng=rng,
    )

    expected_ultimate = (
        config.base_loss
        * trend
        * seasonality
    )

    ultimate_loss = (
        expected_ultimate
        * occurrence_noise
    )

    return pd.DataFrame(
        {
            "accident_period": periods,
            "month_index": month_index,
            "trend_factor": trend,
            "seasonality_factor": seasonality,
            "occurrence_noise": occurrence_noise,
            "expected_ultimate": expected_ultimate,
            "ultimate_loss": ultimate_loss,
        }
    )


def simulate_scenario(
    scenario_name: ScenarioName,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    """
    Simula la base longitudinal de un escenario de Loss Incurred.

    Parameters
    ----------
    scenario_name : {"creciente", "decreciente", "mixto"}
        Escenario que se desea simular.

    config : SimulationConfig, optional
        Configuración general de la simulación.

        Si no se proporciona se utiliza ``SimulationConfig()``.

    Returns
    -------
    pd.DataFrame
        Base simulada con una observación por período de ocurrencia
        y edad de desarrollo.

    Notes
    -----
    La función genera la trayectoria COMPLETA de desarrollo para cada
    período de ocurrencia.

    Por tanto, algunas observaciones tendrán ``calendar_period`` posterior
    al último período de ocurrencia de la simulación.

    Esto es intencional.

    En las fases posteriores, una valuación en una fecha determinada
    deberá construirse filtrando:

        calendar_period <= valuation_period

    Esta separación entre:

        verdad simulada

    y

        información disponible a una fecha de valuación

    es fundamental para realizar posteriormente backtesting temporal sin
    fuga de información.
    """

    if config is None:
        config = SimulationConfig()

    scenario = SCENARIO_CONFIGS[scenario_name]

    # Generadores separados por escenario para mantener reproducibilidad.
    scenario_offset = {
        "creciente": 0,
        "decreciente": 1,
        "mixto": 2,
    }

    rng = np.random.default_rng(
        config.random_seed + scenario_offset[scenario_name]
    )

    occurrence = create_occurrence_table(
        config=config,
        rng=rng,
    )

    dev_months = np.arange(
        0,
        scenario.dev_M + 1,
    )

    # Producto cartesiano:
    # accident_period x dev_month
    base = (
        occurrence
        .merge(
            pd.DataFrame({"dev_month": dev_months}),
            how="cross",
        )
    )

    # ---------------------------------------------------------
    # Período calendario.
    # ---------------------------------------------------------

    base["calendar_period"] = (
        base["accident_period"]
        + base["dev_month"]
    )

    # ---------------------------------------------------------
    # Patrón teórico de desarrollo.
    # ---------------------------------------------------------

    base["development_factor"] = development_curve(
        dev_month=base["dev_month"].to_numpy(),
        scenario=scenario,
    )

    # ---------------------------------------------------------
    # Loss Incurred esperado antes del ruido de desarrollo.
    # ---------------------------------------------------------

    base["expected_loss_incurred"] = (
        base["ultimate_loss"]
        * base["development_factor"]
    )

    # ---------------------------------------------------------
    # Variación específica de cada celda.
    # ---------------------------------------------------------

    development_noise = lognormal_noise(
        size=len(base),
        sigma=config.development_noise_sigma,
        rng=rng,
    )

    # En la madurez hacemos ruido = 1 para mantener:
    #
    # Loss Incurred(dev_M) = ultimate_loss
    #
    is_mature = (
        base["dev_month"] == scenario.dev_M
    )

    development_noise[is_mature] = 1.0

    base["development_noise"] = development_noise

    base["loss_incurred"] = (
        base["expected_loss_incurred"]
        * base["development_noise"]
    )

    # ---------------------------------------------------------
    # Metadatos.
    # ---------------------------------------------------------

    base["scenario"] = scenario.name
    base["dev_M"] = scenario.dev_M

    # ---------------------------------------------------------
    # Orden final.
    # ---------------------------------------------------------

    column_order = [
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

    return (
        base[column_order]
        .sort_values(
            ["accident_period", "dev_month"]
        )
        .reset_index(drop=True)
    )


def simulate_all_scenarios(
    config: SimulationConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Genera las tres bases maestras del proyecto.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario con las claves:

        - ``creciente``
        - ``decreciente``
        - ``mixto``

        Cada valor contiene la base longitudinal correspondiente.
    """

    return {
        scenario_name: simulate_scenario(
            scenario_name=scenario_name,
            config=config,
        )
        for scenario_name in SCENARIO_CONFIGS
    }