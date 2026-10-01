"""
Componentes temporales utilizados en la simulación.

Este módulo modela los principales elementos que determinan el nivel
esperado de pérdidas de cada período de ocurrencia:

- tendencia;
- estacionalidad;
- variación aleatoria.

Estas funciones no contienen ninguna lógica relacionada con el patrón
de desarrollo del Loss Incurred. Esa responsabilidad corresponde al
módulo ``development.py``.
"""

import numpy as np
import pandas as pd


def trend_factor(
    month_index: np.ndarray,
    annual_trend: float,
) -> np.ndarray:
    """
    Calcula el factor de tendencia acumulada para cada mes.

    Se utiliza capitalización compuesta a partir de una tasa anual.

    Parameters
    ----------
    month_index : np.ndarray
        Número de meses transcurridos desde el inicio de la simulación.

    annual_trend : float
        Tasa anual de tendencia.

    Returns
    -------
    np.ndarray
        Factor multiplicativo de tendencia.

    Notes
    -----
    El factor utilizado es:

        (1 + annual_trend) ** (month_index / 12)

    Por tanto, para ``month_index = 0`` el factor es exactamente 1.
    """

    month_index = np.asarray(month_index)

    return (1.0 + annual_trend) ** (month_index / 12.0)


def seasonality_factor(
    periods: pd.PeriodIndex,
    amplitude: float,
) -> np.ndarray:
    """
    Genera un patrón estacional mensual suave.

    La estacionalidad se modela mediante una función sinusoidal anual.

    Parameters
    ----------
    periods : pd.PeriodIndex
        Períodos mensuales de ocurrencia.

    amplitude : float
        Amplitud de la estacionalidad.

    Returns
    -------
    np.ndarray
        Factores multiplicativos centrados aproximadamente alrededor
        de 1.

    Examples
    --------
    Con ``amplitude = 0.08`` los factores oscilarán aproximadamente
    entre 0.92 y 1.08.
    """

    months = periods.month.to_numpy()

    return 1.0 + amplitude * np.sin(
        2.0 * np.pi * (months - 1) / 12.0
    )


def lognormal_noise(
    size: int | tuple[int, ...],
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Genera ruido multiplicativo lognormal con media cercana a 1.

    El uso de ruido lognormal presenta dos ventajas para este proyecto:

    1. garantiza factores estrictamente positivos;
    2. representa mejor fluctuaciones relativas que ruido aditivo.

    Parameters
    ----------
    size : int or tuple[int, ...]
        Dimensiones del arreglo generado.

    sigma : float
        Desviación estándar del componente normal subyacente.

    rng : np.random.Generator
        Generador pseudoaleatorio de NumPy.

    Returns
    -------
    np.ndarray
        Factores aleatorios positivos.

    Notes
    -----
    Se utiliza ``mean = -sigma**2 / 2`` para que la esperanza
    aproximada del factor lognormal sea igual a 1.
    """

    if sigma < 0:
        raise ValueError("sigma debe ser mayor o igual que 0.")

    return rng.lognormal(
        mean=-0.5 * sigma**2,
        sigma=sigma,
        size=size,
    )