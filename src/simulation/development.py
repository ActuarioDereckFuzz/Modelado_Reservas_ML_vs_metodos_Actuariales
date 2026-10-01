"""
Curvas de desarrollo del Loss Incurred.

Cada función devuelve un multiplicador acumulado de desarrollo para
las edades comprendidas entre 0 y ``dev_M``.

Interpretación
--------------
Si el Loss Incurred definitivo de un período es U y la curva devuelve:

    development_factor = 0.75

entonces:

    Loss Incurred acumulado = 0.75 * U

Las curvas están construidas de forma que el factor en ``dev_M`` sea
exactamente igual a 1.0.

Esto permite interpretar ``ultimate_loss`` como el nivel definitivo
de la simulación.
"""

import numpy as np

from .config import ScenarioConfig


def increasing_curve(
    dev_month: np.ndarray,
    dev_M: int,
    start_level: float,
    speed: float = 3.0,
) -> np.ndarray:
    """
    Curva acumulada monótonamente creciente.

    La curva parte de ``start_level`` y converge suavemente a 1.

    Parameters
    ----------
    dev_month : np.ndarray
        Edades de desarrollo.

    dev_M : int
        Edad de madurez.

    start_level : float
        Nivel relativo inicial.

    speed : float, default=3.0
        Controla la rapidez con la que se reconoce el Loss Incurred.

    Returns
    -------
    np.ndarray
        Factores acumulados de desarrollo.
    """

    x = np.asarray(dev_month, dtype=float) / dev_M

    raw = 1.0 - np.exp(-speed * x)
    raw_M = 1.0 - np.exp(-speed)

    normalized = raw / raw_M

    return start_level + (1.0 - start_level) * normalized


def decreasing_curve(
    dev_month: np.ndarray,
    dev_M: int,
    start_level: float,
    speed: float = 3.0,
) -> np.ndarray:
    """
    Curva acumulada monótonamente decreciente.

    El Loss Incurred inicia por encima de su valor definitivo y
    posteriormente se ajusta hasta converger a 1.

    Returns
    -------
    np.ndarray
        Factores acumulados de desarrollo.
    """

    x = np.asarray(dev_month, dtype=float) / dev_M

    raw = np.exp(-speed * x)
    raw_M = np.exp(-speed)

    normalized = (raw - raw_M) / (1.0 - raw_M)

    return 1.0 + (start_level - 1.0) * normalized


def mixed_curve(
    dev_month: np.ndarray,
    dev_M: int,
    start_level: float,
    peak_level: float,
    peak_month: int,
) -> np.ndarray:
    """
    Curva de desarrollo con crecimiento temprano y ajuste posterior.

    La curva se divide conceptualmente en dos regiones:

    1. ``0 <= dev_month <= peak_month``:
       crecimiento desde ``start_level`` hasta ``peak_level``.

    2. ``peak_month < dev_month <= dev_M``:
       disminución desde ``peak_level`` hasta 1.

    Se utilizan funciones ``smoothstep`` para evitar cambios abruptos
    de pendiente.

    Parameters
    ----------
    dev_month : np.ndarray
        Edades mensuales de desarrollo.

    dev_M : int
        Edad de madurez.

    start_level : float
        Nivel relativo inicial.

    peak_level : float
        Nivel máximo alcanzado.

    peak_month : int
        Mes aproximado del máximo.

    Returns
    -------
    np.ndarray
        Factores acumulados de desarrollo.
    """

    dev_month = np.asarray(dev_month, dtype=float)

    if not 0 < peak_month < dev_M:
        raise ValueError(
            "peak_month debe estar estrictamente entre 0 y dev_M."
        )

    result = np.empty_like(dev_month)

    early = dev_month <= peak_month
    late = ~early

    # ---------------------------------------------------------
    # Primera etapa: crecimiento.
    # ---------------------------------------------------------

    x_early = dev_month[early] / peak_month

    smooth_early = (
        3.0 * x_early**2
        - 2.0 * x_early**3
    )

    result[early] = (
        start_level
        + (peak_level - start_level) * smooth_early
    )

    # ---------------------------------------------------------
    # Segunda etapa: disminución hacia la madurez.
    # ---------------------------------------------------------

    x_late = (
        (dev_month[late] - peak_month)
        / (dev_M - peak_month)
    )

    smooth_late = (
        3.0 * x_late**2
        - 2.0 * x_late**3
    )

    result[late] = (
        peak_level
        + (1.0 - peak_level) * smooth_late
    )

    return result


def development_curve(
    dev_month: np.ndarray,
    scenario: ScenarioConfig,
) -> np.ndarray:
    """
    Devuelve la curva correspondiente al escenario solicitado.

    Esta función constituye la interfaz principal del módulo. Los
    demás componentes del proyecto no necesitan conocer qué función
    específica utiliza cada escenario.

    Parameters
    ----------
    dev_month : np.ndarray
        Edades mensuales de desarrollo.

    scenario : ScenarioConfig
        Configuración del escenario.

    Returns
    -------
    np.ndarray
        Factores acumulados de desarrollo.

    Raises
    ------
    ValueError
        Si el escenario recibido no está implementado.
    """

    if scenario.name == "creciente":

        return increasing_curve(
            dev_month=dev_month,
            dev_M=scenario.dev_M,
            start_level=scenario.start_level,
        )

    if scenario.name == "decreciente":

        return decreasing_curve(
            dev_month=dev_month,
            dev_M=scenario.dev_M,
            start_level=scenario.start_level,
        )

    if scenario.name == "mixto":

        if (
            scenario.peak_level is None
            or scenario.peak_month is None
        ):
            raise ValueError(
                "El escenario mixto requiere peak_level y peak_month."
            )

        return mixed_curve(
            dev_month=dev_month,
            dev_M=scenario.dev_M,
            start_level=scenario.start_level,
            peak_level=scenario.peak_level,
            peak_month=scenario.peak_month,
        )

    raise ValueError(
        f"Escenario no reconocido: {scenario.name!r}"
    )