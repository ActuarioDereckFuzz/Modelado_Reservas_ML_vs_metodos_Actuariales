"""
Configuración de la simulación de Loss Incurred.

Este módulo contiene las estructuras de configuración utilizadas para
generar los escenarios simulados del proyecto.

La separación entre configuración y lógica permite:

1. Reproducir simulaciones mediante semillas aleatorias.
2. Modificar parámetros sin alterar las funciones de simulación.
3. Documentar explícitamente las hipótesis de cada escenario.
4. Mantener los notebooks libres de parámetros dispersos.

Escenarios considerados
------------------------
- creciente:
    El Loss Incurred acumulado aumenta progresivamente hasta alcanzar
    su nivel de madurez.

- decreciente:
    El Loss Incurred parte de un nivel superior al definitivo y se
    ajusta gradualmente hacia abajo.

- mixto:
    El Loss Incurred presenta crecimiento en edades tempranas,
    sobrepasa temporalmente el nivel definitivo y posteriormente
    disminuye hacia la madurez.
"""

from dataclasses import dataclass
from typing import Literal


ScenarioName = Literal["creciente", "decreciente", "mixto"]


@dataclass(frozen=True)
class SimulationConfig:
    """
    Parámetros generales de la simulación.

    Parameters
    ----------
    start_period : str
        Primer mes de ocurrencia simulado en formato ``YYYY-MM``.

    end_period : str
        Último mes de ocurrencia simulado.

    base_loss : float
        Nivel esperado de Loss Incurred definitivo para el primer
        período de ocurrencia antes de tendencia y estacionalidad.

    annual_trend : float
        Tendencia anual esperada del costo.

        Por ejemplo, ``0.04`` representa una tendencia aproximada
        de 4 % anual.

    seasonality_amplitude : float
        Magnitud relativa de la estacionalidad mensual.

        ``0.08`` representa oscilaciones estacionales aproximadas
        de ±8 % alrededor del nivel esperado.

    occurrence_noise_sigma : float
        Desviación estándar del ruido multiplicativo asociado al
        mes de ocurrencia.

    development_noise_sigma : float
        Desviación estándar del ruido multiplicativo aplicado a cada
        observación de desarrollo.

    random_seed : int
        Semilla utilizada por el generador pseudoaleatorio.
    """

    start_period: str = "2015-01"
    end_period: str = "2025-12"

    base_loss: float = 1_000_000.0

    annual_trend: float = 0.04
    seasonality_amplitude: float = 0.08

    occurrence_noise_sigma: float = 0.08
    development_noise_sigma: float = 0.015

    random_seed: int = 42


@dataclass(frozen=True)
class ScenarioConfig:
    """
    Configuración específica de un escenario de desarrollo.

    Parameters
    ----------
    name : ScenarioName
        Nombre identificador del escenario.

    dev_M : int
        Edad máxima de desarrollo considerada como madurez.

    description : str
        Descripción conceptual del patrón esperado.

    start_level : float
        Proporción aproximada del Loss Incurred definitivo observada
        en ``dev_month = 0``.

    peak_level : float, optional
        Nivel máximo utilizado principalmente por el escenario mixto.

    peak_month : int, optional
        Edad de desarrollo en la que aproximadamente se alcanza
        ``peak_level``.
    """

    name: ScenarioName
    dev_M: int
    description: str

    start_level: float
    peak_level: float | None = None
    peak_month: int | None = None


SCENARIO_CONFIGS: dict[ScenarioName, ScenarioConfig] = {

    "creciente": ScenarioConfig(
        name="creciente",
        dev_M=60,
        start_level=0.55,
        description=(
            "Desarrollo acumulado creciente. El Loss Incurred parte "
            "por debajo de su nivel definitivo y converge gradualmente "
            "a la madurez."
        ),
    ),

    "decreciente": ScenarioConfig(
        name="decreciente",
        dev_M=48,
        start_level=1.25,
        description=(
            "Desarrollo acumulado decreciente. El Loss Incurred parte "
            "por encima de su nivel definitivo debido a estimaciones "
            "inicialmente conservadoras y posteriormente se ajusta "
            "hacia la madurez."
        ),
    ),

    "mixto": ScenarioConfig(
        name="mixto",
        dev_M=48,
        start_level=0.65,
        peak_level=1.15,
        peak_month=18,
        description=(
            "Desarrollo mixto. El Loss Incurred aumenta durante las "
            "edades tempranas, alcanza un nivel superior al definitivo "
            "y posteriormente disminuye hasta converger a la madurez."
        ),
    ),
}