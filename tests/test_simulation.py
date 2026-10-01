"""
Pruebas unitarias del módulo ``src.simulation``.

Estas pruebas verifican que la simulación utilizada en el proyecto
cumpla las propiedades de diseño definidas para los tres escenarios
de Loss Incurred.

Las pruebas se concentran en cuatro aspectos:

1. Configuración
   Verificar que los horizontes de madurez y parámetros principales
   correspondan con el diseño del proyecto.

2. Curvas de desarrollo
   Comprobar matemáticamente que:
       - creciente: aumenta hasta 1;
       - decreciente: disminuye hasta 1;
       - mixto: aumenta, alcanza un máximo y posteriormente disminuye.

3. Generación de datos
   Verificar dimensiones, períodos, positividad y consistencia entre
   accident_period, dev_month y calendar_period.

4. Reproducibilidad
   Garantizar que una misma semilla produzca exactamente la misma
   simulación.

Importante
----------
La monotonicidad se verifica sobre ``development_factor`` y no sobre
``loss_incurred``.

``development_factor`` representa el patrón teórico de desarrollo.

``loss_incurred`` incorpora además ruido aleatorio de desarrollo, por
lo que pequeñas desviaciones respecto al patrón teórico son esperables
y forman parte del diseño de la simulación.
"""

import numpy as np
import pandas as pd
import pytest

from src.simulation import (
    SCENARIO_CONFIGS,
    SimulationConfig,
    create_occurrence_table,
    simulate_all_scenarios,
    simulate_scenario,
)

from src.simulation.components import (
    lognormal_noise,
    seasonality_factor,
    trend_factor,
)

from src.simulation.development import (
    development_curve,
)


# ============================================================
# CONFIGURACIÓN AUXILIAR PARA PRUEBAS
# ============================================================

@pytest.fixture
def small_config() -> SimulationConfig:
    """
    Configuración reducida utilizada en varias pruebas.

    Se utiliza un solo año de períodos de ocurrencia para mantener
    las pruebas rápidas y fáciles de inspeccionar.
    """

    return SimulationConfig(
        start_period="2020-01",
        end_period="2020-12",
        base_loss=1_000_000.0,
        annual_trend=0.04,
        seasonality_amplitude=0.08,
        occurrence_noise_sigma=0.08,
        development_noise_sigma=0.015,
        random_seed=123,
    )


# ============================================================
# 1. PRUEBAS DE CONFIGURACIÓN
# ============================================================

def test_scenario_configuration():
    """
    Verifica los parámetros estructurales de los tres escenarios.
    """

    increasing = SCENARIO_CONFIGS["creciente"]
    decreasing = SCENARIO_CONFIGS["decreciente"]
    mixed = SCENARIO_CONFIGS["mixto"]

    # Horizonte de madurez.
    assert increasing.dev_M == 60
    assert decreasing.dev_M == 48
    assert mixed.dev_M == 48

    # Niveles iniciales coherentes con cada escenario.
    assert increasing.start_level < 1.0
    assert decreasing.start_level > 1.0
    assert mixed.start_level < 1.0

    # El escenario mixto necesita un máximo superior al definitivo.
    assert mixed.peak_level is not None
    assert mixed.peak_level > 1.0

    assert mixed.peak_month is not None
    assert 0 < mixed.peak_month < mixed.dev_M


# ============================================================
# 2. PRUEBAS DE COMPONENTES TEMPORALES
# ============================================================

def test_trend_factor():
    """
    Verifica la fórmula de tendencia.

    Para una tendencia anual de 4 %:

        Trend(t) = (1 + 0.04) ** (t / 12)

    Por lo tanto:

        Trend(0)  = 1
        Trend(12) = 1.04
    """

    month_index = np.array([0, 6, 12])

    trend = trend_factor(
        month_index=month_index,
        annual_trend=0.04,
    )

    assert trend[0] == pytest.approx(1.0)
    assert trend[2] == pytest.approx(1.04)

    # Una tendencia positiva debe ser creciente.
    assert np.all(np.diff(trend) > 0)


def test_seasonality_factor():
    """
    Verifica que la estacionalidad mensual:

    - genere 12 factores;
    - permanezca alrededor de 1;
    - respete aproximadamente la amplitud especificada.
    """

    periods = pd.period_range(
        start="2020-01",
        end="2020-12",
        freq="M",
    )

    amplitude = 0.08

    seasonality = seasonality_factor(
        periods=periods,
        amplitude=amplitude,
    )

    assert len(seasonality) == 12

    assert seasonality.min() >= (
        1.0 - amplitude - 1e-10
    )

    assert seasonality.max() <= (
        1.0 + amplitude + 1e-10
    )

    assert np.mean(seasonality) == pytest.approx(
        1.0,
        abs=1e-10,
    )


def test_lognormal_noise_is_positive_and_reproducible():
    """
    Verifica dos propiedades importantes del ruido lognormal:

    1. todos los factores deben ser positivos;
    2. una misma semilla debe reproducir exactamente los mismos valores.
    """

    rng_1 = np.random.default_rng(42)
    rng_2 = np.random.default_rng(42)

    noise_1 = lognormal_noise(
        size=100,
        sigma=0.08,
        rng=rng_1,
    )

    noise_2 = lognormal_noise(
        size=100,
        sigma=0.08,
        rng=rng_2,
    )

    assert np.all(noise_1 > 0)

    np.testing.assert_allclose(
        noise_1,
        noise_2,
    )


# ============================================================
# 3. PRUEBAS DE CURVAS DE DESARROLLO
# ============================================================

@pytest.mark.parametrize(
    "scenario_name",
    [
        "creciente",
        "decreciente",
        "mixto",
    ],
)
def test_development_curves_start_and_finish_correctly(
    scenario_name,
):
    """
    Verifica que cada curva:

    - comience en su nivel inicial definido;
    - permanezca positiva;
    - termine exactamente en 1 al alcanzar dev_M.

    La condición final permite interpretar ``ultimate_loss`` como el
    Loss Incurred definitivo.
    """

    scenario = SCENARIO_CONFIGS[scenario_name]

    dev_month = np.arange(
        scenario.dev_M + 1
    )

    curve = development_curve(
        dev_month=dev_month,
        scenario=scenario,
    )

    assert len(curve) == scenario.dev_M + 1

    assert np.all(curve > 0)

    assert curve[0] == pytest.approx(
        scenario.start_level
    )

    assert curve[-1] == pytest.approx(
        1.0
    )


def test_increasing_curve_is_monotonic():
    """
    El escenario creciente debe presentar:

        D_0 < D_1 < ... < D_M = 1
    """

    scenario = SCENARIO_CONFIGS["creciente"]

    dev_month = np.arange(
        scenario.dev_M + 1
    )

    curve = development_curve(
        dev_month=dev_month,
        scenario=scenario,
    )

    assert np.all(
        np.diff(curve) > 0
    )


def test_decreasing_curve_is_monotonic():
    """
    El escenario decreciente debe presentar:

        D_0 > D_1 > ... > D_M = 1
    """

    scenario = SCENARIO_CONFIGS["decreciente"]

    dev_month = np.arange(
        scenario.dev_M + 1
    )

    curve = development_curve(
        dev_month=dev_month,
        scenario=scenario,
    )

    assert np.all(
        np.diff(curve) < 0
    )


def test_mixed_curve_has_expected_peak():
    """
    Verifica la estructura del escenario mixto:

    - crecimiento hasta peak_month;
    - máximo en peak_month;
    - disminución después del máximo;
    - convergencia a 1 en dev_M.
    """

    scenario = SCENARIO_CONFIGS["mixto"]

    dev_month = np.arange(
        scenario.dev_M + 1
    )

    curve = development_curve(
        dev_month=dev_month,
        scenario=scenario,
    )

    peak_month = scenario.peak_month
    peak_level = scenario.peak_level

    assert peak_month is not None
    assert peak_level is not None

    # Máximo exactamente donde fue diseñado.
    assert np.argmax(curve) == peak_month

    assert curve[peak_month] == pytest.approx(
        peak_level
    )

    # Antes del máximo la curva aumenta.
    assert np.all(
        np.diff(curve[: peak_month + 1]) > 0
    )

    # Después del máximo disminuye.
    assert np.all(
        np.diff(curve[peak_month:]) < 0
    )

    assert curve[-1] == pytest.approx(1.0)


# ============================================================
# 4. TABLA DE PERÍODOS DE OCURRENCIA
# ============================================================

def test_occurrence_table_structure(
    small_config,
):
    """
    Verifica que la tabla de períodos de ocurrencia tenga:

    - una fila por mes;
    - pérdidas positivas;
    - factores temporales positivos;
    - períodos mensuales correctamente construidos.
    """

    rng = np.random.default_rng(
        small_config.random_seed
    )

    occurrence = create_occurrence_table(
        config=small_config,
        rng=rng,
    )

    # Enero-diciembre = 12 meses.
    assert len(occurrence) == 12

    assert occurrence["accident_period"].min() == pd.Period(
        "2020-01",
        freq="M",
    )

    assert occurrence["accident_period"].max() == pd.Period(
        "2020-12",
        freq="M",
    )

    # Índice temporal.
    np.testing.assert_array_equal(
        occurrence["month_index"].to_numpy(),
        np.arange(12),
    )

    # Todos los componentes multiplicativos deben ser positivos.
    assert (
        occurrence["trend_factor"] > 0
    ).all()

    assert (
        occurrence["seasonality_factor"] > 0
    ).all()

    assert (
        occurrence["occurrence_noise"] > 0
    ).all()

    assert (
        occurrence["expected_ultimate"] > 0
    ).all()

    assert (
        occurrence["ultimate_loss"] > 0
    ).all()


# ============================================================
# 5. GENERACIÓN COMPLETA DEL ESCENARIO
# ============================================================

@pytest.mark.parametrize(
    "scenario_name",
    [
        "creciente",
        "decreciente",
        "mixto",
    ],
)
def test_simulated_scenario_dimensions(
    scenario_name,
    small_config,
):
    """
    Verifica el producto cartesiano:

        accident_period × dev_month

    Para 12 períodos de ocurrencia:

        filas = 12 × (dev_M + 1)
    """

    scenario = SCENARIO_CONFIGS[scenario_name]

    df = simulate_scenario(
        scenario_name=scenario_name,
        config=small_config,
    )

    expected_rows = (
        12 * (scenario.dev_M + 1)
    )

    assert len(df) == expected_rows

    assert df["accident_period"].nunique() == 12

    assert df["dev_month"].min() == 0

    assert df["dev_month"].max() == scenario.dev_M

    assert (
        df["scenario"] == scenario_name
    ).all()

    assert (
        df["dev_M"] == scenario.dev_M
    ).all()


@pytest.mark.parametrize(
    "scenario_name",
    [
        "creciente",
        "decreciente",
        "mixto",
    ],
)
def test_calendar_period_is_consistent(
    scenario_name,
    small_config,
):
    """
    Verifica la identidad fundamental:

        calendar_period
        =
        accident_period + dev_month

    Esta relación será esencial para construir posteriormente
    diagonales de valuación y backtesting temporal.
    """

    df = simulate_scenario(
        scenario_name=scenario_name,
        config=small_config,
    )

    expected_calendar = (
        df["accident_period"]
        + df["dev_month"]
    )

    pd.testing.assert_series_equal(
        df["calendar_period"],
        expected_calendar,
        check_names=False,
    )


@pytest.mark.parametrize(
    "scenario_name",
    [
        "creciente",
        "decreciente",
        "mixto",
    ],
)
def test_all_losses_are_positive(
    scenario_name,
    small_config,
):
    """
    Verifica que ningún componente monetario simulado sea negativo.
    """

    df = simulate_scenario(
        scenario_name=scenario_name,
        config=small_config,
    )

    monetary_columns = [
        "expected_ultimate",
        "ultimate_loss",
        "expected_loss_incurred",
        "loss_incurred",
    ]

    for column in monetary_columns:

        assert (
            df[column] > 0
        ).all()


# ============================================================
# 6. IDENTIDAD EN LA MADUREZ
# ============================================================

@pytest.mark.parametrize(
    "scenario_name",
    [
        "creciente",
        "decreciente",
        "mixto",
    ],
)
def test_maturity_equals_ultimate(
    scenario_name,
    small_config,
):
    """
    Verifica una de las identidades centrales de la simulación:

        Loss Incurred(dev_M)
        =
        Ultimate Loss

    Esto ocurre porque en madurez:

        development_factor = 1
        development_noise  = 1
    """

    scenario = SCENARIO_CONFIGS[scenario_name]

    df = simulate_scenario(
        scenario_name=scenario_name,
        config=small_config,
    )

    mature = df.loc[
        df["dev_month"] == scenario.dev_M
    ]

    np.testing.assert_allclose(
        mature["development_factor"],
        1.0,
    )

    np.testing.assert_allclose(
        mature["development_noise"],
        1.0,
    )

    np.testing.assert_allclose(
        mature["expected_loss_incurred"],
        mature["ultimate_loss"],
    )

    np.testing.assert_allclose(
        mature["loss_incurred"],
        mature["ultimate_loss"],
    )


# ============================================================
# 7. TRAYECTORIAS COMPLETAS
# ============================================================

@pytest.mark.parametrize(
    "scenario_name",
    [
        "creciente",
        "decreciente",
        "mixto",
    ],
)
def test_full_future_development_is_generated(
    scenario_name,
    small_config,
):
    """
    Verifica que la base maestra incluya desarrollo futuro.

    Por ejemplo, un período de ocurrencia 2020-12 con dev_month > 0
    tendrá un calendar_period posterior a 2020-12.

    Esta propiedad es intencional.

    La base representa la 'verdad completa' simulada.

    Posteriormente, para construir una valuación histórica, se deberá
    aplicar:

        calendar_period <= valuation_period

    evitando así utilizar información futura.
    """

    df = simulate_scenario(
        scenario_name=scenario_name,
        config=small_config,
    )

    final_occurrence = pd.Period(
        small_config.end_period,
        freq="M",
    )

    assert (
        df["calendar_period"] > final_occurrence
    ).any()


# ============================================================
# 8. REPRODUCIBILIDAD
# ============================================================

@pytest.mark.parametrize(
    "scenario_name",
    [
        "creciente",
        "decreciente",
        "mixto",
    ],
)
def test_simulation_is_reproducible(
    scenario_name,
    small_config,
):
    """
    Dos ejecuciones con exactamente la misma configuración deben
    producir exactamente la misma base.
    """

    df_1 = simulate_scenario(
        scenario_name=scenario_name,
        config=small_config,
    )

    df_2 = simulate_scenario(
        scenario_name=scenario_name,
        config=small_config,
    )

    pd.testing.assert_frame_equal(
        df_1,
        df_2,
    )


# ============================================================
# 9. GENERACIÓN DE LOS TRES ESCENARIOS
# ============================================================

def test_simulate_all_scenarios(
    small_config,
):
    """
    Verifica que la función de alto nivel produzca exactamente los
    tres escenarios definidos para el proyecto.
    """

    simulations = simulate_all_scenarios(
        config=small_config
    )

    assert set(simulations.keys()) == {
        "creciente",
        "decreciente",
        "mixto",
    }

    for scenario_name, df in simulations.items():

        assert not df.empty

        assert (
            df["scenario"] == scenario_name
        ).all()