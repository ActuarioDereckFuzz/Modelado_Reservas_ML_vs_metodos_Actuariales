import pytest

from src.simulation import (
    SimulationConfig,
    simulate_all_scenarios,
)


@pytest.fixture(scope="session")
def simulation_config():
    """
    Configuración base utilizada por los tests.
    """
    return SimulationConfig()


@pytest.fixture(scope="session")
def simulations_base(simulation_config):
    """
    Genera las simulaciones una sola vez durante
    toda la sesión de pytest.
    """
    return simulate_all_scenarios(simulation_config)


@pytest.fixture
def simulations(simulations_base):
    """
    Entrega una copia independiente de las simulaciones
    a cada test para evitar modificaciones accidentales
    entre pruebas.
    """
    return {
        scenario: df.copy(deep=True)
        for scenario, df in simulations_base.items()
    }