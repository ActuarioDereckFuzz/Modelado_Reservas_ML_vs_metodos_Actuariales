from pathlib import Path
import sys

import pandas as pd
import pytest


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

from src.simulation import (
    SimulationConfig,
    simulate_all_scenarios,
)


# ============================================================
# FIXTURES DE SIMULACIÓN
# ============================================================

@pytest.fixture(scope="session")
def simulation_config():
    """
    Configuración base utilizada por los tests.
    """
    return SimulationConfig()


@pytest.fixture(scope="session")
def simulations_base(simulation_config):
    """
    Genera las tres simulaciones una sola vez durante
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


# ============================================================
# FIXTURES DEL DISEÑO EXPERIMENTAL
# ============================================================

@pytest.fixture
def accident_periods():
    """
    Períodos de ocurrencia utilizados en las pruebas
    del diseño experimental.
    """
    return pd.period_range(
        "2015-01",
        "2025-12",
        freq="M",
    )