"""
Herramientas para la simulación de escenarios de Loss Incurred.
"""

from .config import (
    SCENARIO_CONFIGS,
    ScenarioConfig,
    SimulationConfig,
)

from .generator import (
    create_occurrence_table,
    simulate_all_scenarios,
    simulate_scenario,
)

__all__ = [
    "SCENARIO_CONFIGS",
    "ScenarioConfig",
    "SimulationConfig",
    "create_occurrence_table",
    "simulate_scenario",
    "simulate_all_scenarios",
]