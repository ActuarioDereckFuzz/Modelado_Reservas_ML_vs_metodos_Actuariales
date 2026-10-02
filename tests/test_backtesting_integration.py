# tests/test_backtesting_integration.py

import numpy as np
import pandas as pd
import pytest

from src.simulation import (
    SimulationConfig,
    simulate_all_scenarios,
)

from src.experiment.backtesting import (
    run_development_backtest_valuation,
)


# ============================================================
# Configuración congelada para la prueba
# ============================================================

DEV_M = {
    "creciente": 60,
    "decreciente": 48,
    "mixto": 48,
}

FINAL_PERIOD = pd.Period(
    "2025-12",
    freq="M",
)


DEVELOPMENT_ALTERNATIVES = [
    # Simple
    ("S", "all", "none"),
    ("S", "all", "minmax"),
    ("S", "all", "std2"),
    ("S", 24, "none"),
    ("S", 24, "minmax"),
    ("S", 24, "std2"),

    # Volume weighted
    ("VW", "all", "none"),
    ("VW", "all", "minmax"),
    ("VW", "all", "std2"),
    ("VW", 24, "none"),
    ("VW", 24, "minmax"),
    ("VW", 24, "std2"),

    # Median
    ("M", "all", "none"),
    ("M", "all", "minmax"),
    ("M", "all", "std2"),
    ("M", 24, "none"),
    ("M", 24, "minmax"),
    ("M", 24, "std2"),

    # Ventanas adicionales VW
    ("VW", 6, "none"),
    ("VW", 12, "none"),
    ("VW", 36, "none"),
]


# ============================================================
# Fixture de simulación real
# ============================================================

@pytest.fixture(scope="module")
def simulations():
    """
    Reconstruye las tres simulaciones utilizando la misma
    infraestructura real del proyecto.
    """
    config = SimulationConfig()

    return simulate_all_scenarios(
        config
    )


# ============================================================
# 1. Integración completa por escenario
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    [
        "creciente",
        "decreciente",
        "mixto",
    ],
)
def test_real_backtesting_single_valuation(
    simulations,
    scenario,
):
    """
    Ejecuta una valuación histórica completa utilizando:

        simulación real
        -> snapshots reales
        -> modelos de Desarrollo reales
        -> freeze
        -> reveal_targets
        -> errores
        -> métricas
    """

    valuation = pd.Period(
        "2022-06",
        freq="M",
    )

    result = (
        run_development_backtest_valuation(
            df=simulations[scenario],
            scenario=scenario,
            model_valuation_period=valuation,
            dev_M=DEV_M[scenario],
            alternatives=(
                DEVELOPMENT_ALTERNATIVES
            ),
            final_period=FINAL_PERIOD,
        )
    )

    # --------------------------------------------------------
    # Resultado general
    # --------------------------------------------------------

    assert result.scenario == scenario

    assert (
        result.model_valuation_period
        == valuation
    )

    assert not result.train_snapshot.empty
    assert not result.prediction_snapshot.empty

    # --------------------------------------------------------
    # Las 21 alternativas se ejecutaron.
    # --------------------------------------------------------

    assert (
        result.frozen_predictions[
            "method_name"
        ].nunique()
        == 21
    )

    assert (
        result.metrics[
            "method_name"
        ].nunique()
        == 21
    )

    # --------------------------------------------------------
    # Cada método predice todas las cohortes.
    # --------------------------------------------------------

    n_prediction_cohorts = len(
        result.prediction_snapshot
    )

    expected_predictions = (
        n_prediction_cohorts * 21
    )

    assert len(
        result.frozen_predictions
    ) == expected_predictions

    assert len(
        result.evaluated_predictions
    ) == expected_predictions

    # --------------------------------------------------------
    # Factores completos.
    # --------------------------------------------------------

    expected_factor_rows = (
        DEV_M[scenario] * 21
    )

    assert len(
        result.development_factors
    ) == expected_factor_rows

    # --------------------------------------------------------
    # No leakage antes de reveal.
    # --------------------------------------------------------

    assert (
        "target_amount"
        not in result.frozen_predictions.columns
    )

    # target_period sí es información estructural conocida.
    assert (
        "target_period"
        in result.frozen_predictions.columns
    )

    assert (
        result.frozen_predictions[
            "target_period"
        ].notna().all()
    )

    assert (
        result.frozen_predictions[
            "target_period"
        ]
        >
        result.frozen_predictions[
            "model_valuation_period"
        ]
    ).all()

    # --------------------------------------------------------
    # Target disponible después de reveal.
    # --------------------------------------------------------

    assert (
        "target_amount"
        in result.evaluated_predictions.columns
    )

    assert (
        "target_period"
        in result.evaluated_predictions.columns
    )

    assert (
        result.evaluated_predictions[
            "target_amount"
        ]
        .notna()
        .all()
    )

    # --------------------------------------------------------
    # Temporalidad.
    # --------------------------------------------------------

    assert (
        result.evaluated_predictions[
            "target_period"
        ]
        >
        result.evaluated_predictions[
            "model_valuation_period"
        ]
    ).all()

    # --------------------------------------------------------
    # Predicciones válidas.
    # --------------------------------------------------------

    predictions = (
        result.frozen_predictions[
            "prediction"
        ].to_numpy(dtype=float)
    )

    assert np.isfinite(
        predictions
    ).all()

    assert (
        predictions > 0
    ).all()

    # --------------------------------------------------------
    # Errores válidos.
    # --------------------------------------------------------

    error_columns = [
        "error",
        "absolute_error",
        "squared_error",
        "relative_error",
        "absolute_percentage_error",
    ]

    assert set(
        error_columns
    ).issubset(
        result.evaluated_predictions.columns
    )

    assert np.isfinite(
        result.evaluated_predictions[
            error_columns
        ].to_numpy(dtype=float)
    ).all()


# ============================================================
# 2. Todas las alternativas usan las mismas cohortes
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    [
        "creciente",
        "decreciente",
        "mixto",
    ],
)
def test_real_methods_use_same_cohorts(
    simulations,
    scenario,
):
    valuation = pd.Period(
        "2022-06",
        freq="M",
    )

    result = (
        run_development_backtest_valuation(
            df=simulations[scenario],
            scenario=scenario,
            model_valuation_period=valuation,
            dev_M=DEV_M[scenario],
            alternatives=(
                DEVELOPMENT_ALTERNATIVES
            ),
            final_period=FINAL_PERIOD,
        )
    )

    cohorts = (
        result.frozen_predictions
        .groupby("method_name")[
            "accident_period"
        ]
        .apply(
            lambda x: tuple(
                sorted(x.tolist())
            )
        )
    )

    reference = cohorts.iloc[0]

    assert cohorts.apply(
        lambda x: x == reference
    ).all()


# ============================================================
# 3. Targets idénticos entre métodos
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    [
        "creciente",
        "decreciente",
        "mixto",
    ],
)
def test_real_targets_are_same_across_methods(
    simulations,
    scenario,
):
    result = (
        run_development_backtest_valuation(
            df=simulations[scenario],
            scenario=scenario,
            model_valuation_period="2022-06",
            dev_M=DEV_M[scenario],
            alternatives=(
                DEVELOPMENT_ALTERNATIVES
            ),
            final_period=FINAL_PERIOD,
        )
    )

    targets_per_cohort = (
        result.evaluated_predictions
        .groupby("accident_period")[
            "target_amount"
        ]
        .nunique()
    )

    assert (
        targets_per_cohort == 1
    ).all()


# ============================================================
# 4. FINAL_PERIOD es exclusivamente diagnóstico
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    [
        "creciente",
        "decreciente",
        "mixto",
    ],
)
def test_real_backtesting_keeps_targets_after_final_period(
    simulations,
    scenario,
):
    """
    Prueba de regresión del problema que motivó la
    modificación del diseño.

    Deben existir cohortes cuyo target se revela después de
    FINAL_PERIOD y esas cohortes deben seguir siendo evaluadas.
    """

    result = (
        run_development_backtest_valuation(
            df=simulations[scenario],
            scenario=scenario,
            model_valuation_period="2022-06",
            dev_M=DEV_M[scenario],
            alternatives=(
                DEVELOPMENT_ALTERNATIVES
            ),
            final_period=FINAL_PERIOD,
        )
    )

    evaluated = (
        result.evaluated_predictions
    )

    assert (
        "target_revealed_by_experiment_end"
        in evaluated.columns
    )

    # Debe existir al menos una cohorte cuyo target
    # esté después del cierre original.
    after_end = (
        ~evaluated[
            "target_revealed_by_experiment_end"
        ]
    )

    assert after_end.any()

    # Esas observaciones siguen teniendo target y error.
    assert evaluated.loc[
        after_end,
        "target_amount",
    ].notna().all()

    assert evaluated.loc[
        after_end,
        "error",
    ].notna().all()


# ============================================================
# 5. Edades jóvenes permanecen en el backtesting
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    [
        "creciente",
        "decreciente",
        "mixto",
    ],
)
def test_real_backtesting_includes_young_ages(
    simulations,
    scenario,
):
    """
    Comprueba directamente que el nuevo diseño sí incorpora
    edades jóvenes en la población de predicción.
    """

    result = (
        run_development_backtest_valuation(
            df=simulations[scenario],
            scenario=scenario,
            model_valuation_period="2022-06",
            dev_M=DEV_M[scenario],
            alternatives=(
                DEVELOPMENT_ALTERNATIVES
            ),
            final_period=FINAL_PERIOD,
        )
    )

    ages = (
        result.prediction_snapshot[
            "snapshot_dev_month"
        ]
    )

    # La cohorte del mismo mes de valuación
    # debe aparecer con edad 0.
    assert ages.min() == 0

    assert (
        0
        in set(
            ages.tolist()
        )
    )


# ============================================================
# 6. Fingerprint válido
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    [
        "creciente",
        "decreciente",
        "mixto",
    ],
)
def test_real_backtesting_creates_fingerprint(
    simulations,
    scenario,
):
    result = (
        run_development_backtest_valuation(
            df=simulations[scenario],
            scenario=scenario,
            model_valuation_period="2022-06",
            dev_M=DEV_M[scenario],
            alternatives=(
                DEVELOPMENT_ALTERNATIVES
            ),
            final_period=FINAL_PERIOD,
        )
    )

    fingerprint = (
        result.prediction_fingerprint
    )

    assert isinstance(
        fingerprint,
        str,
    )

    assert len(
        fingerprint
    ) == 64