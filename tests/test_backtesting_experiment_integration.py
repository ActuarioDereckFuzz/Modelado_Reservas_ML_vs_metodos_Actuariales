# tests/test_backtesting_experiment_integration.py

import numpy as np
import pandas as pd
import pytest

from src.simulation import (
    SimulationConfig,
    simulate_all_scenarios,
)

from src.experiment.backtesting import (
    DevelopmentBacktestExperimentResult,
    run_development_backtest_experiment,
)


# ============================================================
# Configuración real del experimento
# ============================================================

SCENARIOS = [
    "creciente",
    "decreciente",
    "mixto",
]

DEV_M = {
    "creciente": 60,
    "decreciente": 48,
    "mixto": 48,
}

FINAL_PERIOD = pd.Period(
    "2025-12",
    freq="M",
)


# Seleccionamos una valuación temprana,
# una intermedia y la última.
VALUATION_SAMPLE = [
    pd.Period("2022-06", freq="M"),
    pd.Period("2023-06", freq="M"),
    pd.Period("2025-11", freq="M"),
]


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
# Fixture
# ============================================================

@pytest.fixture(scope="module")
def experiment_result():

    simulations = simulate_all_scenarios(
        SimulationConfig()
    )

    return run_development_backtest_experiment(
        simulations,
        scenarios=SCENARIOS,
        valuation_periods=VALUATION_SAMPLE,
        dev_M_by_scenario=DEV_M,
        alternatives=DEVELOPMENT_ALTERNATIVES,
        final_period=FINAL_PERIOD,
    )


# ============================================================
# 1. Resultado general
# ============================================================

def test_real_experiment_returns_expected_type(
    experiment_result,
):
    assert isinstance(
        experiment_result,
        DevelopmentBacktestExperimentResult,
    )


def test_real_experiment_has_nine_valuations(
    experiment_result,
):
    """
    3 escenarios × 3 fechas = 9 valuaciones.
    """

    summary = (
        experiment_result.valuation_summary
    )

    assert len(summary) == 9

    observed = set(
        zip(
            summary["scenario"],
            summary[
                "model_valuation_period"
            ],
        )
    )

    expected = {
        (scenario, valuation)
        for scenario in SCENARIOS
        for valuation in VALUATION_SAMPLE
    }

    assert observed == expected


# ============================================================
# 2. Las 21 alternativas aparecen en cada valuación
# ============================================================

def test_every_valuation_has_21_methods(
    experiment_result,
):
    metrics = (
        experiment_result.valuation_metrics
    )

    counts = (
        metrics
        .groupby(
            [
                "scenario",
                "model_valuation_period",
            ]
        )[
            "method_name"
        ]
        .nunique()
    )

    assert (
        counts == 21
    ).all()

    # 9 valuaciones × 21 métodos.
    assert len(metrics) == (
        9 * 21
    )


# ============================================================
# 3. Factores completos por valuación
# ============================================================

def test_real_experiment_has_complete_factor_curves(
    experiment_result,
):
    factors = (
        experiment_result
        .development_factors
    )

    counts = (
        factors
        .groupby(
            [
                "scenario",
                "model_valuation_period",
                "method_name",
            ]
        )
        .size()
    )

    for (
        scenario,
        valuation,
        method_name,
    ), n_rows in counts.items():

        assert n_rows == DEV_M[
            scenario
        ], (
            scenario,
            valuation,
            method_name,
            n_rows,
        )

    # 3 fechas por cada escenario.
    expected_total = (
        3
        * 21
        * sum(
            DEV_M.values()
        )
    )

    assert len(factors) == expected_total


# ============================================================
# 4. Congelamiento vs evaluación
# ============================================================

def test_target_amount_is_hidden_before_reveal(
    experiment_result,
):
    frozen = (
        experiment_result
        .frozen_predictions
    )

    assert (
        "target_amount"
        not in frozen.columns
    )

    assert (
        "target_period"
        in frozen.columns
    )


def test_target_amount_exists_after_reveal(
    experiment_result,
):
    evaluated = (
        experiment_result
        .evaluated_predictions
    )

    assert (
        "target_amount"
        in evaluated.columns
    )

    assert evaluated[
        "target_amount"
    ].notna().all()

    assert (
        evaluated[
            "target_amount"
        ] > 0
    ).all()


def test_frozen_and_evaluated_have_same_number_of_rows(
    experiment_result,
):
    assert len(
        experiment_result
        .frozen_predictions
    ) == len(
        experiment_result
        .evaluated_predictions
    )


# ============================================================
# 5. Una predicción por método y cohorte
# ============================================================

def test_prediction_keys_are_unique(
    experiment_result,
):
    frozen = (
        experiment_result
        .frozen_predictions
    )

    key = [
        "scenario",
        "model_valuation_period",
        "accident_period",
        "method_name",
    ]

    assert not frozen.duplicated(
        key
    ).any()


# ============================================================
# 6. Todos los métodos usan las mismas cohortes
# ============================================================

def test_methods_use_same_population_per_valuation(
    experiment_result,
):
    frozen = (
        experiment_result
        .frozen_predictions
    )

    for (
        scenario,
        valuation,
    ), group in frozen.groupby(
        [
            "scenario",
            "model_valuation_period",
        ]
    ):

        cohorts = (
            group
            .groupby(
                "method_name"
            )[
                "accident_period"
            ]
            .apply(
                lambda x: tuple(
                    sorted(
                        x.tolist()
                    )
                )
            )
        )

        reference = cohorts.iloc[0]

        assert cohorts.apply(
            lambda x:
                x == reference
        ).all(), (
            scenario,
            valuation,
        )


# ============================================================
# 7. Métricas válidas
# ============================================================

def test_real_experiment_metrics_are_finite(
    experiment_result,
):
    metrics = (
        experiment_result
        .valuation_metrics
    )

    metric_columns = [
        "mae",
        "rmse",
        "mape",
        "bias",
        "relative_bias",
    ]

    assert np.isfinite(
        metrics[
            metric_columns
        ].to_numpy(
            dtype=float
        )
    ).all()

    assert (
        metrics["mae"] >= 0
    ).all()

    assert (
        metrics["rmse"] >= 0
    ).all()

    assert (
        metrics["mape"] >= 0
    ).all()


# ============================================================
# 8. Edades jóvenes
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
)
@pytest.mark.parametrize(
    "valuation",
    VALUATION_SAMPLE,
)
def test_real_experiment_keeps_age_zero(
    experiment_result,
    scenario,
    valuation,
):
    frozen = (
        experiment_result
        .frozen_predictions
    )

    subset = frozen.loc[
        (
            frozen["scenario"]
            == scenario
        )
        &
        (
            frozen[
                "model_valuation_period"
            ]
            == valuation
        )
    ]

    assert not subset.empty

    assert (
        subset[
            "snapshot_dev_month"
        ].min()
        == 0
    )


# ============================================================
# 9. Temporalidad
# ============================================================

def test_all_targets_are_future_relative_to_model_valuation(
    experiment_result,
):
    frozen = (
        experiment_result
        .frozen_predictions
    )

    assert (
        frozen["target_period"]
        >
        frozen[
            "model_valuation_period"
        ]
    ).all()


# ============================================================
# 10. FINAL_PERIOD sigue siendo diagnóstico
# ============================================================

def test_targets_after_final_period_are_kept(
    experiment_result,
):
    evaluated = (
        experiment_result
        .evaluated_predictions
    )

    assert (
        "target_revealed_by_experiment_end"
        in evaluated.columns
    )

    after_end = (
        ~evaluated[
            "target_revealed_by_experiment_end"
        ]
    )

    assert after_end.any()

    # Estas predicciones siguen evaluadas.
    assert evaluated.loc[
        after_end,
        "target_amount",
    ].notna().all()

    assert evaluated.loc[
        after_end,
        "error",
    ].notna().all()


# ============================================================
# 11. Resumen de cobertura coherente
# ============================================================

def test_valuation_summary_is_consistent(
    experiment_result,
):
    summary = (
        experiment_result
        .valuation_summary
    )

    assert (
        summary[
            "n_train_cohorts"
        ] > 0
    ).all()

    assert (
        summary[
            "n_prediction_cohorts"
        ] > 0
    ).all()

    assert (
        summary[
            "prediction_age_min"
        ] == 0
    ).all()

    assert (
        summary[
            "n_methods"
        ] == 21
    ).all()

    assert (
        summary[
            "n_predictions"
        ]
        ==
        summary[
            "n_prediction_cohorts"
        ] * 21
    ).all()


# ============================================================
# 12. Última valuación permanece válida
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
)
def test_last_valuation_is_fully_evaluable(
    experiment_result,
    scenario,
):
    """
    En 2025-11 muchas cohortes tendrán target posterior a
    FINAL_PERIOD.

    Deben permanecer dentro del experimento.
    """

    evaluated = (
        experiment_result
        .evaluated_predictions
    )

    subset = evaluated.loc[
        (
            evaluated["scenario"]
            == scenario
        )
        &
        (
            evaluated[
                "model_valuation_period"
            ]
            == pd.Period(
                "2025-11",
                freq="M",
            )
        )
    ]

    assert not subset.empty

    assert subset[
        "target_amount"
    ].notna().all()

    assert subset[
        "error"
    ].notna().all()

    assert (
        ~subset[
            "target_revealed_by_experiment_end"
        ]
    ).any()