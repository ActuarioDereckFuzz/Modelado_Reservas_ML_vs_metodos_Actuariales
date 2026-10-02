import numpy as np
import pandas as pd
import pytest

import src.experiment.backtesting as bt

from src.experiment.backtesting import (
    DevelopmentBacktestResult,
    DevelopmentBacktestExperimentResult,
    _normalize_valuation_periods,
    _build_valuation_summary,
    run_development_backtest_experiment,
)


# ============================================================
# Configuración artificial
# ============================================================

SCENARIOS = [
    "creciente",
    "decreciente",
]

VALUATIONS = [
    pd.Period("2022-06", freq="M"),
    pd.Period("2022-07", freq="M"),
]

DEV_M = {
    "creciente": 3,
    "decreciente": 3,
}

ALTERNATIVES = [
    ("VW", "all", "none"),
    ("S", "all", "none"),
]


# ============================================================
# Helpers
# ============================================================

def make_fake_result(
    scenario,
    valuation,
):
    """
    Construye un DevelopmentBacktestResult pequeño y totalmente
    controlado.

    Dos métodos predicen las mismas dos cohortes.
    """

    valuation = pd.Period(
        valuation,
        freq="M",
    )

    train_snapshot = pd.DataFrame(
        {
            "scenario": [
                scenario,
                scenario,
            ],
            "accident_period": [
                pd.Period("2020-01", freq="M"),
                pd.Period("2020-02", freq="M"),
            ],
            "snapshot_dev_month": [
                0,
                0,
            ],
            "observed_amount": [
                100.0,
                110.0,
            ],
            "target_amount": [
                150.0,
                160.0,
            ],
        }
    )

    prediction_snapshot = pd.DataFrame(
        {
            "scenario": [
                scenario,
                scenario,
            ],
            "model_valuation_period": [
                valuation,
                valuation,
            ],
            "accident_period": [
                valuation - 1,
                valuation,
            ],
            "snapshot_dev_month": [
                1,
                0,
            ],
            "observed_amount": [
                120.0,
                100.0,
            ],
            "target_period": [
                valuation + 2,
                valuation + 3,
            ],
        }
    )

    factor_rows = []

    for method in [
        "VW_all_none",
        "S_all_none",
    ]:
        for dev_month in range(3):
            factor_rows.append(
                {
                    "dev_month":
                        dev_month,

                    "development_factor":
                        1.1,

                    "method_name":
                        method,
                }
            )

    factors = pd.DataFrame(
        factor_rows
    )

    frozen_rows = []

    method_predictions = {
        "VW_all_none": [
            145.0,
            155.0,
        ],
        "S_all_none": [
            140.0,
            150.0,
        ],
    }

    for method_name, predictions in (
        method_predictions.items()
    ):

        for row, prediction in zip(
            prediction_snapshot.itertuples(
                index=False
            ),
            predictions,
        ):
            frozen_rows.append(
                {
                    "scenario":
                        scenario,

                    "model_valuation_period":
                        valuation,

                    "accident_period":
                        row.accident_period,

                    "snapshot_dev_month":
                        row.snapshot_dev_month,

                    "observed_amount":
                        row.observed_amount,

                    "target_period":
                        row.target_period,

                    "method_family":
                        "development",

                    "method_name":
                        method_name,

                    "prediction":
                        prediction,
                }
            )

    frozen = pd.DataFrame(
        frozen_rows
    )

    evaluated = frozen.copy()

    target_map = {
        valuation - 1: 150.0,
        valuation: 160.0,
    }

    evaluated["target_amount"] = (
        evaluated[
            "accident_period"
        ].map(target_map)
    )

    evaluated[
        "target_revealed_by_experiment_end"
    ] = (
        evaluated["target_period"]
        <= pd.Period(
            "2022-08",
            freq="M",
        )
    )

    evaluated["error"] = (
        evaluated["prediction"]
        - evaluated["target_amount"]
    )

    evaluated["absolute_error"] = (
        evaluated["error"].abs()
    )

    evaluated["squared_error"] = (
        evaluated["error"] ** 2
    )

    evaluated["relative_error"] = (
        evaluated["error"]
        / evaluated["target_amount"]
    )

    evaluated[
        "absolute_percentage_error"
    ] = (
        evaluated[
            "relative_error"
        ].abs()
    )

    metrics_rows = []

    for method_name, group in (
        evaluated.groupby(
            "method_name"
        )
    ):
        error = (
            group["prediction"]
            - group["target_amount"]
        )

        relative_error = (
            error
            / group["target_amount"]
        )

        metrics_rows.append(
            {
                "method_name":
                    method_name,

                "n_predictions":
                    len(group),

                "mae":
                    np.abs(error).mean(),

                "rmse":
                    np.sqrt(
                        np.mean(
                            error ** 2
                        )
                    ),

                "mape":
                    np.abs(
                        relative_error
                    ).mean(),

                "bias":
                    error.mean(),

                "relative_bias":
                    relative_error.mean(),
            }
        )

    metrics = pd.DataFrame(
        metrics_rows
    )

    return DevelopmentBacktestResult(
        scenario=scenario,
        model_valuation_period=valuation,
        train_snapshot=train_snapshot,
        prediction_snapshot=prediction_snapshot,
        development_factors=factors,
        frozen_predictions=frozen,
        evaluated_predictions=evaluated,
        metrics=metrics,
        prediction_fingerprint=(
            f"fake-{scenario}-{valuation}"
        ),
    )


# ============================================================
# 1. Normalización de valuaciones
# ============================================================

def test_normalize_valuation_periods():
    periods = (
        _normalize_valuation_periods(
            [
                "2022-06",
                pd.Period(
                    "2022-07",
                    freq="M",
                ),
            ]
        )
    )

    assert periods == [
        pd.Period(
            "2022-06",
            freq="M",
        ),
        pd.Period(
            "2022-07",
            freq="M",
        ),
    ]


def test_normalize_valuation_periods_rejects_empty():
    with pytest.raises(
        ValueError,
        match="vacío",
    ):
        _normalize_valuation_periods(
            []
        )


def test_normalize_valuation_periods_rejects_duplicates():
    with pytest.raises(
        ValueError,
        match="duplicadas",
    ):
        _normalize_valuation_periods(
            [
                "2022-06",
                "2022-06",
            ]
        )


# ============================================================
# 2. Resumen de valuación
# ============================================================

def test_build_valuation_summary():
    result = make_fake_result(
        "creciente",
        "2022-06",
    )

    summary = (
        _build_valuation_summary(
            result,
            final_period="2022-08",
        )
    )

    assert (
        summary["scenario"]
        == "creciente"
    )

    assert (
        summary[
            "model_valuation_period"
        ]
        == pd.Period(
            "2022-06",
            freq="M",
        )
    )

    assert (
        summary["n_train_cohorts"]
        == 2
    )

    assert (
        summary[
            "n_prediction_cohorts"
        ]
        == 2
    )

    assert (
        summary[
            "prediction_age_min"
        ]
        == 0
    )

    assert (
        summary[
            "prediction_age_max"
        ]
        == 1
    )

    assert (
        summary["n_methods"]
        == 2
    )

    assert (
        summary["n_predictions"]
        == 4
    )


# ============================================================
# 3. Experimento completo
# ============================================================

def test_run_development_backtest_experiment(
    monkeypatch,
):
    """
    Prueba el orquestador multi-valuación sin volver a ejecutar
    snapshots, modelos ni reveal_targets.

    Aquí únicamente nos interesa comprobar la consolidación.
    """

    calls = []

    def fake_run_valuation(
        df,
        *,
        scenario,
        model_valuation_period,
        dev_M,
        alternatives,
        final_period=None,
    ):
        calls.append(
            (
                scenario,
                model_valuation_period,
            )
        )

        assert dev_M == 3

        assert len(
            alternatives
        ) == 2

        return make_fake_result(
            scenario,
            model_valuation_period,
        )

    monkeypatch.setattr(
        bt,
        "run_development_backtest_valuation",
        fake_run_valuation,
    )

    simulations = {
        "creciente": pd.DataFrame(
            {
                "dummy": [1]
            }
        ),
        "decreciente": pd.DataFrame(
            {
                "dummy": [1]
            }
        ),
    }

    result = (
        run_development_backtest_experiment(
            simulations,
            scenarios=SCENARIOS,
            valuation_periods=VALUATIONS,
            dev_M_by_scenario=DEV_M,
            alternatives=ALTERNATIVES,
            final_period="2022-08",
        )
    )

    # --------------------------------------------------------
    # Tipo de resultado.
    # --------------------------------------------------------

    assert isinstance(
        result,
        DevelopmentBacktestExperimentResult,
    )

    # --------------------------------------------------------
    # 2 escenarios × 2 valuaciones.
    # --------------------------------------------------------

    assert len(
        result.valuation_summary
    ) == 4

    assert len(calls) == 4

    assert set(calls) == {
        (
            "creciente",
            pd.Period(
                "2022-06",
                freq="M",
            ),
        ),
        (
            "creciente",
            pd.Period(
                "2022-07",
                freq="M",
            ),
        ),
        (
            "decreciente",
            pd.Period(
                "2022-06",
                freq="M",
            ),
        ),
        (
            "decreciente",
            pd.Period(
                "2022-07",
                freq="M",
            ),
        ),
    }

    # --------------------------------------------------------
    # Cada valuación contiene:
    #
    # 2 métodos × 2 cohortes = 4 predicciones.
    #
    # 4 valuaciones × 4 = 16.
    # --------------------------------------------------------

    assert len(
        result.frozen_predictions
    ) == 16

    assert len(
        result.evaluated_predictions
    ) == 16

    # --------------------------------------------------------
    # 3 factores × 2 métodos × 4 valuaciones.
    # --------------------------------------------------------

    assert len(
        result.development_factors
    ) == 24

    # --------------------------------------------------------
    # 2 métodos × 4 valuaciones.
    # --------------------------------------------------------

    assert len(
        result.valuation_metrics
    ) == 8


# ============================================================
# 4. Identidad de factores
# ============================================================

def test_experiment_adds_scenario_and_valuation_to_factors(
    monkeypatch,
):
    def fake_run_valuation(
        df,
        *,
        scenario,
        model_valuation_period,
        dev_M,
        alternatives,
        final_period=None,
    ):
        return make_fake_result(
            scenario,
            model_valuation_period,
        )

    monkeypatch.setattr(
        bt,
        "run_development_backtest_valuation",
        fake_run_valuation,
    )

    simulations = {
        "creciente": pd.DataFrame(
            {"dummy": [1]}
        ),
    }

    result = (
        run_development_backtest_experiment(
            simulations,
            scenarios=[
                "creciente"
            ],
            valuation_periods=[
                "2022-06",
                "2022-07",
            ],
            dev_M_by_scenario={
                "creciente": 3
            },
            alternatives=(
                ALTERNATIVES
            ),
            final_period="2022-08",
        )
    )

    factors = (
        result.development_factors
    )

    assert (
        "scenario"
        in factors.columns
    )

    assert (
        "model_valuation_period"
        in factors.columns
    )

    assert set(
        factors[
            "model_valuation_period"
        ]
    ) == {
        pd.Period(
            "2022-06",
            freq="M",
        ),
        pd.Period(
            "2022-07",
            freq="M",
        ),
    }


# ============================================================
# 5. Métricas correctamente identificadas
# ============================================================

def test_experiment_metrics_have_full_key(
    monkeypatch,
):
    def fake_run_valuation(
        df,
        *,
        scenario,
        model_valuation_period,
        dev_M,
        alternatives,
        final_period=None,
    ):
        return make_fake_result(
            scenario,
            model_valuation_period,
        )

    monkeypatch.setattr(
        bt,
        "run_development_backtest_valuation",
        fake_run_valuation,
    )

    simulations = {
        "creciente": pd.DataFrame(
            {"dummy": [1]}
        ),
    }

    result = (
        run_development_backtest_experiment(
            simulations,
            scenarios=[
                "creciente"
            ],
            valuation_periods=[
                "2022-06",
                "2022-07",
            ],
            dev_M_by_scenario={
                "creciente": 3
            },
            alternatives=(
                ALTERNATIVES
            ),
        )
    )

    metrics = (
        result.valuation_metrics
    )

    key = [
        "scenario",
        "model_valuation_period",
        "method_name",
    ]

    assert not metrics.duplicated(
        key
    ).any()


# ============================================================
# 6. target_amount permanece separado
# ============================================================

def test_experiment_preserves_target_separation(
    monkeypatch,
):
    def fake_run_valuation(
        df,
        *,
        scenario,
        model_valuation_period,
        dev_M,
        alternatives,
        final_period=None,
    ):
        return make_fake_result(
            scenario,
            model_valuation_period,
        )

    monkeypatch.setattr(
        bt,
        "run_development_backtest_valuation",
        fake_run_valuation,
    )

    result = (
        run_development_backtest_experiment(
            {
                "creciente":
                    pd.DataFrame(
                        {"dummy": [1]}
                    ),
            },
            scenarios=[
                "creciente"
            ],
            valuation_periods=[
                "2022-06"
            ],
            dev_M_by_scenario={
                "creciente": 3
            },
            alternatives=(
                ALTERNATIVES
            ),
        )
    )

    assert (
        "target_amount"
        not in result
        .frozen_predictions
        .columns
    )

    assert (
        "target_amount"
        in result
        .evaluated_predictions
        .columns
    )


# ============================================================
# 7. Configuraciones inválidas
# ============================================================

def test_experiment_rejects_missing_simulation():
    with pytest.raises(
        KeyError,
        match="simulación",
    ):
        run_development_backtest_experiment(
            {},
            scenarios=[
                "creciente"
            ],
            valuation_periods=[
                "2022-06"
            ],
            dev_M_by_scenario={
                "creciente": 3
            },
            alternatives=(
                ALTERNATIVES
            ),
        )


def test_experiment_rejects_missing_dev_M():
    with pytest.raises(
        KeyError,
        match="dev_M",
    ):
        run_development_backtest_experiment(
            {
                "creciente":
                    pd.DataFrame(
                        {"dummy": [1]}
                    ),
            },
            scenarios=[
                "creciente"
            ],
            valuation_periods=[
                "2022-06"
            ],
            dev_M_by_scenario={},
            alternatives=(
                ALTERNATIVES
            ),
        )


def test_experiment_rejects_duplicate_scenarios():
    with pytest.raises(
        ValueError,
        match="duplicados",
    ):
        run_development_backtest_experiment(
            {
                "creciente":
                    pd.DataFrame(
                        {"dummy": [1]}
                    ),
            },
            scenarios=[
                "creciente",
                "creciente",
            ],
            valuation_periods=[
                "2022-06"
            ],
            dev_M_by_scenario={
                "creciente": 3
            },
            alternatives=(
                ALTERNATIVES
            ),
        )

def test_normalize_valuation_periods_accepts_period_index():
    periods = pd.period_range(
        "2022-06",
        "2022-08",
        freq="M",
    )

    result = (
        _normalize_valuation_periods(
            periods
        )
    )

    assert result == [
        pd.Period("2022-06", freq="M"),
        pd.Period("2022-07", freq="M"),
        pd.Period("2022-08", freq="M"),
    ]