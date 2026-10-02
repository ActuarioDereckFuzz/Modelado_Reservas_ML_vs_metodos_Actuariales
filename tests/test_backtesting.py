# tests/test_backtesting.py

import numpy as np
import pandas as pd
import pytest

import src.experiment.backtesting as bt

from src.experiment.backtesting import (
    normalize_development_alternative,
    normalize_development_alternatives,
    prediction_fingerprint,
    validate_prediction_population,
    validate_same_cohorts_across_methods,
    freeze_predictions,
    validate_revealed_predictions,
    add_experiment_end_diagnostic,
    add_backtest_errors,
    summarize_backtest_metrics,
    validate_same_targets_across_methods,
    run_development_backtest_valuation,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def prediction_table():
    """
    Predicciones ya generadas para dos métodos y tres cohortes.

    Representan el estado inmediatamente anterior a
    reveal_targets().
    """

    rows = []

    valuation = pd.Period(
        "2022-06",
        freq="M",
    )

    cohorts = [
        (
            pd.Period("2022-04", freq="M"),
            2,
            100.0,
        ),
        (
            pd.Period("2022-05", freq="M"),
            1,
            110.0,
        ),
        (
            pd.Period("2022-06", freq="M"),
            0,
            120.0,
        ),
    ]

    predictions = {
        "VW_24_none": [
            150.0,
            165.0,
            180.0,
        ],
        "S_all_none": [
            148.0,
            162.0,
            177.0,
        ],
    }

    for method_name, method_predictions in (
        predictions.items()
    ):
        for (
            accident_period,
            snapshot_dev_month,
            observed_amount,
        ), prediction in zip(
            cohorts,
            method_predictions,
        ):
            rows.append(
                {
                    "scenario":
                        "creciente",

                    "model_valuation_period":
                        valuation,

                    "accident_period":
                        accident_period,

                    "snapshot_dev_month":
                        snapshot_dev_month,

                    "observed_amount":
                        observed_amount,

                    "method_family":
                        "development",

                    "method_name":
                        method_name,

                    "development_cdf":
                        prediction
                        / observed_amount,

                    "prediction":
                        prediction,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def evaluated_table(
    prediction_table,
):
    """
    Misma tabla de predicciones después de revelar targets.
    """

    result = prediction_table.copy()

    targets = {
        pd.Period(
            "2022-04",
            freq="M",
        ): (
            160.0,
            pd.Period("2022-10", freq="M"),
        ),

        pd.Period(
            "2022-05",
            freq="M",
        ): (
            170.0,
            pd.Period("2022-11", freq="M"),
        ),

        pd.Period(
            "2022-06",
            freq="M",
        ): (
            190.0,
            pd.Period("2022-12", freq="M"),
        ),
    }

    result["target_amount"] = (
        result["accident_period"]
        .map(
            lambda x: targets[x][0]
        )
    )

    result["target_period"] = (
        result["accident_period"]
        .map(
            lambda x: targets[x][1]
        )
    )

    return result


# ============================================================
# 1. Normalización de alternativas
# ============================================================


@pytest.mark.parametrize(
    "alternative, expected",
    [
        (
            ("VW", 24, "none"),
            ("VW", 24, "none"),
        ),
        (
            ("vw", "all", "none"),
            ("VW", "all", "none"),
        ),
        (
            {
                "method": "S",
                "window": 24,
                "exclusion": "minmax",
            },
            (
                "S",
                24,
                "minmax",
            ),
        ),
        (
            {
                "aggregation": "M",
                "window": "all",
                "outlier_rule": "std2",
            },
            (
                "M",
                "all",
                "std2",
            ),
        ),
    ],
)
def test_normalize_development_alternative(
    alternative,
    expected,
):
    result = (
        normalize_development_alternative(
            alternative
        )
    )

    assert result == expected


def test_normalize_rejects_invalid_tuple():
    with pytest.raises(
        ValueError,
        match="tuple",
    ):
        normalize_development_alternative(
            ("VW", 24)
        )


def test_normalize_rejects_missing_method():
    with pytest.raises(
        ValueError,
        match="method",
    ):
        normalize_development_alternative(
            {
                "window": 24,
                "exclusion": "none",
            }
        )


def test_normalize_catalog_rejects_duplicates():
    alternatives = [
        ("VW", 24, "none"),
        ("VW", 24, "none"),
    ]

    with pytest.raises(
        ValueError,
        match="duplicadas",
    ):
        normalize_development_alternatives(
            alternatives
        )


def test_normalize_catalog_rejects_empty():
    with pytest.raises(
        ValueError,
        match="vacío",
    ):
        normalize_development_alternatives(
            []
        )


# ============================================================
# 2. Fingerprint
# ============================================================


def test_prediction_fingerprint_is_stable(
    prediction_table,
):
    fingerprint_1 = (
        prediction_fingerprint(
            prediction_table
        )
    )

    fingerprint_2 = (
        prediction_fingerprint(
            prediction_table.copy(
                deep=True
            )
        )
    )

    assert fingerprint_1 == fingerprint_2


def test_prediction_fingerprint_ignores_row_order(
    prediction_table,
):
    original = prediction_fingerprint(
        prediction_table
    )

    shuffled = (
        prediction_table
        .sample(
            frac=1.0,
            random_state=123,
        )
        .reset_index(drop=True)
    )

    shuffled_hash = (
        prediction_fingerprint(
            shuffled
        )
    )

    assert original == shuffled_hash


def test_prediction_fingerprint_changes_if_prediction_changes(
    prediction_table,
):
    original = prediction_fingerprint(
        prediction_table
    )

    modified = prediction_table.copy(
        deep=True
    )

    modified.loc[
        modified.index[0],
        "prediction",
    ] += 1.0

    changed = prediction_fingerprint(
        modified
    )

    assert original != changed


# ============================================================
# 3. Validación de predicciones
# ============================================================


def test_validate_prediction_population_passes(
    prediction_table,
):
    validate_prediction_population(
        prediction_table
    )

def test_prediction_population_allows_target_period(
    prediction_table,
):
    predictions = prediction_table.copy()

    predictions["target_period"] = (
        predictions["accident_period"]
        + 6
    )

    validate_prediction_population(
        predictions
    )


def test_prediction_population_rejects_duplicates(
    prediction_table,
):
    duplicated = pd.concat(
        [
            prediction_table,
            prediction_table.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicadas",
    ):
        validate_prediction_population(
            duplicated
        )


def test_prediction_population_rejects_nan(
    prediction_table,
):
    invalid = prediction_table.copy()

    invalid.loc[
        invalid.index[0],
        "prediction",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="faltantes",
    ):
        validate_prediction_population(
            invalid
        )


def test_prediction_population_rejects_nonpositive(
    prediction_table,
):
    invalid = prediction_table.copy()

    invalid.loc[
        invalid.index[0],
        "prediction",
    ] = 0.0

    with pytest.raises(
        ValueError,
        match="positivas",
    ):
        validate_prediction_population(
            invalid
        )


# ============================================================
# 4. Comparabilidad entre métodos
# ============================================================


def test_same_cohorts_across_methods(
    prediction_table,
):
    validate_same_cohorts_across_methods(
        prediction_table
    )


def test_different_cohorts_across_methods_rejected(
    prediction_table,
):
    invalid = prediction_table.copy()

    mask = (
        invalid["method_name"]
        == "S_all_none"
    )

    row_to_remove = (
        invalid.loc[
            mask
        ]
        .index[0]
    )

    invalid = invalid.drop(
        index=row_to_remove
    )

    with pytest.raises(
        ValueError,
        match="mismas cohortes",
    ):
        validate_same_cohorts_across_methods(
            invalid
        )


# ============================================================
# 5. Congelamiento
# ============================================================


def test_freeze_predictions(
    prediction_table,
):
    frozen, fingerprint = (
        freeze_predictions(
            prediction_table
        )
    )

    pd.testing.assert_frame_equal(
        frozen,
        prediction_table,
    )

    assert frozen is not prediction_table

    assert isinstance(
        fingerprint,
        str,
    )

    assert len(fingerprint) == 64


def test_frozen_copy_is_independent(
    prediction_table,
):
    frozen, _ = freeze_predictions(
        prediction_table
    )

    original_value = (
        frozen.loc[
            frozen.index[0],
            "prediction",
        ]
    )

    prediction_table.loc[
        prediction_table.index[0],
        "prediction",
    ] += 1000.0

    assert (
        frozen.loc[
            frozen.index[0],
            "prediction",
        ]
        == original_value
    )


# ============================================================
# 6. Reveal validation
# ============================================================


def test_validate_revealed_predictions(
    prediction_table,
    evaluated_table,
):
    frozen, fingerprint = (
        freeze_predictions(
            prediction_table
        )
    )

    validate_revealed_predictions(
        frozen,
        evaluated_table,
        expected_fingerprint=fingerprint,
    )


def test_reveal_rejects_changed_prediction(
    prediction_table,
    evaluated_table,
):
    frozen, fingerprint = (
        freeze_predictions(
            prediction_table
        )
    )

    modified = evaluated_table.copy()

    modified.loc[
        modified.index[0],
        "prediction",
    ] += 1.0

    with pytest.raises(
        ValueError,
        match="modificadas",
    ):
        validate_revealed_predictions(
            frozen,
            modified,
            expected_fingerprint=(
                fingerprint
            ),
        )


def test_reveal_rejects_removed_row(
    prediction_table,
    evaluated_table,
):
    frozen, fingerprint = (
        freeze_predictions(
            prediction_table
        )
    )

    invalid = evaluated_table.iloc[:-1]

    with pytest.raises(
        ValueError,
        match="número",
    ):
        validate_revealed_predictions(
            frozen,
            invalid,
            expected_fingerprint=(
                fingerprint
            ),
        )


def test_reveal_rejects_past_target(
    prediction_table,
    evaluated_table,
):
    frozen, fingerprint = (
        freeze_predictions(
            prediction_table
        )
    )

    invalid = evaluated_table.copy()

    invalid.loc[
        invalid.index[0],
        "target_period",
    ] = pd.Period(
        "2022-05",
        freq="M",
    )

    with pytest.raises(
        ValueError,
        match="observable",
    ):
        validate_revealed_predictions(
            frozen,
            invalid,
            expected_fingerprint=(
                fingerprint
            ),
        )


# ============================================================
# 7. Diagnóstico FINAL_PERIOD
# ============================================================


def test_experiment_end_diagnostic(
    evaluated_table,
):
    result = (
        add_experiment_end_diagnostic(
            evaluated_table,
            final_period="2022-11",
        )
    )

    assert (
        "target_revealed_by_experiment_end"
        in result.columns
    )

    expected = (
        result["target_period"]
        <= pd.Period(
            "2022-11",
            freq="M",
        )
    )

    pd.testing.assert_series_equal(
        result[
            "target_revealed_by_experiment_end"
        ].reset_index(drop=True),

        expected.reset_index(
            drop=True
        ),

        check_names=False,
    )


def test_final_period_is_diagnostic_only(
    evaluated_table,
):
    result = (
        add_experiment_end_diagnostic(
            evaluated_table,
            final_period="2022-10",
        )
    )

    # Ninguna observación debe desaparecer.
    assert len(result) == len(
        evaluated_table
    )


# ============================================================
# 8. Errores
# ============================================================


def test_add_backtest_errors(
    evaluated_table,
):
    result = add_backtest_errors(
        evaluated_table
    )

    row = result.iloc[0]

    expected_error = (
        row["prediction"]
        - row["target_amount"]
    )

    assert row["error"] == pytest.approx(
        expected_error
    )

    assert row[
        "absolute_error"
    ] == pytest.approx(
        abs(expected_error)
    )

    assert row[
        "squared_error"
    ] == pytest.approx(
        expected_error ** 2
    )

    assert row[
        "relative_error"
    ] == pytest.approx(
        expected_error
        / row["target_amount"]
    )

    assert row[
        "absolute_percentage_error"
    ] == pytest.approx(
        abs(
            expected_error
            / row["target_amount"]
        )
    )


def test_error_sign_convention(
    evaluated_table,
):
    result = add_backtest_errors(
        evaluated_table
    )

    first = result.iloc[0]

    # 150 - 160 = -10:
    # predicción menor al real = subestimación.
    assert first["error"] == pytest.approx(
        -10.0
    )


# ============================================================
# 9. Métricas
# ============================================================


def test_summarize_backtest_metrics(
    evaluated_table,
):
    metrics = summarize_backtest_metrics(
        evaluated_table
    )

    assert len(metrics) == 2

    assert set(
        metrics["method_name"]
    ) == {
        "VW_24_none",
        "S_all_none",
    }

    assert (
        metrics["n_predictions"]
        == 3
    ).all()

    numeric_columns = [
        "mae",
        "rmse",
        "mape",
        "bias",
        "relative_bias",
    ]

    assert np.isfinite(
        metrics[
            numeric_columns
        ].to_numpy()
    ).all()


def test_metrics_manual_calculation(
    evaluated_table,
):
    metrics = summarize_backtest_metrics(
        evaluated_table
    )

    row = (
        metrics.loc[
            metrics["method_name"]
            == "VW_24_none"
        ]
        .iloc[0]
    )

    predictions = np.array(
        [
            150.0,
            165.0,
            180.0,
        ]
    )

    targets = np.array(
        [
            160.0,
            170.0,
            190.0,
        ]
    )

    errors = predictions - targets

    expected_mae = np.mean(
        np.abs(errors)
    )

    expected_rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    expected_mape = np.mean(
        np.abs(
            errors / targets
        )
    )

    expected_bias = np.mean(
        errors
    )

    expected_relative_bias = (
        np.mean(
            errors / targets
        )
    )

    assert row["mae"] == pytest.approx(
        expected_mae
    )

    assert row["rmse"] == pytest.approx(
        expected_rmse
    )

    assert row["mape"] == pytest.approx(
        expected_mape
    )

    assert row["bias"] == pytest.approx(
        expected_bias
    )

    assert row[
        "relative_bias"
    ] == pytest.approx(
        expected_relative_bias
    )


# ============================================================
# 10. Targets comunes
# ============================================================


def test_same_targets_across_methods(
    evaluated_table,
):
    validate_same_targets_across_methods(
        evaluated_table
    )


def test_different_targets_across_methods_rejected(
    evaluated_table,
):
    invalid = evaluated_table.copy()

    mask = (
        invalid["method_name"]
        == "S_all_none"
    )

    idx = invalid.loc[
        mask
    ].index[0]

    invalid.loc[
        idx,
        "target_amount",
    ] += 1.0

    with pytest.raises(
        ValueError,
        match="targets distintos",
    ):
        validate_same_targets_across_methods(
            invalid
        )


# ============================================================
# 11. Orquestación completa de una valuación
# ============================================================


def test_run_development_backtest_valuation(
    monkeypatch,
    prediction_table,
    evaluated_table,
):
    """
    Test unitario del orquestador completo.

    Las dependencias se sustituyen por resultados controlados.
    Así este test verifica backtesting.py sin volver a probar
    snapshots.py ni models/development.py.
    """

    # --------------------------------------------------------
    # Snapshots artificiales
    # --------------------------------------------------------

    train_snapshot = pd.DataFrame(
        {
            "scenario": [
                "creciente"
            ],
            "accident_period": [
                pd.Period(
                    "2020-01",
                    freq="M",
                )
            ],
            "snapshot_dev_month": [
                0
            ],
            "observed_amount": [
                100.0
            ],
            "target_amount": [
                150.0
            ],
        }
    )

    prediction_snapshot = (
        prediction_table.loc[
            prediction_table[
                "method_name"
            ] == "VW_24_none",
            [
                "scenario",
                "model_valuation_period",
                "accident_period",
                "snapshot_dev_month",
                "observed_amount",
            ],
        ]
        .reset_index(drop=True)
    )

    fake_factors = pd.DataFrame(
        {
            "dev_month": [
                0,
                1,
                2,
            ],
            "development_factor": [
                1.2,
                1.1,
                1.05,
            ],
            "method_name": [
                "fake",
                "fake",
                "fake",
            ],
        }
    )

    # --------------------------------------------------------
    # Sustituir dependencias externas del orquestador
    # --------------------------------------------------------

    def fake_build_snapshots(
        *,
        df,
        scenario,
        valuation_period,
    ):
        assert scenario == "creciente"

        assert valuation_period == (
            pd.Period(
                "2022-06",
                freq="M",
            )
        )

        return (
            train_snapshot.copy(),
            prediction_snapshot.copy(),
        )

    def fake_run_alternatives(
        train_snapshot,
        prediction_snapshot,
        *,
        alternatives,
        dev_M,
    ):
        assert dev_M == 3

        return (
            fake_factors.copy(),
            prediction_table.copy(),
        )

    def fake_reveal_targets(
        *,
        predictions,
        df,
        scenario,
    ):
        assert (
            "target_amount"
            not in predictions.columns
        )

        assert scenario == "creciente"

        return evaluated_table.copy()

    monkeypatch.setattr(
        bt,
        "build_train_prediction_snapshots",
        fake_build_snapshots,
    )

    monkeypatch.setattr(
        bt,
        "run_development_alternatives",
        fake_run_alternatives,
    )

    monkeypatch.setattr(
        bt,
        "reveal_targets",
        fake_reveal_targets,
    )

    # --------------------------------------------------------
    # Ejecutar
    # --------------------------------------------------------

    result = (
        run_development_backtest_valuation(
            df=pd.DataFrame(
                {
                    "dummy": [1]
                }
            ),
            scenario="creciente",
            model_valuation_period=(
                "2022-06"
            ),
            dev_M=3,
            alternatives=[
                (
                    "VW",
                    24,
                    "none",
                ),
                (
                    "S",
                    "all",
                    "none",
                ),
            ],
            final_period="2022-11",
        )
    )

    # --------------------------------------------------------
    # Validar resultado estructurado
    # --------------------------------------------------------

    assert (
        result.scenario
        == "creciente"
    )

    assert (
        result.model_valuation_period
        == pd.Period(
            "2022-06",
            freq="M",
        )
    )

    assert len(
        result.frozen_predictions
    ) == len(
        prediction_table
    )

    assert len(
        result.evaluated_predictions
    ) == len(
        evaluated_table
    )

    assert (
        result.metrics[
            "method_name"
        ].nunique()
        == 2
    )

    assert (
        "target_revealed_by_experiment_end"
        in result.evaluated_predictions.columns
    )

    assert (
        "error"
        in result.evaluated_predictions.columns
    )

    assert (
        "absolute_error"
        in result.evaluated_predictions.columns
    )

    assert isinstance(
        result.prediction_fingerprint,
        str,
    )

    assert len(
        result.prediction_fingerprint
    ) == 64


# ============================================================
# 12. FINAL_PERIOD no filtra la evaluación
# ============================================================


def test_backtesting_keeps_targets_after_final_period(
    evaluated_table,
):
    """
    Reproduce explícitamente la corrección conceptual principal:

    target_period > FINAL_PERIOD NO excluye una observación.
    """

    final_period = pd.Period(
        "2022-10",
        freq="M",
    )

    result = (
        add_experiment_end_diagnostic(
            evaluated_table,
            final_period=final_period,
        )
    )

    # Existen observaciones cuyo target está después
    # del cierre.
    assert (
        ~result[
            "target_revealed_by_experiment_end"
        ]
    ).any()

    # Pero siguen presentes.
    assert len(result) == len(
        evaluated_table
    )

    # Y pueden evaluarse normalmente.
    evaluated = add_backtest_errors(
        result
    )

    assert evaluated[
        "error"
    ].notna().all()

def test_prediction_population_rejects_nonfuture_target_period(
    prediction_table,
):
    invalid = prediction_table.copy()

    invalid["target_period"] = (
        invalid["model_valuation_period"]
    )

    with pytest.raises(
        ValueError,
        match="posterior",
    ):
        validate_prediction_population(
            invalid
        )