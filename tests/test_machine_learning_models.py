# tests/test_machine_learning_models.py

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline

from src.models.machine_learning import (
    ML_MODEL_NAMES,
    MLModelArtifact,
    build_ml_features,
    build_ml_target,
    make_ml_estimator,
    fit_ml_model,
    predict_ml_snapshot,
    run_ml_model,
    run_ml_models,
)


# ============================================================
# Configuración
# ============================================================

DEV_M = 6


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def train_snapshot():
    """
    Snapshot artificial de entrenamiento.

    Contiene 24 observaciones históricas con:

    - edad de desarrollo conocida;
    - observed_amount positivo;
    - target_amount positivo;
    - accident_period mensual.

    El target se expresa directamente en la misma escala
    monetaria utilizada por Desarrollo.
    """

    periods = pd.period_range(
        "2020-01",
        periods=24,
        freq="M",
    )

    ages = np.tile(
        np.arange(DEV_M),
        4,
    )

    observed = (
        100_000
        + np.arange(24) * 2_500
        + ages * 1_000
    ).astype(float)

    # Target directo en escala monetaria.
    target = (
        observed
        + 20_000
        + (DEV_M - ages) * 3_000
    ).astype(float)

    return pd.DataFrame(
        {
            "scenario":
                ["test"] * 24,

            "model_valuation_period":
                [
                    pd.Period(
                        "2022-12",
                        freq="M",
                    )
                ] * 24,

            "accident_period":
                periods,

            "snapshot_dev_month":
                ages,

            "observed_amount":
                observed,

            "target_amount":
                target,
        }
    )


@pytest.fixture
def prediction_snapshot():
    """
    Snapshot artificial de predicción.

    Deliberadamente NO contiene target_amount.
    """

    periods = pd.period_range(
        "2023-01",
        periods=6,
        freq="M",
    )

    ages = np.arange(
        DEV_M
    )

    observed = (
        180_000
        + np.arange(6) * 5_000
        + ages * 1_000
    ).astype(float)

    return pd.DataFrame(
        {
            "scenario":
                ["test"] * 6,

            "model_valuation_period":
                [
                    pd.Period(
                        "2023-06",
                        freq="M",
                    )
                ] * 6,

            "accident_period":
                periods,

            "snapshot_dev_month":
                ages,

            "observed_amount":
                observed,
        }
    )


# ============================================================
# 1. Catálogo ML
# ============================================================

def test_ml_catalog_contains_expected_models():

    assert set(
        ML_MODEL_NAMES
    ) == {
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    }

    assert len(
        ML_MODEL_NAMES
    ) == 3


# ============================================================
# 2. Features
# ============================================================

def test_build_ml_features_shape(
    train_snapshot,
):

    X = build_ml_features(
        train_snapshot,
        dev_M=DEV_M,
    )

    assert len(X) == len(
        train_snapshot
    )

    assert list(
        X.columns
    ) == [
        "snapshot_dev_month",
        "development_fraction",
        "observed_amount",
        "accident_year",
        "accident_month_sin",
        "accident_month_cos",
    ]


def test_ml_features_do_not_include_target(
    train_snapshot,
):

    X = build_ml_features(
        train_snapshot,
        dev_M=DEV_M,
    )

    assert (
        "target_amount"
        not in X.columns
    )


def test_development_fraction_is_correct(
    train_snapshot,
):

    X = build_ml_features(
        train_snapshot,
        dev_M=DEV_M,
    )

    expected = (
        train_snapshot[
            "snapshot_dev_month"
        ].to_numpy(dtype=float)
        / DEV_M
    )

    np.testing.assert_allclose(
        X[
            "development_fraction"
        ].to_numpy(),
        expected,
    )


def test_observed_amount_is_preserved(
    train_snapshot,
):

    X = build_ml_features(
        train_snapshot,
        dev_M=DEV_M,
    )

    np.testing.assert_allclose(
        X[
            "observed_amount"
        ].to_numpy(),
        train_snapshot[
            "observed_amount"
        ].to_numpy(),
    )


def test_accident_year_is_correct(
    train_snapshot,
):

    X = build_ml_features(
        train_snapshot,
        dev_M=DEV_M,
    )

    expected = (
        pd.PeriodIndex(
            train_snapshot[
                "accident_period"
            ],
            freq="M",
        )
        .year
        .astype(float)
    )

    np.testing.assert_allclose(
        X[
            "accident_year"
        ].to_numpy(),
        expected,
    )


def test_month_cyclical_features():
    """
    Para enero:

        sin(2*pi*1/12) = 0.5
        cos(2*pi*1/12) = sqrt(3)/2
    """

    df = pd.DataFrame(
        {
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
                100_000.0
            ],
        }
    )

    X = build_ml_features(
        df,
        dev_M=DEV_M,
    )

    assert X.loc[
        0,
        "accident_month_sin",
    ] == pytest.approx(
        0.5
    )

    assert X.loc[
        0,
        "accident_month_cos",
    ] == pytest.approx(
        np.sqrt(3) / 2
    )


def test_ml_features_are_finite(
    train_snapshot,
):

    X = build_ml_features(
        train_snapshot,
        dev_M=DEV_M,
    )

    assert np.isfinite(
        X.to_numpy(
            dtype=float
        )
    ).all()


# ============================================================
# 3. Target
# ============================================================

def test_build_ml_target_is_target_amount(
    train_snapshot,
):
    """
    Este test es particularmente importante.

    Machine Learning debe predecir exactamente el mismo
    target monetario que Desarrollo.
    """

    y = build_ml_target(
        train_snapshot
    )

    np.testing.assert_allclose(
        y.to_numpy(),
        train_snapshot[
            "target_amount"
        ].to_numpy(),
    )

    assert y.name == (
        "target_amount"
    )


def test_ml_target_has_no_transformation(
    train_snapshot,
):
    """
    Comprueba explícitamente que NO estamos usando:

        log(target)
        log(target / observed)
        target / observed
    """

    y = build_ml_target(
        train_snapshot
    )

    expected = (
        train_snapshot[
            "target_amount"
        ].astype(float)
    )

    pd.testing.assert_series_equal(
        y,
        expected.rename(
            "target_amount"
        ),
    )


# ============================================================
# 4. Validaciones de datos
# ============================================================

def test_invalid_dev_M_rejected(
    train_snapshot,
):

    with pytest.raises(
        ValueError,
        match="mayor que cero",
    ):
        build_ml_features(
            train_snapshot,
            dev_M=0,
        )


def test_invalid_age_rejected(
    train_snapshot,
):

    invalid = (
        train_snapshot.copy()
    )

    invalid.loc[
        invalid.index[0],
        "snapshot_dev_month",
    ] = DEV_M

    with pytest.raises(
        ValueError,
        match="edades",
    ):
        build_ml_features(
            invalid,
            dev_M=DEV_M,
        )


def test_negative_age_rejected(
    train_snapshot,
):

    invalid = (
        train_snapshot.copy()
    )

    invalid.loc[
        invalid.index[0],
        "snapshot_dev_month",
    ] = -1

    with pytest.raises(
        ValueError,
        match="edades",
    ):
        build_ml_features(
            invalid,
            dev_M=DEV_M,
        )


def test_nonpositive_observed_amount_rejected(
    train_snapshot,
):

    invalid = (
        train_snapshot.copy()
    )

    invalid.loc[
        invalid.index[0],
        "observed_amount",
    ] = 0.0

    with pytest.raises(
        ValueError,
        match="no positivos",
    ):
        build_ml_features(
            invalid,
            dev_M=DEV_M,
        )


def test_nonpositive_target_rejected(
    train_snapshot,
):

    invalid = (
        train_snapshot.copy()
    )

    invalid.loc[
        invalid.index[0],
        "target_amount",
    ] = 0.0

    with pytest.raises(
        ValueError,
        match="positivo",
    ):
        build_ml_target(
            invalid
        )


def test_missing_feature_column_rejected(
    train_snapshot,
):

    invalid = (
        train_snapshot.drop(
            columns=[
                "observed_amount"
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="columnas requeridas",
    ):
        build_ml_features(
            invalid,
            dev_M=DEV_M,
        )


# ============================================================
# 5. Construcción de estimadores
# ============================================================

def test_make_ridge_estimator():

    estimator = (
        make_ml_estimator(
            "ridge"
        )
    )

    assert isinstance(
        estimator,
        Pipeline,
    )


def test_make_random_forest_estimator():

    estimator = (
        make_ml_estimator(
            "random_forest"
        )
    )

    assert isinstance(
        estimator,
        RandomForestRegressor,
    )


def test_make_hist_gradient_boosting_estimator():

    estimator = (
        make_ml_estimator(
            "hist_gradient_boosting"
        )
    )

    assert isinstance(
        estimator,
        HistGradientBoostingRegressor,
    )


def test_invalid_model_name_rejected():

    with pytest.raises(
        ValueError,
        match="desconocido",
    ):
        make_ml_estimator(
            "invalid_model"
        )


# ============================================================
# 6. Entrenamiento
# ============================================================

@pytest.mark.parametrize(
    "model_name",
    ML_MODEL_NAMES,
)
def test_fit_ml_model(
    train_snapshot,
    model_name,
):

    artifact = fit_ml_model(
        train_snapshot,
        dev_M=DEV_M,
        model_name=model_name,
        random_state=42,
    )

    assert isinstance(
        artifact,
        MLModelArtifact,
    )

    assert (
        artifact.model_name
        == model_name
    )

    assert (
        artifact.dev_M
        == DEV_M
    )

    assert artifact.feature_names == (
        "snapshot_dev_month",
        "development_fraction",
        "observed_amount",
        "accident_year",
        "accident_month_sin",
        "accident_month_cos",
    )

    assert artifact.estimator is not None


# ============================================================
# 7. Predicción
# ============================================================

@pytest.mark.parametrize(
    "model_name",
    ML_MODEL_NAMES,
)
def test_predict_ml_snapshot(
    train_snapshot,
    prediction_snapshot,
    model_name,
):

    artifact = fit_ml_model(
        train_snapshot,
        dev_M=DEV_M,
        model_name=model_name,
        random_state=42,
    )

    predictions = (
        predict_ml_snapshot(
            artifact,
            prediction_snapshot,
        )
    )

    assert len(
        predictions
    ) == len(
        prediction_snapshot
    )

    assert (
        "prediction"
        in predictions.columns
    )

    assert (
        "target_amount"
        not in predictions.columns
    )

    assert (
        predictions[
            "method_family"
        ]
        == "machine_learning"
    ).all()

    assert (
        predictions[
            "method_name"
        ]
        == f"ML_{model_name}"
    ).all()

    assert np.isfinite(
        predictions[
            "prediction"
        ].to_numpy(
            dtype=float
        )
    ).all()


def test_prediction_rejects_target_amount(
    train_snapshot,
    prediction_snapshot,
):

    artifact = fit_ml_model(
        train_snapshot,
        dev_M=DEV_M,
        model_name="ridge",
    )

    leaked = (
        prediction_snapshot.copy()
    )

    leaked[
        "target_amount"
    ] = 999_999.0

    with pytest.raises(
        ValueError,
        match="target_amount",
    ):
        predict_ml_snapshot(
            artifact,
            leaked,
        )


# ============================================================
# 8. Modelo completo
# ============================================================

@pytest.mark.parametrize(
    "model_name",
    ML_MODEL_NAMES,
)
def test_run_ml_model(
    train_snapshot,
    prediction_snapshot,
    model_name,
):

    artifact, predictions = (
        run_ml_model(
            train_snapshot,
            prediction_snapshot,
            dev_M=DEV_M,
            model_name=model_name,
            random_state=42,
        )
    )

    assert isinstance(
        artifact,
        MLModelArtifact,
    )

    assert len(
        predictions
    ) == len(
        prediction_snapshot
    )

    assert (
        predictions[
            "method_name"
        ]
        == f"ML_{model_name}"
    ).all()


# ============================================================
# 9. Catálogo completo
# ============================================================

def test_run_all_ml_models(
    train_snapshot,
    prediction_snapshot,
):

    artifacts, predictions = (
        run_ml_models(
            train_snapshot,
            prediction_snapshot,
            dev_M=DEV_M,
            model_names=(
                ML_MODEL_NAMES
            ),
            random_state=42,
        )
    )

    # --------------------------------------------------------
    # Tres artefactos
    # --------------------------------------------------------

    assert set(
        artifacts.keys()
    ) == set(
        ML_MODEL_NAMES
    )

    assert len(
        artifacts
    ) == 3

    # --------------------------------------------------------
    # 3 modelos × 6 cohortes
    # --------------------------------------------------------

    expected_rows = (
        len(
            prediction_snapshot
        )
        * len(
            ML_MODEL_NAMES
        )
    )

    assert len(
        predictions
    ) == expected_rows

    # --------------------------------------------------------
    # Tres nombres únicos
    # --------------------------------------------------------

    assert (
        predictions[
            "method_name"
        ].nunique()
        == 3
    )

    assert set(
        predictions[
            "method_name"
        ].unique()
    ) == {
        "ML_ridge",
        "ML_random_forest",
        "ML_hist_gradient_boosting",
    }

    # --------------------------------------------------------
    # Sin target
    # --------------------------------------------------------

    assert (
        "target_amount"
        not in predictions.columns
    )

    # --------------------------------------------------------
    # Predicciones finitas
    # --------------------------------------------------------

    assert np.isfinite(
        predictions[
            "prediction"
        ].to_numpy(
            dtype=float
        )
    ).all()


# ============================================================
# 10. Todos los modelos usan las mismas cohortes
# ============================================================

def test_all_ml_models_predict_same_cohorts(
    train_snapshot,
    prediction_snapshot,
):

    _, predictions = (
        run_ml_models(
            train_snapshot,
            prediction_snapshot,
            dev_M=DEV_M,
        )
    )

    cohorts_by_method = (
        predictions
        .groupby(
            "method_name"
        )[
            "accident_period"
        ]
        .apply(
            lambda values:
                tuple(
                    sorted(
                        values.tolist()
                    )
                )
        )
    )

    reference = (
        cohorts_by_method.iloc[0]
    )

    assert cohorts_by_method.apply(
        lambda cohorts:
            cohorts == reference
    ).all()


# ============================================================
# 11. Claves originales se conservan
# ============================================================

@pytest.mark.parametrize(
    "model_name",
    ML_MODEL_NAMES,
)
def test_prediction_preserves_original_cohorts(
    train_snapshot,
    prediction_snapshot,
    model_name,
):

    _, predictions = (
        run_ml_model(
            train_snapshot,
            prediction_snapshot,
            dev_M=DEV_M,
            model_name=model_name,
        )
    )

    pd.testing.assert_series_equal(
        predictions[
            "accident_period"
        ].reset_index(
            drop=True
        ),
        prediction_snapshot[
            "accident_period"
        ].reset_index(
            drop=True
        ),
        check_names=True,
    )

    pd.testing.assert_series_equal(
        predictions[
            "snapshot_dev_month"
        ].reset_index(
            drop=True
        ),
        prediction_snapshot[
            "snapshot_dev_month"
        ].reset_index(
            drop=True
        ),
        check_names=True,
    )


# ============================================================
# 12. Duplicados en catálogo
# ============================================================

def test_run_ml_models_rejects_duplicate_models(
    train_snapshot,
    prediction_snapshot,
):

    with pytest.raises(
        ValueError,
        match="duplicados",
    ):
        run_ml_models(
            train_snapshot,
            prediction_snapshot,
            dev_M=DEV_M,
            model_names=[
                "ridge",
                "ridge",
            ],
        )


def test_run_ml_models_rejects_empty_catalog(
    train_snapshot,
    prediction_snapshot,
):

    with pytest.raises(
        ValueError,
        match="vacío",
    ):
        run_ml_models(
            train_snapshot,
            prediction_snapshot,
            dev_M=DEV_M,
            model_names=[],
        )