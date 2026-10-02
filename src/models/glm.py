# src/models/glm.py

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.machine_learning import (
    build_ml_features,
    build_ml_target,
)


# ============================================================
# Configuración
# ============================================================

GLM_MODEL_NAMES = (
    "gamma_log",
    "gaussian_identity",
)


# ============================================================
# Artefacto entrenado
# ============================================================

@dataclass
class GLMModelArtifact:
    """
    Artefacto de un GLM entrenado en una fecha histórica.

    Attributes
    ----------
    model_name:
        Nombre del GLM.

    estimator:
        Pipeline sklearn ajustado.

    dev_M:
        Horizonte de madurez del escenario.

    feature_names:
        Variables utilizadas durante entrenamiento.
    """

    model_name: str
    estimator: object
    dev_M: int
    feature_names: tuple[str, ...]


# ============================================================
# Estimadores
# ============================================================

def make_glm_estimator(
    model_name: str,
):
    """
    Construye un GLM con especificación congelada.

    gamma_log
        Gamma con link log.

        power = 2
        link = log

    gaussian_identity
        Gaussian con link identidad.

        power = 0
        link = identity

    Notes
    -----
    alpha=0 elimina regularización para aproximarnos a una
    formulación GLM clásica.
    """

    model_name = str(
        model_name
    ).lower()

    # --------------------------------------------------------
    # Gamma + log
    # --------------------------------------------------------

    if model_name == "gamma_log":

        return Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    TweedieRegressor(
                        power=2.0,
                        alpha=0.0,
                        link="log",
                        max_iter=2_000,
                        tol=1e-7,
                    ),
                ),
            ]
        )

    # --------------------------------------------------------
    # Gaussian + identity
    # --------------------------------------------------------

    if model_name == "gaussian_identity":

        return Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    TweedieRegressor(
                        power=0.0,
                        alpha=0.0,
                        link="identity",
                        max_iter=2_000,
                        tol=1e-7,
                    ),
                ),
            ]
        )

    raise ValueError(
        f"GLM desconocido: {model_name!r}. "
        f"Opciones disponibles: {GLM_MODEL_NAMES}"
    )


# ============================================================
# Entrenamiento
# ============================================================

def fit_glm_model(
    train_snapshot: pd.DataFrame,
    *,
    dev_M: int,
    model_name: str,
) -> GLMModelArtifact:
    """
    Ajusta un GLM utilizando exactamente las mismas features
    y el mismo target empleados por Machine Learning.

    Target
    ------

        y = target_amount

    No se transforma el target.
    """

    # --------------------------------------------------------
    # Features
    #
    # build_ml_features ya contiene las validaciones de:
    #
    # - columnas;
    # - edades;
    # - observed_amount;
    # - dev_M.
    # --------------------------------------------------------

    X = build_ml_features(
        train_snapshot,
        dev_M=dev_M,
    )

    # --------------------------------------------------------
    # Target directo
    # --------------------------------------------------------

    y = build_ml_target(
        train_snapshot
    )

    # --------------------------------------------------------
    # Estimador
    # --------------------------------------------------------

    estimator = make_glm_estimator(
        model_name
    )

    # --------------------------------------------------------
    # Ajuste
    # --------------------------------------------------------

    estimator.fit(
        X,
        y,
    )

    return GLMModelArtifact(
        model_name=str(
            model_name
        ).lower(),
        estimator=estimator,
        dev_M=dev_M,
        feature_names=tuple(
            X.columns
        ),
    )


# ============================================================
# Predicción
# ============================================================

def predict_glm_snapshot(
    artifact: GLMModelArtifact,
    prediction_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predice directamente target_amount para las cohortes
    inmaduras.

    target_amount no puede estar presente durante predicción.
    """

    # --------------------------------------------------------
    # Guardrail contra leakage
    # --------------------------------------------------------

    if (
        "target_amount"
        in prediction_snapshot.columns
    ):
        raise ValueError(
            "prediction_snapshot no debe contener "
            "target_amount durante la predicción."
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = build_ml_features(
        prediction_snapshot,
        dev_M=artifact.dev_M,
    )

    # --------------------------------------------------------
    # Consistencia train / prediction
    # --------------------------------------------------------

    if tuple(
        X.columns
    ) != artifact.feature_names:

        raise ValueError(
            "Las features de predicción no coinciden "
            "con las utilizadas durante entrenamiento."
        )

    # --------------------------------------------------------
    # Predicción
    # --------------------------------------------------------

    prediction = np.asarray(
        artifact.estimator.predict(
            X
        ),
        dtype=float,
    )

    if not np.isfinite(
        prediction
    ).all():

        raise ValueError(
            "El GLM generó predicciones no finitas."
        )

    # No hacemos clipping.
    #
    # Gamma-log producirá predicciones positivas por
    # construcción.
    #
    # Gaussian-identity puede matemáticamente producir valores
    # negativos. Si ocurriera, las validaciones posteriores del
    # backtesting lo detectarán.

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    result = (
        prediction_snapshot
        .copy()
    )

    result[
        "prediction"
    ] = prediction

    result[
        "method_family"
    ] = "glm"

    result[
        "method_name"
    ] = (
        "GLM_"
        + artifact.model_name
    )

    return result


# ============================================================
# Ejecutar un GLM
# ============================================================

def run_glm_model(
    train_snapshot: pd.DataFrame,
    prediction_snapshot: pd.DataFrame,
    *,
    dev_M: int,
    model_name: str,
) -> tuple[
    GLMModelArtifact,
    pd.DataFrame,
]:
    """
    Ajusta un GLM y genera predicciones para una fecha
    histórica.
    """

    artifact = fit_glm_model(
        train_snapshot,
        dev_M=dev_M,
        model_name=model_name,
    )

    predictions = (
        predict_glm_snapshot(
            artifact,
            prediction_snapshot,
        )
    )

    return (
        artifact,
        predictions,
    )


# ============================================================
# Ejecutar catálogo GLM
# ============================================================

def run_glm_models(
    train_snapshot: pd.DataFrame,
    prediction_snapshot: pd.DataFrame,
    *,
    dev_M: int,
    model_names: Sequence[str] = GLM_MODEL_NAMES,
) -> tuple[
    dict[str, GLMModelArtifact],
    pd.DataFrame,
]:
    """
    Ejecuta todos los GLM utilizando exactamente:

    - el mismo train_snapshot;
    - el mismo prediction_snapshot;
    - las mismas features;
    - el mismo target_amount.
    """

    if model_names is None:
        raise ValueError(
            "model_names no puede ser None."
        )

    if len(
        model_names
    ) == 0:
        raise ValueError(
            "model_names no puede estar vacío."
        )

    normalized_names = [
        str(name).lower()
        for name in model_names
    ]

    if len(
        normalized_names
    ) != len(
        set(normalized_names)
    ):
        raise ValueError(
            "model_names contiene duplicados."
        )

    artifacts = {}
    prediction_tables = []

    # --------------------------------------------------------
    # Ejecutar catálogo
    # --------------------------------------------------------

    for model_name in (
        normalized_names
    ):

        artifact, predictions = (
            run_glm_model(
                train_snapshot,
                prediction_snapshot,
                dev_M=dev_M,
                model_name=model_name,
            )
        )

        artifacts[
            model_name
        ] = artifact

        prediction_tables.append(
            predictions
        )

    # --------------------------------------------------------
    # Consolidar
    # --------------------------------------------------------

    all_predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Misma población para todos los GLM
    # --------------------------------------------------------

    cohorts_by_method = (
        all_predictions
        .groupby(
            "method_name",
            observed=True,
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

    reference_cohorts = (
        cohorts_by_method.iloc[0]
    )

    if not cohorts_by_method.apply(
        lambda cohorts:
            cohorts
            == reference_cohorts
    ).all():

        raise ValueError(
            "No todos los GLM predijeron "
            "exactamente las mismas cohortes."
        )

    # --------------------------------------------------------
    # Número de predicciones
    # --------------------------------------------------------

    expected_rows = (
        len(
            prediction_snapshot
        )
        * len(
            normalized_names
        )
    )

    if (
        len(
            all_predictions
        )
        != expected_rows
    ):

        raise RuntimeError(
            "El número de predicciones GLM no "
            "coincide con el esperado."
        )

    return (
        artifacts,
        all_predictions,
    )