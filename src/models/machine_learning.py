# src/models/machine_learning.py

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuración
# ============================================================

ML_MODEL_NAMES = (
    "ridge",
    "random_forest",
    "hist_gradient_boosting",
)


ML_REQUIRED_COLUMNS = {
    "accident_period",
    "snapshot_dev_month",
    "observed_amount",
}


ML_TRAIN_REQUIRED_COLUMNS = (
    ML_REQUIRED_COLUMNS
    | {
        "target_amount",
    }
)


# ============================================================
# Artefacto entrenado
# ============================================================

@dataclass
class MLModelArtifact:
    """
    Artefacto de un modelo de Machine Learning entrenado en una
    fecha histórica de valuación.

    Attributes
    ----------
    model_name:
        Nombre interno del modelo.

    estimator:
        Estimador de scikit-learn ya ajustado.

    dev_M:
        Horizonte de madurez correspondiente al escenario.

    feature_names:
        Variables utilizadas durante el entrenamiento.
    """

    model_name: str
    estimator: object
    dev_M: int
    feature_names: tuple[str, ...]


# ============================================================
# Helpers
# ============================================================

def _require_columns(
    df: pd.DataFrame,
    required: set[str],
    *,
    name: str,
) -> None:
    """
    Verifica que un DataFrame contenga las columnas requeridas.
    """

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"{name} no contiene las columnas requeridas: "
            f"{sorted(missing)}"
        )


def _validate_dev_M(
    dev_M: int,
) -> None:
    """
    Valida el horizonte de madurez.
    """

    if not isinstance(
        dev_M,
        (int, np.integer),
    ):
        raise TypeError(
            "dev_M debe ser entero."
        )

    if dev_M <= 0:
        raise ValueError(
            "dev_M debe ser mayor que cero."
        )


def _validate_snapshot_values(
    df: pd.DataFrame,
    *,
    dev_M: int,
    name: str,
) -> None:
    """
    Valida las variables básicas utilizadas por ML.
    """

    if df.empty:
        raise ValueError(
            f"{name} está vacío."
        )

    age = pd.to_numeric(
        df[
            "snapshot_dev_month"
        ],
        errors="raise",
    )

    invalid_age = (
        (age < 0)
        | (age >= dev_M)
    )

    if invalid_age.any():
        raise ValueError(
            f"{name} contiene edades fuera "
            f"del intervalo [0, {dev_M - 1}]."
        )

    observed_amount = (
        pd.to_numeric(
            df[
                "observed_amount"
            ],
            errors="raise",
        )
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        observed_amount
    ).all():
        raise ValueError(
            f"{name} contiene observed_amount "
            "no finitos."
        )

    if (
        observed_amount <= 0
    ).any():
        raise ValueError(
            f"{name} contiene observed_amount "
            "no positivos."
        )


# ============================================================
# Features
# ============================================================

def build_ml_features(
    snapshot: pd.DataFrame,
    *,
    dev_M: int,
) -> pd.DataFrame:
    """
    Construye las variables explicativas disponibles en la
    fecha histórica de valuación.

    Features
    --------
    snapshot_dev_month
        Edad de desarrollo observada.

    development_fraction
        Proporción del horizonte de madurez ya observada:

            snapshot_dev_month / dev_M

    observed_amount
        Loss Incurred disponible en la fecha histórica.

    accident_year
        Año de ocurrencia de la cohorte.

    accident_month_sin
        Componente seno de la estacionalidad mensual.

    accident_month_cos
        Componente coseno de la estacionalidad mensual.

    Notes
    -----
    No se utiliza target_amount ni ninguna variable derivada
    de información futura.
    """

    _validate_dev_M(
        dev_M
    )

    _require_columns(
        snapshot,
        ML_REQUIRED_COLUMNS,
        name="snapshot",
    )

    _validate_snapshot_values(
        snapshot,
        dev_M=dev_M,
        name="snapshot",
    )

    # --------------------------------------------------------
    # Edad de desarrollo
    # --------------------------------------------------------

    snapshot_dev_month = (
        pd.to_numeric(
            snapshot[
                "snapshot_dev_month"
            ],
            errors="raise",
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Monto observado
    # --------------------------------------------------------

    observed_amount = (
        pd.to_numeric(
            snapshot[
                "observed_amount"
            ],
            errors="raise",
        )
        .astype(float)
    )

    # --------------------------------------------------------
    # Periodo de ocurrencia
    # --------------------------------------------------------

    accident_period = (
        pd.PeriodIndex(
            snapshot[
                "accident_period"
            ],
            freq="M",
        )
    )

    accident_year = (
        accident_period.year
        .astype(float)
    )

    accident_month = (
        accident_period.month
        .astype(float)
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = pd.DataFrame(
        {
            "snapshot_dev_month":
                snapshot_dev_month.to_numpy(),

            "development_fraction":
                (
                    snapshot_dev_month.to_numpy(
                        dtype=float
                    )
                    / float(dev_M)
                ),

            "observed_amount":
                observed_amount.to_numpy(
                    dtype=float
                ),

            "accident_year":
                accident_year,

            "accident_month_sin":
                np.sin(
                    2.0
                    * np.pi
                    * accident_month
                    / 12.0
                ),

            "accident_month_cos":
                np.cos(
                    2.0
                    * np.pi
                    * accident_month
                    / 12.0
                ),
        },
        index=snapshot.index,
    )

    # --------------------------------------------------------
    # Validación final
    # --------------------------------------------------------

    feature_values = (
        X.to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        feature_values
    ).all():
        raise ValueError(
            "Se generaron features ML no finitas."
        )

    return X


# ============================================================
# Target
# ============================================================

def build_ml_target(
    train_snapshot: pd.DataFrame,
) -> pd.Series:
    """
    Construye el objetivo del modelo ML.

    Target
    ------

        y = target_amount

    Es decir, Machine Learning predice directamente el mismo
    Loss Incurred en dev_M utilizado para evaluar los métodos
    actuariales de Desarrollo.

    No se aplican transformaciones logarítmicas ni
    retransformation.
    """

    _require_columns(
        train_snapshot,
        {
            "target_amount",
        },
        name="train_snapshot",
    )

    target = pd.to_numeric(
        train_snapshot[
            "target_amount"
        ],
        errors="raise",
    ).astype(float)

    target_values = (
        target.to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        target_values
    ).all():
        raise ValueError(
            "target_amount contiene valores "
            "no finitos."
        )

    if (
        target_values <= 0
    ).any():
        raise ValueError(
            "target_amount debe ser positivo."
        )

    return pd.Series(
        target_values,
        index=train_snapshot.index,
        name="target_amount",
    )


# ============================================================
# Estimadores
# ============================================================

def make_ml_estimator(
    model_name: str,
    *,
    random_state: int = 42,
):
    """
    Construye un estimador ML con hiperparámetros congelados.

    Modelos
    -------
    ridge
        Regresión lineal regularizada.

    random_forest
        Random Forest no lineal.

    hist_gradient_boosting
        Gradient Boosting basado en histogramas.

    Notes
    -----
    Los hiperparámetros se fijan antes de ejecutar el
    backtesting completo. No se optimizan utilizando los
    resultados futuros del experimento.
    """

    model_name = str(
        model_name
    ).lower()

    # --------------------------------------------------------
    # Ridge
    # --------------------------------------------------------

    if model_name == "ridge":

        return Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    Ridge(
                        alpha=1.0,
                    ),
                ),
            ]
        )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    if model_name == "random_forest":

        return (
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=random_state,
                n_jobs=-1,
            )
        )

    # --------------------------------------------------------
    # Hist Gradient Boosting
    # --------------------------------------------------------

    if (
        model_name
        == "hist_gradient_boosting"
    ):

        return (
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=31,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=random_state,
            )
        )

    raise ValueError(
        f"Modelo ML desconocido: "
        f"{model_name!r}. "
        f"Opciones disponibles: "
        f"{ML_MODEL_NAMES}"
    )


# ============================================================
# Entrenamiento
# ============================================================

def fit_ml_model(
    train_snapshot: pd.DataFrame,
    *,
    dev_M: int,
    model_name: str,
    random_state: int = 42,
) -> MLModelArtifact:
    """
    Ajusta un modelo ML utilizando exclusivamente información
    disponible en el train_snapshot histórico.

    El objetivo es target_amount, es decir, el Loss Incurred
    observado en dev_M.

    No se utiliza ninguna transformación del target.
    """

    _validate_dev_M(
        dev_M
    )

    _require_columns(
        train_snapshot,
        ML_TRAIN_REQUIRED_COLUMNS,
        name="train_snapshot",
    )

    _validate_snapshot_values(
        train_snapshot,
        dev_M=dev_M,
        name="train_snapshot",
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = build_ml_features(
        train_snapshot,
        dev_M=dev_M,
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = build_ml_target(
        train_snapshot
    )

    # --------------------------------------------------------
    # Estimador
    # --------------------------------------------------------

    estimator = make_ml_estimator(
        model_name,
        random_state=random_state,
    )

    # --------------------------------------------------------
    # Entrenamiento
    # --------------------------------------------------------

    estimator.fit(
        X,
        y,
    )

    # --------------------------------------------------------
    # Artefacto
    # --------------------------------------------------------

    return MLModelArtifact(
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

def predict_ml_snapshot(
    artifact: MLModelArtifact,
    prediction_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera predicciones de target_amount para todas las
    cohortes inmaduras de una fecha histórica.

    El prediction_snapshot no puede contener target_amount.
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

    _require_columns(
        prediction_snapshot,
        ML_REQUIRED_COLUMNS,
        name="prediction_snapshot",
    )

    _validate_snapshot_values(
        prediction_snapshot,
        dev_M=artifact.dev_M,
        name="prediction_snapshot",
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = build_ml_features(
        prediction_snapshot,
        dev_M=artifact.dev_M,
    )

    # --------------------------------------------------------
    # Comprobar consistencia entre entrenamiento/predicción
    # --------------------------------------------------------

    if tuple(
        X.columns
    ) != artifact.feature_names:
        raise ValueError(
            "Las features de predicción no coinciden "
            "con las utilizadas durante entrenamiento."
        )

    # --------------------------------------------------------
    # Predicción directa de target_amount
    # --------------------------------------------------------

    prediction = (
        np.asarray(
            artifact.estimator.predict(
                X
            ),
            dtype=float,
        )
    )

    if not np.isfinite(
        prediction
    ).all():
        raise ValueError(
            "El modelo ML generó predicciones "
            "no finitas."
        )

    # No se modifica artificialmente una predicción negativa.
    # Si ocurriera, se conserva para que las validaciones del
    # experimento detecten que ese estimador produjo un valor
    # incompatible con la naturaleza positiva del target.

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
    ] = "machine_learning"

    result[
        "method_name"
    ] = (
        "ML_"
        + artifact.model_name
    )

    return result


# ============================================================
# Ejecutar un modelo
# ============================================================

def run_ml_model(
    train_snapshot: pd.DataFrame,
    prediction_snapshot: pd.DataFrame,
    *,
    dev_M: int,
    model_name: str,
    random_state: int = 42,
) -> tuple[
    MLModelArtifact,
    pd.DataFrame,
]:
    """
    Ajusta un modelo ML y genera las predicciones para un
    prediction_snapshot histórico.
    """

    artifact = fit_ml_model(
        train_snapshot,
        dev_M=dev_M,
        model_name=model_name,
        random_state=random_state,
    )

    predictions = (
        predict_ml_snapshot(
            artifact,
            prediction_snapshot,
        )
    )

    return (
        artifact,
        predictions,
    )


# ============================================================
# Ejecutar catálogo ML
# ============================================================

def run_ml_models(
    train_snapshot: pd.DataFrame,
    prediction_snapshot: pd.DataFrame,
    *,
    dev_M: int,
    model_names: Sequence[str] = ML_MODEL_NAMES,
    random_state: int = 42,
) -> tuple[
    dict[str, MLModelArtifact],
    pd.DataFrame,
]:
    """
    Ejecuta un catálogo de modelos ML sobre exactamente los
    mismos snapshots históricos.

    Todos los modelos:

    - utilizan el mismo train_snapshot;
    - utilizan el mismo prediction_snapshot;
    - predicen el mismo target_amount;
    - producen una predicción por cohorte.
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
    # Ejecutar todos los modelos
    # --------------------------------------------------------

    for model_name in (
        normalized_names
    ):

        artifact, predictions = (
            run_ml_model(
                train_snapshot,
                prediction_snapshot,
                dev_M=dev_M,
                model_name=model_name,
                random_state=random_state,
            )
        )

        artifacts[
            model_name
        ] = artifact

        prediction_tables.append(
            predictions
        )

    # --------------------------------------------------------
    # Consolidar predicciones
    # --------------------------------------------------------

    all_predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Validar misma población
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
            "No todos los modelos ML predijeron "
            "exactamente las mismas cohortes."
        )

    # --------------------------------------------------------
    # Validar número de predicciones
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
            "El número de predicciones ML no coincide "
            "con el número esperado."
        )

    return (
        artifacts,
        all_predictions,
    )