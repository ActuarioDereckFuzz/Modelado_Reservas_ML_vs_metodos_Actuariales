# src/experiment/backtesting.py

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from collections.abc import Mapping , Sequence


import numpy as np
import pandas as pd

from src.experiment.snapshots import (
    build_train_prediction_snapshots,
    reveal_targets,
)

from src.models.development import (
    DevelopmentAlternative,
    run_development_alternatives,
)

from src.models.machine_learning import (
    ML_MODEL_NAMES,
    MLModelArtifact,
    run_ml_models,
)

from src.models.glm import (
    GLM_MODEL_NAMES,
    GLMModelArtifact,
    run_glm_models,
)


# ============================================================
# Constantes internas
# ============================================================

PREDICTION_KEY = [
    "scenario",
    "model_valuation_period",
    "accident_period",
    "method_name",
]

FREEZE_COLUMNS = [
    "scenario",
    "model_valuation_period",
    "accident_period",
    "method_name",
    "prediction",
]

REQUIRED_PREDICTION_COLUMNS = {
    "scenario",
    "model_valuation_period",
    "accident_period",
    "snapshot_dev_month",
    "observed_amount",
    "method_name",
    "prediction",
}

REQUIRED_EVALUATION_COLUMNS = {
    "scenario",
    "model_valuation_period",
    "accident_period",
    "snapshot_dev_month",
    "method_name",
    "prediction",
    "target_amount",
    "target_period",
}


# ============================================================
# Resultado de una valuación
# ============================================================

@dataclass
class DevelopmentBacktestResult:
    """
    Resultado completo de una fecha histórica de backtesting.

    Attributes
    ----------
    scenario:
        Escenario simulado.

    model_valuation_period:
        Fecha histórica en la que se supone que se ejecutó
        el modelo.

    train_snapshot:
        Información disponible para entrenamiento.

    prediction_snapshot:
        Cohortes observables pero inmaduras en la fecha de
        valuación.

    development_factors:
        Factores estimados por todas las alternativas de
        Desarrollo.

    frozen_predictions:
        Predicciones generadas antes de revelar targets.

    evaluated_predictions:
        Predicciones después de incorporar el target real.

    metrics:
        Métricas agregadas por método.

    prediction_fingerprint:
        Huella de las predicciones congeladas.
    """

    scenario: str
    model_valuation_period: pd.Period

    train_snapshot: pd.DataFrame
    prediction_snapshot: pd.DataFrame

    development_factors: pd.DataFrame

    frozen_predictions: pd.DataFrame
    evaluated_predictions: pd.DataFrame

    metrics: pd.DataFrame

    prediction_fingerprint: str

@dataclass
class DevelopmentBacktestExperimentResult:
    """
    Resultado consolidado del backtesting de Desarrollo para
    múltiples escenarios y fechas de valuación.

    Attributes
    ----------
    valuation_summary:
        Una fila por escenario y fecha de valuación con
        información de cobertura del experimento.

    development_factors:
        Factores estimados para todas las alternativas,
        escenarios y fechas de valuación.

    frozen_predictions:
        Todas las predicciones generadas antes de revelar
        target_amount.

    evaluated_predictions:
        Predicciones después de revelar targets y calcular
        errores individuales.

    valuation_metrics:
        Métricas por escenario, fecha de valuación y método.
    """

    valuation_summary: pd.DataFrame
    development_factors: pd.DataFrame
    frozen_predictions: pd.DataFrame
    evaluated_predictions: pd.DataFrame
    valuation_metrics: pd.DataFrame

@dataclass
class MLBacktestResult:
    """
    Resultado completo de una fecha histórica de backtesting ML.
    """

    scenario: str
    model_valuation_period: pd.Period

    train_snapshot: pd.DataFrame
    prediction_snapshot: pd.DataFrame

    model_artifacts: dict[str, MLModelArtifact]

    frozen_predictions: pd.DataFrame
    evaluated_predictions: pd.DataFrame

    metrics: pd.DataFrame

    prediction_fingerprint: str

@dataclass
class MLBacktestExperimentResult:
    """
    Resultado consolidado del backtesting de Machine Learning.
    """

    valuation_summary: pd.DataFrame
    frozen_predictions: pd.DataFrame
    evaluated_predictions: pd.DataFrame
    valuation_metrics: pd.DataFrame

@dataclass
class GLMBacktestResult:
    """
    Resultado completo de una fecha histórica de backtesting
    para los modelos GLM.
    """

    scenario: str
    model_valuation_period: pd.Period

    train_snapshot: pd.DataFrame
    prediction_snapshot: pd.DataFrame

    model_artifacts: dict[
        str,
        GLMModelArtifact,
    ]

    frozen_predictions: pd.DataFrame
    evaluated_predictions: pd.DataFrame

    metrics: pd.DataFrame

    prediction_fingerprint: str


@dataclass
class GLMBacktestExperimentResult:
    """
    Resultado consolidado del backtesting GLM para múltiples
    escenarios y fechas de valuación.
    """

    valuation_summary: pd.DataFrame
    frozen_predictions: pd.DataFrame
    evaluated_predictions: pd.DataFrame
    valuation_metrics: pd.DataFrame
# ============================================================
# Helpers generales
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
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} no contiene las columnas requeridas: "
            f"{sorted(missing)}"
        )


def _as_month_period(
    value,
    *,
    name: str,
) -> pd.Period:
    """
    Convierte un valor a Period mensual.
    """
    try:
        period = pd.Period(
            value,
            freq="M",
        )
    except Exception as exc:
        raise ValueError(
            f"{name} no puede convertirse a Period[M]: "
            f"{value!r}"
        ) from exc

    return period


# ============================================================
# Normalización del catálogo de Desarrollo
# ============================================================

def normalize_development_alternative(
    alternative,
) -> DevelopmentAlternative:
    """
    Convierte una alternativa del experimento al formato:

        (method, window, exclusion)

    Se aceptan:

    - tuples;
    - diccionarios con method / aggregation;
    - diccionarios con exclusion / outlier_rule.

    Esta función únicamente adapta representación. No modifica
    ninguna regla del experimento.
    """

    if isinstance(alternative, tuple):

        if len(alternative) != 3:
            raise ValueError(
                "Una alternativa tuple debe contener "
                "(method, window, exclusion)."
            )

        method, window, exclusion = alternative

        return (
            str(method).upper(),
            window,
            str(exclusion),
        )

    if isinstance(alternative, dict):

        method = alternative.get(
            "method",
            alternative.get("aggregation"),
        )

        window = alternative.get(
            "window"
        )

        exclusion = alternative.get(
            "exclusion",
            alternative.get("outlier_rule"),
        )

        if method is None:
            raise ValueError(
                "No se encontró method/aggregation "
                f"en {alternative}."
            )

        if window is None:
            raise ValueError(
                "No se encontró window "
                f"en {alternative}."
            )

        if exclusion is None:
            raise ValueError(
                "No se encontró exclusion/outlier_rule "
                f"en {alternative}."
            )

        return (
            str(method).upper(),
            window,
            str(exclusion),
        )

    raise TypeError(
        "Formato de alternativa no reconocido: "
        f"{type(alternative)}."
    )


def normalize_development_alternatives(
    alternatives: Sequence,
) -> list[DevelopmentAlternative]:
    """
    Normaliza un catálogo completo de alternativas y verifica
    que no existan duplicados.
    """

    if not alternatives:
        raise ValueError(
            "alternatives no puede estar vacío."
        )

    normalized = [
        normalize_development_alternative(
            alternative
        )
        for alternative in alternatives
    ]

    if len(normalized) != len(set(normalized)):
        raise ValueError(
            "El catálogo contiene alternativas duplicadas."
        )

    return normalized


# ============================================================
# Congelamiento de predicciones
# ============================================================

def prediction_fingerprint(
    predictions: pd.DataFrame,
) -> str:
    """
    Genera una huella SHA-256 de las claves y predicciones.

    El fingerprint permite comprobar que revelar los targets no
    modifica las predicciones previamente generadas.
    """

    _require_columns(
        predictions,
        set(FREEZE_COLUMNS),
        name="predictions",
    )

    frozen_view = (
        predictions[
            FREEZE_COLUMNS
        ]
        .sort_values(
            PREDICTION_KEY
        )
        .reset_index(drop=True)
    )

    hashes = pd.util.hash_pandas_object(
        frozen_view,
        index=False,
    ).to_numpy()

    return sha256(
        hashes.tobytes()
    ).hexdigest()


def validate_prediction_population(
    predictions: pd.DataFrame,
) -> None:
    """
    Valida una tabla de predicciones antes de revelar targets.
    """

    _require_columns(
        predictions,
        REQUIRED_PREDICTION_COLUMNS,
        name="predictions",
    )

    if "target_amount" in predictions.columns:
        raise ValueError(
            "Las predicciones no pueden contener "
            "target_amount antes de reveal_targets()."
        )

    if "target_period" in predictions.columns:

        if predictions[
            "target_period"
        ].isna().any():
            raise ValueError(
                "Existen target_period faltantes."
            )

        if not (
            predictions["target_period"]
            >
            predictions[
                "model_valuation_period"
            ]
        ).all():
            raise ValueError(
                "Las cohortes de predicción deben tener "
                "target_period posterior a "
                "model_valuation_period."
            )

    if predictions.empty:
        raise ValueError(
            "No existen predicciones para congelar."
        )

    if predictions.duplicated(
        PREDICTION_KEY
    ).any():
        raise ValueError(
            "Existen predicciones duplicadas por "
            f"{PREDICTION_KEY}."
        )

    if predictions["prediction"].isna().any():
        raise ValueError(
            "Existen predicciones faltantes."
        )

    prediction_values = predictions[
        "prediction"
    ].to_numpy(dtype=float)

    if not np.isfinite(
        prediction_values
    ).all():
        raise ValueError(
            "Existen predicciones no finitas."
        )

    if (prediction_values <= 0).any():
        raise ValueError(
            "Las predicciones deben ser positivas."
        )


def validate_same_cohorts_across_methods(
    predictions: pd.DataFrame,
) -> None:
    """
    Verifica que todos los métodos hayan predicho exactamente
    las mismas cohortes.
    """

    _require_columns(
        predictions,
        {
            "method_name",
            "accident_period",
        },
        name="predictions",
    )

    cohorts_by_method = (
        predictions
        .groupby(
            "method_name",
            observed=True,
        )["accident_period"]
        .apply(
            lambda values: tuple(
                sorted(values.tolist())
            )
        )
    )

    if cohorts_by_method.empty:
        raise ValueError(
            "No existen métodos para comparar."
        )

    reference = cohorts_by_method.iloc[0]

    same_population = cohorts_by_method.apply(
        lambda cohort_set:
            cohort_set == reference
    )

    if not same_population.all():
        failing_methods = (
            same_population.loc[
                ~same_population
            ]
            .index
            .tolist()
        )

        raise ValueError(
            "No todos los métodos utilizan las mismas "
            f"cohortes. Métodos inconsistentes: "
            f"{failing_methods}"
        )


def freeze_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Valida y congela conceptualmente una tabla de predicciones.

    Returns
    -------
    frozen_predictions:
        Copia profunda de las predicciones.

    fingerprint:
        SHA-256 de las claves y valores predichos.
    """

    validate_prediction_population(
        predictions
    )

    validate_same_cohorts_across_methods(
        predictions
    )

    frozen = predictions.copy(
        deep=True
    )

    fingerprint = prediction_fingerprint(
        frozen
    )

    return frozen, fingerprint


# ============================================================
# Validación posterior a reveal_targets
# ============================================================

def validate_revealed_predictions(
    frozen_predictions: pd.DataFrame,
    evaluated_predictions: pd.DataFrame,
    *,
    expected_fingerprint: str,
) -> None:
    """
    Verifica que reveal_targets() únicamente haya agregado
    información futura y no haya alterado las predicciones.
    """

    _require_columns(
        evaluated_predictions,
        REQUIRED_EVALUATION_COLUMNS,
        name="evaluated_predictions",
    )

    # --------------------------------------------------------
    # 1. La tabla congelada debe seguir intacta.
    # --------------------------------------------------------

    current_fingerprint = prediction_fingerprint(
        frozen_predictions
    )

    if (
        current_fingerprint
        != expected_fingerprint
    ):
        raise ValueError(
            "Las predicciones congeladas fueron modificadas."
        )

    # --------------------------------------------------------
    # 2. Debe mantenerse exactamente el mismo número de filas.
    # --------------------------------------------------------

    if (
        len(frozen_predictions)
        != len(evaluated_predictions)
    ):
        raise ValueError(
            "reveal_targets() modificó el número de "
            "predicciones."
        )

    # --------------------------------------------------------
    # 3. Las claves deben ser exactamente iguales.
    # --------------------------------------------------------

    before_keys = (
        frozen_predictions[
            PREDICTION_KEY
        ]
        .sort_values(
            PREDICTION_KEY
        )
        .reset_index(drop=True)
    )

    after_keys = (
        evaluated_predictions[
            PREDICTION_KEY
        ]
        .sort_values(
            PREDICTION_KEY
        )
        .reset_index(drop=True)
    )

    if not before_keys.equals(
        after_keys
    ):
        raise ValueError(
            "Las claves cambiaron durante reveal_targets()."
        )

    # --------------------------------------------------------
    # 4. Las predicciones deben ser idénticas.
    # --------------------------------------------------------

    before_prediction = (
        frozen_predictions[
            PREDICTION_KEY
            + ["prediction"]
        ]
        .sort_values(
            PREDICTION_KEY
        )
        .reset_index(drop=True)
    )

    after_prediction = (
        evaluated_predictions[
            PREDICTION_KEY
            + ["prediction"]
        ]
        .sort_values(
            PREDICTION_KEY
        )
        .reset_index(drop=True)
    )

    if not np.allclose(
        before_prediction[
            "prediction"
        ].to_numpy(dtype=float),
        after_prediction[
            "prediction"
        ].to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "Las predicciones fueron modificadas al "
            "revelar los targets."
        )

    # --------------------------------------------------------
    # 5. Targets válidos.
    # --------------------------------------------------------

    if evaluated_predictions[
        "target_amount"
    ].isna().any():
        raise ValueError(
            "Existen target_amount faltantes."
        )

    target_values = evaluated_predictions[
        "target_amount"
    ].to_numpy(dtype=float)

    if not np.isfinite(
        target_values
    ).all():
        raise ValueError(
            "Existen target_amount no finitos."
        )

    if (target_values <= 0).any():
        raise ValueError(
            "target_amount debe ser positivo."
        )

    # --------------------------------------------------------
    # 6. El target debe pertenecer al futuro respecto a la
    #    valuación del modelo.
    # --------------------------------------------------------

    if not (
        evaluated_predictions["target_period"]
        >
        evaluated_predictions[
            "model_valuation_period"
        ]
    ).all():
        raise ValueError(
            "Se encontró al menos un target que ya era "
            "observable en model_valuation_period."
        )


# ============================================================
# Diagnóstico temporal
# ============================================================

def add_experiment_end_diagnostic(
    evaluated_predictions: pd.DataFrame,
    *,
    final_period,
) -> pd.DataFrame:
    """
    Añade un indicador diagnóstico de si el target habría sido
    observable antes del cierre original del experimento.

    IMPORTANTE
    ----------
    Esta columna NO se utiliza para filtrar predicciones ni
    métricas.

    Su propósito es únicamente documentar qué observaciones
    habría eliminado el diseño anterior.
    """

    final_period = _as_month_period(
        final_period,
        name="final_period",
    )

    _require_columns(
        evaluated_predictions,
        {"target_period"},
        name="evaluated_predictions",
    )

    result = evaluated_predictions.copy()

    result[
        "target_revealed_by_experiment_end"
    ] = (
        result["target_period"]
        <= final_period
    )

    return result


# ============================================================
# Errores individuales
# ============================================================

def add_backtest_errors(
    evaluated_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula errores individuales de backtesting.

    Convención
    ----------
    error = prediction - target_amount

    Por tanto:

    error > 0
        sobreestimación.

    error < 0
        subestimación.
    """

    _require_columns(
        evaluated_predictions,
        {
            "prediction",
            "target_amount",
        },
        name="evaluated_predictions",
    )

    result = evaluated_predictions.copy()

    result["error"] = (
        result["prediction"]
        - result["target_amount"]
    )

    result["absolute_error"] = (
        result["error"].abs()
    )

    result["squared_error"] = (
        result["error"] ** 2
    )

    result["relative_error"] = (
        result["error"]
        / result["target_amount"]
    )

    result[
        "absolute_percentage_error"
    ] = (
        result["relative_error"].abs()
    )

    error_columns = [
        "error",
        "absolute_error",
        "squared_error",
        "relative_error",
        "absolute_percentage_error",
    ]

    values = result[
        error_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Se generaron errores no finitos."
        )

    return result


# ============================================================
# Métricas
# ============================================================

def summarize_backtest_metrics(
    evaluated_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resume métricas de backtesting por método.

    Métricas
    --------
    MAE
        Mean Absolute Error.

    RMSE
        Root Mean Squared Error.

    MAPE
        Mean Absolute Percentage Error.

    Bias
        Error promedio.

    Relative bias
        Error relativo promedio.
    """

    required = {
        "method_name",
        "prediction",
        "target_amount",
    }

    _require_columns(
        evaluated_predictions,
        required,
        name="evaluated_predictions",
    )

    if evaluated_predictions.empty:
        raise ValueError(
            "evaluated_predictions está vacío."
        )

    rows = []

    for (
        method_name,
        group,
    ) in evaluated_predictions.groupby(
        "method_name",
        observed=True,
        sort=True,
    ):

        prediction = group[
            "prediction"
        ].to_numpy(dtype=float)

        target = group[
            "target_amount"
        ].to_numpy(dtype=float)

        error = (
            prediction - target
        )

        relative_error = (
            error / target
        )

        rows.append(
            {
                "method_name":
                    method_name,

                "n_predictions":
                    int(len(group)),

                "mae":
                    float(
                        np.mean(
                            np.abs(error)
                        )
                    ),

                "rmse":
                    float(
                        np.sqrt(
                            np.mean(
                                error ** 2
                            )
                        )
                    ),

                "mape":
                    float(
                        np.mean(
                            np.abs(
                                relative_error
                            )
                        )
                    ),

                "bias":
                    float(
                        np.mean(error)
                    ),

                "relative_bias":
                    float(
                        np.mean(
                            relative_error
                        )
                    ),
            }
        )

    metrics = pd.DataFrame(
        rows
    )

    # Todos deben utilizar exactamente
    # la misma población.
    if (
        metrics[
            "n_predictions"
        ].nunique()
        != 1
    ):
        raise ValueError(
            "Los métodos no fueron evaluados sobre "
            "la misma cantidad de predicciones."
        )

    return metrics.reset_index(
        drop=True
    )


# ============================================================
# Validar targets comunes
# ============================================================

def validate_same_targets_across_methods(
    evaluated_predictions: pd.DataFrame,
) -> None:
    """
    Verifica que una misma cohorte tenga exactamente el mismo
    target independientemente del método utilizado.
    """

    _require_columns(
        evaluated_predictions,
        {
            "accident_period",
            "target_amount",
        },
        name="evaluated_predictions",
    )

    target_counts = (
        evaluated_predictions
        .groupby(
            "accident_period",
            observed=True,
        )["target_amount"]
        .nunique()
    )

    if not (
        target_counts == 1
    ).all():
        raise ValueError(
            "Una o más cohortes presentan targets distintos "
            "entre métodos."
        )


# ============================================================
# Backtesting de una valuación
# ============================================================

def run_development_backtest_valuation(
    df: pd.DataFrame,
    *,
    scenario: str,
    model_valuation_period,
    dev_M: int,
    alternatives: Sequence,
    final_period=None,
) -> DevelopmentBacktestResult:
    """
    Ejecuta una valuación histórica completa para las
    alternativas del Método de Desarrollo.

    Flujo
    -----
    1. construir train_snapshot;
    2. construir prediction_snapshot;
    3. estimar alternativas de Desarrollo;
    4. generar predicciones;
    5. congelar predicciones;
    6. revelar targets;
    7. validar integridad temporal;
    8. calcular errores;
    9. calcular métricas.

    Notes
    -----
    `final_period`, cuando se proporciona, se utiliza
    exclusivamente como diagnóstico.

    Nunca participa en la selección del prediction_snapshot.
    """

    model_valuation_period = (
        _as_month_period(
            model_valuation_period,
            name="model_valuation_period",
        )
    )

    if dev_M <= 0:
        raise ValueError(
            "dev_M debe ser mayor que cero."
        )

    normalized_alternatives = (
        normalize_development_alternatives(
            alternatives
        )
    )

    # --------------------------------------------------------
    # 1. Snapshots históricos
    # --------------------------------------------------------

    (
        train_snapshot,
        prediction_snapshot,
    ) = build_train_prediction_snapshots(
        df=df,
        scenario=scenario,
        valuation_period=(
            model_valuation_period
        ),
    )

    if train_snapshot.empty:
        raise ValueError(
            "train_snapshot está vacío para "
            f"{scenario=} y "
            f"{model_valuation_period=}."
        )

    if prediction_snapshot.empty:
        raise ValueError(
            "prediction_snapshot está vacío para "
            f"{scenario=} y "
            f"{model_valuation_period=}."
        )

    # --------------------------------------------------------
    # 2. Ejecutar Desarrollo
    # --------------------------------------------------------

    (
        development_factors,
        development_predictions,
    ) = run_development_alternatives(
        train_snapshot,
        prediction_snapshot,
        alternatives=(
            normalized_alternatives
        ),
        dev_M=dev_M,
    )

    # --------------------------------------------------------
    # 3. Congelar predicciones
    # --------------------------------------------------------

    (
        frozen_predictions,
        fingerprint,
    ) = freeze_predictions(
        development_predictions
    )

    # --------------------------------------------------------
    # 4. Revelar targets
    # --------------------------------------------------------

    evaluated_predictions = reveal_targets(
        predictions=(
            frozen_predictions
        ),
        df=df,
        scenario=scenario,
    )

    # --------------------------------------------------------
    # 5. Validar integridad
    # --------------------------------------------------------

    validate_revealed_predictions(
        frozen_predictions,
        evaluated_predictions,
        expected_fingerprint=(
            fingerprint
        ),
    )

    validate_same_targets_across_methods(
        evaluated_predictions
    )

    # --------------------------------------------------------
    # 6. Diagnóstico respecto al cierre original
    # --------------------------------------------------------

    if final_period is not None:

        evaluated_predictions = (
            add_experiment_end_diagnostic(
                evaluated_predictions,
                final_period=final_period,
            )
        )

    # --------------------------------------------------------
    # 7. Errores
    # --------------------------------------------------------

    evaluated_predictions = (
        add_backtest_errors(
            evaluated_predictions
        )
    )

    # --------------------------------------------------------
    # 8. Métricas
    # --------------------------------------------------------

    metrics = summarize_backtest_metrics(
        evaluated_predictions
    )

    # --------------------------------------------------------
    # 9. Resultado estructurado
    # --------------------------------------------------------

    return DevelopmentBacktestResult(
        scenario=scenario,
        model_valuation_period=(
            model_valuation_period
        ),
        train_snapshot=train_snapshot,
        prediction_snapshot=(
            prediction_snapshot
        ),
        development_factors=(
            development_factors
        ),
        frozen_predictions=(
            frozen_predictions
        ),
        evaluated_predictions=(
            evaluated_predictions
        ),
        metrics=metrics,
        prediction_fingerprint=(
            fingerprint
        ),
    )

def _normalize_valuation_periods(
    valuation_periods: Sequence,
) -> list[pd.Period]:
    """
    Normaliza una secuencia de fechas de valuación a Period[M]
    y verifica que no existan duplicados.

    Acepta listas, tuplas, Index y PeriodIndex.
    """

    if valuation_periods is None:
        raise ValueError(
            "valuation_periods no puede ser None."
        )

    if len(valuation_periods) == 0:
        raise ValueError(
            "valuation_periods no puede estar vacío."
        )

    periods = [
        _as_month_period(
            value,
            name="valuation_period",
        )
        for value in valuation_periods
    ]

    if len(periods) != len(set(periods)):
        raise ValueError(
            "valuation_periods contiene fechas duplicadas."
        )

    return periods

def _build_valuation_summary(
    result: DevelopmentBacktestResult,
    *,
    final_period=None,
) -> dict:
    """
    Construye un resumen compacto de una valuación histórica.
    """

    prediction_snapshot = (
        result.prediction_snapshot
    )

    evaluated = (
        result.evaluated_predictions
    )

    summary = {
        "scenario":
            result.scenario,

        "model_valuation_period":
            result.model_valuation_period,

        "n_train_rows":
            int(
                len(
                    result.train_snapshot
                )
            ),

        "n_train_cohorts":
            int(
                result.train_snapshot[
                    "accident_period"
                ].nunique()
            ),

        "n_prediction_cohorts":
            int(
                prediction_snapshot[
                    "accident_period"
                ].nunique()
            ),

        "prediction_age_min":
            int(
                prediction_snapshot[
                    "snapshot_dev_month"
                ].min()
            ),

        "prediction_age_max":
            int(
                prediction_snapshot[
                    "snapshot_dev_month"
                ].max()
            ),

        "n_methods":
            int(
                result.frozen_predictions[
                    "method_name"
                ].nunique()
            ),

        "n_predictions":
            int(
                len(
                    result.frozen_predictions
                )
            ),

        "prediction_fingerprint":
            result.prediction_fingerprint,
    }

    if (
        final_period is not None
        and
        "target_revealed_by_experiment_end"
        in evaluated.columns
    ):
        cohort_status = (
            evaluated[
                [
                    "accident_period",
                    "target_revealed_by_experiment_end",
                ]
            ]
            .drop_duplicates()
        )

        summary[
            "n_target_by_experiment_end"
        ] = int(
            cohort_status[
                "target_revealed_by_experiment_end"
            ].sum()
        )

        summary[
            "n_target_after_experiment_end"
        ] = int(
            (
                ~cohort_status[
                    "target_revealed_by_experiment_end"
                ]
            ).sum()
        )

    return summary

def run_development_backtest_experiment(
    simulations: Mapping[str, pd.DataFrame],
    *,
    scenarios: Sequence[str],
    valuation_periods: Sequence,
    dev_M_by_scenario: Mapping[str, int],
    alternatives: Sequence,
    final_period=None,
) -> DevelopmentBacktestExperimentResult:
    """
    Ejecuta el backtesting completo del Método de Desarrollo
    para múltiples escenarios y fechas históricas.

    Cada combinación:

        scenario × model_valuation_period

    se ejecuta independientemente mediante
    `run_development_backtest_valuation()`.

    Por tanto, esta función no implementa lógica actuarial ni
    lógica adicional de elegibilidad. Únicamente orquesta y
    consolida resultados.

    Parameters
    ----------
    simulations:
        Mapping:

            scenario -> DataFrame simulado

    scenarios:
        Escenarios que forman parte del experimento.

    valuation_periods:
        Fechas históricas de valuación.

    dev_M_by_scenario:
        Mapping:

            scenario -> dev_M

    alternatives:
        Catálogo congelado de alternativas de Desarrollo.

    final_period:
        Cierre original del experimento.

        Se utiliza únicamente como diagnóstico.

    Returns
    -------
    DevelopmentBacktestExperimentResult
        Resultados consolidados.
    """

    if not scenarios:
        raise ValueError(
            "scenarios no puede estar vacío."
        )

    scenarios = list(
        scenarios
    )

    if len(scenarios) != len(
        set(scenarios)
    ):
        raise ValueError(
            "scenarios contiene duplicados."
        )

    normalized_valuations = (
        _normalize_valuation_periods(
            valuation_periods
        )
    )

    normalized_alternatives = (
        normalize_development_alternatives(
            alternatives
        )
    )

    # --------------------------------------------------------
    # Validar configuración por escenario.
    # --------------------------------------------------------

    for scenario in scenarios:

        if scenario not in simulations:
            raise KeyError(
                f"No existe simulación para "
                f"{scenario!r}."
            )

        if scenario not in dev_M_by_scenario:
            raise KeyError(
                f"No existe dev_M para "
                f"{scenario!r}."
            )

        if dev_M_by_scenario[
            scenario
        ] <= 0:
            raise ValueError(
                f"dev_M inválido para "
                f"{scenario!r}."
            )

    # --------------------------------------------------------
    # Contenedores consolidados.
    # --------------------------------------------------------

    all_summaries = []

    all_factors = []
    all_frozen_predictions = []
    all_evaluated_predictions = []
    all_metrics = []

    # --------------------------------------------------------
    # Experimento.
    # --------------------------------------------------------

    for scenario in scenarios:

        df = simulations[
            scenario
        ]

        dev_M = int(
            dev_M_by_scenario[
                scenario
            ]
        )

        for model_valuation_period in (
            normalized_valuations
        ):

            result = (
                run_development_backtest_valuation(
                    df=df,
                    scenario=scenario,
                    model_valuation_period=(
                        model_valuation_period
                    ),
                    dev_M=dev_M,
                    alternatives=(
                        normalized_alternatives
                    ),
                    final_period=final_period,
                )
            )

            # ------------------------------------------------
            # Resumen de la valuación.
            # ------------------------------------------------

            all_summaries.append(
                _build_valuation_summary(
                    result,
                    final_period=(
                        final_period
                    ),
                )
            )

            # ------------------------------------------------
            # Factores.
            #
            # Añadimos explícitamente scenario y fecha porque
            # constituyen parte de la identidad del ajuste.
            # ------------------------------------------------

            factors = (
                result.development_factors
                .copy()
            )

            factors["scenario"] = (
                scenario
            )

            factors[
                "model_valuation_period"
            ] = (
                model_valuation_period
            )

            all_factors.append(
                factors
            )

            # ------------------------------------------------
            # Predicciones congeladas.
            # ------------------------------------------------

            all_frozen_predictions.append(
                result.frozen_predictions.copy()
            )

            # ------------------------------------------------
            # Predicciones evaluadas.
            # ------------------------------------------------

            all_evaluated_predictions.append(
                result.evaluated_predictions.copy()
            )

            # ------------------------------------------------
            # Métricas.
            # ------------------------------------------------

            metrics = (
                result.metrics.copy()
            )

            metrics.insert(
                0,
                "model_valuation_period",
                model_valuation_period,
            )

            metrics.insert(
                0,
                "scenario",
                scenario,
            )

            all_metrics.append(
                metrics
            )

    # --------------------------------------------------------
    # Consolidación.
    # --------------------------------------------------------

    valuation_summary = pd.DataFrame(
        all_summaries
    )

    development_factors = pd.concat(
        all_factors,
        ignore_index=True,
    )

    frozen_predictions = pd.concat(
        all_frozen_predictions,
        ignore_index=True,
    )

    evaluated_predictions = pd.concat(
        all_evaluated_predictions,
        ignore_index=True,
    )

    valuation_metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Validaciones finales del experimento.
    # --------------------------------------------------------

    expected_valuations = (
        len(scenarios)
        * len(normalized_valuations)
    )

    if (
        len(valuation_summary)
        != expected_valuations
    ):
        raise RuntimeError(
            "El número de valuaciones consolidadas "
            "no coincide con el diseño experimental."
        )

    expected_methods = len(
        normalized_alternatives
    )

    methods_by_valuation = (
        valuation_metrics
        .groupby(
            [
                "scenario",
                "model_valuation_period",
            ],
            observed=True,
        )["method_name"]
        .nunique()
    )

    if not (
        methods_by_valuation
        == expected_methods
    ).all():
        raise RuntimeError(
            "Una o más valuaciones no contienen "
            "todas las alternativas de Desarrollo."
        )

    # Cada predicción evaluada debe corresponder exactamente
    # a una predicción previamente congelada.
    if (
        len(frozen_predictions)
        != len(evaluated_predictions)
    ):
        raise RuntimeError(
            "El número consolidado de predicciones "
            "cambió después de reveal_targets()."
        )

    # target_amount nunca debe aparecer en la tabla congelada.
    if (
        "target_amount"
        in frozen_predictions.columns
    ):
        raise RuntimeError(
            "Las predicciones congeladas contienen "
            "target_amount."
        )

    if (
        "target_amount"
        not in evaluated_predictions.columns
    ):
        raise RuntimeError(
            "Las predicciones evaluadas no contienen "
            "target_amount."
        )

    return (
        DevelopmentBacktestExperimentResult(
            valuation_summary=(
                valuation_summary
            ),
            development_factors=(
                development_factors
            ),
            frozen_predictions=(
                frozen_predictions
            ),
            evaluated_predictions=(
                evaluated_predictions
            ),
            valuation_metrics=(
                valuation_metrics
            ),
        )
    )

def run_ml_backtest_valuation(
    df: pd.DataFrame,
    *,
    scenario: str,
    model_valuation_period,
    dev_M: int,
    model_names: Sequence[str] = ML_MODEL_NAMES,
    final_period=None,
    random_state: int = 42,
) -> MLBacktestResult:
    """
    Ejecuta una valuación histórica completa para los modelos
    de Machine Learning.

    Flujo
    -----
    1. construir train_snapshot;
    2. construir prediction_snapshot;
    3. entrenar modelos ML;
    4. generar predicciones;
    5. congelar predicciones;
    6. revelar target_amount;
    7. validar integridad temporal;
    8. calcular errores;
    9. calcular métricas.

    Todos los modelos predicen directamente target_amount.
    """

    model_valuation_period = (
        _as_month_period(
            model_valuation_period,
            name="model_valuation_period",
        )
    )

    if dev_M <= 0:
        raise ValueError(
            "dev_M debe ser mayor que cero."
        )

    if model_names is None:
        raise ValueError(
            "model_names no puede ser None."
        )

    if len(model_names) == 0:
        raise ValueError(
            "model_names no puede estar vacío."
        )

    # ========================================================
    # 1. Snapshots históricos
    # ========================================================

    (
        train_snapshot,
        prediction_snapshot,
    ) = build_train_prediction_snapshots(
        df=df,
        scenario=scenario,
        valuation_period=model_valuation_period,
    )

    if train_snapshot.empty:
        raise ValueError(
            "train_snapshot está vacío para "
            f"{scenario=} y "
            f"{model_valuation_period=}."
        )

    if prediction_snapshot.empty:
        raise ValueError(
            "prediction_snapshot está vacío para "
            f"{scenario=} y "
            f"{model_valuation_period=}."
        )

    # ========================================================
    # 2. Modelos ML
    # ========================================================

    (
        model_artifacts,
        ml_predictions,
    ) = run_ml_models(
        train_snapshot,
        prediction_snapshot,
        dev_M=dev_M,
        model_names=model_names,
        random_state=random_state,
    )

    # ========================================================
    # 3. Congelar predicciones
    # ========================================================

    (
        frozen_predictions,
        fingerprint,
    ) = freeze_predictions(
        ml_predictions
    )

    # ========================================================
    # 4. Revelar targets
    # ========================================================

    evaluated_predictions = (
        reveal_targets(
            predictions=frozen_predictions,
            df=df,
            scenario=scenario,
        )
    )

    # ========================================================
    # 5. Validar integridad
    # ========================================================

    validate_revealed_predictions(
        frozen_predictions,
        evaluated_predictions,
        expected_fingerprint=fingerprint,
    )

    validate_same_targets_across_methods(
        evaluated_predictions
    )

    # ========================================================
    # 6. Diagnóstico FINAL_PERIOD
    # ========================================================

    if final_period is not None:

        evaluated_predictions = (
            add_experiment_end_diagnostic(
                evaluated_predictions,
                final_period=final_period,
            )
        )

    # ========================================================
    # 7. Errores
    # ========================================================

    evaluated_predictions = (
        add_backtest_errors(
            evaluated_predictions
        )
    )

    # ========================================================
    # 8. Métricas
    # ========================================================

    metrics = (
        summarize_backtest_metrics(
            evaluated_predictions
        )
    )

    return MLBacktestResult(
        scenario=scenario,
        model_valuation_period=(
            model_valuation_period
        ),
        train_snapshot=train_snapshot,
        prediction_snapshot=(
            prediction_snapshot
        ),
        model_artifacts=(
            model_artifacts
        ),
        frozen_predictions=(
            frozen_predictions
        ),
        evaluated_predictions=(
            evaluated_predictions
        ),
        metrics=metrics,
        prediction_fingerprint=(
            fingerprint
        ),
    )

def run_ml_backtest_experiment(
    simulations: Mapping[str, pd.DataFrame],
    *,
    scenarios: Sequence[str],
    valuation_periods: Sequence,
    dev_M_by_scenario: Mapping[str, int],
    model_names: Sequence[str] = ML_MODEL_NAMES,
    final_period=None,
    random_state: int = 42,
) -> MLBacktestExperimentResult:
    """
    Ejecuta el backtesting ML para múltiples escenarios y
    fechas históricas de valuación.

    Cada combinación:

        scenario × model_valuation_period

    se ejecuta independientemente mediante
    run_ml_backtest_valuation().
    """

    if not scenarios:
        raise ValueError(
            "scenarios no puede estar vacío."
        )

    scenarios = list(
        scenarios
    )

    if len(scenarios) != len(
        set(scenarios)
    ):
        raise ValueError(
            "scenarios contiene duplicados."
        )

    normalized_valuations = (
        _normalize_valuation_periods(
            valuation_periods
        )
    )

    if model_names is None:
        raise ValueError(
            "model_names no puede ser None."
        )

    if len(model_names) == 0:
        raise ValueError(
            "model_names no puede estar vacío."
        )

    normalized_model_names = [
        str(name).lower()
        for name in model_names
    ]

    if len(
        normalized_model_names
    ) != len(
        set(normalized_model_names)
    ):
        raise ValueError(
            "model_names contiene duplicados."
        )

    # ========================================================
    # Validar escenarios
    # ========================================================

    for scenario in scenarios:

        if scenario not in simulations:
            raise KeyError(
                f"No existe simulación para "
                f"{scenario!r}."
            )

        if (
            scenario
            not in dev_M_by_scenario
        ):
            raise KeyError(
                f"No existe dev_M para "
                f"{scenario!r}."
            )

        if (
            dev_M_by_scenario[
                scenario
            ] <= 0
        ):
            raise ValueError(
                f"dev_M inválido para "
                f"{scenario!r}."
            )

    # ========================================================
    # Contenedores
    # ========================================================

    all_summaries = []
    all_frozen_predictions = []
    all_evaluated_predictions = []
    all_metrics = []

    # ========================================================
    # Experimento
    # ========================================================

    for scenario in scenarios:

        df = simulations[
            scenario
        ]

        dev_M = int(
            dev_M_by_scenario[
                scenario
            ]
        )

        for model_valuation_period in (
            normalized_valuations
        ):

            result = (
                run_ml_backtest_valuation(
                    df=df,
                    scenario=scenario,
                    model_valuation_period=(
                        model_valuation_period
                    ),
                    dev_M=dev_M,
                    model_names=(
                        normalized_model_names
                    ),
                    final_period=(
                        final_period
                    ),
                    random_state=(
                        random_state
                    ),
                )
            )

            # ------------------------------------------------
            # Resumen de valuación
            # ------------------------------------------------

            evaluated = (
                result.evaluated_predictions
            )

            summary = {
                "scenario":
                    scenario,

                "model_valuation_period":
                    model_valuation_period,

                "n_train_rows":
                    int(
                        len(
                            result.train_snapshot
                        )
                    ),

                "n_train_cohorts":
                    int(
                        result.train_snapshot[
                            "accident_period"
                        ].nunique()
                    ),

                "n_prediction_cohorts":
                    int(
                        result.prediction_snapshot[
                            "accident_period"
                        ].nunique()
                    ),

                "prediction_age_min":
                    int(
                        result.prediction_snapshot[
                            "snapshot_dev_month"
                        ].min()
                    ),

                "prediction_age_max":
                    int(
                        result.prediction_snapshot[
                            "snapshot_dev_month"
                        ].max()
                    ),

                "n_models":
                    int(
                        result.frozen_predictions[
                            "method_name"
                        ].nunique()
                    ),

                "n_predictions":
                    int(
                        len(
                            result.frozen_predictions
                        )
                    ),

                "prediction_fingerprint":
                    result.prediction_fingerprint,
            }

            if (
                final_period is not None
                and
                "target_revealed_by_experiment_end"
                in evaluated.columns
            ):

                cohort_status = (
                    evaluated[
                        [
                            "accident_period",
                            "target_revealed_by_experiment_end",
                        ]
                    ]
                    .drop_duplicates()
                )

                summary[
                    "n_target_by_experiment_end"
                ] = int(
                    cohort_status[
                        "target_revealed_by_experiment_end"
                    ].sum()
                )

                summary[
                    "n_target_after_experiment_end"
                ] = int(
                    (
                        ~cohort_status[
                            "target_revealed_by_experiment_end"
                        ]
                    ).sum()
                )

            all_summaries.append(
                summary
            )

            # ------------------------------------------------
            # Predicciones
            # ------------------------------------------------

            all_frozen_predictions.append(
                result
                .frozen_predictions
                .copy()
            )

            all_evaluated_predictions.append(
                result
                .evaluated_predictions
                .copy()
            )

            # ------------------------------------------------
            # Métricas
            # ------------------------------------------------

            metrics = (
                result.metrics.copy()
            )

            metrics.insert(
                0,
                "model_valuation_period",
                model_valuation_period,
            )

            metrics.insert(
                0,
                "scenario",
                scenario,
            )

            all_metrics.append(
                metrics
            )

    # ========================================================
    # Consolidación
    # ========================================================

    valuation_summary = (
        pd.DataFrame(
            all_summaries
        )
    )

    frozen_predictions = (
        pd.concat(
            all_frozen_predictions,
            ignore_index=True,
        )
    )

    evaluated_predictions = (
        pd.concat(
            all_evaluated_predictions,
            ignore_index=True,
        )
    )

    valuation_metrics = (
        pd.concat(
            all_metrics,
            ignore_index=True,
        )
    )

    # ========================================================
    # Validaciones globales mínimas
    # ========================================================

    expected_valuations = (
        len(scenarios)
        * len(
            normalized_valuations
        )
    )

    if (
        len(
            valuation_summary
        )
        != expected_valuations
    ):
        raise RuntimeError(
            "El número de valuaciones ML "
            "no coincide con el esperado."
        )

    expected_models = len(
        normalized_model_names
    )

    models_by_valuation = (
        valuation_metrics
        .groupby(
            [
                "scenario",
                "model_valuation_period",
            ],
            observed=True,
        )[
            "method_name"
        ]
        .nunique()
    )

    if not (
        models_by_valuation
        == expected_models
    ).all():
        raise RuntimeError(
            "Una o más valuaciones no contienen "
            "todos los modelos ML."
        )

    if (
        len(
            frozen_predictions
        )
        != len(
            evaluated_predictions
        )
    ):
        raise RuntimeError(
            "El número de predicciones ML "
            "cambió después de reveal_targets()."
        )

    if (
        "target_amount"
        in frozen_predictions.columns
    ):
        raise RuntimeError(
            "Las predicciones ML congeladas "
            "contienen target_amount."
        )

    if (
        "target_amount"
        not in evaluated_predictions.columns
    ):
        raise RuntimeError(
            "Las predicciones ML evaluadas "
            "no contienen target_amount."
        )

    return (
        MLBacktestExperimentResult(
            valuation_summary=(
                valuation_summary
            ),
            frozen_predictions=(
                frozen_predictions
            ),
            evaluated_predictions=(
                evaluated_predictions
            ),
            valuation_metrics=(
                valuation_metrics
            ),
        )
    )

def run_glm_backtest_valuation(
    df: pd.DataFrame,
    *,
    scenario: str,
    model_valuation_period,
    dev_M: int,
    model_names: Sequence[str] = GLM_MODEL_NAMES,
    final_period=None,
) -> GLMBacktestResult:
    """
    Ejecuta una valuación histórica completa para los modelos
    GLM.

    Flujo
    -----
    1. construir train_snapshot;
    2. construir prediction_snapshot;
    3. ajustar modelos GLM;
    4. generar predicciones;
    5. congelar predicciones;
    6. revelar target_amount;
    7. validar integridad temporal;
    8. calcular errores;
    9. calcular métricas.

    Todos los GLM predicen directamente target_amount.
    """

    # ========================================================
    # 1. Normalizar valuación
    # ========================================================

    model_valuation_period = (
        _as_month_period(
            model_valuation_period,
            name="model_valuation_period",
        )
    )

    if dev_M <= 0:
        raise ValueError(
            "dev_M debe ser mayor que cero."
        )

    if model_names is None:
        raise ValueError(
            "model_names no puede ser None."
        )

    if len(model_names) == 0:
        raise ValueError(
            "model_names no puede estar vacío."
        )

    normalized_model_names = [
        str(name).lower()
        for name in model_names
    ]

    if len(
        normalized_model_names
    ) != len(
        set(normalized_model_names)
    ):
        raise ValueError(
            "model_names contiene duplicados."
        )

    # ========================================================
    # 2. Snapshots históricos
    # ========================================================

    (
        train_snapshot,
        prediction_snapshot,
    ) = build_train_prediction_snapshots(
        df=df,
        scenario=scenario,
        valuation_period=(
            model_valuation_period
        ),
    )

    if train_snapshot.empty:
        raise ValueError(
            "train_snapshot está vacío para "
            f"{scenario=} y "
            f"{model_valuation_period=}."
        )

    if prediction_snapshot.empty:
        raise ValueError(
            "prediction_snapshot está vacío para "
            f"{scenario=} y "
            f"{model_valuation_period=}."
        )

    # ========================================================
    # 3. Ajustar y ejecutar GLM
    # ========================================================

    (
        model_artifacts,
        glm_predictions,
    ) = run_glm_models(
        train_snapshot,
        prediction_snapshot,
        dev_M=dev_M,
        model_names=(
            normalized_model_names
        ),
    )

    # ========================================================
    # 4. Congelar predicciones
    # ========================================================

    (
        frozen_predictions,
        fingerprint,
    ) = freeze_predictions(
        glm_predictions
    )

    # ========================================================
    # 5. Revelar targets
    # ========================================================

    evaluated_predictions = (
        reveal_targets(
            predictions=(
                frozen_predictions
            ),
            df=df,
            scenario=scenario,
        )
    )

    # ========================================================
    # 6. Validar integridad temporal
    # ========================================================

    validate_revealed_predictions(
        frozen_predictions,
        evaluated_predictions,
        expected_fingerprint=(
            fingerprint
        ),
    )

    validate_same_targets_across_methods(
        evaluated_predictions
    )

    # ========================================================
    # 7. Diagnóstico FINAL_PERIOD
    # ========================================================

    if final_period is not None:

        evaluated_predictions = (
            add_experiment_end_diagnostic(
                evaluated_predictions,
                final_period=final_period,
            )
        )

    # ========================================================
    # 8. Errores individuales
    # ========================================================

    evaluated_predictions = (
        add_backtest_errors(
            evaluated_predictions
        )
    )

    # ========================================================
    # 9. Métricas
    # ========================================================

    metrics = (
        summarize_backtest_metrics(
            evaluated_predictions
        )
    )

    # ========================================================
    # 10. Resultado
    # ========================================================

    return GLMBacktestResult(
        scenario=scenario,
        model_valuation_period=(
            model_valuation_period
        ),
        train_snapshot=(
            train_snapshot
        ),
        prediction_snapshot=(
            prediction_snapshot
        ),
        model_artifacts=(
            model_artifacts
        ),
        frozen_predictions=(
            frozen_predictions
        ),
        evaluated_predictions=(
            evaluated_predictions
        ),
        metrics=metrics,
        prediction_fingerprint=(
            fingerprint
        ),
    )

def run_glm_backtest_experiment(
    simulations: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    scenarios: Sequence[str],
    valuation_periods: Sequence,
    dev_M_by_scenario: Mapping[
        str,
        int,
    ],
    model_names: Sequence[str] = (
        GLM_MODEL_NAMES
    ),
    final_period=None,
) -> GLMBacktestExperimentResult:
    """
    Ejecuta el backtesting GLM completo para múltiples
    escenarios y fechas históricas de valuación.

    Cada combinación:

        scenario × model_valuation_period

    se ejecuta independientemente mediante
    run_glm_backtest_valuation().
    """

    # ========================================================
    # 1. Validar escenarios
    # ========================================================

    if scenarios is None:
        raise ValueError(
            "scenarios no puede ser None."
        )

    if len(scenarios) == 0:
        raise ValueError(
            "scenarios no puede estar vacío."
        )

    scenarios = list(
        scenarios
    )

    if len(scenarios) != len(
        set(scenarios)
    ):
        raise ValueError(
            "scenarios contiene duplicados."
        )

    # ========================================================
    # 2. Normalizar valuaciones
    # ========================================================

    normalized_valuations = (
        _normalize_valuation_periods(
            valuation_periods
        )
    )

    # ========================================================
    # 3. Validar modelos
    # ========================================================

    if model_names is None:
        raise ValueError(
            "model_names no puede ser None."
        )

    if len(model_names) == 0:
        raise ValueError(
            "model_names no puede estar vacío."
        )

    normalized_model_names = [
        str(name).lower()
        for name in model_names
    ]

    if len(
        normalized_model_names
    ) != len(
        set(normalized_model_names)
    ):
        raise ValueError(
            "model_names contiene duplicados."
        )

    # ========================================================
    # 4. Validar configuración por escenario
    # ========================================================

    for scenario in scenarios:

        if scenario not in simulations:
            raise KeyError(
                f"No existe simulación para "
                f"{scenario!r}."
            )

        if (
            scenario
            not in dev_M_by_scenario
        ):
            raise KeyError(
                f"No existe dev_M para "
                f"{scenario!r}."
            )

        if (
            dev_M_by_scenario[
                scenario
            ] <= 0
        ):
            raise ValueError(
                f"dev_M inválido para "
                f"{scenario!r}."
            )

    # ========================================================
    # 5. Contenedores
    # ========================================================

    all_summaries = []
    all_frozen_predictions = []
    all_evaluated_predictions = []
    all_metrics = []

    # ========================================================
    # 6. Experimento
    # ========================================================

    for scenario in scenarios:

        df = simulations[
            scenario
        ]

        dev_M = int(
            dev_M_by_scenario[
                scenario
            ]
        )

        for model_valuation_period in (
            normalized_valuations
        ):

            result = (
                run_glm_backtest_valuation(
                    df=df,
                    scenario=scenario,
                    model_valuation_period=(
                        model_valuation_period
                    ),
                    dev_M=dev_M,
                    model_names=(
                        normalized_model_names
                    ),
                    final_period=(
                        final_period
                    ),
                )
            )

            # =================================================
            # Resumen de valuación
            # =================================================

            evaluated = (
                result.evaluated_predictions
            )

            summary = {
                "scenario":
                    scenario,

                "model_valuation_period":
                    model_valuation_period,

                "n_train_rows":
                    int(
                        len(
                            result.train_snapshot
                        )
                    ),

                "n_train_cohorts":
                    int(
                        result.train_snapshot[
                            "accident_period"
                        ].nunique()
                    ),

                "n_prediction_cohorts":
                    int(
                        result.prediction_snapshot[
                            "accident_period"
                        ].nunique()
                    ),

                "prediction_age_min":
                    int(
                        result.prediction_snapshot[
                            "snapshot_dev_month"
                        ].min()
                    ),

                "prediction_age_max":
                    int(
                        result.prediction_snapshot[
                            "snapshot_dev_month"
                        ].max()
                    ),

                "n_models":
                    int(
                        result.frozen_predictions[
                            "method_name"
                        ].nunique()
                    ),

                "n_predictions":
                    int(
                        len(
                            result.frozen_predictions
                        )
                    ),

                "prediction_fingerprint":
                    result.prediction_fingerprint,
            }

            # =================================================
            # Diagnóstico FINAL_PERIOD
            # =================================================

            if (
                final_period is not None
                and
                "target_revealed_by_experiment_end"
                in evaluated.columns
            ):

                cohort_status = (
                    evaluated[
                        [
                            "accident_period",
                            "target_revealed_by_experiment_end",
                        ]
                    ]
                    .drop_duplicates()
                )

                summary[
                    "n_target_by_experiment_end"
                ] = int(
                    cohort_status[
                        "target_revealed_by_experiment_end"
                    ].sum()
                )

                summary[
                    "n_target_after_experiment_end"
                ] = int(
                    (
                        ~cohort_status[
                            "target_revealed_by_experiment_end"
                        ]
                    ).sum()
                )

            all_summaries.append(
                summary
            )

            # =================================================
            # Predicciones congeladas
            # =================================================

            all_frozen_predictions.append(
                result
                .frozen_predictions
                .copy()
            )

            # =================================================
            # Predicciones evaluadas
            # =================================================

            all_evaluated_predictions.append(
                result
                .evaluated_predictions
                .copy()
            )

            # =================================================
            # Métricas
            # =================================================

            metrics = (
                result.metrics.copy()
            )

            metrics.insert(
                0,
                "model_valuation_period",
                model_valuation_period,
            )

            metrics.insert(
                0,
                "scenario",
                scenario,
            )

            all_metrics.append(
                metrics
            )

    # ========================================================
    # 7. Consolidación
    # ========================================================

    valuation_summary = (
        pd.DataFrame(
            all_summaries
        )
    )

    frozen_predictions = (
        pd.concat(
            all_frozen_predictions,
            ignore_index=True,
        )
    )

    evaluated_predictions = (
        pd.concat(
            all_evaluated_predictions,
            ignore_index=True,
        )
    )

    valuation_metrics = (
        pd.concat(
            all_metrics,
            ignore_index=True,
        )
    )

    # ========================================================
    # 8. Validaciones globales
    # ========================================================

    expected_valuations = (
        len(scenarios)
        * len(
            normalized_valuations
        )
    )

    if (
        len(
            valuation_summary
        )
        != expected_valuations
    ):
        raise RuntimeError(
            "El número de valuaciones GLM "
            "no coincide con el esperado."
        )

    expected_models = len(
        normalized_model_names
    )

    models_by_valuation = (
        valuation_metrics
        .groupby(
            [
                "scenario",
                "model_valuation_period",
            ],
            observed=True,
        )[
            "method_name"
        ]
        .nunique()
    )

    if not (
        models_by_valuation
        == expected_models
    ).all():
        raise RuntimeError(
            "Una o más valuaciones no contienen "
            "todos los modelos GLM."
        )

    # --------------------------------------------------------
    # Frozen vs evaluated
    # --------------------------------------------------------

    if (
        len(
            frozen_predictions
        )
        != len(
            evaluated_predictions
        )
    ):
        raise RuntimeError(
            "El número de predicciones GLM cambió "
            "después de reveal_targets()."
        )

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

    if (
        "target_amount"
        in frozen_predictions.columns
    ):
        raise RuntimeError(
            "Las predicciones GLM congeladas "
            "contienen target_amount."
        )

    if (
        "target_amount"
        not in evaluated_predictions.columns
    ):
        raise RuntimeError(
            "Las predicciones GLM evaluadas "
            "no contienen target_amount."
        )

    # ========================================================
    # 9. Resultado consolidado
    # ========================================================

    return (
        GLMBacktestExperimentResult(
            valuation_summary=(
                valuation_summary
            ),
            frozen_predictions=(
                frozen_predictions
            ),
            evaluated_predictions=(
                evaluated_predictions
            ),
            valuation_metrics=(
                valuation_metrics
            ),
        )
    )