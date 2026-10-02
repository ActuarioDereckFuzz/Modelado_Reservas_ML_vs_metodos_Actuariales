from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


# ============================================================
# Tipos y configuración
# ============================================================

DevelopmentAlternative = tuple[str, int | str, str]


VALID_METHODS = {"S", "VW", "M"}
VALID_EXCLUSIONS = {"none", "minmax", "std2"}


TRAIN_REQUIRED_COLUMNS = {
    "accident_period",
    "snapshot_dev_month",
    "observed_amount",
    "target_amount",
}

PREDICTION_REQUIRED_COLUMNS = {
    "accident_period",
    "snapshot_dev_month",
    "observed_amount",
}


# ============================================================
# Validaciones internas
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


def _validate_dev_M(dev_M: int) -> None:
    """
    Valida el horizonte de madurez.
    """
    if not isinstance(dev_M, (int, np.integer)):
        raise TypeError("dev_M debe ser entero.")

    if dev_M <= 0:
        raise ValueError("dev_M debe ser mayor que cero.")


def _cohort_columns(df: pd.DataFrame) -> list[str]:
    """
    Define las columnas que identifican una cohorte.
    """
    cols = ["accident_period"]

    if "scenario" in df.columns:
        cols.insert(0, "scenario")

    return cols


def _factor_group_columns(df: pd.DataFrame) -> list[str]:
    """
    Columnas para agrupar factores por edad de desarrollo.
    """
    cols = ["dev_month"]

    if "scenario" in df.columns:
        cols.insert(0, "scenario")

    return cols


# ============================================================
# Nombre de alternativa
# ============================================================

def development_alternative_name(
    alternative: DevelopmentAlternative,
) -> str:
    """
    Convierte una alternativa de Desarrollo a su identificador.

    Ejemplos
    --------
    ("VW", 24, "none") -> "VW_24_none"
    ("S", "all", "std2") -> "S_all_std2"
    """
    method, window, exclusion = alternative

    return f"{method}_{window}_{exclusion}"


# ============================================================
# 1. Factores individuales
# ============================================================

def calculate_individual_factors(
    train_snapshot: pd.DataFrame,
    *,
    dev_M: int,
) -> pd.DataFrame:
    """
    Calcula factores edad-a-edad individuales a partir de un
    snapshot de entrenamiento.

    Para edades d < dev_M - 1:

        F_(i,d) = L_(i,d+1) / L_(i,d)

    Para la transición final:

        F_(i,dev_M-1) = target_amount / L_(i,dev_M-1)

    `target_amount` puede utilizarse aquí porque las cohortes del
    conjunto de entrenamiento ya alcanzaron dev_M antes de la
    fecha histórica de valuación.

    La función NO utiliza información de cohortes de predicción.
    """
    _validate_dev_M(dev_M)

    _require_columns(
        train_snapshot,
        TRAIN_REQUIRED_COLUMNS,
        name="train_snapshot",
    )

    if train_snapshot.empty:
        raise ValueError("train_snapshot está vacío.")

    df = train_snapshot.copy()

    df["snapshot_dev_month"] = pd.to_numeric(
        df["snapshot_dev_month"],
        errors="raise",
    ).astype(int)

    # --------------------------------------------------------
    # Solo esperamos edades 0,...,dev_M-1 en entrenamiento.
    # --------------------------------------------------------

    invalid_age = (
        (df["snapshot_dev_month"] < 0)
        | (df["snapshot_dev_month"] >= dev_M)
    )

    if invalid_age.any():
        raise ValueError(
            "train_snapshot contiene edades fuera del intervalo "
            f"[0, {dev_M - 1}]."
        )

    cohort_cols = _cohort_columns(df)

    key_cols = cohort_cols + ["snapshot_dev_month"]

    if df.duplicated(key_cols).any():
        raise ValueError(
            "train_snapshot contiene duplicados por "
            f"{key_cols}."
        )

    # --------------------------------------------------------
    # Cada cohorte madura debe contener toda la trayectoria
    # 0,...,dev_M-1.
    # --------------------------------------------------------

    age_summary = (
        df.groupby(cohort_cols, observed=True)
        ["snapshot_dev_month"]
        .agg(["min", "max", "nunique"])
    )

    incomplete = (
        (age_summary["min"] != 0)
        | (age_summary["max"] != dev_M - 1)
        | (age_summary["nunique"] != dev_M)
    )

    if incomplete.any():
        raise ValueError(
            "Existen cohortes de entrenamiento con trayectoria "
            "incompleta antes de dev_M."
        )

    if (df["observed_amount"] <= 0).any():
        raise ValueError(
            "observed_amount debe ser positivo."
        )

    if (df["target_amount"] <= 0).any():
        raise ValueError(
            "target_amount debe ser positivo."
        )

    # --------------------------------------------------------
    # Orden cronológico dentro de cada cohorte.
    # --------------------------------------------------------

    df = df.sort_values(
        cohort_cols + ["snapshot_dev_month"]
    ).reset_index(drop=True)

    grouped = df.groupby(
        cohort_cols,
        observed=True,
        sort=False,
    )

    df["next_dev_month"] = grouped[
        "snapshot_dev_month"
    ].shift(-1)

    df["next_amount"] = grouped[
        "observed_amount"
    ].shift(-1)

    # --------------------------------------------------------
    # Para la transición M-1 -> M utilizamos target_amount.
    # --------------------------------------------------------

    final_transition = (
        df["snapshot_dev_month"] == dev_M - 1
    )

    df.loc[
        final_transition,
        "next_dev_month",
    ] = dev_M

    df.loc[
        final_transition,
        "next_amount",
    ] = df.loc[
        final_transition,
        "target_amount",
    ]

    # --------------------------------------------------------
    # Verificar continuidad d -> d+1.
    # --------------------------------------------------------

    expected_next_age = (
        df["snapshot_dev_month"] + 1
    )

    if not np.array_equal(
        df["next_dev_month"].to_numpy(dtype=int),
        expected_next_age.to_numpy(dtype=int),
    ):
        raise ValueError(
            "La trayectoria de desarrollo no es consecutiva."
        )

    if df["next_amount"].isna().any():
        raise ValueError(
            "No fue posible obtener el monto del siguiente "
            "periodo de desarrollo."
        )

    if (df["next_amount"] <= 0).any():
        raise ValueError(
            "Los montos de desarrollo siguientes deben ser "
            "positivos."
        )

    df["individual_factor"] = (
        df["next_amount"]
        / df["observed_amount"]
    )

    if not np.isfinite(
        df["individual_factor"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Se generaron factores individuales no finitos."
        )

    if (df["individual_factor"] <= 0).any():
        raise ValueError(
            "Los factores individuales deben ser positivos."
        )

    output_cols = []

    if "scenario" in df.columns:
        output_cols.append("scenario")

    output_cols.extend(
        [
            "accident_period",
            "snapshot_dev_month",
            "observed_amount",
            "next_amount",
            "individual_factor",
        ]
    )

    result = df[output_cols].rename(
        columns={
            "snapshot_dev_month": "dev_month",
            "observed_amount": "current_amount",
        }
    )

    return result.reset_index(drop=True)


# ============================================================
# 2. Ventana histórica
# ============================================================

def apply_history_window(
    individual_factors: pd.DataFrame,
    window: int | str,
) -> pd.DataFrame:
    """
    Selecciona las cohortes históricas utilizadas para estimar
    cada factor edad-a-edad.

    window = "all"
        utiliza toda la historia disponible.

    window = N
        utiliza las N cohortes más recientes disponibles para
        cada edad de desarrollo.
    """
    required = {
        "accident_period",
        "dev_month",
        "individual_factor",
    }

    _require_columns(
        individual_factors,
        required,
        name="individual_factors",
    )

    if individual_factors.empty:
        raise ValueError(
            "individual_factors está vacío."
        )

    if window == "all":
        return individual_factors.copy().reset_index(drop=True)

    if not isinstance(window, (int, np.integer)):
        raise ValueError(
            "window debe ser 'all' o un entero positivo."
        )

    if window <= 0:
        raise ValueError(
            "window debe ser mayor que cero."
        )

    group_cols = _factor_group_columns(
        individual_factors
    )

    ordered = individual_factors.sort_values(
        group_cols + ["accident_period"]
    )

    selected = (
        ordered
        .groupby(
            group_cols,
            observed=True,
            group_keys=False,
        )
        .tail(int(window))
    )

    return selected.reset_index(drop=True)


# ============================================================
# 3. Reglas de exclusión
# ============================================================

def _exclude_group(
    group: pd.DataFrame,
    rule: str,
) -> pd.DataFrame:
    """
    Aplica una regla de exclusión a una sola edad.
    """
    if rule == "none":
        return group.copy()

    factors = group["individual_factor"]

    # Con muestras demasiado pequeñas no eliminamos datos.
    if len(group) <= 2:
        return group.copy()

    if rule == "minmax":
        # Si todos son iguales, no existe un extremo real
        # que tenga sentido eliminar.
        if factors.nunique() <= 1:
            return group.copy()

        min_idx = factors.idxmin()
        max_idx = factors.idxmax()

        return group.drop(
            index={min_idx, max_idx}
        )

    if rule == "std2":
        mean = factors.mean()
        std = factors.std(ddof=1)

        if (
            not np.isfinite(std)
            or np.isclose(std, 0.0)
        ):
            return group.copy()

        lower = mean - 2.0 * std
        upper = mean + 2.0 * std

        return group.loc[
            factors.between(
                lower,
                upper,
                inclusive="both",
            )
        ].copy()

    raise ValueError(
        f"Regla de exclusión desconocida: {rule!r}"
    )


def apply_exclusion_rule(
    individual_factors: pd.DataFrame,
    exclusion: str,
) -> pd.DataFrame:
    """
    Aplica la regla de exclusión independientemente para cada
    edad de desarrollo.

    Reglas disponibles
    ------------------
    none
        No elimina observaciones.

    minmax
        Elimina una observación mínima y una máxima por edad,
        siempre que exista suficiente información.

    std2
        Conserva observaciones dentro de media ± 2 desviaciones
        estándar.
    """
    if exclusion not in VALID_EXCLUSIONS:
        raise ValueError(
            f"exclusion debe pertenecer a "
            f"{sorted(VALID_EXCLUSIONS)}."
        )

    required = {
        "dev_month",
        "individual_factor",
    }

    _require_columns(
        individual_factors,
        required,
        name="individual_factors",
    )

    if individual_factors.empty:
        raise ValueError(
            "individual_factors está vacío."
        )

    group_cols = _factor_group_columns(
        individual_factors
    )

    pieces = []

    for _, group in individual_factors.groupby(
        group_cols,
        observed=True,
        sort=False,
    ):
        pieces.append(
            _exclude_group(
                group,
                exclusion,
            )
        )

    if not pieces:
        raise ValueError(
            "La regla de exclusión eliminó todas las "
            "observaciones."
        )

    result = pd.concat(
        pieces,
        ignore_index=True,
    )

    if result.empty:
        raise ValueError(
            "La regla de exclusión eliminó todas las "
            "observaciones."
        )

    return result


# ============================================================
# 4. Agregación de factores
# ============================================================

def aggregate_development_factors(
    individual_factors: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """
    Agrega factores individuales por edad.

    Métodos
    -------
    S
        Promedio simple.

    VW
        Promedio ponderado por volumen:

            sum(L_(i,d+1)) / sum(L_(i,d))

    M
        Mediana de factores individuales.
    """
    method = method.upper()

    if method not in VALID_METHODS:
        raise ValueError(
            f"method debe pertenecer a "
            f"{sorted(VALID_METHODS)}."
        )

    required = {
        "dev_month",
        "current_amount",
        "next_amount",
        "individual_factor",
    }

    _require_columns(
        individual_factors,
        required,
        name="individual_factors",
    )

    if individual_factors.empty:
        raise ValueError(
            "individual_factors está vacío."
        )

    group_cols = _factor_group_columns(
        individual_factors
    )

    rows = []

    for keys, group in individual_factors.groupby(
        group_cols,
        observed=True,
        sort=True,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))

        if method == "S":
            factor = group[
                "individual_factor"
            ].mean()

        elif method == "M":
            factor = group[
                "individual_factor"
            ].median()

        else:  # VW
            denominator = group[
                "current_amount"
            ].sum()

            if denominator <= 0:
                raise ValueError(
                    "La suma de current_amount debe ser "
                    "positiva para VW."
                )

            factor = (
                group["next_amount"].sum()
                / denominator
            )

        row.update(
            {
                "development_factor": float(factor),
                "n_obs": int(len(group)),
                "method": method,
            }
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    if not np.isfinite(
        result["development_factor"]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Se generaron factores agregados no finitos."
        )

    if (
        result["development_factor"] <= 0
    ).any():
        raise ValueError(
            "Los factores agregados deben ser positivos."
        )

    return result.sort_values(
        group_cols
    ).reset_index(drop=True)


# ============================================================
# 5. Ajuste completo de una alternativa
# ============================================================

def fit_development_factors(
    train_snapshot: pd.DataFrame,
    *,
    dev_M: int,
    method: str,
    window: int | str,
    exclusion: str,
) -> pd.DataFrame:
    """
    Ajusta una curva completa de factores de Desarrollo usando
    exclusivamente el snapshot de entrenamiento.

    Flujo
    -----
    train_snapshot
        -> factores individuales
        -> ventana histórica
        -> regla de exclusión
        -> agregación
        -> factores seleccionados
    """
    _validate_dev_M(dev_M)

    individual = calculate_individual_factors(
        train_snapshot,
        dev_M=dev_M,
    )

    windowed = apply_history_window(
        individual,
        window,
    )

    filtered = apply_exclusion_rule(
        windowed,
        exclusion,
    )

    factors = aggregate_development_factors(
        filtered,
        method,
    )

    observed_ages = set(
        factors["dev_month"].astype(int)
    )

    expected_ages = set(range(dev_M))

    missing_ages = sorted(
        expected_ages - observed_ages
    )

    if missing_ages:
        raise ValueError(
            "No fue posible estimar factores para las "
            f"edades: {missing_ages}"
        )

    factors = factors.loc[
        factors["dev_month"].between(
            0,
            dev_M - 1,
        )
    ].copy()

    if len(factors) != dev_M:
        raise ValueError(
            f"Se esperaban {dev_M} factores y se "
            f"obtuvieron {len(factors)}."
        )

    factors["window"] = window
    factors["exclusion"] = exclusion

    return factors.reset_index(drop=True)


# ============================================================
# 6. CDF / predicción individual
# ============================================================

def build_development_cdf(
    factors: pd.DataFrame,
    *,
    dev_M: int,
) -> pd.Series:
    """
    Construye el factor acumulado desde cada edad d hasta dev_M:

        CDF_d = product(f_j), j=d,...,dev_M-1
    """
    _validate_dev_M(dev_M)

    _require_columns(
        factors,
        {
            "dev_month",
            "development_factor",
        },
        name="factors",
    )

    series = (
        factors
        .set_index("dev_month")
        ["development_factor"]
        .sort_index()
    )

    expected_index = pd.Index(
        range(dev_M),
        name="dev_month",
    )

    series = series.reindex(
        expected_index
    )

    if series.isna().any():
        missing = series.index[
            series.isna()
        ].tolist()

        raise ValueError(
            "Faltan factores para las edades: "
            f"{missing}"
        )

    cdf = (
        series.iloc[::-1]
        .cumprod()
        .iloc[::-1]
    )

    cdf.name = "development_cdf"

    return cdf


def predict_development_ultimate(
    *,
    observed_amount: float,
    current_dev_month: int,
    development_cdf: pd.Series,
    dev_M: int,
) -> float:
    """
    Predice el Loss Incurred a dev_M para una observación.

        L_hat_M = L_d * CDF_d
    """
    _validate_dev_M(dev_M)

    if not np.isfinite(observed_amount):
        raise ValueError(
            "observed_amount debe ser finito."
        )

    if observed_amount <= 0:
        raise ValueError(
            "observed_amount debe ser positivo."
        )

    current_dev_month = int(
        current_dev_month
    )

    if not 0 <= current_dev_month < dev_M:
        raise ValueError(
            "current_dev_month debe satisfacer "
            f"0 <= d < {dev_M}."
        )

    try:
        cdf = float(
            development_cdf.loc[
                current_dev_month
            ]
        )

    except KeyError as exc:
        raise ValueError(
            "No existe CDF para "
            f"dev_month={current_dev_month}."
        ) from exc

    prediction = (
        float(observed_amount)
        * cdf
    )

    if (
        not np.isfinite(prediction)
        or prediction <= 0
    ):
        raise ValueError(
            "La predicción resultante no es "
            "positiva y finita."
        )

    return prediction


# ============================================================
# 7. Predicción de un snapshot completo
# ============================================================

def predict_development_snapshot(
    prediction_snapshot: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    dev_M: int,
    method_name: str,
) -> pd.DataFrame:
    """
    Genera predicciones de Desarrollo para todas las cohortes
    inmaduras de una fecha de valuación.

    El prediction_snapshot NO puede contener target_amount.
    Esto constituye una barrera explícita contra leakage.
    """
    _validate_dev_M(dev_M)

    _require_columns(
        prediction_snapshot,
        PREDICTION_REQUIRED_COLUMNS,
        name="prediction_snapshot",
    )

    if "target_amount" in prediction_snapshot.columns:
        raise ValueError(
            "prediction_snapshot no debe contener "
            "target_amount durante la predicción."
        )

    if prediction_snapshot.empty:
        return prediction_snapshot.copy().assign(
            method_family=pd.Series(dtype="object"),
            method_name=pd.Series(dtype="object"),
            development_cdf=pd.Series(dtype="float64"),
            prediction=pd.Series(dtype="float64"),
        )

    df = prediction_snapshot.copy()

    invalid_age = (
        (df["snapshot_dev_month"] < 0)
        | (df["snapshot_dev_month"] >= dev_M)
    )

    if invalid_age.any():
        raise ValueError(
            "prediction_snapshot contiene edades "
            "fuera del intervalo permitido."
        )

    if (df["observed_amount"] <= 0).any():
        raise ValueError(
            "observed_amount debe ser positivo."
        )

    cdf = build_development_cdf(
        factors,
        dev_M=dev_M,
    )

    df["development_cdf"] = (
        df["snapshot_dev_month"]
        .astype(int)
        .map(cdf)
    )

    if df["development_cdf"].isna().any():
        raise ValueError(
            "No fue posible asignar CDF a todas "
            "las cohortes de predicción."
        )

    df["prediction"] = (
        df["observed_amount"]
        * df["development_cdf"]
    )

    if not np.isfinite(
        df["prediction"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Se generaron predicciones no finitas."
        )

    if (df["prediction"] <= 0).any():
        raise ValueError(
            "Las predicciones deben ser positivas."
        )

    df["method_family"] = "development"
    df["method_name"] = method_name

    return df


# ============================================================
# 8. Ejecutar una alternativa completa
# ============================================================

def run_development_alternative(
    train_snapshot: pd.DataFrame,
    prediction_snapshot: pd.DataFrame,
    *,
    alternative: DevelopmentAlternative,
    dev_M: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ajusta y ejecuta una alternativa completa de Desarrollo.

    Returns
    -------
    factors:
        Tabla de factores seleccionados por edad.

    predictions:
        Predicciones para todas las cohortes inmaduras.
    """
    method, window, exclusion = alternative

    method = method.upper()

    if method not in VALID_METHODS:
        raise ValueError(
            f"Método desconocido: {method!r}"
        )

    if exclusion not in VALID_EXCLUSIONS:
        raise ValueError(
            f"Exclusión desconocida: {exclusion!r}"
        )

    name = development_alternative_name(
        alternative
    )

    factors = fit_development_factors(
        train_snapshot,
        dev_M=dev_M,
        method=method,
        window=window,
        exclusion=exclusion,
    )

    factors["method_name"] = name

    predictions = predict_development_snapshot(
        prediction_snapshot,
        factors,
        dev_M=dev_M,
        method_name=name,
    )

    return factors, predictions


# ============================================================
# 9. Ejecutar catálogo completo
# ============================================================

def run_development_alternatives(
    train_snapshot: pd.DataFrame,
    prediction_snapshot: pd.DataFrame,
    *,
    alternatives: Sequence[DevelopmentAlternative],
    dev_M: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta todas las alternativas de Desarrollo sobre los
    mismos snapshots de entrenamiento y predicción.
    """
    if not alternatives:
        raise ValueError(
            "alternatives no puede estar vacío."
        )

    all_factors = []
    all_predictions = []

    for alternative in alternatives:
        factors, predictions = (
            run_development_alternative(
                train_snapshot,
                prediction_snapshot,
                alternative=alternative,
                dev_M=dev_M,
            )
        )

        all_factors.append(factors)
        all_predictions.append(predictions)

    factor_table = pd.concat(
        all_factors,
        ignore_index=True,
    )

    prediction_table = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    return factor_table, prediction_table