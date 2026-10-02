from __future__ import annotations

import pandas as pd


# ============================================================
# HORIZONTES DE MADUREZ
# ============================================================

DEV_M = {
    "creciente": 60,
    "decreciente": 48,
    "mixto": 48,
}


# ============================================================
# CALENDARIO DEL EXPERIMENTO
# ============================================================

BACKTEST_START = pd.Period("2020-01", freq="M")
BACKTEST_END = pd.Period("2025-11", freq="M")
FINAL_PERIOD = pd.Period("2025-12", freq="M")

FIRST_ACCIDENT_PERIOD = pd.Period("2015-01", freq="M")

VALUATION_PERIODS = pd.period_range(
    BACKTEST_START,
    BACKTEST_END,
    freq="M",
)


# ============================================================
# MODELOS PREDICTIVOS
# ============================================================

MIN_MATURE_COHORTS_ML = 12


GLM_FEATURES = (
    "log_observed_amount",
    "dev_month",
    "dev_month_sq",
    "accident_time_idx",
    "accident_month_sin",
    "accident_month_cos",
    "delta_1m",
)


TREE_FEATURES = (
    "observed_amount",
    "dev_month",
    "accident_time_idx",
    "accident_month_sin",
    "accident_month_cos",
    "delta_1m",
    "mean_delta_3m",
    "std_delta_3m",
    "has_1m_history",
    "has_3m_history",
)


FORBIDDEN_PREDICTORS = (
    "target_amount",
    "target_period",
    "is_mature_at_valuation",
    "is_target_revealed_by_end",
)


# ============================================================
# ALTERNATIVAS DEL MÉTODO DE DESARROLLO
# ============================================================

DEVELOPMENT_ALTERNATIVES = (
    # Promedio simple
    ("S", 24, "none"),
    ("S", 24, "minmax"),
    ("S", 24, "std2"),
    ("S", "all", "none"),
    ("S", "all", "minmax"),
    ("S", "all", "std2"),

    # Ponderado por volumen
    ("VW", 24, "none"),
    ("VW", 24, "minmax"),
    ("VW", 24, "std2"),
    ("VW", "all", "none"),
    ("VW", "all", "minmax"),
    ("VW", "all", "std2"),

    # Mediana
    ("M", 24, "none"),
    ("M", 24, "minmax"),
    ("M", 24, "std2"),
    ("M", "all", "none"),
    ("M", "all", "minmax"),
    ("M", "all", "std2"),

    # Ventanas adicionales
    ("VW", 6, "none"),
    ("VW", 12, "none"),
    ("VW", 36, "none"),
)


# ============================================================
# EVALUACIÓN
# ============================================================

RELATIVE_SELECTION_THRESHOLD = 0.05


DEV_BANDS = {
    60: (
        (0, 5),
        (6, 11),
        (12, 23),
        (24, 35),
        (36, 47),
        (48, 59),
    ),
    48: (
        (0, 5),
        (6, 11),
        (12, 23),
        (24, 35),
        (36, 47),
    ),
}