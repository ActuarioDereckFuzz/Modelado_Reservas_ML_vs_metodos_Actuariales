from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def plot_development_factors(
    development_factors: pd.DataFrame,
    figsize: tuple[int, int] = (12, 6),
):
    """
    Grafica los factores edad-a-edad volumen-ponderados
    de todos los escenarios.

    Se incluye una referencia horizontal en 1.0 para
    distinguir crecimiento y reducción.
    """

    required = {
        "scenario",
        "dev_month",
        "volume_weighted_factor",
    }

    missing = (
        required
        - set(development_factors.columns)
    )

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: "
            f"{sorted(missing)}"
        )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    for scenario, group in (
        development_factors
        .groupby("scenario")
    ):

        group = group.sort_values(
            "dev_month"
        )

        ax.plot(
            group["dev_month"],
            group["volume_weighted_factor"],
            marker="o",
            markersize=3,
            label=scenario,
        )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Edad de desarrollo"
    )

    ax.set_ylabel(
        "Factor edad-a-edad"
    )

    ax.set_title(
        "Factores edad-a-edad volumen-ponderados"
    )

    ax.legend()

    fig.tight_layout()

    return fig, ax