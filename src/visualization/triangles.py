from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_triangle_heatmap(
    triangle: pd.DataFrame,
    title: str,
    figsize: tuple[int, int] = (13, 7),
    x_step: int = 6,
    y_step: int = 12,
):
    """
    Representa un triángulo actuarial mediante un mapa de calor.

    Los valores NaN corresponden a celdas no observables
    a la fecha de valuación y se mantienen fuera de la escala.

    Parameters
    ----------
    triangle:
        Triángulo con accident_period en filas y dev_month
        en columnas.

    title:
        Título del gráfico.

    figsize:
        Tamaño de la figura.

    x_step:
        Separación entre etiquetas de edad de desarrollo.

    y_step:
        Separación entre etiquetas de periodos de ocurrencia.

    Returns
    -------
    fig, ax
    """

    if triangle.empty:
        raise ValueError(
            "El triángulo está vacío."
        )

    values = triangle.to_numpy(dtype=float)

    masked_values = np.ma.masked_invalid(
        values
    )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    image = ax.imshow(
        masked_values,
        aspect="auto",
    )

    # ---------------------------------------------------------
    # Eje X: edad de desarrollo
    # ---------------------------------------------------------
    x_positions = np.arange(
        0,
        len(triangle.columns),
        x_step,
    )

    ax.set_xticks(x_positions)

    ax.set_xticklabels(
        [
            triangle.columns[i]
            for i in x_positions
        ]
    )

    # ---------------------------------------------------------
    # Eje Y: periodo de ocurrencia
    # ---------------------------------------------------------
    y_positions = np.arange(
        0,
        len(triangle.index),
        y_step,
    )

    ax.set_yticks(y_positions)

    ax.set_yticklabels(
        [
            str(triangle.index[i])
            for i in y_positions
        ]
    )

    ax.set_xlabel(
        "Edad de desarrollo"
    )

    ax.set_ylabel(
        "Periodo de ocurrencia"
    )

    ax.set_title(title)

    fig.colorbar(
        image,
        ax=ax,
        label="Loss Incurred",
    )

    fig.tight_layout()

    return fig, ax
