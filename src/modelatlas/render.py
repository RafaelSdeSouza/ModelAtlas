"""SOM codebook and component plane visualizations."""

from collections.abc import Sequence
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from ._lattice_plot import INK, add_colorbar, color_scale, draw_lattice
from .architecture import _plot_som_with_inputs
from .geometry import ObliqueProjection
from .lattice import Cell
from .som import SOMMap


def _metric(som: SOMMap, color_by: int | str) -> tuple[NDArray, str]:
    if isinstance(color_by, Integral):
        k = int(color_by)
        if not 0 <= k < som.n_features:
            raise ValueError("color feature out of range")
        display = som.codebook if som.display_codebook is None else som.display_codebook
        label = som.feature_labels[k] if som.feature_labels else f"Feature {k + 1}"
        if som.feature_units and som.feature_units[k]:
            label += f" ({som.feature_units[k]})"
        return display[..., k], label
    if color_by == "occupancy":
        return som.occupancy(), "Observations per prototype"
    if color_by == "u_matrix":
        return som.u_matrix(), "Mean neighbor distance (fitted feature units)"
    raise ValueError("color_by must be a feature index, 'occupancy', or 'u_matrix'")


def plot_som(
    map_or_weights: SOMMap | ArrayLike,
    *,
    color_by: int | str = 0,
    labels: Sequence[str] | None = None,
    highlight: Cell | None = None,
    depth: float = 0.36,
    elevation: str = "constant",
    connections: bool = True,
    colorbar: bool = True,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (5.8, 4.5),
    azimuth: float = 1.0,
    projection: ObliqueProjection | None = None,
    value_range: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Render a square or hexagonal codebook as oblique, vector-native prisms.

    Color selects a codebook component, occupancy or U-distance. By default
    all prisms have the same thickness. With elevation='occupancy', thickness
    is depth times [0.26 + 0.74 sqrt(count/max_count)]. The legacy azimuth
    factor controls tilt when no explicit ObliqueProjection is supplied.
    """
    som = (
        map_or_weights if isinstance(map_or_weights, SOMMap) else SOMMap(np.asarray(map_or_weights))
    )
    if not 0.04 <= depth <= 1.0:
        raise ValueError("depth must be in [0.04, 1.0]")
    if elevation not in ("constant", "occupancy"):
        raise ValueError("elevation must be 'constant' or 'occupancy'")
    if not 0.5 <= azimuth <= 1.4:
        raise ValueError("azimuth factor must be in [0.5, 1.4]")
    values, label = _metric(som, color_by)
    thickness: ArrayLike = depth
    depth_label = None
    if elevation == "occupancy":
        count = som.occupancy()
        thickness = depth * (0.26 + 0.74 * np.sqrt(count / max(1, count.max())))
        depth_label = "Thickness increases with √occupancy"
    return _plot_som_with_inputs(
        som,
        values,
        label,
        feature_labels=labels,
        cmap=cmap,
        value_range=value_range,
        highlight=highlight,
        show_connections="all" if connections else "none",
        emphasize_highlight=False,
        show_neuron_centers=False,
        show_colorbar=colorbar,
        depth=thickness,
        depth_label=depth_label,
        figsize=figsize,
        projection=projection or ObliqueProjection(elevation=35 * azimuth),
        ax=ax,
    )


def plot_metric(
    map_: SOMMap | ArrayLike,
    metric: int | str = "u_matrix",
    *,
    cmap: str = "magma",
    figsize: tuple[float, float] = (3.6, 3.4),
    cell_labels: bool = False,
    value_range: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a top-down component, U-matrix or occupancy with true lattice cells."""
    som = map_ if isinstance(map_, SOMMap) else SOMMap(np.asarray(map_))
    values, label = _metric(som, metric)
    norm, ticks = color_scale(values, value_range)
    color_map = colormaps[cmap]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure
    centers = draw_lattice(
        ax,
        som.lattice,
        values,
        norm=norm,
        cmap=color_map,
        projection=ObliqueProjection(azimuth=0, elevation=90),
        depth=0,
    )
    if cell_labels:
        for center, value in zip(centers.reshape(-1, 2), values.ravel()):
            rgb = np.asarray(color_map(norm(value))[:3])
            contrast = INK if rgb @ [0.2126, 0.7152, 0.0722] > 0.53 else "white"
            ax.text(
                *center,
                f"{value:.0f}" if metric == "occupancy" else f"{value:.2g}",
                color=contrast,
                fontsize=8,
                ha="center",
                va="center",
                zorder=4,
            )
    add_colorbar(fig, ax, norm, color_map, label, ticks)
    return fig, ax
