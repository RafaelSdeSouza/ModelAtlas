"""Vector diagrams of input features and fitted SOM prototypes."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Patch
from numpy.typing import ArrayLike

from ._lattice_plot import INK, add_colorbar, color_scale, draw_lattice
from .geometry import ObliqueProjection
from .lattice import Cell
from .photoz import PhotoZBounds
from .som import SOMMap

FEATURE_COLORS = (
    "#2457a5",
    "#bd5140",
    "#168076",
    "#80599a",
    "#a87422",
    "#327b96",
    "#ad567c",
    "#69783b",
)


def plot_som_architecture(
    som_or_codebook: SOMMap | ArrayLike,
    *,
    values: ArrayLike | None = None,
    metric_label: str | None = None,
    feature_labels: Sequence[str] | None = None,
    cmap: str = "viridis",
    value_range: tuple[float, float] | None = None,
    highlight: Cell | None = None,
    show_connections: str = "all",
    emphasize_highlight: bool = True,
    show_neuron_centers: bool = True,
    show_colorbar: bool = True,
    depth: float = 0.40,
    figsize: tuple[float, float] = (6.0, 4.6),
    hatch_mask: ArrayLike | None = None,
    hatch_label: str | None = None,
    projection: ObliqueProjection | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Draw a fitted square or hexagonal SOM with input-to-prototype strands.

    Cap color is the supplied cell scalar or prototype component zero. Set
    value_range and projection explicitly for comparable figures. One line
    represents one codebook component. Faint lines use feature color only.
    Highlighted line widths encode the absolute component divided by the
    90th percentile for that feature, with clipping. Prism depth is constant
    in lattice units. show_connections accepts 'all', 'selected' or 'none'.

    A raw array uses rectangular topology. Pass a SOMMap for hexagonal
    topology. At most eight input features fit this compact labeled view.
    """
    som = (
        som_or_codebook
        if isinstance(som_or_codebook, SOMMap)
        else SOMMap(np.asarray(som_or_codebook))
    )
    if som.n_features > 8:
        raise ValueError("select or group input features explicitly for more than eight inputs")
    if not 0.10 <= depth <= 0.90:
        raise ValueError("depth must be between 0.10 and 0.90 lattice units")
    if feature_labels is not None and len(feature_labels) != som.n_features:
        raise ValueError("one label per feature is required")
    if values is None:
        display = som.codebook if som.display_codebook is None else som.display_codebook
        values = display[..., 0]
        names = feature_labels if feature_labels is not None else som.feature_labels
        name = "Feature 1" if names is None else names[0]
        unit = som.feature_units[0] if som.feature_units else ""
        metric_label = metric_label or (f"{name} ({unit})" if unit else f"{name} prototype value")
    return _plot_som_with_inputs(
        som,
        values,
        metric_label or "Cell metric",
        feature_labels=feature_labels,
        cmap=cmap,
        value_range=value_range,
        highlight=highlight,
        show_connections=show_connections,
        emphasize_highlight=emphasize_highlight,
        show_neuron_centers=show_neuron_centers,
        show_colorbar=show_colorbar,
        depth=depth,
        figsize=figsize,
        hatch_mask=hatch_mask,
        hatch_label=hatch_label,
        projection=projection or ObliqueProjection(),
        ax=ax,
    )


def _plot_som_with_inputs(
    som: SOMMap,
    values: ArrayLike,
    metric_label: str,
    *,
    feature_labels: Sequence[str] | None,
    cmap: str,
    value_range: tuple[float, float] | None,
    highlight: Cell | None,
    show_connections: str,
    emphasize_highlight: bool,
    show_neuron_centers: bool,
    show_colorbar: bool,
    depth: ArrayLike,
    figsize: tuple[float, float],
    projection: ObliqueProjection,
    ax: Axes | None,
    hatch_mask: ArrayLike | None = None,
    hatch_label: str | None = None,
    depth_label: str | None = None,
) -> tuple[Figure, Axes]:
    values = np.asarray(values, dtype=float)
    nf = som.n_features
    if values.shape != som.lattice.shape:
        raise ValueError("values must match the SOM lattice")
    if show_connections not in ("all", "selected", "none"):
        raise ValueError("show_connections must be 'all', 'selected', or 'none'")
    if show_connections == "selected" and highlight is None:
        raise ValueError("selected connections require a highlight neuron")
    if highlight is not None:
        highlight = tuple(highlight)
        som.lattice.hop_distances(highlight)
    labels = som.feature_labels if feature_labels is None else feature_labels
    if labels is None:
        labels = [f"x{j + 1}" for j in range(nf)]
    if len(labels) != nf:
        raise ValueError("one label per feature is required")
    hatch = None if hatch_mask is None else np.asarray(hatch_mask, dtype=bool)
    if hatch is not None and hatch.shape != values.shape:
        raise ValueError("hatch_mask must match the SOM lattice")
    norm, ticks = color_scale(values, value_range)
    color_map = colormaps[cmap]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure

    positions = draw_lattice(
        ax,
        som.lattice,
        values,
        norm=norm,
        cmap=color_map,
        projection=projection,
        depth=depth,
        highlight=highlight,
        hatch=hatch,
        show_centers=show_neuron_centers,
    )
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    left = xmin - max(0.68, 0.16 * (xmax - xmin))
    pad = max(0.40, 0.15 * (ymax - ymin))
    ys = np.linspace(ymax - pad, ymin + pad, nf) if nf > 1 else [(ymax + ymin) / 2]
    colors = [FEATURE_COLORS[k % len(FEATURE_COLORS)] for k in range(nf)]
    segments = []
    if show_connections == "all":
        links = positions.reshape(-1, 2)
        segments = [[(left, y), p] for y in ys for p in links]
        strand_colors = [(*to_rgb(color), 0.16) for color in colors for _ in links]
        ax.add_collection(LineCollection(segments, colors=strand_colors, lw=0.42, zorder=0.5))

    highlighted = highlight is not None and show_connections != "none" and emphasize_highlight
    if highlight is not None and (highlighted or show_connections == "selected"):
        strengths = np.abs(som.codebook[highlight])
        normalizer = np.percentile(np.abs(som.codebook), 90, axis=(0, 1)).clip(1e-8)
        position = positions[highlight]
        for y, color, strength in zip(ys, colors, np.clip(strengths / normalizer, 0, 1)):
            ax.plot(
                [left, position[0]],
                [y, position[1]],
                color=color,
                alpha=0.45 + 0.3 * strength if highlighted else 0.38,
                lw=0.75 + 1.15 * strength if highlighted else 0.55,
                zorder=1,
            )
    for y, color, label in zip(ys, colors, labels):
        ax.add_patch(Circle((left, y), 0.078, fc=color, ec="white", lw=0.7, zorder=4))
        ax.text(left - 0.15, y, label, fontsize=9, ha="right", va="center", color=INK)
    ax.set_xlim(left - max(0.66, (xmax - xmin) * 0.18), xmax)
    if show_colorbar:
        add_colorbar(fig, ax, norm, color_map, metric_label, ticks)
    if hatch_label and hatch is not None and np.any(hatch):
        ax.legend(
            handles=[Patch(fc="white", ec=INK, hatch="///", label=hatch_label)],
            loc="upper left",
            fontsize=7.5,
            frameon=False,
        )
    if depth_label:
        ax.text(0.02, 0.02, depth_label, transform=ax.transAxes, fontsize=7.5, color=INK)
    setattr(
        ax,
        "_modelatlas_architecture",
        {
            "features": nf,
            "neurons": som.rows * som.cols,
            "connections": len(segments),
            "highlighted_connections": nf if highlighted else 0,
            "highlight": highlight,
            "units": "fitted codebook components",
            "colored_metric": metric_label,
            "topology": som.topology,
        },
    )
    return fig, ax


def plot_photoz_architecture(
    bounds: PhotoZBounds,
    som: SOMMap | ArrayLike,
    *,
    metric: str = "identified_width",
    **kwargs,
) -> tuple[Figure, Axes]:
    """Color the fitted SOM architecture by a photo-z cell diagnostic."""
    from .photoz_plot import _METRICS, _FRACTION_METRICS

    if not isinstance(bounds, PhotoZBounds):
        raise TypeError("bounds must be a PhotoZBounds")
    if metric not in _METRICS:
        raise ValueError(f"unknown photo-z metric: {metric}")
    m = som if isinstance(som, SOMMap) else SOMMap(np.asarray(som))
    if m.lattice.shape != bounds.n_target.shape or m.topology != bounds.topology:
        raise ValueError("som and bounds must have matching shape and topology")
    label, cmap = _METRICS[metric]
    defaults = {
        "metric_label": label,
        "cmap": cmap,
        "hatch_mask": bounds.unidentified,
        "hatch_label": "No usable spectroscopic redshift",
    }
    if metric in _FRACTION_METRICS:
        defaults["value_range"] = (0.0, 1.0)
    defaults.update(kwargs)
    return plot_som_architecture(m, values=bounds.metric(metric), **defaults)
