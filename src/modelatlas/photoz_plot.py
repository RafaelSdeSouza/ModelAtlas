"""Plot observed spectroscopy and assumption-dependent photo-z bounds."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import ArrayLike, NDArray

from ._lattice_plot import INK, add_colorbar, color_scale, draw_lattice
from .geometry import ObliqueProjection
from .lattice import Cell, SOMLattice, Topology
from .photoz import PhotoZBounds, PhotoZCells

_METRICS = {
    "target_occupancy": ("Target galaxies per cell", "viridis"),
    "spectroscopic_coverage": ("Usable spectra / target", "cividis"),
    "attempt_fraction": ("Spectroscopic attempts / target", "cividis"),
    "success_given_attempt": ("Usable spectra / attempts", "cividis"),
    "identified_width": ("Width of identified fraction", "magma"),
    "population_width_contribution": ("Contribution to population bound width", "magma"),
    "lower_bound": ("Lower bound for P(z > z₀)", "viridis"),
    "upper_bound": ("Upper bound for P(z > z₀)", "viridis"),
    "observed_high_fraction": ("P(z > z₀) among usable spectra", "viridis"),
}
_FRACTION_METRICS = _METRICS.keys() - {"target_occupancy", "population_width_contribution"}


def plot_cell_plane(
    values: ArrayLike,
    *,
    label: str = "",
    cmap: str = "viridis",
    value_range: tuple[float, float] | None = None,
    depth: float = 0.48,
    elevation: str = "constant",
    figsize: tuple[float, float] = (4.6, 3.9),
    highlight: Cell | None = None,
    hatch_mask: ArrayLike | None = None,
    hatch_label: str | None = None,
    label_ticks: ArrayLike | None = None,
    ax: Axes | None = None,
    topology: Topology = "rectangular",
    projection: ObliqueProjection | None = None,
) -> tuple[Figure, Axes]:
    """Plot a cell scalar on a rectangular or hexagonal oblique lattice.

    Gray cells represent missing scalars. In constant mode, thickness is a
    graphical cue. In metric mode, cap height above the reference plane equals
    the color-normalized metric in lattice units (clipped to [0, 1]).
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D lattice")
    lattice = SOMLattice((values.shape[0], values.shape[1]), topology)
    hatch = None if hatch_mask is None else np.asarray(hatch_mask, dtype=bool)
    if hatch is not None and hatch.shape != values.shape:
        raise ValueError("hatch_mask must match values")
    if not 0.10 <= depth <= 0.85:
        raise ValueError("depth must be between 0.10 and 0.85")
    if elevation not in ("constant", "metric"):
        raise ValueError("elevation must be 'constant' or 'metric'")
    if highlight is not None:
        highlight = tuple(highlight)
        lattice.hop_distances(highlight)
    norm, ticks = color_scale(values, value_range)
    color_map = colormaps[cmap]
    lift: ArrayLike = 0.0
    if elevation == "metric":
        lift = np.where(np.isfinite(values), norm(values), 0.0)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure
    draw_lattice(
        ax,
        lattice,
        values,
        norm=norm,
        cmap=color_map,
        projection=projection or ObliqueProjection(),
        depth=depth,
        lift=lift,
        highlight=highlight,
        hatch=hatch,
    )
    add_colorbar(fig, ax, norm, color_map, label, ticks if label_ticks is None else label_ticks)
    if hatch_label and hatch is not None and np.any(hatch):
        ax.legend(
            handles=[Patch(fc="white", ec=INK, hatch="///", label=hatch_label)],
            loc="upper center",
            fontsize=7.5,
            frameon=False,
        )
    return fig, ax


def plot_photoz_map(
    bounds: PhotoZBounds, metric: str = "identified_width", **kwargs
) -> tuple[Figure, Axes]:
    """Plot one cellwise photo-z diagnostic, retaining the fitted topology."""
    if not isinstance(bounds, PhotoZBounds):
        raise TypeError("expected PhotoZBounds")
    if metric not in _METRICS:
        raise ValueError(f"unsupported photo-z metric: {metric}")
    label, cmap = _METRICS[metric]
    defaults = {"cmap": cmap, "label": label}
    if metric in _FRACTION_METRICS:
        defaults["value_range"] = (0.0, 1.0)
    if metric in (
        "identified_width",
        "lower_bound",
        "upper_bound",
        "population_width_contribution",
    ):
        defaults["hatch_mask"] = bounds.unidentified
        defaults["hatch_label"] = "No usable spectroscopic redshift"
    defaults.update(kwargs)
    return plot_cell_plane(bounds.metric(metric), topology=bounds.topology, **defaults)


def plot_photoz_diagnostics(
    bounds: PhotoZBounds,
    *,
    metrics: tuple[str, str, str, str] = (
        "target_occupancy",
        "spectroscopic_coverage",
        "observed_high_fraction",
        "identified_width",
    ),
    highlight: Cell | None = None,
    figsize: tuple[float, float] = (8.4, 7.2),
) -> tuple[Figure, NDArray]:
    """Draw four scalar diagnostics on identical SOM positions and camera."""
    if len(metrics) != 4:
        raise ValueError("four cell metrics are required")
    fig, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained")
    for ax, metric in zip(axes.ravel(), metrics):
        plot_photoz_map(bounds, metric=metric, highlight=highlight, ax=ax)
    return fig, axes


def plot_identification_sensitivity(
    cells: PhotoZCells,
    radii: ArrayLike,
    *,
    threshold: float = 1.2,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Draw population tail-bound endpoints against the assumed W1 radius.

    Each vertical range is a plug-in identified interval conditional on that
    radius. The shaded region is a sensitivity display without sampling
    coverage. Lines guide the eye between evaluated radii. The zero-radius
    result is plotted separately if the reference has an atom at the
    threshold. Dashed lines show the unrestricted missing-redshift bounds.
    """
    radii = np.asarray(radii, dtype=float)
    if (
        radii.ndim != 1
        or len(radii) < 2
        or not np.isfinite(radii).all()
        or radii[0] < 0
        or np.any(np.diff(radii) <= 0)
    ):
        raise ValueError("radii must contain at least two finite, increasing nonnegative values")
    lo, hi = np.array([cells.identify(threshold, r).population_interval() for r in radii]).T
    unrestricted = cells.identify(threshold, None).population_interval()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
    else:
        fig = ax.figure
    separate_zero = radii[0] == 0 and np.any(cells.spectroscopic_redshift == threshold)
    start = int(separate_zero)
    ax.fill_between(
        radii[start:], lo[start:], hi[start:], color="#3d7198", alpha=0.16, label="Identified range"
    )
    ax.plot(radii[start:], lo[start:], color="#245c83", marker=".", lw=1.5)
    ax.plot(radii[start:], hi[start:], color="#245c83", marker=".", lw=1.5)
    if separate_zero:
        ax.vlines(0, lo[0], hi[0], color="#245c83", lw=1.2)
        ax.scatter([0, 0], [lo[0], hi[0]], s=14, color="#245c83", zorder=4)
    for i, endpoint in enumerate(unrestricted):
        ax.axhline(
            endpoint,
            color="#86523d",
            lw=1,
            ls="--",
            label="Unrestricted endpoints" if i == 0 else None,
        )
    ax.set(
        xlabel=r"Assumed Wasserstein-1 radius (redshift units)",
        ylabel=rf"Target fraction with $z>{threshold:g}$",
        ylim=(0, 1),
    )
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax
