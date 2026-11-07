"""Oblique prototype graph views and regularized GTM lattice plots."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from operator import index
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Patch, Polygon
from numpy.typing import ArrayLike, NDArray

from ._lattice_plot import INK, MISSING, add_colorbar, color_scale
from .adaptive import GTMModel, PrototypeGraph
from .export import save_figure
from .geometry import ObliqueProjection, ProjectionDiagnostics, projection_diagnostics
from .photoz import _validate_partition
from .photoz_plot import plot_cell_plane


@dataclass(frozen=True)
class PrototypeLayout:
    """Reusable PCA frame and camera, fitted once in shared feature coordinates."""

    center: NDArray[np.float64]
    basis: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    projection: ObliqueProjection
    bounds: tuple[float, float, float, float]

    def transform(self, prototypes: ArrayLike) -> NDArray[np.float64]:
        """Return projected centers in the fitted screen frame."""
        w = np.asarray(prototypes, dtype=float)
        if w.ndim != 2 or w.shape[1] != len(self.center) or not np.isfinite(w).all():
            raise ValueError("prototypes must be finite and match the fitted layout dimension")
        pc = (w - self.center) @ self.basis.T
        xyz = np.pad(pc, ((0, 0), (0, 3 - pc.shape[1])))
        return self.projection.project(xyz)

    def diagnostics(self, prototypes: ArrayLike) -> ProjectionDiagnostics:
        """Measure fitted-space versus screen-space distance distortion."""
        return projection_diagnostics(prototypes, self.transform(prototypes))


def fit_prototype_layout(
    prototypes: ArrayLike, *, projection: ObliqueProjection | None = None
) -> PrototypeLayout:
    """Fit up to three unwhitened principal axes and a shared oblique camera.

    Fit on pooled prototypes to compare models in the same feature space.
    Components use a deterministic sign convention. Nearly tied eigenvalues
    can still rotate the fitted basis. A layout never changes graph adjacency.
    """
    w = np.asarray(prototypes, dtype=float)
    if w.ndim != 2 or len(w) < 2 or w.shape[1] < 1 or not np.isfinite(w).all():
        raise ValueError("prototypes must be finite (n >= 2, d >= 1)")
    center = w.mean(axis=0)
    _, singular, basis = np.linalg.svd(w - center, full_matrices=False)
    basis = basis[:3]
    pivots = basis[np.arange(len(basis)), np.argmax(np.abs(basis), axis=1)]
    basis *= np.where(pivots < 0, -1.0, 1.0)[:, None]
    total = np.sum(singular**2)
    ratios = singular[:3] ** 2 / total if total > 0 else np.zeros(len(basis))
    camera = projection or ObliqueProjection(azimuth=20, elevation=52)
    pc = (w - center) @ basis.T
    xy = camera.project(np.pad(pc, ((0, 0), (0, 3 - pc.shape[1]))))
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    return PrototypeLayout(center, basis, ratios, camera, (xmin, xmax, ymin, ymax))


def _prism(ax: Axes, point: NDArray, width: float, color, z: float) -> None:
    """Draw a screen-space glyph anchored at the projected prototype center."""
    offsets = np.array([[-0.66, -0.13], [-0.14, 0.59], [0.70, 0.26], [0.18, -0.46]])
    cap = point + width * (offsets - offsets.mean(axis=0))
    extrusion = np.array([0, -0.07])
    for (i, j), shade in [((2, 3), 0.64), ((3, 0), 0.79)]:
        a, b = cap[i], cap[j]
        ax.add_patch(
            Polygon(
                [a, b, b + extrusion, a + extrusion],
                fc=np.asarray(to_rgb(color)) * shade,
                ec=INK,
                lw=0.36,
                zorder=z,
            )
        )
    ax.add_patch(Polygon(cap, fc=color, ec=INK, lw=0.48, zorder=z + 0.001))
    ax.add_patch(Circle(point, 0.07 * width, fc=INK, ec="none", zorder=z + 0.002))


def plot_prototype_graph(
    graph: PrototypeGraph,
    assignments: ArrayLike | None = None,
    spectroscopic_z: ArrayLike | None = None,
    *,
    layout: PrototypeLayout | None = None,
    feature_labels: Sequence[str] | None = None,
    show_inputs: bool = True,
    max_input_targets: int = 11,
    occupancy_max: int | None = None,
    figsize: tuple[float, float] = (6.8, 5.2),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Draw learned graph edges over an explicit PCA-to-screen frame.

    Glyph width has a fixed baseline plus a term proportional to the square
    root of the target count divided by occupancy_max. Supply the same
    occupancy_max and pooled layout when comparing models. Cap hue shows
    usable spectroscopy / target on [0, 1], or occupancy if spectroscopy is
    omitted. Empty cells have gray caps. With no assignments all glyphs are
    neutral and equally wide. Extrusion is decorative in screen units.
    """
    if not isinstance(graph, PrototypeGraph):
        raise TypeError("graph must be a PrototypeGraph")
    w = graph.prototypes
    n, n_features = w.shape
    max_input_targets = index(max_input_targets)
    if max_input_targets < 0:
        raise ValueError("max_input_targets must be nonnegative")
    labels = [f"x{j + 1}" for j in range(n_features)] if feature_labels is None else feature_labels
    if len(labels) != n_features:
        raise ValueError("one label is required per feature")
    if assignments is None and spectroscopic_z is not None:
        raise ValueError("spectroscopic_z requires assignments")
    counts = np.zeros(n, dtype=int)
    norm = ticks = None
    cmap = colormaps["viridis"]
    fills = ["#96b6c9"] * n
    label = ""
    if assignments is not None:
        ids, z = _validate_partition(assignments, n, spectroscopic_z)
        counts = np.bincount(ids, minlength=n)
        if z is None:
            norm, ticks = color_scale(
                counts.astype(float), (0, float(occupancy_max or counts.max()))
            )
            vals = counts
            label = "Target observations per prototype"
        else:
            spec = np.bincount(ids[np.isfinite(z)], minlength=n)
            vals = np.divide(spec, counts, out=np.zeros(n, dtype=float), where=counts > 0)
            norm = Normalize(0, 1)
            label = "Usable spectra / target"
        fills = [cmap(norm(v)) if count else MISSING for count, v in zip(counts, vals)]
    count_scale = max(1, counts.max()) if occupancy_max is None else index(occupancy_max)
    if count_scale < 1 or count_scale < counts.max():
        raise ValueError("occupancy_max must be positive and at least the maximum observed count")

    if layout is None:
        layout = fit_prototype_layout(w if n > 1 else np.repeat(w, 2, axis=0))
    xy = layout.transform(w)
    xmin, xmax, ymin, ymax = layout.bounds
    scale = max(xmax - xmin, ymax - ymin, 1.0)
    # Draw in normalized screen units, using a uniform scale in x and y.
    xy = (xy - [xmin, ymin]) / scale * 6
    width, height = 6 * (xmax - xmin) / scale, 6 * (ymax - ymin) / scale
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.add_collection(
        LineCollection(xy[graph.edges], colors="#597e99", lw=0.70, alpha=0.45, zorder=1)
    )
    left = -1.2
    if show_inputs:
        ys = (
            np.linspace(height + 0.06, -0.06, n_features)
            if height >= 1
            else np.linspace(0.55, -0.55, n_features)
        )
        if n_features == 1:
            ys = np.array([height / 2])
        selected = np.argsort(-counts, kind="stable")[:max_input_targets]
        standardized = (w - w.mean(axis=0)) / w.std(axis=0).clip(0.1)
        segments, strand_colors, widths = [], [], []
        for j, (y, feature) in enumerate(zip(ys, labels)):
            for i in selected:
                val = standardized[i, j]
                segments.append([(left, y), xy[i]])
                strand_colors.append("#ba5d42" if val >= 0 else "#39749e")
                widths.append(0.28 + 0.25 * min(abs(val), 1.6))
            ax.add_patch(Circle((left, y), 0.052, fc="#d6e4ee", ec=INK, lw=0.6, zorder=5))
            ax.text(left - 0.10, y, feature, ha="right", va="center", color=INK, fontsize=8.5)
        if segments:
            ax.add_collection(
                LineCollection(segments, colors=strand_colors, lw=widths, alpha=0.22, zorder=0)
            )
    tile_widths = 0.13 + 0.17 * np.sqrt(counts / count_scale)
    order = np.argsort(-xy[:, 1], kind="stable")
    for rank, i in enumerate(order):
        _prism(ax, xy[i], tile_widths[i], fills[i], 2 + rank / (n + 1))
    ax.set_xlim(
        min(-1.95 if show_inputs else -0.30, xy[:, 0].min() - 0.3),
        max(width, xy[:, 0].max()) + 0.30,
    )
    ax.set_ylim(min(-0.65, xy[:, 1].min() - 0.30), max(0.65, height + 0.32, xy[:, 1].max() + 0.30))
    if norm is not None:
        add_colorbar(fig, ax, norm, cmap, label, ticks)
        if np.any(counts == 0):
            ax.legend(
                handles=[Patch(fc=MISSING, ec=INK, label="No assigned target objects")],
                frameon=False,
                fontsize=7.5,
                loc="lower right",
            )
    return fig, ax


def plot_graph_atlas(
    graph,
    assignments,
    spectroscopic_z,
    *,
    feature_labels=None,
    show_inputs=True,
    max_input_targets=11,
    figsize=(6.8, 6.0),
    output=None,
    layout=None,
):
    """Legacy graph renderer with a third return value of layout diagnostics.

    Use plot_prototype_graph for the standard (figure, axes) return value.
    """
    if not isinstance(graph, PrototypeGraph):
        raise TypeError("graph must be a PrototypeGraph")
    w = graph.prototypes
    layout = layout or fit_prototype_layout(w if len(w) > 1 else np.repeat(w, 2, axis=0))
    fig, ax = plot_prototype_graph(
        graph,
        assignments,
        spectroscopic_z,
        layout=layout,
        feature_labels=feature_labels,
        show_inputs=show_inputs,
        max_input_targets=max_input_targets,
        figsize=figsize,
    )
    if output is not None:
        save_figure(fig, Path(output).with_suffix(""))
    return (
        fig,
        ax,
        {
            "pca_explained_variance_ratio": layout.explained_variance_ratio.tolist(),
            "n_graph_edges": len(graph.edges),
            "displayed_input_target_count": min(max_input_targets, len(w)) if show_inputs else 0,
            "max_coverage_colorbar": 1.0,
            "projection": asdict(layout.diagnostics(w)) if len(w) > 1 else None,
        },
    )


def plot_gtm(model: GTMModel, *, component: int = 0, **kwargs) -> tuple[Figure, Axes]:
    """Color the fixed latent grid by one fitted GTM prototype component.

    Grid coordinates are latent locations of the GTM basis. No learned graph
    edges or posterior membership probabilities are encoded by the prisms.
    """
    if not isinstance(model, GTMModel):
        raise TypeError("model must be a GTMModel")
    component = index(component)
    if not 0 <= component < model.prototypes.shape[1]:
        raise ValueError("component is outside the fitted feature dimension")
    rows, cols = (np.unique(model.latent[:, axis]) for axis in (0, 1))
    expected = np.column_stack((np.repeat(rows, len(cols)), np.tile(cols, len(rows))))
    if not np.array_equal(model.latent, expected) or len(expected) != len(model.prototypes):
        raise ValueError("GTM latent coordinates must be a complete row-major grid")
    kwargs.setdefault("label", f"GTM prototype component {component + 1} (fitted units)")
    return plot_cell_plane(model.prototypes[:, component].reshape(len(rows), len(cols)), **kwargs)
