"""Shared vector artists for top-down and oblique lattice figures."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon
from numpy.typing import ArrayLike, NDArray

from .geometry import ObliqueProjection
from .lattice import Cell, SOMLattice

INK = "#24354b"
MISSING = "#e0e5eb"


def color_scale(
    values: NDArray[np.float64], value_range: tuple[float, float] | None
) -> tuple[Normalize, list[float] | None]:
    if np.isinf(values).any():
        raise ValueError("values may contain NaN for missing cells, but not infinities")
    if value_range is None:
        valid = values[np.isfinite(values)]
        if not len(valid):
            raise ValueError("all cells are missing; supply an explicit value_range")
        lo, hi = float(valid.min()), float(valid.max())
        if lo == hi:
            pad = max(abs(lo) * 0.05, 0.05)
            return Normalize(lo - pad, hi + pad, clip=True), [lo]
    else:
        lo, hi = map(float, value_range)
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError("value_range must be a finite, increasing pair")
    return Normalize(lo, hi, clip=True), None


def add_colorbar(
    fig: Figure,
    ax: Axes,
    norm: Normalize,
    cmap: Colormap,
    label: str,
    ticks: ArrayLike | None,
) -> None:
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation="horizontal",
        fraction=0.075,
        pad=0.055,
        aspect=28,
        shrink=0.68,
        ticks=ticks,
    )
    cb.set_label(label, fontsize=9, color=INK)
    cb.ax.tick_params(labelsize=8, length=2, colors=INK)
    cb.outline.set_linewidth(0.4)
    if cb.solids is not None:
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")


def draw_lattice(
    ax: Axes,
    lattice: SOMLattice,
    values: NDArray[np.float64],
    *,
    norm: Normalize,
    cmap: Colormap,
    projection: ObliqueProjection,
    depth: ArrayLike,
    lift: ArrayLike = 0.0,
    highlight: Cell | None = None,
    hatch: NDArray[np.bool_] | None = None,
    show_centers: bool = False,
) -> NDArray[np.float64]:
    """Draw unit polygons with shared camera, retaining the cap centers."""
    if highlight is not None:
        lattice.hop_distances(highlight)
    centers = lattice.centers()
    corners = lattice.corners(gap=0.055)
    planar = centers[:, :, None, :] + corners
    upper = np.broadcast_to(lift, lattice.shape)
    lower = -np.broadcast_to(depth, lattice.shape)
    top = np.concatenate(
        (planar, np.broadcast_to(upper[..., None, None], (*planar.shape[:-1], 1))), axis=-1
    )
    bottom = top.copy()
    bottom[..., 2] = lower[..., None]
    top_xy, bottom_xy = projection.project(top), projection.project(bottom)
    center_xyz = np.concatenate((centers, upper[..., None]), axis=-1)
    center_xy = projection.project(center_xyz)
    view = projection.viewing_direction
    edges = np.roll(corners, -1, axis=0) - corners
    outward = np.column_stack((edges[:, 1], -edges[:, 0]))
    visible = np.flatnonzero(outward @ view[:2] > 1e-10)
    order = np.argsort((center_xyz @ view).ravel(), kind="stable")
    step = 1 / (len(order) + 1)
    for rank, flat in enumerate(order):
        i, j = np.unravel_index(flat, lattice.shape)
        z = 2 + rank * step
        cap, base = top_xy[i, j], bottom_xy[i, j]
        color = cmap(norm(values[i, j])) if np.isfinite(values[i, j]) else MISSING
        if lower[i, j] != upper[i, j]:
            for edge in visible:
                nxt = (edge + 1) % len(corners)
                shade = 0.62 + 0.18 * abs(outward[edge, 0]) / np.linalg.norm(outward[edge])
                face = np.asarray(to_rgb(color)) * shade
                ax.add_patch(
                    Polygon(
                        [cap[edge], cap[nxt], base[nxt], base[edge]],
                        facecolor=face,
                        edgecolor=INK,
                        lw=0.35,
                        zorder=z,
                    )
                )
        ax.add_patch(
            Polygon(
                cap,
                facecolor=color,
                edgecolor=INK,
                lw=0.43,
                zorder=z + 0.3 * step,
            )
        )
        if hatch is not None and hatch[i, j]:
            ax.add_patch(
                Polygon(
                    cap,
                    facecolor="none",
                    edgecolor=INK,
                    lw=0.15,
                    hatch="///",
                    zorder=z + 0.4 * step,
                )
            )
        if show_centers:
            ax.add_patch(Circle(center_xy[i, j], 0.035, fc=INK, ec="none", zorder=z + 0.5 * step))
        if highlight == (i, j):
            ax.add_patch(
                Polygon(
                    cap,
                    facecolor="none",
                    edgecolor="white",
                    lw=1.5,
                    zorder=z + 0.6 * step,
                )
            )
            ax.add_patch(
                Circle(
                    center_xy[i, j],
                    0.09,
                    fc="white",
                    ec=INK,
                    lw=0.9,
                    zorder=z + 0.7 * step,
                )
            )
    vertices = np.concatenate((top_xy.reshape(-1, 2), bottom_xy.reshape(-1, 2)))
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    ax.set_xlim(lo[0] - 0.14, hi[0] + 0.14)
    ax.set_ylim(lo[1] - 0.16, hi[1] + 0.16)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    return center_xy
