"""Fitted coefficients and predictive checks on explicitly chosen scales."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from ._lattice_plot import INK


def _intervals(intervals, shape, label) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not label:
        raise ValueError("interval_label must name the meaning of the supplied intervals")
    a = np.asarray(intervals, dtype=float)
    if a.shape != (2, *shape) or not np.isfinite(a).all() or np.any(a[0] > a[1]):
        raise ValueError("intervals must be two finite, ordered arrays matching the data")
    return a[0], a[1]


def plot_coefficients(
    model_or_coefficients,
    *,
    labels: Sequence[str] | None = None,
    figsize: tuple[float, float] = (5.0, 3.25),
    xlabel: str = "Fitted coefficient (model scale)",
    intervals: tuple[ArrayLike, ArrayLike] | None = None,
    interval_label: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Draw single-equation regression or GLM coefficients.

    Accepts a finite vector, or a fitted estimator with a vector or one-row
    ``coef_`` (such as binary logistic regression). The intercept is excluded
    unless included explicitly in a vector. Label GLM slopes on the link scale
    and provide the meaning of any externally computed uncertainty intervals.
    """
    coef = np.asarray(getattr(model_or_coefficients, "coef_", model_or_coefficients), dtype=float)
    if coef.ndim == 2 and coef.shape[0] == 1:
        coef = coef[0]
    if coef.ndim != 1 or not len(coef) or not np.isfinite(coef).all():
        raise ValueError("single-equation coefficients must be a nonempty finite vector")
    n = len(coef)
    labels = [f"x{i + 1}" for i in range(n)] if labels is None else labels
    if len(labels) != n:
        raise ValueError("one label per coefficient is required")
    bounds = None if intervals is None else _intervals(intervals, coef.shape, interval_label)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure
    y = np.arange(n)[::-1]
    if bounds is not None:
        ax.hlines(y, *bounds, color="#63788d", lw=1.6, label=interval_label)
    for v, z, color in zip(coef, y, np.where(coef >= 0, "#2d679f", "#b5553c")):
        if bounds is None:
            ax.plot([0, v], [z, z], color=color, lw=3, alpha=0.70, solid_capstyle="round")
        ax.scatter([v], [z], s=50, color=color, zorder=5, edgecolors="white", linewidths=0.7)
    ax.axvline(0, color="#98a4b0", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel, color=INK)
    ax.tick_params(length=0, labelsize=9)
    ax.grid(axis="x", alpha=0.16, zorder=-3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    max_abs = max(np.max(np.abs(coef if bounds is None else np.concatenate((coef, *bounds)))), 1e-5)
    ax.set(xlim=(-1.12 * max_abs, 1.12 * max_abs), ylim=(-0.6, n - 0.4))
    if bounds is not None:
        ax.legend(frameon=False, fontsize=8, loc="best")
    return fig, ax


def plot_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    x: ArrayLike | None = None,
    baseline: ArrayLike | None = None,
    xlabel: str = "Observation index",
    ylabel: str = "Response",
    figsize: tuple[float, float] = (5.9, 3.0),
    intervals: tuple[ArrayLike, ArrayLike] | None = None,
    interval_label: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot observed and predicted values in the supplied observation order.

    Optional intervals are precomputed by the caller. Name their statistical
    meaning through interval_label. An optional baseline is a point sequence.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if not len(y) or len(y) != len(pred) or not np.isfinite([y, pred]).all():
        raise ValueError("y_true and y_pred must be nonempty, finite and equal length")
    index = np.arange(len(y)) if x is None else np.asarray(x)
    if index.shape != y.shape:
        raise ValueError("x must have one entry per response")
    b = None if baseline is None else np.asarray(baseline, dtype=float).reshape(-1)
    if b is not None and (b.shape != y.shape or not np.isfinite(b).all()):
        raise ValueError("baseline must be finite and match y")
    bounds = None if intervals is None else _intervals(intervals, y.shape, interval_label)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure
    if bounds is not None:
        ax.fill_between(index, *bounds, color="#5d8cac", alpha=0.20, label=interval_label, zorder=1)
    if b is not None:
        ax.plot(index, b, lw=1.1, color="#728498", alpha=0.85, label="Baseline", zorder=2)
    ax.plot(index, y, lw=1.3, color=INK, label="Observed", zorder=4)
    ax.plot(index, pred, lw=1.6, color="#b6533c", label="Model prediction", zorder=3)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.tick_params(length=0, labelsize=8)
    ax.grid(lw=0.6, color="#7d8fa6", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    return fig, ax
