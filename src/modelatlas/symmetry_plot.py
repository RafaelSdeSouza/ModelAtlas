"""Matplotlib comparisons in shared hidden-unit coordinates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Circle, Polygon

from ._lattice_plot import INK, add_colorbar
from .export import save_figure
from .symmetry import _paired_corr, _tanh_parameters, activations

BLUE, RUST = "#33799e", "#ae553d"


def _layout(widths):
    """Use fixed input slots and shared, affine projections of hidden grids."""
    n_in, n_1, n_2, _ = widths
    ys = np.linspace(2.45, 0.48, n_in) if n_in > 1 else [1.45]
    points = [np.column_stack((np.full(n_in, 0.60), ys))]
    corners = []
    for n, anchor in zip((n_1, n_2), ([2.55, 2.60], [5.30, 2.40])):
        cols = int(np.ceil(np.sqrt(n)))
        row, col = np.divmod(np.arange(n), cols)
        size = 4 / max(4, cols)
        vrow = size * np.array([0.17, -0.40])
        vcol = size * np.array([0.45, -0.13])
        center = np.asarray(anchor) + row[:, None] * vrow + col[:, None] * vcol
        points.append(center)
        offsets = 0.45 * np.stack((-vrow - vcol, -vrow + vcol, vrow + vcol, vrow - vcol))
        corners.append(center[:, None, :] + offsets)
    points.append(np.array([[8.1, 1.37]]))
    return points, corners


def _draw_network(ax, points, corners, matrices, scales, hidden, norm, cmap, inputs, out, rings=()):
    ax.set(xlim=(0.26, 8.66), ylim=(0.04, 3.07), aspect="equal")
    ax.set_axis_off()
    for matrix, scale, left, right in zip(matrices, scales, points[:-1], points[1:]):
        endpoints = [[a, b] for a in left for b in right]
        strengths = np.clip(np.abs(matrix.ravel()) / scale, 0, 1)
        colors = [
            (*to_rgb(BLUE if w < 0 else RUST), 0.04 + 0.35 * strength)
            for w, strength in zip(matrix.ravel(), strengths)
        ]
        ax.add_collection(
            LineCollection(endpoints, colors=colors, lw=0.25 + 0.9 * strengths, zorder=1)
        )
    for positions, faces, values in zip(points[1:-1], corners, hidden):
        for rank, k in enumerate(np.argsort(-positions[:, 1], kind="stable")):
            z = 2 + rank / (len(positions) + 1)
            tile = faces[k]
            base = tile + [0.10, -0.14]
            color = cmap(norm(values[k]))
            for (i, j), shade in [((1, 2), 0.62), ((2, 3), 0.78)]:
                face = np.asarray(to_rgb(color)) * shade
                ax.add_patch(
                    Polygon(
                        [tile[i], tile[j], base[j], base[i]], fc=face, ec=INK, lw=0.40, zorder=z
                    )
                )
            ax.add_patch(Polygon(tile, fc=color, ec=INK, lw=0.48, zorder=z + 0.004))
            ax.add_patch(Circle(positions[k], 0.024, fc=INK, ec="none", zorder=z + 0.007))
    for point, color in zip(points[0], inputs):
        ax.add_patch(Circle(point, 0.077, fc=color, ec=INK, lw=0.65, zorder=4))
    for idx in rings:
        ax.add_patch(Circle(points[-2][idx], 0.080, fc="none", ec="white", lw=1.7, zorder=5))
    ax.add_patch(Circle(points[-1][0], 0.155, fc="#e9c594", ec=INK, lw=0.85, zorder=4))
    ax.annotate(
        out,
        points[-1][0],
        xytext=(0, -23),
        textcoords="offset points",
        ha="center",
        color=INK,
        fontsize=9,
    )


def _comparison_layout(model, aligned):
    shapes = [np.shape(w) for w in model.coefs_]
    if len(shapes) != 3 or shapes != [w.shape for w in aligned.coefs]:
        raise ValueError("oblique comparison requires two matching hidden layers")
    widths = (shapes[0][0], *(s[1] for s in shapes))
    if widths[0] > 18 or widths[1] > 36 or widths[2] > 25:
        raise ValueError("compact comparison supports at most 18 inputs and 36/25 hidden units")
    return _layout(widths)


def _weight_scales(first, second=None):
    groups = (
        first if second is None else [np.r_[a.ravel(), b.ravel()] for a, b in zip(first, second)]
    )
    return [max(float(np.percentile(np.abs(group), 95)), 1e-12) for group in groups]


def plot_independent_pair(
    modelA,
    alignedB,
    x,
    title="Independent tanh MLPs",
    label="positive class",
    metrics=None,
    directory=None,
):
    """Plot two fitted tanh MLPs with identical slots, cameras and weight scales.

    Prism hue is signed tanh activation (fixed range -1 to 1). Edges show
    signed fitted weights with pooled per-layer 95th percentile scaling.
    Rings show the largest absolute final logit terms, excluding the bias.
    Optional metrics supplies held-out AUC labels. Directory saves all formats.
    """
    points, corners = _comparison_layout(modelA, alignedB)
    x = np.asarray(x, dtype=float).reshape(1, -1)
    first = activations(*_tanh_parameters(modelA), x)
    second = activations(alignedB.coefs, alignedB.intercepts, x)
    scales = _weight_scales(modelA.coefs_, alignedB.coefs)
    cmap = colormaps["RdBu_r"]
    norm = Normalize(-1, 1)
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.35), layout="constrained")
    inputs = cmap(norm(np.tanh(x[0] / 2)))
    for i, (ax, params, result) in enumerate(
        zip(axes, (modelA.coefs_, alignedB.coefs), (first, second))
    ):
        layers, _, prob = result
        rings = np.argsort(np.abs(layers[-1][0] * params[-1][:, 0]), kind="stable")[-3:]
        tag = ("Reference fit", "Aligned independent fit")[i]
        key = ("test_auc_A", "test_auc_B")[i]
        if metrics is not None and key in metrics:
            tag += f"  ·  held-out AUC {metrics[key]:.3f}"
        ax.set_title(tag, loc="left", color=INK, fontsize=11, pad=0)
        _draw_network(
            ax,
            points,
            corners,
            params,
            scales,
            [layer[0] for layer in layers[1:]],
            norm,
            cmap,
            inputs,
            f"P({label}) = {prob[0]:.3f}",
            rings,
        )
    fig.suptitle(title, color=INK, fontsize=14)
    fig.supxlabel("Edges show fitted weights · rust positive, blue negative", color=INK, fontsize=8)
    add_colorbar(
        fig, axes[-1], norm, cmap, "Hidden tanh activation · same scale for both fits", [-1, 0, 1]
    )
    if directory is not None:
        save_figure(fig, Path(directory) / "independent_networks_aligned")
    return fig, axes


def plot_residual_atlas(
    modelA, alignedB, X_holdout, title="Residual after alignment", metrics=None, directory=None
):
    """Show held-out RMS activation discrepancy in the shared neuron slots.

    Edge signs and widths represent aligned weight differences B - A, with
    scaling within each layer. The output summarizes mean absolute probability
    disagreement on the supplied observations. Directory optionally exports.
    """
    points, corners = _comparison_layout(modelA, alignedB)
    first = activations(*_tanh_parameters(modelA), X_holdout)
    second = activations(alignedB.coefs, alignedB.intercepts, X_holdout)
    hidden = [np.sqrt(np.mean((a - b) ** 2, axis=0)) for a, b in zip(first[0][1:], second[0][1:])]
    matrices = [b - a for a, b in zip(modelA.coefs_, alignedB.coefs)]
    mean_difference = np.mean(np.abs(first[-1] - second[-1]))
    norm = Normalize(0, max(0.05, max(v.max() for v in hidden)))
    cmap = colormaps["magma"]
    fig, ax = plt.subplots(figsize=(8.1, 3.9), layout="constrained")
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    _draw_network(
        ax,
        points,
        corners,
        matrices,
        _weight_scales(matrices),
        hidden,
        norm,
        cmap,
        ["#dce5ed"] * len(points[0]),
        f"Mean |Δp| = {mean_difference:.3f}",
    )
    add_colorbar(fig, ax, norm, cmap, "RMS hidden activation discrepancy (held out)", None)
    fig.supxlabel("Edges show B − A weights · rust positive, blue negative", color=INK, fontsize=8)
    if directory is not None:
        save_figure(fig, Path(directory) / "residual_atlas")
    return fig, ax


def plot_alignment_diagnostic(
    modelA,
    modelB,
    alignedB,
    X_align,
    X_test,
    y_test,
    title="Activation matching and predictive agreement",
    metrics=None,
    directory=None,
):
    """Separate alignment-sample correlations from held-out prediction agreement."""
    _comparison_layout(modelA, alignedB)
    params_a, params_b = _tanh_parameters(modelA), _tanh_parameters(modelB)
    if [w.shape for w in params_a[0]] != [w.shape for w in params_b[0]]:
        raise ValueError("Both fitted networks must have the same architecture")
    a = activations(*params_a, X_align)[0][1:]
    b = activations(*params_b, X_align)[0][1:]
    aligned = activations(alignedB.coefs, alignedB.intercepts, X_align)[0][1:]
    pred_a = activations(*params_a, X_test)[-1]
    before = activations(*params_b, X_test)[-1]
    after = activations(alignedB.coefs, alignedB.intercepts, X_test)[-1]
    if not np.allclose(before, after, rtol=0, atol=1e-11):
        raise ValueError("alignedB does not preserve the candidate predictions on X_test")
    y = np.asarray(y_test)
    if y.shape != pred_a.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("y_test must contain one binary 0/1 label per held-out prediction")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.65), layout="constrained")
    left, right = axes
    hidden = np.arange(1, len(a) + 1)
    left.plot(
        hidden,
        [_paired_corr(u, v) for u, v in zip(a, b)],
        "o--",
        color=RUST,
        label="Original order",
    )
    left.plot(
        hidden, [_paired_corr(u, v) for u, v in zip(a, aligned)], "s-", color=BLUE, label="Aligned"
    )
    left.set(
        xticks=hidden,
        ylim=(-1.05, 1.05),
        xlabel="Hidden layer",
        ylabel="Mean activation correlation",
    )
    left.legend(frameon=False, fontsize=8)
    right.plot([0, 1], [0, 1], color="#93a1ae", lw=0.9, ls="--")
    for value, marker, color in ((0, "o", BLUE), (1, "^", RUST)):
        mask = y == value
        right.scatter(
            pred_a[mask],
            before[mask],
            marker=marker,
            s=22,
            alpha=0.72,
            color=color,
            label=f"Class {value}",
        )
    right.set(
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        aspect="equal",
        xlabel="Reference probability",
        ylabel="Candidate probability",
    )
    right.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(title, color=INK, fontsize=13)
    if directory is not None:
        save_figure(fig, Path(directory) / "matching_diagnostic")
    return fig, axes
