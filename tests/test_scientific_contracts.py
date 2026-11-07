"""Independent statistical checks and stable plotting contracts."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression

from modelatlas import (
    PhotoZCells,
    PrototypeGraph,
    SOMMap,
    explain_mlp,
    explanation_layout,
    fit_prototype_layout,
    match_tanh_mlp,
    partition_tail_bounds,
    plot_coefficients,
    plot_explanation_atlas,
    plot_identification_sensitivity,
    plot_independent_pair,
    plot_photoz_map,
    plot_predictions,
    plot_prototype_graph,
    plot_residual_atlas,
    save_figure,
    sharp_tail_bounds,
    train_gtm,
    transform_signed_permutation,
    visualize,
)


@pytest.mark.parametrize("budget", [0.015, 0.07, 0.2, 0.7])
def test_tail_extrema_match_independent_transport_lp(budget):
    z, threshold = np.array([0.15, 0.85, 1.0, 1.12, 1.7]), 1.0
    p = np.mean(z > threshold)
    lp = []
    for costs in (z[z > threshold] - threshold, threshold - z[z <= threshold]):
        result = linprog(-np.ones(len(costs)), A_ub=[costs], b_ub=[budget], bounds=(0, 1 / len(z)))
        assert result.success
        lp.append(-result.fun)
    np.testing.assert_allclose(sharp_tail_bounds(z, threshold, budget), [p - lp[0], p + lp[1]])


def test_transport_respects_small_positive_cost_and_strict_threshold():
    z = 1 + np.spacing(1.0)
    lower, upper = sharp_tail_bounds([z], 1.0, (z - 1) / 4)
    assert (lower, upper) == pytest.approx((0.75, 1))
    assert sharp_tail_bounds([1.0], 1.0, 0) == (0, 0)
    assert sharp_tail_bounds([1.0], 1.0, 1e-6) == (0, 1)


def test_photoz_hex_topology_survives_empty_spectroscopic_sample():
    w = np.arange(9.0).reshape(3, 3, 1)
    som = SOMMap(w, topology="hexagonal")
    cells = PhotoZCells(som, w.reshape(-1, 1), np.full(9, np.nan))
    bounds = cells.identify(gamma=0)
    assert bounds.topology == "hexagonal"
    assert bounds.population_interval() == (0, 1)
    fig, ax = plot_photoz_map(bounds, "observed_high_fraction")
    assert sum(len(p.get_xy()) == 7 for p in ax.patches if hasattr(p, "get_xy")) == 9
    plt.close(fig)
    fig, ax = plot_identification_sensitivity(cells, [0.0, 0.05, 0.1])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0, 0, 0])
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), [1, 1, 1])
    plt.close(fig)


def test_attempt_diagnostics_require_observed_attempts():
    w = np.arange(4.0).reshape(2, 2, 1)
    cells = PhotoZCells(SOMMap(w), w.reshape(-1, 1), np.array([0.8, np.nan, 1.4, np.nan]))
    bounds = cells.identify()
    assert bounds.summary()["N_attempted"] is None
    assert np.isnan(bounds.attempted_fraction).all()
    assert np.isnan(bounds.success_given_attempt).all()
    assert np.isfinite(bounds.coverage).all()


def test_sensitivity_figure_separates_zero_radius_at_threshold_atom():
    w = np.arange(4.0).reshape(2, 2, 1)
    x = np.array([[0.0], [0.0]])
    cells = PhotoZCells(SOMMap(w), x, np.array([1.2, np.nan]))
    fig, ax = plot_identification_sensitivity(cells, [0, 0.01, 0.10])
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [0.01, 0.10])
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), [0.5, 0.5])
    plt.close(fig)


@pytest.mark.parametrize("ids,z", [([0.1], [0.2]), ([0], [np.inf]), ([0], [-1]), ([], [])])
def test_partition_input_is_not_silently_recoded(ids, z):
    with pytest.raises(ValueError):
        partition_tail_bounds(ids, z, 2)


def _mlp(weights, biases):
    return SimpleNamespace(
        coefs_=weights,
        intercepts_=biases,
        n_outputs_=1,
        out_activation_="logistic",
        activation="tanh",
    )


def test_edge_attributions_are_equivariant_to_signed_permutations():
    rng = np.random.default_rng(41)
    w = [rng.normal(0, 0.45, size=shape) for shape in [(3, 5), (5, 4), (4, 1)]]
    b = [rng.normal(0, 0.3, size=layer.shape[1]) for layer in w]
    model = _mlp(w, b)
    p = [rng.permutation(5), rng.permutation(4)]
    signs = [rng.choice([-1, 1], len(order)) for order in p]
    new_w, new_b = transform_signed_permutation(w, b, p, signs)
    other = _mlp(new_w, new_b)
    x = rng.normal(size=(20, 3))
    first = explain_mlp(model, x[0], baseline=x[1], steps=480)
    second = explain_mlp(other, x[0], baseline=x[1], steps=480)
    orders = [np.arange(3), *p, np.arange(1)]
    for i, edge in enumerate(first.edge_contributions):
        np.testing.assert_allclose(
            second.edge_contributions[i], edge[np.ix_(*orders[i : i + 2])], atol=1e-12
        )
    assert first.max_layer_residual < 1e-6
    assert first.max_node_residual < 1e-12

    align = match_tanh_mlp(model, other, x[:13], X_check=x[13:])
    fig, axes = plot_independent_pair(model, align, x[0])
    for a, b in zip(axes[0].collections, axes[1].collections):
        np.testing.assert_allclose(a.get_segments(), b.get_segments())
        np.testing.assert_allclose(a.get_linewidths(), b.get_linewidths())
    plt.close(fig)
    fig, ax = plot_residual_atlas(model, align, x[13:])
    assert len(ax.collections) == 3
    plt.close(fig)

    layout = explanation_layout(model, x[:13])
    fig, ax = plot_explanation_atlas(
        model, first, grid_layout=layout, attribution_scale=1, edge_scale=1
    )
    assert len(ax.collections) == 1
    plt.close(fig)


def test_graph_renderer_preserves_edges_and_pooled_layout():
    w = np.array([[0, 0], [0, 2], [2, 0], [2, 2]], dtype=float)
    adjacency = np.array([[0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]], dtype=bool)
    graph = PrototypeGraph(w, adjacency, np.zeros(4))
    layout = fit_prototype_layout(np.vstack([w, w + 0.1]))
    fig, ax = plot_prototype_graph(
        graph,
        [0, 1, 1, 2],
        [0.2, np.nan, 0.7, 0.8],
        layout=layout,
        show_inputs=False,
        occupancy_max=3,
    )
    segments = np.asarray(ax.collections[0].get_segments())
    xy = layout.transform(w)
    xmin, xmax, ymin, ymax = layout.bounds
    displayed = (xy - [xmin, ymin]) / max(xmax - xmin, ymax - ymin, 1) * 6
    np.testing.assert_allclose(segments, displayed[graph.edges])
    assert fig.axes[-1].get_xlim() == (0, 1)
    fig2, ax2 = plot_prototype_graph(
        graph, [2, 2, 3], layout=layout, show_inputs=False, occupancy_max=3
    )
    np.testing.assert_allclose(segments, ax2.collections[0].get_segments())
    assert ax2.get_xlim() == ax.get_xlim() and ax2.get_ylim() == ax.get_ylim()
    plt.close(fig)
    plt.close(fig2)


def test_gtm_membership_and_glm_and_interval_views():
    x = np.random.default_rng(71).normal(size=(90, 2))
    gtm = train_gtm(x, grid=(3, 4), n_iter=5)
    r = gtm.responsibilities(x)
    np.testing.assert_allclose(r.sum(axis=1), 1)
    np.testing.assert_array_equal(r.argmax(axis=1), gtm.assign(x))
    fig, ax = visualize(gtm)
    assert ax.figure is fig
    plt.close(fig)
    glm = LogisticRegression().fit(x, x[:, 1] > 0)
    fig, ax = plot_coefficients(glm, xlabel="Log-odds per input unit")
    assert len(ax.collections) == 2
    plt.close(fig)
    with pytest.raises(ValueError, match="interval_label"):
        plot_predictions([0, 1], [0, 1], intervals=([-1, 0], [1, 2]))
    fig, ax = plot_predictions(
        [0, 1], [0, 1], intervals=([-1, 0], [1, 2]), interval_label="Prediction interval"
    )
    assert ax.collections[0].get_label() == "Prediction interval"
    plt.close(fig)


def test_vector_and_raster_exports_reproduce_bytes(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    ax.set_xlabel("Scientific figure")
    first = save_figure(fig, tmp_path / "first")
    second = save_figure(fig, tmp_path / "second")
    for a, b in zip(first, second):
        assert a.read_bytes() == b.read_bytes()
        assert a.stat().st_size > 1000
    assert "<text" in first[0].read_text()
    assert b"ModelAtlas " in first[1].read_bytes()
    assert first[2].read_bytes().startswith(b"\x89PNG")
    plt.close(fig)
