"""Numerical and artist-level checks of lattice and camera semantics."""

from collections import deque
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Polygon
from minisom import MiniSom

from modelatlas import (
    ObliqueProjection,
    SOMLattice,
    SOMMap,
    fit_som,
    plot_metric,
    plot_som_architecture,
    projection_diagnostics,
    som_from_minisom,
    train_som,
)


@pytest.mark.parametrize("shape", [(2, 2), (4, 5), (5, 4)])
def test_hex_neighbors_match_centers_and_shortest_paths(shape):
    lattice = SOMLattice(shape, "hexagonal")
    centers = lattice.centers().reshape(-1, 2)
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    neighbors = np.isclose(distances, 1)
    edges = set(map(tuple, lattice.edges()))
    assert edges == set(map(tuple, np.argwhere(np.triu(neighbors, k=1))))
    for i in range(len(centers)):
        found = {i: 0}
        queue = deque([i])
        while queue:
            j = queue.popleft()
            for k in np.flatnonzero(neighbors[j]):
                if k not in found:
                    found[k] = found[j] + 1
                    queue.append(k)
        hops = lattice.hop_distances(np.unravel_index(i, shape)).ravel()
        assert hops.tolist() == [found[j] for j in range(len(centers))]
    if min(shape) > 2:
        assert np.count_nonzero(lattice.hop_distances((1, 1)) == 1) == 6


def test_hex_neighborhood_is_used_during_training():
    x = np.random.default_rng(63).normal(size=(70, 2))
    rect = train_som(x, (4, 5), steps=120, seed=8)
    hex_som = fit_som(x, (4, 5), topology="hexagonal", steps=120, seed=8)
    repeat = train_som(x, (4, 5), steps=120, seed=8, topology="hexagonal")
    np.testing.assert_array_equal(hex_som.codebook, repeat)
    assert not np.allclose(rect, repeat)

    # These two BMUs lie on a diagonal that is only an edge for the hex grid.
    w = np.array([[[8], [0]], [[1], [8]]], dtype=float)
    sample = np.array([[0.1]])
    square = SOMMap(w, sample)
    hexagonal = SOMMap(w, sample, topology="hexagonal")
    assert square.diagnostics()["topographic_error"] == 1
    assert hexagonal.diagnostics()["topographic_error"] == 0
    assert hexagonal.u_matrix()[0, 1] == pytest.approx((8 + 8 + 1) / 3)


def test_single_hex_training_step_matches_independent_gaussian():
    x = np.random.default_rng(52).normal(size=(30, 2))
    rows, cols = 4, 5
    lattice = SOMLattice((rows, cols), "hexagonal")
    centers = lattice.centers()
    mean = x.mean(axis=0)
    basis = np.linalg.svd(x - mean, full_matrices=False)[2]
    row = np.linspace(-1, 1, rows)[:, None, None]
    col = (2 * centers[..., 0] / centers[..., 0].max() - 1)[..., None]
    w = mean + 0.72 * (row * basis[0] + col * basis[1]) * max(x.std(axis=0).mean(), 0.25)
    sample = x[np.random.default_rng(17).integers(len(x))]
    win = np.unravel_index(np.sum((w - sample) ** 2, axis=-1).argmin(), (rows, cols))
    expected = w.copy()
    # Breadth-first hop counts from Euclidean unit-spaced neighbors.
    reachable = {win}
    frontier = {win}
    for distance in range(rows + cols):
        following = set()
        for cell in frontier:
            expected[cell] += 0.42 * np.exp(-(distance**2) / (2 * 1.4**2)) * (sample - w[cell])
            following.update(
                other
                for other in product(range(rows), range(cols))
                if np.isclose(np.linalg.norm(centers[other] - centers[cell]), 1)
            )
        frontier = following - reachable
        reachable.update(frontier)
    np.testing.assert_allclose(
        train_som(x, (rows, cols), steps=1, seed=17, sigma=1.4, topology="hexagonal"),
        expected,
    )


@pytest.mark.parametrize("ny", [4, 5])
def test_minisom_hex_adapter_retains_real_neighbor_pairs(ny):
    fitted = MiniSom(3, ny, 2, topology="hexagonal", random_seed=12)
    model = som_from_minisom(fitted)
    assert model.topology == "hexagonal"
    original = fitted.get_weights().reshape(-1, 2)
    positions = np.column_stack([a.ravel() for a in fitted.get_euclidean_coordinates()])
    translated = np.argmin(
        np.linalg.norm(model.codebook.reshape(-1, 2)[:, None] - original[None, :], axis=2), axis=1
    )
    assert len(np.unique(translated)) == len(original)
    edges = np.sort(translated[model.lattice.edges()], axis=1)
    distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    assert set(map(tuple, edges)) == set(
        map(tuple, np.argwhere(np.triu(np.isclose(distance, 1), k=1)))
    )


def test_orthographic_geometry_and_distance_stress():
    camera = ObliqueProjection(azimuth=33, elevation=41)
    p = camera.matrix
    np.testing.assert_allclose(p @ p.T, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(p @ camera.viewing_direction, 0, atol=1e-14)
    assert abs(np.linalg.det(p[:, :2])) == pytest.approx(np.sin(np.deg2rad(41)))
    xyz = np.random.default_rng(6).normal(size=(19, 3))
    np.testing.assert_allclose(
        camera.project(xyz + 3 * camera.viewing_direction), camera.project(xyz)
    )
    assert projection_diagnostics(xyz, 4 * xyz + 3).normalized_stress < 1e-12
    diagnostics = projection_diagnostics(xyz, camera.project(xyz))
    assert 0 < diagnostics.normalized_stress < 1
    assert diagnostics.n_pairs == 171


def test_hexagonal_geometry_is_drawn_in_both_views():
    som = SOMMap(np.arange(40.0).reshape(4, 5, 2), topology="hexagonal")
    expected = som.lattice.centers().reshape(-1, 2)
    for draw, camera in [
        (plot_som_architecture, ObliqueProjection()),
        (plot_metric, ObliqueProjection(azimuth=0, elevation=90)),
    ]:
        fig, ax = draw(som)
        caps = [
            p.get_xy()[:-1] for p in ax.patches if isinstance(p, Polygon) and len(p.get_xy()) == 7
        ]
        assert len(caps) == 20
        world = np.column_stack((expected, np.zeros(len(expected))))
        observed = np.array([p.mean(axis=0) for p in caps])
        np.testing.assert_allclose(
            np.sort(observed, axis=0), np.sort(camera.project(world), axis=0), atol=1e-12
        )
        plt.close(fig)


@pytest.mark.parametrize("bad", [(-1, 3), (3.5, 4), (4, 1)])
def test_invalid_lattice_is_rejected(bad):
    with pytest.raises(ValueError):
        SOMLattice(bad, "hexagonal")
