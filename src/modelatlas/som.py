"""Self-organizing map fitting and diagnostics in fitted feature coordinates."""

from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist

from .lattice import SOMLattice, Topology


@dataclass(frozen=True)
class SOMMap:
    """A fitted (row, column, feature) codebook and optional observations.

    Samples must share the codebook's fitted feature scaling. A separate
    display_codebook may contain prototype values converted to physical units.
    Topology describes the fitted lattice and is used by all map diagnostics.
    """

    codebook: NDArray[np.float64]
    samples: NDArray[np.float64] | None = None
    feature_labels: tuple[str, ...] | None = None
    display_codebook: NDArray[np.float64] | None = None
    feature_units: tuple[str, ...] | None = None
    topology: Topology = "rectangular"

    def __post_init__(self) -> None:
        w = np.asarray(self.codebook, dtype=float)
        if w.ndim != 3 or w.shape[-1] < 1 or not np.isfinite(w).all():
            raise ValueError("SOM codebook must be finite (rows, cols, features)")
        SOMLattice((w.shape[0], w.shape[1]), self.topology)
        object.__setattr__(self, "codebook", w)
        if self.samples is not None:
            object.__setattr__(self, "samples", self._observations(self.samples))
        if self.display_codebook is not None:
            display = np.asarray(self.display_codebook, dtype=float)
            if display.shape != w.shape or not np.isfinite(display).all():
                raise ValueError("display_codebook must have codebook shape and finite values")
            object.__setattr__(self, "display_codebook", display)
        if self.feature_labels is not None and len(self.feature_labels) != self.n_features:
            raise ValueError("feature_labels length must equal n_features")
        if self.feature_units is not None and len(self.feature_units) != self.n_features:
            raise ValueError("feature_units length must equal n_features")

    @property
    def rows(self) -> int:
        return self.codebook.shape[0]

    @property
    def cols(self) -> int:
        return self.codebook.shape[1]

    @property
    def n_features(self) -> int:
        return self.codebook.shape[2]

    @property
    def lattice(self) -> SOMLattice:
        return SOMLattice((self.rows, self.cols), self.topology)

    def _observations(self, x: ArrayLike) -> NDArray[np.float64]:
        a = np.asarray(x, dtype=float)
        if a.ndim != 2 or a.shape[1] != self.n_features or not np.isfinite(a).all():
            raise ValueError("observations must be finite (n, n_features), matching codebook")
        return a

    def distances(self, x: ArrayLike | None = None) -> NDArray[np.float64]:
        """Euclidean distances from observations to row-major codebook entries."""
        if x is None:
            if self.samples is None:
                raise ValueError("observations required for this diagnostic")
            x = self.samples
        return cdist(self._observations(x), self.codebook.reshape(-1, self.n_features))

    def best_matching_units(self, x: ArrayLike | None = None) -> NDArray[np.intp]:
        """Return (row, column) of the closest unit, breaking ties by flat index."""
        flat = np.argmin(self.distances(x), axis=1)
        return np.column_stack(np.unravel_index(flat, self.lattice.shape))

    def occupancy(self) -> NDArray[np.intp]:
        """Count supplied observations per fitted prototype."""
        bmu = self.best_matching_units()
        return np.bincount(
            np.ravel_multi_index(bmu.T, self.lattice.shape), minlength=self.rows * self.cols
        ).reshape(self.lattice.shape)

    def u_matrix(self) -> NDArray[np.float64]:
        """Mean fitted-space Euclidean distance to each cell's lattice neighbors."""
        w = self.codebook.reshape(-1, self.n_features)
        start, end = self.lattice.edges().T
        distance = np.linalg.norm(w[start] - w[end], axis=1)
        vertices = np.concatenate((start, end))
        sums = np.bincount(vertices, weights=np.tile(distance, 2), minlength=len(w))
        degree = np.bincount(vertices, minlength=len(w))
        return (sums / degree).reshape(self.lattice.shape)

    def diagnostics(self) -> dict[str, float | int]:
        """Quantization, topographic error and occupancy of the supplied sample."""
        if self.samples is None or not len(self.samples):
            raise ValueError("a nonempty sample is required for diagnostics")
        d = self.distances()
        n = len(self.samples)
        first, second = np.argsort(d, axis=1, kind="stable")[:, :2].T
        cell_edges = self.lattice.edges()
        neighbor_pairs = cell_edges[:, 0] * d.shape[1] + cell_edges[:, 1]
        pairs = np.minimum(first, second) * d.shape[1] + np.maximum(first, second)
        counts = np.bincount(first, minlength=d.shape[1])
        occupied = int(np.count_nonzero(counts))
        return {
            "n_samples": n,
            "n_neurons": d.shape[1],
            "quantization_error": float(d[np.arange(n), first].mean()),
            "topographic_error": float(1 - np.isin(pairs, neighbor_pairs).mean()),
            "occupied_neurons": occupied,
            "empty_neurons": d.shape[1] - occupied,
            "max_occupancy": int(counts.max()),
        }


def train_som(
    x: ArrayLike,
    grid: tuple[int, int] = (7, 8),
    steps: int = 9000,
    seed: int | None = 21,
    learning_rate: float = 0.42,
    sigma: float | None = None,
    *,
    topology: Topology = "rectangular",
) -> NDArray[np.float64]:
    """Fit an online Euclidean SOM and return its codebook array.

    Scale input features before fitting. The rectangular trainer retains its
    Euclidean lattice-coordinate Gaussian. Hexagonal training uses a Gaussian
    of shortest hex-lattice hop distance, with six equidistant neighbors in
    the interior. A fixed seed reproduces fits in a fixed numerical environment.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("x must be a finite (n_samples >= 2, n_features >= 1) array")
    lattice = SOMLattice(grid, topology)
    rows, cols = lattice.shape
    steps = index(steps)
    if steps < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    sigma = max(rows, cols) * 0.54 if sigma is None else sigma
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")

    rng = np.random.default_rng(seed)
    center = x.mean(axis=0)
    _, _, vt = np.linalg.svd(x - center, full_matrices=False)
    basis = np.zeros((2, x.shape[1]))
    basis[: min(2, len(vt))] = vt[: min(2, len(vt))]
    rr, cc = np.indices(lattice.shape)
    coord0 = np.linspace(-1, 1, rows)[:, None, None]
    coord1 = np.linspace(-1, 1, cols)[None, :, None]
    if topology == "hexagonal":
        centers = lattice.centers()[..., 0]
        coord1 = (2 * centers / centers.max() - 1)[..., None]
    initial = center + 0.72 * (coord0 * basis[0] + coord1 * basis[1]) * max(
        np.mean(np.std(x, axis=0)), 0.25
    )
    w = np.array(initial, dtype=float, copy=True)
    for it in range(steps):
        sample = x[rng.integers(len(x))]
        r, c = np.unravel_index(np.sum((w - sample) ** 2, axis=-1).argmin(), lattice.shape)
        frac = it / max(steps - 1, 1)
        lr = learning_rate * (0.07 / learning_rate) ** frac
        radius = sigma * (0.72 / sigma) ** frac
        if topology == "hexagonal":
            distance_squared = lattice.hop_distances((r, c)) ** 2
        else:
            distance_squared = (rr - r) ** 2 + (cc - c) ** 2
        kernel = np.exp(-distance_squared / (2 * radius**2))
        w += lr * kernel[:, :, None] * (sample - w)
    return w


def fit_som(
    x: ArrayLike,
    grid: tuple[int, int] = (7, 8),
    *,
    topology: Topology = "rectangular",
    steps: int = 9000,
    seed: int | None = 21,
    learning_rate: float = 0.42,
    sigma: float | None = None,
    feature_labels: tuple[str, ...] | None = None,
    feature_units: tuple[str, ...] | None = None,
) -> SOMMap:
    """Fit and return a SOMMap that retains its topology and training sample."""
    codebook = train_som(x, grid, steps, seed, learning_rate, sigma, topology=topology)
    return SOMMap(
        codebook,
        np.asarray(x, dtype=float),
        feature_labels,
        feature_units=feature_units,
        topology=topology,
    )


def som_from_minisom(
    model: object,
    X: ArrayLike | None = None,
    feature_labels: tuple[str, ...] | None = None,
    display_codebook: ArrayLike | None = None,
) -> SOMMap:
    """Adapt a fitted MiniSom, using its public Euclidean coordinates for hex maps.

    Rectangular arrays retain their original indexing. For hex maps the axes
    are transposed to (row, column) and rows are reversed when needed to match
    the odd-row offset convention. Supply feature-space observations as X.
    """
    if not hasattr(model, "get_weights"):
        raise TypeError("model must provide get_weights()")
    weights = np.asarray(model.get_weights(), dtype=float)
    display = None if display_codebook is None else np.asarray(display_codebook, dtype=float)
    topology: Topology = "rectangular"
    if (
        weights.ndim == 3
        and min(weights.shape[:2]) >= 2
        and hasattr(model, "get_euclidean_coordinates")
    ):
        x, _ = model.get_euclidean_coordinates()
        if not np.allclose(x[:, 1:], x[:, :1]):
            topology = "hexagonal"
            row_order = slice(None) if x[0, 1] > x[0, 0] else slice(None, None, -1)
            weights = weights.transpose(1, 0, 2)[row_order]
            if display is not None:
                display = display.transpose(1, 0, 2)[row_order]
    return SOMMap(
        weights,
        None if X is None else np.asarray(X, dtype=float),
        feature_labels,
        display,
        topology=topology,
    )
