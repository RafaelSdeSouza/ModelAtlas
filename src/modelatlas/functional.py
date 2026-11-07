"""Model-agnostic functional geometry and finite-information lattices.

This module deliberately requires only NumPy. It operates on predictions, not
on model parameters, and is compatible with regression or probabilistic models
whose outputs can be represented by common arrays on shared inputs.

The reference inputs/distribution and the output scale are essential parts of
the definition of functional distance. Equality on a finite reference sample
is *not* proof of global functional equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial
from typing import Iterable, Mapping
import numpy as np


@dataclass(frozen=True)
class FunctionalMap:
    """A low-dimensional representation of reference-prediction vectors.

    coords is metric MDS/PCA on the prediction profiles. Coordinates are
    centered and measured on the RMS prediction scale. ``variance_retained``
    is the fraction of between-model squared distance preserved by two axes.
    """
    names: tuple[str, ...]
    profiles: np.ndarray
    distances: np.ndarray
    coords: np.ndarray
    variance_retained: float
    stress: float
    coords3: np.ndarray
    variance_retained_3d: float
    stress3: float


def model_predictions(models: Mapping[str, object], X: np.ndarray, *, class_index: int = 1) -> dict[str, np.ndarray]:
    """Evaluate models on exactly the same finite reference sample.

    If ``predict_proba`` is available, extract the same column from each model.
    Otherwise use ``predict``. Callables are also supported. Multiclass output
    must be selected by the caller rather than implicitly flattened.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a finite, nonempty 2D reference array")
    out = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            matrix = np.asarray(model.predict_proba(X), dtype=float)
            if matrix.ndim != 2 or not 0 <= class_index < matrix.shape[1]:
                raise ValueError(f"{name}: incompatible predict_proba output")
            pred = matrix[:, class_index]
        elif hasattr(model, "predict"):
            pred = np.asarray(model.predict(X), dtype=float)
        elif callable(model):
            pred = np.asarray(model(X), dtype=float)
        else:
            raise TypeError(f"{name}: model must predict or be callable")
        if pred.shape != (len(X),) or not np.isfinite(pred).all():
            raise ValueError(f"{name}: expected one finite scalar prediction per input")
        out[str(name)] = pred
    if not out:
        raise ValueError("at least one model is required")
    return out


def functional_map(predictions: Mapping[str, np.ndarray], *, sample_weights: np.ndarray | None = None) -> FunctionalMap:
    """Embed fitted models by empirical RMS differences in their predictions.

    The input consists of comparable scalar outputs on one common reference
    dataset, e.g. P(Y=1|X) across fitted classifiers. For vector outputs,
    callers may supply an appropriate comparison operator in a future adapter.

    The 2D coordinates are classical metric MDS computed via a centered
    prediction matrix. An exact prediction duplicate shares its coordinates.
    The stress is the relative Frobenius norm of pairwise distance residuals.
    """
    names = tuple(predictions)
    P = np.stack([np.asarray(predictions[name], dtype=float) for name in names])
    if P.ndim != 2 or P.shape[1] == 0 or not np.isfinite(P).all():
        raise ValueError("predictions must have equal nonempty finite shapes")
    n_model, n_ref = P.shape
    if sample_weights is None:
        weights = np.full(n_ref, 1.0 / n_ref)
    else:
        weights = np.asarray(sample_weights, dtype=float)
        if weights.shape != (n_ref,) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("invalid nonnegative reference sample weights")
        weights = weights / weights.sum()
    centered = (P-P.mean(axis=0, keepdims=True)) * np.sqrt(weights)[None, :]
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_axis = min(3, len(singular_values))
    coords3 = np.zeros((n_model, 3), dtype=float)
    coords3[:, :n_axis] = (centered @ vt[:n_axis].T)
    coords = coords3[:, :2].copy()
    differences = (P[:, None, :] - P[None, :, :])
    D = np.sqrt(np.sum(weights[None, None, :] * differences**2, axis=-1))
    d2 = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    d3 = np.linalg.norm(coords3[:, None, :] - coords3[None, :, :], axis=-1)
    var_total = float(np.sum(singular_values**2))
    explained = float(np.sum(singular_values[:2]**2)/var_total) if var_total > 0 else 1.0
    denom = float(np.linalg.norm(D))
    stress = float(np.linalg.norm(D-d2)/denom) if denom > 0 else 0.0
    explained3 = float(np.sum(singular_values[:3]**2)/var_total) if var_total > 0 else 1.0
    stress3 = float(np.linalg.norm(D-d3)/denom) if denom > 0 else 0.0
    # Fix eigenvector orientation reproducibly, including cases with tiny eigenvalue
    for j in range(2):
        if np.max(np.abs(coords[:, j])) > 0:
            idx = int(np.argmax(np.abs(coords[:, j])))
            if coords[idx, j] < 0:
                coords[:, j] *= -1
    for j in range(2):
        if np.max(np.abs(coords[:, j])) > 0 and np.dot(coords3[:, j], coords[:, j]) < 0:
            coords3[:, j] *= -1
    return FunctionalMap(names, P, D, coords, explained, stress, coords3, explained3, stress3)


def find_empirical_equivalences(atlas: FunctionalMap, *, tolerance: float = 1e-11) -> list[tuple[int, ...]]:
    """Cluster predictions equal up to RMS tolerance on the reference sample.

    This is an empirical equivalence diagnostic, not a formal orbit classifier.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    groups = []
    seen = set()
    for i in range(len(atlas.names)):
        if i in seen:
            continue
        group = [i]
        for j in range(i+1, len(atlas.names)):
            if j not in seen and all(atlas.distances[j, k] <= tolerance for k in group):
                group.append(j)
        seen.update(group)
        groups.append(tuple(group))
    return groups


@dataclass(frozen=True)
class LatticeNode:
    subset: tuple[int, ...]
    residual_variance: float
    n_groups: int
    labels: tuple[str, ...]


@dataclass(frozen=True)
class InformationLattice:
    """The Boolean lattice of coarsenings induced by binary feature bins."""
    nodes: tuple[LatticeNode, ...]
    feature_names: tuple[str, ...]
    baseline_variance: float
    shapley_variance_reduction: np.ndarray


def _residual_given_codes(p: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    _, inv = np.unique(labels, return_inverse=True, axis=0)
    sizes = np.bincount(inv)
    means = np.bincount(inv, weights=p) / sizes
    residual = float(np.mean((p-means[inv])**2))
    return residual, len(sizes)


def information_lattice(X: np.ndarray, f_predictions: np.ndarray, *, feature_indices: Iterable[int],
                        thresholds: np.ndarray | None = None, feature_names: Iterable[str] | None = None) -> InformationLattice:
    """Compute E_emp Var_emp[f(X) | observed feature-bin partition].

    Feature thresholds must be fixed across all subsets. The finite partitions
    are nested as features are added. The residual variation averaged over all
    observations is therefore monotone nonincreasing with set inclusion.

    This is prediction variability after compression, NOT estimator uncertainty,
    causal feature importance, a confidence interval, or partial identification.
    """
    X = np.asarray(X, dtype=float)
    p = np.asarray(f_predictions, dtype=float)
    if X.ndim != 2 or p.shape != (len(X),) or len(X) < 2 or not np.isfinite(X).all() or not np.isfinite(p).all():
        raise ValueError("require finite X and one scalar prediction per row")
    idx = tuple(int(v) for v in feature_indices)
    if len(idx) < 1 or len(idx) > 7 or len(set(idx)) != len(idx) or any(i < 0 or i >= X.shape[1] for i in idx):
        raise ValueError("one to seven distinct valid feature indices required")
    names = tuple(feature_names) if feature_names is not None else tuple(f"x{i+1}" for i in idx)
    if len(names) != len(idx):
        raise ValueError("feature_names length mismatch")
    thresholds = np.median(X[:, idx], axis=0) if thresholds is None else np.asarray(thresholds, dtype=float)
    if thresholds.shape != (len(idx),) or not np.isfinite(thresholds).all():
        raise ValueError("one finite threshold per selected feature required")
    bits = (X[:, idx] > thresholds[None, :]).astype(np.int8)
    R = {}
    nodes = []
    for k in range(len(idx)+1):
        for subset in combinations(range(len(idx)), k):
            labels = bits[:, subset] if subset else np.zeros((len(X), 1), dtype=np.int8)
            residual, n_groups = _residual_given_codes(p, labels)
            R[subset] = residual
            nodes.append(LatticeNode(subset, residual, n_groups, tuple(names[j] for j in subset)))
    m = len(idx)
    # Feature contributions to average reduction in unexplained prediction
    # variance: Shapley value for the set function v(S)=R(empty)-R(S).
    contributions = np.zeros(m)
    for j in range(m):
        for k in range(m):
            for S in combinations((q for q in range(m) if q != j), k):
                T = tuple(sorted(S+(j,)))
                coeff = factorial(k) * factorial(m-k-1) / factorial(m)
                contributions[j] += coeff * (R[S]-R[T])
    return InformationLattice(tuple(nodes), names, R[()], contributions)


def monotonicity_violations(lattice: InformationLattice, *, atol: float = 1e-12) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    R = {node.subset: node.residual_variance for node in lattice.nodes}
    errors = []
    for S in R:
        for j in range(len(lattice.feature_names)):
            if j not in S:
                T = tuple(sorted(S + (j,)))
                if R[T] > R[S] + atol:
                    errors.append((S, T))
    return errors
