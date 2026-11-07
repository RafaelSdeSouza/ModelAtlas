"""Explicit camera geometry and distance-distortion diagnostics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist


@dataclass(frozen=True)
class ObliqueProjection:
    """Orthographic camera seen from an oblique direction.

    Azimuth is measured from positive y towards positive x, in degrees.
    Elevation is measured above the xy plane, in degrees. World z points
    upwards. The two screen axes are orthonormal, so distances along the
    viewing direction collapse and in-plane distances are preserved.
    """

    azimuth: float = 32.0
    elevation: float = 35.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.azimuth) or not 0 < self.elevation <= 90:
            raise ValueError("azimuth must be finite and elevation must be in (0, 90]")

    @property
    def matrix(self) -> NDArray[np.float64]:
        """Return the 2 × 3 world-to-screen linear map."""
        az, el = np.deg2rad([self.azimuth, self.elevation])
        return np.array(
            [
                [np.cos(az), -np.sin(az), 0],
                [-np.sin(el) * np.sin(az), -np.sin(el) * np.cos(az), np.cos(el)],
            ]
        )

    @property
    def viewing_direction(self) -> NDArray[np.float64]:
        """Unit vector from the origin towards the viewer."""
        horizontal, vertical = self.matrix
        return np.cross(vertical, horizontal)

    def project(self, points: ArrayLike) -> NDArray[np.float64]:
        """Project finite Cartesian triples, keeping leading dimensions."""
        xyz = np.asarray(points, dtype=float)
        if xyz.ndim == 0 or xyz.shape[-1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("points must have finite Cartesian triples on the last axis")
        return xyz @ self.matrix.T


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Pairwise distance distortion after allowing one uniform display scale."""

    normalized_stress: float
    distance_correlation: float | None
    display_scale: float
    n_pairs: int


def projection_diagnostics(reference: ArrayLike, displayed: ArrayLike) -> ProjectionDiagnostics:
    """Compare Euclidean distances between corresponding points.

    Stress is ||d - s·d_display|| / ||d||, with s chosen by least squares.
    Correlation is Pearson's r across distinct pairs, or None if undefined.
    The calculation uses all point pairs and takes quadratic time and memory.
    """
    native = np.asarray(reference, dtype=float)
    screen = np.asarray(displayed, dtype=float)
    if (
        native.ndim != 2
        or screen.ndim != 2
        or len(native) != len(screen)
        or len(native) < 2
        or min(native.shape[1], screen.shape[1]) < 1
        or not np.isfinite(native).all()
        or not np.isfinite(screen).all()
    ):
        raise ValueError("reference and displayed must be finite matrices with matching rows")
    d, e = pdist(native), pdist(screen)
    norm = np.linalg.norm(d)
    denominator = e @ e
    scale = float(d @ e / denominator) if denominator > 0 else 0.0
    stress = float(np.linalg.norm(d - scale * e) / norm) if norm > 0 else 0.0
    corr = float(np.corrcoef(d, e)[0, 1]) if np.ptp(d) > 0 and np.ptp(e) > 0 else None
    return ProjectionDiagnostics(stress, corr, scale, len(d))
