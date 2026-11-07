"""Finite rectangular and odd-row offset hexagonal SOM lattices."""

from dataclasses import dataclass
from operator import index
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Topology = Literal["rectangular", "hexagonal"]
Cell = tuple[int, int]


@dataclass(frozen=True)
class SOMLattice:
    """A nonperiodic lattice indexed by (row, column).

    Rectangular cells are four-connected. Hexagons use pointy tops and
    odd-numbered rows shifted right by half a cell spacing. The Euclidean
    center-to-center distance of every adjacent pair is one lattice unit.
    """

    shape: tuple[int, int]
    topology: Topology = "rectangular"

    def __post_init__(self) -> None:
        try:
            rows, columns = map(index, self.shape)
        except (TypeError, ValueError) as exc:
            raise ValueError("shape must contain two integer lattice dimensions") from exc
        if min(rows, columns) < 2:
            raise ValueError("SOM lattice must have at least two rows and columns")
        if self.topology not in ("rectangular", "hexagonal"):
            raise ValueError("topology must be 'rectangular' or 'hexagonal'")
        object.__setattr__(self, "shape", (rows, columns))

    def centers(self) -> NDArray[np.float64]:
        """Return (rows, columns, 2) planar coordinates in lattice units."""
        row, col = np.indices(self.shape)
        if self.topology == "hexagonal":
            return np.stack((col + 0.5 * (row % 2), row * np.sqrt(3) / 2), axis=-1)
        return np.stack((col, row), axis=-1).astype(float)

    def corners(self, gap: float = 0.0) -> NDArray[np.float64]:
        """Return counterclockwise polygon offsets, shrunk by a fraction gap."""
        if not 0 <= gap < 1:
            raise ValueError("gap must be in [0, 1)")
        if self.topology == "hexagonal":
            angle = np.pi / 6 + np.arange(6) * np.pi / 3
            polygon = np.column_stack((np.cos(angle), np.sin(angle))) / np.sqrt(3)
        else:
            polygon = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        return polygon * (1 - gap)

    def hop_distances(self, cell: Cell) -> NDArray[np.intp]:
        """Shortest-path distances from cell, measured in lattice edges."""
        row, col = map(index, cell)
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            raise ValueError("cell is outside the lattice")
        rr, cc = np.indices(self.shape)
        dr = rr - row
        if self.topology == "rectangular":
            return np.abs(dr) + np.abs(cc - col)
        dq = cc - rr // 2 - (col - row // 2)
        return np.maximum.reduce((np.abs(dq), np.abs(dr), np.abs(dq + dr)))

    def edges(self) -> NDArray[np.intp]:
        """Unique undirected neighbor pairs in row-major flattened indices."""
        flat = np.arange(np.prod(self.shape)).reshape(self.shape)
        pairs = [
            np.column_stack((flat[:, :-1].ravel(), flat[:, 1:].ravel())),
            np.column_stack((flat[:-1].ravel(), flat[1:].ravel())),
        ]
        if self.topology == "hexagonal":
            pairs.extend(
                (
                    np.column_stack((flat[:-1:2, 1:].ravel(), flat[1::2, :-1].ravel())),
                    np.column_stack((flat[1:-1:2, :-1].ravel(), flat[2::2, 1:].ravel())),
                )
            )
        return np.concatenate(pairs)
