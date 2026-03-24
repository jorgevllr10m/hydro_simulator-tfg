from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GridDefinition:
    """Regular 2D rectangular grid definition for the simulator domain.

    - Spatial array shape is always (ny, nx) in order to (y, x).
    - x and y coordinates refer to cell centers.
    - dx and dy are expressed in meters.
    """

    nx: int
    ny: int
    dx: float
    dy: float
    x0: float = 0.0
    y0: float = 0.0
    crs: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate grid parameters after initialization."""
        if not isinstance(self.nx, int) or self.nx <= 0:
            raise ValueError(f"'nx' must be a positive integer, got {self.nx!r}")

        if not isinstance(self.ny, int) or self.ny <= 0:
            raise ValueError(f"'ny' must be a positive integer, got {self.ny!r}")

        if not isinstance(self.dx, (int, float)) or self.dx <= 0:
            raise ValueError(f"'dx' must be a positive number, got {self.dx!r}")

        if not isinstance(self.dy, (int, float)) or self.dy <= 0:
            raise ValueError(f"'dy' must be a positive number, got {self.dy!r}")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the canonical spatial shape as (ny, nx)."""
        return (self.ny, self.nx)

    @property
    def x_coords(self) -> FloatArray:
        """Return x coordinates of cell centers."""
        return self.x0 + (np.arange(self.nx, dtype=float) + 0.5) * self.dx

    @property
    def y_coords(self) -> FloatArray:
        """Return y coordinates of cell centers."""
        return self.y0 + (np.arange(self.ny, dtype=float) + 0.5) * self.dy


@dataclass(frozen=True)
class BasinDefinition:
    """Boolean basin mask over the simulation grid.

    - mask has shape (ny, nx)
    - True means the cell belongs to the basin
    - False means the cell is outside the basin
    """

    mask: BoolArray
    # example 2x4 : mask = np.array([
    #      [False, True,  True,  False],
    #      [True,  True,  True,  False],
    # ], dtype=bool)

    def __post_init__(self) -> None:
        """Validate basin mask."""
        if not isinstance(self.mask, np.ndarray):
            raise TypeError(f"'mask' must be a numpy.ndarray, got {type(self.mask).__name__}")

        if self.mask.ndim != 2:
            raise ValueError(f"'mask' must be a 2D array, got ndim={self.mask.ndim}")

        if self.mask.dtype != np.bool_:
            raise TypeError(f"'mask' must have boolean dtype, got {self.mask.dtype}")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the basin mask."""
        return self.mask.shape

    @property
    def n_active_cells(self) -> int:
        """Return the number of active basin cells."""
        return int(np.count_nonzero(self.mask))

    @property
    def active_fraction(self) -> float:
        """Return the fraction of active basin cells in the full grid."""
        return float(self.n_active_cells / self.mask.size)


@dataclass(frozen=True)
class SpatialDomain:
    """Combine grid and basin mask into a single spatial domain object.

    - grid and basin mask must have the same shape.
    """

    grid: GridDefinition
    basin: BasinDefinition

    def __post_init__(self) -> None:
        """Validate the consistency of grid and basin mask."""
        if self.grid.shape != self.basin.shape:
            raise ValueError(
                f"Grid shape {self.grid.shape} and basin mask shape {self.basin.shape} must match."
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the spatial domain (ny, nx)."""
        return self.grid.shape

    @property
    def x_coords(self) -> FloatArray:
        """Return x coordinates of the grid's cell centers."""
        return self.grid.x_coords

    @property
    def y_coords(self) -> FloatArray:
        """Return y coordinates of the grid's cell centers."""
        return self.grid.y_coords

    @property
    def n_active_cells(self) -> int:
        """Return the number of active cells in the basin."""
        return self.basin.n_active_cells

    @property
    def active_fraction(self) -> float:
        """Return the fraction of the domain that is active."""
        return self.basin.active_fraction
