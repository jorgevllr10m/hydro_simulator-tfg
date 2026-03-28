from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from simulator.core.time import TimeDefinition

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
            raise ValueError(f"Grid shape {self.grid.shape} and basin mask shape {self.basin.shape} must match.")

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


@dataclass(frozen=True)
class SimulationDomain:
    """Static definition of the simulation world.

    This object groups all static components needed by the simulator:
    spatial domain, temporal domain, and static entities such as
    reservoirs and sensors.
    """

    spatial: SpatialDomain
    time: TimeDefinition
    reservoirs: tuple[ReservoirDefinition, ...] = ()
    sensors: tuple[SensorDefinition, ...] = ()

    @property
    def shape(self) -> tuple[int, int]:
        """Return the canonical spatial shape (ny, nx)."""
        return self.spatial.shape

    @property
    def n_steps(self) -> int:
        """Return the number of simulation time steps."""
        return self.time.n_steps


@dataclass(frozen=True)
class ReservoirDefinition:
    """Static definition of a reservoir in the simulation domain."""

    # TODO validar que embalses y sensores caen dentro del grid del dominio. (types.py / futuro loader.py)

    name: str
    cell_y: int
    cell_x: int
    capacity: float
    initial_storage: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Reservoir 'name' must be a non-empty string.")

        if not isinstance(self.cell_y, int) or self.cell_y < 0:
            raise ValueError(f"'cell_y' must be a non-negative integer, got {self.cell_y!r}")

        if not isinstance(self.cell_x, int) or self.cell_x < 0:
            raise ValueError(f"'cell_x' must be a non-negative integer, got {self.cell_x!r}")

        if not isinstance(self.capacity, (int, float)) or self.capacity <= 0:
            raise ValueError(f"'capacity' must be a positive number, got {self.capacity!r}")

        if not isinstance(self.initial_storage, (int, float)) or self.initial_storage < 0:
            raise ValueError(f"'initial_storage' must be a non-negative number, got {self.initial_storage!r}")

        if self.initial_storage > self.capacity:
            raise ValueError(f"'initial_storage' ({self.initial_storage}) cannot exceed 'capacity' ({self.capacity}).")


@dataclass(frozen=True)
class SensorDefinition:
    """Static definition of an observation sensor in the simulation domain."""

    # TODO no valida que cell_y y cell_x caigan dentro del grid. Cuando se conecte config → loader → domain.

    name: str
    sensor_type: str
    cell_y: int
    cell_x: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Sensor 'name' must be a non-empty string.")

        if not self.sensor_type:
            raise ValueError("Sensor 'sensor_type' must be a non-empty string.")

        if not isinstance(self.cell_y, int) or self.cell_y < 0:
            raise ValueError(f"'cell_y' must be a non-negative integer, got {self.cell_y!r}")

        if not isinstance(self.cell_x, int) or self.cell_x < 0:
            raise ValueError(f"'cell_x' must be a non-negative integer, got {self.cell_x!r}")
