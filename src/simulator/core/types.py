from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

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


@dataclass(frozen=True)
class SimulationDomain:
    """Static definition of the simulation world.

    This object groups all static components needed by the simulator:
    spatial domain, temporal domain, and static entities such as
    reservoirs and sensors.
    """

    spatial: SpatialDomain
    time: TimeDefinition
    reservoirs: tuple[Any, ...] = ()
    sensors: tuple[Any, ...] = ()

    @property
    def shape(self) -> tuple[int, int]:
        """Return the canonical spatial shape (ny, nx)."""
        return self.spatial.shape

    @property
    def n_steps(self) -> int:
        """Return the number of simulation time steps."""
        return self.time.n_steps


@dataclass
class SimulationState:
    """Dynamic state of the simulator at a single time step.

    - Spatial fields use shape (ny, nx), i.e. (y, x)
    - Vector-like reservoir fields use shape (n_reservoirs,)
    - Observations are stored separately as generic mappings
    """

    step: int
    timestamp: datetime

    precipitation: FloatArray
    air_temperature: FloatArray
    pet: FloatArray

    soil_moisture: FloatArray
    surface_runoff: FloatArray
    channel_flow: FloatArray

    reservoir_storage: FloatArray | None = None
    reservoir_release: FloatArray | None = None
    observations: dict[str, FloatArray] | None = None

    def __post_init__(self) -> None:
        """Validate dynamic state fields."""
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError(f"'step' must be a non-negative integer, got {self.step!r}")

        self._validate_spatial_field("precipitation", self.precipitation)
        self._validate_spatial_field("air_temperature", self.air_temperature)
        self._validate_spatial_field("pet", self.pet)
        self._validate_spatial_field("soil_moisture", self.soil_moisture)
        self._validate_spatial_field("surface_runoff", self.surface_runoff)
        self._validate_spatial_field("channel_flow", self.channel_flow)

        spatial_shape = self.precipitation.shape
        spatial_fields = {
            "air_temperature": self.air_temperature,
            "pet": self.pet,
            "soil_moisture": self.soil_moisture,
            "surface_runoff": self.surface_runoff,
            "channel_flow": self.channel_flow,
        }

        for name, value in spatial_fields.items():
            if value.shape != spatial_shape:
                raise ValueError(f"'{name}' must have shape {spatial_shape}, got {value.shape}")

        if self.reservoir_storage is not None:
            self._validate_vector_field("reservoir_storage", self.reservoir_storage)

        if self.reservoir_release is not None:
            self._validate_vector_field("reservoir_release", self.reservoir_release)

        if self.observations is not None:
            if not isinstance(self.observations, dict):
                raise TypeError(
                    f"'observations' must be a dict[str, FloatArray] or None, "
                    f"got {type(self.observations).__name__}"
                )
            for key, value in self.observations.items():
                if not isinstance(key, str):
                    raise TypeError(f"Observation keys must be strings, got {type(key).__name__}")
                self._validate_array("observations[{!r}]".format(key), value)

    @staticmethod
    def _validate_array(name: str, value: FloatArray) -> None:
        """Validate that a value is a NumPy float array."""
        if not isinstance(value, np.ndarray):
            raise TypeError(f"'{name}' must be a numpy.ndarray, got {type(value).__name__}")
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"'{name}' must have a floating dtype, got {value.dtype}")

    @classmethod
    def _validate_spatial_field(cls, name: str, value: FloatArray) -> None:
        """Validate a 2D spatial field."""
        cls._validate_array(name, value)
        if value.ndim != 2:
            raise ValueError(
                f"'{name}' must be a 2D array with shape (ny, nx), got ndim={value.ndim}"
            )

    @classmethod
    def _validate_vector_field(cls, name: str, value: FloatArray) -> None:
        """Validate a 1D vector field."""
        cls._validate_array(name, value)
        if value.ndim != 1:
            raise ValueError(f"'{name}' must be a 1D array, got ndim={value.ndim}")

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return the shape of the spatial state fields."""
        return self.precipitation.shape
