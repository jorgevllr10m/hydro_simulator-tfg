from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass
class SimulationState:
    """Dynamic state of the simulator at a single time step.

    - Spatial fields use shape (ny, nx), i.e. (y, x)
    - Vector-like reservoir fields use shape (n_reservoirs,)
    - Observation products are kept outside the physical state
    """

    step: int
    timestamp: datetime

    precipitation: FloatArray
    air_temperature: FloatArray
    pet: FloatArray
    soil_moisture: FloatArray
    surface_runoff: FloatArray
    channel_flow: FloatArray

    background_precipitation: FloatArray | None = None
    storm_mask: FloatArray | None = None
    infiltration: FloatArray | None = None
    subsurface_runoff: FloatArray | None = None

    reservoir_inflow: FloatArray | None = None
    reservoir_storage: FloatArray | None = None
    reservoir_release: FloatArray | None = None
    reservoir_spill: FloatArray | None = None

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

        if self.background_precipitation is not None:
            self._validate_spatial_field("background_precipitation", self.background_precipitation)

        if self.storm_mask is not None:
            self._validate_spatial_field("storm_mask", self.storm_mask)

        if self.infiltration is not None:
            self._validate_spatial_field("infiltration", self.infiltration)

        if self.subsurface_runoff is not None:
            self._validate_spatial_field("subsurface_runoff", self.subsurface_runoff)

        spatial_shape = self.precipitation.shape
        spatial_fields = {
            "air_temperature": self.air_temperature,
            "pet": self.pet,
            "soil_moisture": self.soil_moisture,
            "surface_runoff": self.surface_runoff,
            "channel_flow": self.channel_flow,
        }

        optional_spatial_fields = {
            "background_precipitation": self.background_precipitation,
            "storm_mask": self.storm_mask,
            "infiltration": self.infiltration,
            "subsurface_runoff": self.subsurface_runoff,
        }

        for name, value in spatial_fields.items():
            if value.shape != spatial_shape:
                raise ValueError(f"'{name}' must have shape {spatial_shape}, got {value.shape}")

        for name, value in optional_spatial_fields.items():
            if value is not None and value.shape != spatial_shape:
                raise ValueError(f"'{name}' must have shape {spatial_shape}, got {value.shape}")

        if self.reservoir_inflow is not None:
            self._validate_vector_field("reservoir_inflow", self.reservoir_inflow)

        if self.reservoir_storage is not None:
            self._validate_vector_field("reservoir_storage", self.reservoir_storage)

        if self.reservoir_release is not None:
            self._validate_vector_field("reservoir_release", self.reservoir_release)

        if self.reservoir_spill is not None:
            self._validate_vector_field("reservoir_spill", self.reservoir_spill)

        if self.observations is not None:
            if not isinstance(self.observations, dict):
                raise TypeError(f"'observations' must be a dict[str, FloatArray] or None, got {type(self.observations).__name__}")
            for key, value in self.observations.items():
                if not isinstance(key, str):
                    raise TypeError(f"Observation keys must be strings, got {type(key).__name__}")
                self._validate_array(f"observations[{key!r}]", value)

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
            raise ValueError(f"'{name}' must be a 2D array with shape (ny, nx), got ndim={value.ndim}")

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
