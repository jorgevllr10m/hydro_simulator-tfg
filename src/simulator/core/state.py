from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from simulator.common.validation import (
    validate_bool_array,
    validate_float_array,
    validate_spatial_bool_array,
    validate_spatial_float_array,
    validate_vector_float_array,
)

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
    outlet_discharge: float

    background_precipitation: FloatArray | None = None
    storm_mask: BoolArray | None = None

    aet: FloatArray | None = None
    shortwave_radiation: FloatArray | None = None
    net_radiation: FloatArray | None = None

    infiltration: FloatArray | None = None
    subsurface_runoff: FloatArray | None = None

    reservoir_inflow: FloatArray | None = None
    reservoir_storage: FloatArray | None = None
    reservoir_release: FloatArray | None = None
    reservoir_spill: FloatArray | None = None

    def __post_init__(self) -> None:
        """Validate dynamic state fields."""
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError(f"'step' must be a non-negative integer, got {self.step!r}")

        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"'timestamp' must be a datetime, got {type(self.timestamp).__name__}")

        self._validate_spatial_field("precipitation", self.precipitation)
        self._validate_spatial_field("air_temperature", self.air_temperature)
        self._validate_spatial_field("pet", self.pet)
        self._validate_spatial_field("soil_moisture", self.soil_moisture)
        self._validate_spatial_field("surface_runoff", self.surface_runoff)
        self._validate_spatial_field("channel_flow", self.channel_flow)

        if not isinstance(self.outlet_discharge, (int, float)):
            raise TypeError(f"'outlet_discharge' must be numeric, got {type(self.outlet_discharge).__name__}")

        if self.background_precipitation is not None:
            self._validate_spatial_field("background_precipitation", self.background_precipitation)

        if self.storm_mask is not None:
            self._validate_spatial_bool_field("storm_mask", self.storm_mask)

        if self.aet is not None:
            self._validate_spatial_field("aet", self.aet)

        if self.shortwave_radiation is not None:
            self._validate_spatial_field("shortwave_radiation", self.shortwave_radiation)

        if self.net_radiation is not None:
            self._validate_spatial_field("net_radiation", self.net_radiation)

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
            "aet": self.aet,
            "shortwave_radiation": self.shortwave_radiation,
            "net_radiation": self.net_radiation,
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

        reservoir_fields = {
            "reservoir_inflow": self.reservoir_inflow,
            "reservoir_storage": self.reservoir_storage,
            "reservoir_release": self.reservoir_release,
            "reservoir_spill": self.reservoir_spill,
        }

        present_reservoir_fields = {name: value for name, value in reservoir_fields.items() if value is not None}

        if present_reservoir_fields:
            expected_length = next(iter(present_reservoir_fields.values())).shape[0]

            for name, value in present_reservoir_fields.items():
                if value.shape[0] != expected_length:
                    raise ValueError(f"'{name}' must have length {expected_length}, got {value.shape[0]}")

    @staticmethod
    def _validate_array(name: str, value: FloatArray) -> None:
        """Validate that a value is a NumPy float array."""
        validate_float_array(name, value)

    @classmethod
    def _validate_spatial_field(cls, name: str, value: FloatArray) -> None:
        """Validate a 2D spatial field."""
        validate_spatial_float_array(name, value)

    @classmethod
    def _validate_vector_field(cls, name: str, value: FloatArray) -> None:
        """Validate a 1D vector field."""
        validate_vector_float_array(name, value)

    @staticmethod
    def _validate_bool_array(name: str, value: BoolArray) -> None:
        """Validate that a value is a NumPy boolean array."""
        validate_bool_array(name, value)

    @classmethod
    def _validate_spatial_bool_field(cls, name: str, value: BoolArray) -> None:
        """Validate a 2D boolean spatial field."""
        validate_spatial_bool_array(name, value)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return the shape of the spatial state fields."""
        return self.precipitation.shape
