from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from simulator.core.state import SimulationState
from simulator.core.types import SimulationDomain

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MeteoInput:
    """Typed input contract for the meteorology module."""

    domain: SimulationDomain
    step: int
    timestamp: datetime
    previous_state: SimulationState | None = None


@dataclass(frozen=True)
class MeteoOutput:
    """Typed output contract for the meteorology module."""

    precipitation: FloatArray
    air_temperature: FloatArray
    background_precipitation: FloatArray | None = None
    storm_mask: FloatArray | None = (
        None  # TODO decidir si storm_mask debe ser booleano estricto en vez de float 2D (contracts.py /state.py)
    )

    def __post_init__(self) -> None:
        SimulationState._validate_spatial_field("precipitation", self.precipitation)
        SimulationState._validate_spatial_field("air_temperature", self.air_temperature)

        if self.background_precipitation is not None:
            SimulationState._validate_spatial_field(
                "background_precipitation",
                self.background_precipitation,
            )

        if self.storm_mask is not None:
            if not isinstance(self.storm_mask, np.ndarray):
                raise TypeError(f"'storm_mask' must be a numpy.ndarray, got {type(self.storm_mask).__name__}")
            if self.storm_mask.ndim != 2:
                raise ValueError(f"'storm_mask' must be a 2D array with shape (ny, nx), got ndim={self.storm_mask.ndim}")


@dataclass(frozen=True)
class EnergyInput:
    """Typed input contract for the energy-balance / PET module."""

    domain: SimulationDomain
    step: int
    timestamp: datetime
    precipitation: FloatArray
    air_temperature: FloatArray

    def __post_init__(self) -> None:
        SimulationState._validate_spatial_field("precipitation", self.precipitation)
        SimulationState._validate_spatial_field("air_temperature", self.air_temperature)


@dataclass(frozen=True)
class EnergyOutput:
    """Typed output contract for the energy-balance module."""

    pet: FloatArray
    shortwave_radiation: FloatArray
    net_radiation: FloatArray

    def __post_init__(self) -> None:
        SimulationState._validate_spatial_field("pet", self.pet)
        SimulationState._validate_spatial_field("shortwave_radiation", self.shortwave_radiation)
        SimulationState._validate_spatial_field("net_radiation", self.net_radiation)


@dataclass(frozen=True)
class HydroInput:
    """Typed input contract for the hydrology module."""

    domain: SimulationDomain
    step: int
    timestamp: datetime
    precipitation: FloatArray
    pet: FloatArray
    soil_moisture_prev: FloatArray

    def __post_init__(self) -> None:
        SimulationState._validate_spatial_field("precipitation", self.precipitation)
        SimulationState._validate_spatial_field("pet", self.pet)
        SimulationState._validate_spatial_field("soil_moisture_prev", self.soil_moisture_prev)


@dataclass(frozen=True)
class HydroOutput:
    """Typed output contract for the hydrology module."""

    soil_moisture: FloatArray
    infiltration: FloatArray
    surface_runoff: FloatArray
    channel_flow: FloatArray
    aet: FloatArray
    subsurface_runoff: FloatArray | None = None
    outlet_discharge: float | None = None

    def __post_init__(self) -> None:
        SimulationState._validate_spatial_field("soil_moisture", self.soil_moisture)
        SimulationState._validate_spatial_field("infiltration", self.infiltration)
        SimulationState._validate_spatial_field("surface_runoff", self.surface_runoff)
        SimulationState._validate_spatial_field("channel_flow", self.channel_flow)
        SimulationState._validate_spatial_field("aet", self.aet)

        if self.subsurface_runoff is not None:
            SimulationState._validate_spatial_field("subsurface_runoff", self.subsurface_runoff)

        if self.outlet_discharge is not None and not isinstance(
            self.outlet_discharge,
            (int, float),
        ):
            raise TypeError(f"'outlet_discharge' must be numeric or None, got {type(self.outlet_discharge).__name__}")


@dataclass(frozen=True)
class ReservoirInput:
    """Typed input contract for the reservoir module."""

    domain: SimulationDomain
    step: int
    timestamp: datetime
    reservoir_inflow: FloatArray
    reservoir_storage_prev: FloatArray

    def __post_init__(self) -> None:
        SimulationState._validate_vector_field("reservoir_inflow", self.reservoir_inflow)
        SimulationState._validate_vector_field(
            "reservoir_storage_prev",
            self.reservoir_storage_prev,
        )


@dataclass(frozen=True)
class ReservoirOutput:
    """Typed output contract for the reservoir module."""

    reservoir_storage: FloatArray
    reservoir_release: FloatArray
    reservoir_spill: FloatArray

    def __post_init__(self) -> None:
        SimulationState._validate_vector_field("reservoir_storage", self.reservoir_storage)
        SimulationState._validate_vector_field("reservoir_release", self.reservoir_release)
        SimulationState._validate_vector_field("reservoir_spill", self.reservoir_spill)


@dataclass(frozen=True)
class ObservationInput:
    """Typed input contract for the observation module."""

    domain: SimulationDomain
    step: int
    timestamp: datetime
    precipitation: FloatArray
    channel_flow: FloatArray
    reservoir_storage: FloatArray | None = None

    def __post_init__(self) -> None:
        SimulationState._validate_spatial_field("precipitation", self.precipitation)
        SimulationState._validate_spatial_field("channel_flow", self.channel_flow)

        if self.reservoir_storage is not None:
            SimulationState._validate_vector_field("reservoir_storage", self.reservoir_storage)


@dataclass(frozen=True)
class ObservationOutput:
    """Typed output contract for the observation module."""

    obs_precipitation: FloatArray | None = None
    obs_discharge: FloatArray | None = None
    obs_storage: FloatArray | None = None
    obs_mask: BoolArray | None = None
    obs_quality_flag: NDArray[np.int_] | None = None

    def __post_init__(self) -> None:
        if self.obs_precipitation is not None:
            SimulationState._validate_vector_field("obs_precipitation", self.obs_precipitation)

        if self.obs_discharge is not None:
            SimulationState._validate_vector_field("obs_discharge", self.obs_discharge)

        if self.obs_storage is not None:
            SimulationState._validate_vector_field("obs_storage", self.obs_storage)

        if self.obs_mask is not None:
            if not isinstance(self.obs_mask, np.ndarray):
                raise TypeError(f"'obs_mask' must be a numpy.ndarray, got {type(self.obs_mask).__name__}")
            if self.obs_mask.ndim != 1:
                raise ValueError(f"'obs_mask' must be a 1D array, got ndim={self.obs_mask.ndim}")
            if self.obs_mask.dtype != np.bool_:
                raise TypeError(f"'obs_mask' must have boolean dtype, got {self.obs_mask.dtype}")

        if self.obs_quality_flag is not None:
            if not isinstance(self.obs_quality_flag, np.ndarray):
                raise TypeError(f"'obs_quality_flag' must be a numpy.ndarray, got {type(self.obs_quality_flag).__name__}")
            if self.obs_quality_flag.ndim != 1:
                raise ValueError(f"'obs_quality_flag' must be a 1D array, got ndim={self.obs_quality_flag.ndim}")
            if not np.issubdtype(self.obs_quality_flag.dtype, np.integer):
                raise TypeError(f"'obs_quality_flag' must have integer dtype, got {self.obs_quality_flag.dtype}")
