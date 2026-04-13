from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from simulator.core.state import SimulationState
from simulator.core.types import SimulationDomain
from simulator.routing.rules import ReservoirOperationZone

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
    storm_mask: BoolArray | None = None

    def __post_init__(self) -> None:
        SimulationState._validate_spatial_field("precipitation", self.precipitation)
        SimulationState._validate_spatial_field("air_temperature", self.air_temperature)

        if self.background_precipitation is not None:
            SimulationState._validate_spatial_field(
                "background_precipitation",
                self.background_precipitation,
            )

        if self.storm_mask is not None:
            SimulationState._validate_spatial_bool_field("storm_mask", self.storm_mask)


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

    def __post_init__(self) -> None:
        SimulationState._validate_spatial_field("precipitation", self.precipitation)
        SimulationState._validate_spatial_field("pet", self.pet)


@dataclass(frozen=True)
class HydroOutput:
    """Typed output contract for the hydrology module.

    This output contains only per-cell hydrological products generated before
    channel routing and reservoir regulation.
    """

    soil_moisture: FloatArray
    infiltration: FloatArray
    surface_runoff: FloatArray
    aet: FloatArray
    subsurface_runoff: FloatArray | None = None

    def __post_init__(self) -> None:
        SimulationState._validate_spatial_field("soil_moisture", self.soil_moisture)
        SimulationState._validate_spatial_field("infiltration", self.infiltration)
        SimulationState._validate_spatial_field("surface_runoff", self.surface_runoff)
        SimulationState._validate_spatial_field("aet", self.aet)

        if self.subsurface_runoff is not None:
            SimulationState._validate_spatial_field("subsurface_runoff", self.subsurface_runoff)


@dataclass(frozen=True)
class RegulatedRoutingInput:
    """Typed input contract for the regulated routing module.

    This input contains the local hydrological products required to perform
    channel routing and reservoir regulation.
    """

    domain: SimulationDomain
    step: int
    timestamp: datetime

    surface_runoff: FloatArray
    pet: FloatArray
    subsurface_runoff: FloatArray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.domain, SimulationDomain):
            raise TypeError(f"'domain' must be a SimulationDomain, got {type(self.domain).__name__}")

        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError(f"'step' must be a non-negative integer, got {self.step!r}")

        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"'timestamp' must be a datetime, got {type(self.timestamp).__name__}")

        SimulationState._validate_spatial_field("surface_runoff", self.surface_runoff)
        SimulationState._validate_spatial_field("pet", self.pet)

        if self.surface_runoff.shape != self.domain.shape:
            raise ValueError(f"'surface_runoff' must have shape {self.domain.shape}, got {self.surface_runoff.shape}")

        if self.pet.shape != self.domain.shape:
            raise ValueError(f"'pet' must have shape {self.domain.shape}, got {self.pet.shape}")

        if self.subsurface_runoff is not None:
            SimulationState._validate_spatial_field("subsurface_runoff", self.subsurface_runoff)

            if self.subsurface_runoff.shape != self.domain.shape:
                raise ValueError(f"'subsurface_runoff' must have shape {self.domain.shape}, got {self.subsurface_runoff.shape}")


@dataclass(frozen=True)
class RegulatedRoutingOutput:
    """Typed output contract for the regulated routing module."""

    lateral_inflow_m3s: FloatArray
    cell_inflow_m3s: FloatArray
    channel_flow_m3s: FloatArray
    outlet_discharge_m3s: float

    reservoir_inflow_m3s: FloatArray
    reservoir_requested_release_m3s: FloatArray
    reservoir_storage_fraction: FloatArray
    reservoir_surface_area_m2: FloatArray
    reservoir_evaporation_loss_m3: FloatArray
    reservoir_storage_m3: FloatArray
    reservoir_release_m3s: FloatArray
    reservoir_spill_m3s: FloatArray
    reservoir_total_outflow_m3s: FloatArray
    reservoir_zones: tuple[ReservoirOperationZone | None, ...]

    def __post_init__(self) -> None:
        SimulationState._validate_spatial_field("lateral_inflow_m3s", self.lateral_inflow_m3s)
        SimulationState._validate_spatial_field("cell_inflow_m3s", self.cell_inflow_m3s)
        SimulationState._validate_spatial_field("channel_flow_m3s", self.channel_flow_m3s)

        if not isinstance(self.outlet_discharge_m3s, (int, float)):
            raise TypeError(f"'outlet_discharge_m3s' must be numeric, got {type(self.outlet_discharge_m3s).__name__}")

        SimulationState._validate_vector_field("reservoir_inflow_m3s", self.reservoir_inflow_m3s)
        SimulationState._validate_vector_field(
            "reservoir_requested_release_m3s",
            self.reservoir_requested_release_m3s,
        )
        SimulationState._validate_vector_field(
            "reservoir_storage_fraction",
            self.reservoir_storage_fraction,
        )
        SimulationState._validate_vector_field(
            "reservoir_surface_area_m2",
            self.reservoir_surface_area_m2,
        )
        SimulationState._validate_vector_field(
            "reservoir_evaporation_loss_m3",
            self.reservoir_evaporation_loss_m3,
        )
        SimulationState._validate_vector_field("reservoir_storage_m3", self.reservoir_storage_m3)
        SimulationState._validate_vector_field("reservoir_release_m3s", self.reservoir_release_m3s)
        SimulationState._validate_vector_field("reservoir_spill_m3s", self.reservoir_spill_m3s)
        SimulationState._validate_vector_field(
            "reservoir_total_outflow_m3s",
            self.reservoir_total_outflow_m3s,
        )

        n_reservoirs = self.reservoir_storage_m3.shape[0]

        vector_fields = {
            "reservoir_inflow_m3s": self.reservoir_inflow_m3s,
            "reservoir_requested_release_m3s": self.reservoir_requested_release_m3s,
            "reservoir_storage_fraction": self.reservoir_storage_fraction,
            "reservoir_surface_area_m2": self.reservoir_surface_area_m2,
            "reservoir_evaporation_loss_m3": self.reservoir_evaporation_loss_m3,
            "reservoir_storage_m3": self.reservoir_storage_m3,
            "reservoir_release_m3s": self.reservoir_release_m3s,
            "reservoir_spill_m3s": self.reservoir_spill_m3s,
            "reservoir_total_outflow_m3s": self.reservoir_total_outflow_m3s,
        }

        for name, value in vector_fields.items():
            if value.shape[0] != n_reservoirs:
                raise ValueError(f"'{name}' must have length {n_reservoirs}, got {value.shape[0]}")

        if len(self.reservoir_zones) != n_reservoirs:
            raise ValueError(f"'reservoir_zones' must have length {n_reservoirs}, got {len(self.reservoir_zones)}")

        for zone in self.reservoir_zones:
            if zone is not None and not isinstance(zone, ReservoirOperationZone):
                raise TypeError(f"All items in 'reservoir_zones' must be ReservoirOperationZone or None, got {type(zone).__name__}")


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
