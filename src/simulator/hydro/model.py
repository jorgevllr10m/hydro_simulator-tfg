from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from simulator.core.contracts import HydroInput, HydroOutput
from simulator.core.types import FloatArray
from simulator.hydro.runoff import RunoffConfig, RunoffFields, derive_runoff_fields
from simulator.hydro.soil import (
    SoilConfig,
    SoilStepFields,
    build_initial_soil_moisture_mm,
    update_soil_bucket,
)


def _validate_numeric_scalar(name: str, value: int | float) -> float:
    """Validate a numeric scalar and return it as float."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"'{name}' must be numeric, got {type(value).__name__}")
    return float(value)


def _validate_positive_scalar(name: str, value: int | float) -> float:
    """Validate a strictly positive numeric scalar and return it as float."""
    numeric_value = _validate_numeric_scalar(name, value)
    if numeric_value <= 0.0:
        raise ValueError(f"'name' must be > 0, got {numeric_value}")
    return numeric_value


def _validate_spatial_float_array(name: str, value: FloatArray) -> None:
    """Validate a 2D NumPy float array."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"'{name}' must be a numpy.ndarray, got {type(value).__name__}")
    if value.ndim != 2:
        raise ValueError(f"'{name}' must be a 2D array with shape (ny, nx), got ndim={value.ndim}")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"'{name}' must have a floating dtype, got {value.dtype}")


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Validate a canonical spatial shape (ny, nx)."""
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError(f"'shape' must be a tuple[int, int], got {shape!r}")

    ny, nx = shape
    if not isinstance(ny, int) or not isinstance(nx, int):
        raise TypeError(f"'shape' must contain integers, got {shape!r}")
    if ny <= 0 or nx <= 0:
        raise ValueError(f"'shape' must contain positive integers, got {shape!r}")

    return (ny, nx)


@dataclass(frozen=True)
class HydroConfig:
    """Top-level configuration of the simplified hydrology module."""

    soil: SoilConfig = field(default_factory=SoilConfig)
    runoff: RunoffConfig = field(default_factory=RunoffConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.soil, SoilConfig):
            raise TypeError(f"'soil' must be a SoilConfig, got {type(self.soil).__name__}")

        if not isinstance(self.runoff, RunoffConfig):
            raise TypeError(f"'runoff' must be a RunoffConfig, got {type(self.runoff).__name__}")


@dataclass(frozen=True)
class HydroState:
    """Persistent internal state of the hydrology module."""

    soil_moisture_mm: FloatArray

    def __post_init__(self) -> None:
        _validate_spatial_float_array("soil_moisture_mm", self.soil_moisture_mm)

        if np.any(self.soil_moisture_mm < 0.0):
            raise ValueError("'soil_moisture_mm' must be >= 0 everywhere")


@dataclass(frozen=True)
class HydroStepDiagnostics:
    """Detailed diagnostics of the latest hydrological step."""

    step: int
    timestamp: datetime
    soil: SoilStepFields
    runoff: RunoffFields

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError(f"'step' must be a non-negative integer, got {self.step!r}")

        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"'timestamp' must be a datetime, got {type(self.timestamp).__name__}")

        if not isinstance(self.soil, SoilStepFields):
            raise TypeError(f"'soil' must be a SoilStepFields, got {type(self.soil).__name__}")

        if not isinstance(self.runoff, RunoffFields):
            raise TypeError(f"'runoff' must be a RunoffFields, got {type(self.runoff).__name__}")


class HydroModel:
    """Stateful simplified hydrology model.

    This model couples:
    - the per-cell soil bucket
    - local runoff partition
    - a minimal persistent soil-moisture state

    Notes
    -----
    - AET is computed here from PET and soil water.
    - This model performs only local hydrology at cell level.
    - Channel routing and reservoir regulation are handled later by the routing module.
    """

    def __init__(
        self,
        config: HydroConfig,
        *,
        shape: tuple[int, int],
    ) -> None:
        if not isinstance(config, HydroConfig):
            raise TypeError(f"'config' must be a HydroConfig, got {type(config).__name__}")

        self.config = config
        self._shape = _validate_shape(shape)

        self._latest_state = HydroState(
            soil_moisture_mm=build_initial_soil_moisture_mm(
                self._shape,
                config=self.config.soil,
            )
        )
        self._latest_diagnostics: HydroStepDiagnostics | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the fixed spatial shape of the model state."""
        return self._shape

    @property
    def latest_state(self) -> HydroState:
        """Return the latest persistent internal state."""
        return self._latest_state

    @property
    def latest_diagnostics(self) -> HydroStepDiagnostics | None:
        """Return diagnostics from the most recent completed step."""
        return self._latest_diagnostics

    def reset(self) -> None:
        """Reset persistent state and diagnostics to the initial reproducible state."""
        self._latest_state = HydroState(
            soil_moisture_mm=build_initial_soil_moisture_mm(
                self._shape,
                config=self.config.soil,
            )
        )
        self._latest_diagnostics = None

    def step(
        self,
        hydro_input: HydroInput,
    ) -> HydroOutput:
        """Advance the hydrology model by one simulation step."""
        if not isinstance(hydro_input, HydroInput):
            raise TypeError(f"'hydro_input' must be a HydroInput, got {type(hydro_input).__name__}")

        if hydro_input.domain.shape != self._shape:
            raise ValueError(f"'hydro_input.domain.shape' must be {self._shape}, got {hydro_input.domain.shape}")

        if not isinstance(hydro_input.step, int) or hydro_input.step < 0:
            raise ValueError(f"'hydro_input.step' must be a non-negative integer, got {hydro_input.step!r}")

        if not isinstance(hydro_input.timestamp, datetime):
            raise TypeError(f"'hydro_input.timestamp' must be a datetime, got {type(hydro_input.timestamp).__name__}")

        _validate_positive_scalar("dt_seconds", hydro_input.domain.time.dt_seconds)

        _validate_spatial_float_array("hydro_input.precipitation", hydro_input.precipitation)
        _validate_spatial_float_array("hydro_input.pet", hydro_input.pet)
        _validate_spatial_float_array("hydro_input.soil_moisture_prev", hydro_input.soil_moisture_prev)

        if hydro_input.precipitation.shape != self._shape:
            raise ValueError(f"'hydro_input.precipitation' must have shape {self._shape}, got {hydro_input.precipitation.shape}")
        if hydro_input.pet.shape != self._shape:
            raise ValueError(f"'hydro_input.pet' must have shape {self._shape}, got {hydro_input.pet.shape}")
        if hydro_input.soil_moisture_prev.shape != self._shape:
            raise ValueError(
                f"'hydro_input.soil_moisture_prev' must have shape {self._shape}, got {hydro_input.soil_moisture_prev.shape}"
            )

        # TODO(next): detect desynchronization between hydro_input.soil_moisture_prev
        # and self._latest_state.soil_moisture_mm to avoid silent state mismatches.
        soil = update_soil_bucket(
            soil_moisture_prev_mm=hydro_input.soil_moisture_prev,
            precipitation_mm_dt=hydro_input.precipitation,
            pet_mm_dt=hydro_input.pet,
            config=self.config.soil,
        )

        runoff = derive_runoff_fields(
            soil,
            config=self.config.runoff,
        )

        self._latest_state = HydroState(
            soil_moisture_mm=soil.soil_moisture_mm,
        )

        output = HydroOutput(
            soil_moisture=soil.soil_moisture_mm,
            infiltration=runoff.infiltration_mm_dt,
            surface_runoff=runoff.surface_runoff_mm_dt,
            subsurface_runoff=runoff.subsurface_runoff_mm_dt,
            aet=soil.aet_mm_dt,
        )

        self._latest_diagnostics = HydroStepDiagnostics(
            step=hydro_input.step,
            timestamp=hydro_input.timestamp,
            soil=soil,
            runoff=runoff,
        )

        return output
