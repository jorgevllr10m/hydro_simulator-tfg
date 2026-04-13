from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from simulator.common.validation import (
    validate_latitude_deg as _validate_latitude_deg,
)
from simulator.common.validation import (
    validate_positive_scalar as _validate_positive_scalar,
)
from simulator.common.validation import (
    validate_shape_2d as _validate_shape,
)
from simulator.common.validation import (
    validate_spatial_float_array as _validate_spatial_float_array,
)
from simulator.core.contracts import EnergyInput, EnergyOutput
from simulator.energy.pet import PETConfig, PETFields, compute_priestley_taylor_pet_mm_dt
from simulator.energy.radiation import (
    RadiationFields,
    SolarRadiationConfig,
    compute_radiation_fields,
)
from simulator.energy.solar import SolarGeometry, compute_solar_geometry


@dataclass(frozen=True)
class EnergyBalanceConfig:
    """Top-level configuration of the simplified energy-balance module."""

    latitude_deg: float = 40.0

    solar: SolarRadiationConfig = field(default_factory=SolarRadiationConfig)
    pet: PETConfig = field(default_factory=PETConfig)

    def __post_init__(self) -> None:
        _validate_latitude_deg(self.latitude_deg)

        if not isinstance(self.solar, SolarRadiationConfig):
            raise TypeError(f"'solar' must be a SolarRadiationConfig, got {type(self.solar).__name__}")

        if not isinstance(self.pet, PETConfig):
            raise TypeError(f"'pet' must be a PETConfig, got {type(self.pet).__name__}")


@dataclass(frozen=True)
class EnergyStepDiagnostics:
    """Detailed diagnostics of the latest energy-balance step."""

    step: int
    timestamp: datetime

    solar_geometry: SolarGeometry
    radiation: RadiationFields
    pet: PETFields

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError(f"'step' must be a non-negative integer, got {self.step!r}")

        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"'timestamp' must be a datetime, got {type(self.timestamp).__name__}")

        if not isinstance(self.solar_geometry, SolarGeometry):
            raise TypeError(f"'solar_geometry' must be a SolarGeometry, got {type(self.solar_geometry).__name__}")

        if not isinstance(self.radiation, RadiationFields):
            raise TypeError(f"'radiation' must be a RadiationFields, got {type(self.radiation).__name__}")

        if not isinstance(self.pet, PETFields):
            raise TypeError(f"'pet' must be a PETFields, got {type(self.pet).__name__}")


class EnergyBalanceModel:
    """Stateful simplified energy-balance model.

    This model couples:
    - solar geometry
    - simplified shortwave radiation
    - simplified Priestley-Taylor PET

    Notes
    -----
    - This module does not compute actual evapotranspiration (AET).
    - AET is computed later by the hydrology/soil module from PET and soil water.
    """

    def __init__(
        self,
        config: EnergyBalanceConfig,
        *,
        shape: tuple[int, int],
    ) -> None:
        if not isinstance(config, EnergyBalanceConfig):
            raise TypeError(f"'config' must be an EnergyBalanceConfig, got {type(config).__name__}")

        self.config = config
        self._shape = _validate_shape(shape)
        self._latest_diagnostics: EnergyStepDiagnostics | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the fixed spatial shape of the model state."""
        return self._shape

    @property
    def latest_diagnostics(self) -> EnergyStepDiagnostics | None:
        """Return diagnostics from the most recent completed step."""
        return self._latest_diagnostics

    def reset(self) -> None:
        """Reset diagnostics to the initial reproducible state."""
        self._latest_diagnostics = None

    def step(
        self,
        energy_input: EnergyInput,
    ) -> EnergyOutput:
        """Advance the energy-balance model by one simulation step."""
        if not isinstance(energy_input, EnergyInput):
            raise TypeError(f"'energy_input' must be an EnergyInput, got {type(energy_input).__name__}")

        if energy_input.domain.shape != self._shape:
            raise ValueError(f"'energy_input.domain.shape' must be {self._shape}, got {energy_input.domain.shape}")

        if not isinstance(energy_input.step, int) or energy_input.step < 0:
            raise ValueError(f"'energy_input.step' must be a non-negative integer, got {energy_input.step!r}")

        if not isinstance(energy_input.timestamp, datetime):
            raise TypeError(f"'energy_input.timestamp' must be a datetime, got {type(energy_input.timestamp).__name__}")

        _validate_positive_scalar("dt_seconds", energy_input.domain.time.dt_seconds)

        _validate_spatial_float_array("energy_input.precipitation", energy_input.precipitation)
        _validate_spatial_float_array("energy_input.air_temperature", energy_input.air_temperature)

        if energy_input.precipitation.shape != self._shape:
            raise ValueError(f"'energy_input.precipitation' must have shape {self._shape}, got {energy_input.precipitation.shape}")
        if energy_input.air_temperature.shape != self._shape:
            raise ValueError(f"'energy_input.air_temperature' must have shape {self._shape}, got {energy_input.air_temperature.shape}")

        solar_geometry = compute_solar_geometry(
            timestamp=energy_input.timestamp,
            latitude_deg=self.config.latitude_deg,
        )

        radiation = compute_radiation_fields(
            precipitation_mm_dt=energy_input.precipitation,
            solar_geometry=solar_geometry,
            dt_seconds=energy_input.domain.time.dt_seconds,
            config=self.config.solar,
        )

        pet = compute_priestley_taylor_pet_mm_dt(
            net_radiation_mj_m2_dt=radiation.net_radiation_mj_m2_dt,
            air_temperature_c=energy_input.air_temperature,
            config=self.config.pet,
        )

        output = EnergyOutput(
            pet=pet.pet_mm_dt,
            shortwave_radiation=radiation.shortwave_in_w_m2,
            net_radiation=radiation.net_radiation_mj_m2_dt,
        )

        self._latest_diagnostics = EnergyStepDiagnostics(
            step=energy_input.step,
            timestamp=energy_input.timestamp,
            solar_geometry=solar_geometry,
            radiation=radiation,
            pet=pet,
        )

        return output
