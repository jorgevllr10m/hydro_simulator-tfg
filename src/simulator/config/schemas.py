from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from simulator.meteo.latent_state import MoistureScenario, ThermalScenario
from simulator.meteo.regimes import MeteorologicalRegime


class GridConfig(BaseModel):
    """Configuration schema for the spatial grid."""

    nx: int = Field(..., gt=0, description="Number of grid columns")
    ny: int = Field(..., gt=0, description="Number of grid rows")
    dx: float = Field(..., gt=0, description="Grid spacing in x direction [m]")
    dy: float = Field(..., gt=0, description="Grid spacing in y direction [m]")
    x0: float = Field(0.0, description="X origin of the domain [m]")
    y0: float = Field(0.0, description="Y origin of the domain [m]")
    crs: str | None = Field(None, description="Coordinate reference system identifier")


class ReservoirConfig(BaseModel):
    """Configuration schema for a reservoir definition."""

    name: str = Field(..., min_length=1, description="Reservoir name")
    cell_y: int = Field(..., ge=0, description="Reservoir y-index in the grid")
    cell_x: int = Field(..., ge=0, description="Reservoir x-index in the grid")
    capacity: float = Field(..., gt=0, description="Reservoir maximum storage [m3]")
    initial_storage: float = Field(..., ge=0, description="Initial storage [m3]")

    @model_validator(mode="after")
    def validate_storage_not_above_capacity(self) -> "ReservoirConfig":
        """Ensure initial storage does not exceed capacity."""
        if self.initial_storage > self.capacity:
            raise ValueError("'initial_storage' cannot be greater than 'capacity'")
        return self


class SensorConfig(BaseModel):
    """Configuration schema for an observation sensor."""

    name: str = Field(..., min_length=1, description="Sensor name")
    sensor_type: str = Field(..., min_length=1, description="Type of observed variable")
    cell_y: int = Field(..., ge=0, description="Sensor y-index in the grid")
    cell_x: int = Field(..., ge=0, description="Sensor x-index in the grid")


class RunConfig(BaseModel):
    """Top-level run metadata from config.yaml."""

    name: str = Field(..., min_length=1, description="Run name")
    output_dir: str = Field(..., min_length=1, description="Output directory for run artifacts")


class SimulationWindowConfig(BaseModel):
    """User-facing simulation window configuration from config.yaml.

    Notes
    -----
    This schema intentionally uses:
    - start_date
    - end_date
    - time_step_hours

    rather than the internal TimeDefinition representation
    (start, dt_seconds, n_steps).

    The conversion to internal runtime time settings will be handled later
    by the loader/runner.
    """

    start_date: str = Field(..., description="Simulation start date/datetime in ISO format")
    end_date: str = Field(..., description="Simulation end date/datetime in ISO format (exclusive)")
    time_step_hours: int = Field(..., gt=0, description="Simulation time step in hours")
    calendar_type: Literal["monthly", "seasonal"] = Field(
        "monthly",
        description="Calendar interpretation used for seasonal indexing",
    )

    @model_validator(mode="after")
    def validate_time_window(self) -> "SimulationWindowConfig":
        """Ensure the configured simulation window is valid and divisible by the time step."""
        try:
            start_dt = datetime.fromisoformat(self.start_date)
        except ValueError as exc:
            raise ValueError(f"'start_date' must be a valid ISO date/datetime, got {self.start_date!r}") from exc

        try:
            end_dt = datetime.fromisoformat(self.end_date)
        except ValueError as exc:
            raise ValueError(f"'end_date' must be a valid ISO date/datetime, got {self.end_date!r}") from exc

        if end_dt <= start_dt:
            raise ValueError(f"'end_date' ({self.end_date}) must be later than 'start_date' ({self.start_date})")

        total_seconds = (end_dt - start_dt).total_seconds()
        dt_seconds = self.time_step_hours * 3600

        if total_seconds % dt_seconds != 0:
            raise ValueError("The interval [start_date, end_date) must be exactly divisible by time_step_hours")

        return self


class DomainSelectionConfig(BaseModel):
    """Domain preset selection from config.yaml."""

    preset: str = Field(..., min_length=1, description="Domain preset name")


class ScenarioSelectionConfig(BaseModel):
    """Scenario selection from config.yaml."""

    name: str = Field(..., min_length=1, description="Scenario name")


class MasterConfig(BaseModel):
    """Schema of the master config.yaml file."""

    run: RunConfig
    simulation: SimulationWindowConfig
    domain: DomainSelectionConfig
    scenario: ScenarioSelectionConfig


class DomainPresetConfig(BaseModel):
    """Schema of configs/domain/*.yaml presets."""

    grid: GridConfig
    reservoirs: list[ReservoirConfig] = Field(default_factory=list)
    sensors: list[SensorConfig] = Field(default_factory=list)

    @field_validator("reservoirs")
    @classmethod
    def validate_unique_reservoir_names(
        cls,
        value: list[ReservoirConfig],
    ) -> list[ReservoirConfig]:
        """Ensure reservoir names are unique."""
        names = [item.name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("Reservoir names must be unique")
        return value

    @field_validator("sensors")
    @classmethod
    def validate_unique_sensor_names(
        cls,
        value: list[SensorConfig],
    ) -> list[SensorConfig]:
        """Ensure sensor names are unique."""
        names = [item.name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("Sensor names must be unique")
        return value


# * Meteo module
class LatentEnvironmentOverrideConfig(BaseModel):
    """Optional scenario overrides for the latent meteorological environment.

    Only a minimal subset is exposed for now, matching current phase-3 needs.
    More fields can be added incrementally later.
    """

    random_seed: int | None = Field(None, description="Random seed for latent environment")
    initial_regime: MeteorologicalRegime | None = Field(None, description="Initial weather regime")
    thermal_scenario: ThermalScenario | None = Field(None, description="Thermal scenario")
    moisture_scenario: MoistureScenario | None = Field(None, description="Moisture scenario")
    regime_persistence: float | None = Field(None, ge=0.0, le=1.0, description="Persistence of the weather regime")
    spell_memory: float | None = Field(None, ge=0.0, le=1.0, description="Temporal memory of wet/dry spell state")


class StormBirthOverrideConfig(BaseModel):
    """Optional scenario overrides for storm birth configuration."""

    expected_births_per_step: float | None = Field(
        None,
        ge=0.0,
        description="Baseline expected number of new storms per step",
    )
    mean_peak_intensity_mmph: float | None = Field(
        None,
        gt=0.0,
        description="Mean mature-stage peak rainfall intensity [mm/h]",
    )
    mean_duration_steps: int | None = Field(
        None,
        gt=0,
        description="Mean storm duration in simulation steps",
    )
    band_cluster_probability: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Probability of reorganizing same-step births into a band",
    )


class BackgroundFieldOverrideConfig(BaseModel):
    """Optional scenario overrides for the background precipitation field."""

    enabled: bool | None = Field(None, description="Whether the background precipitation field is enabled")
    random_seed: int | None = Field(None, description="Random seed for the background field")

    temporal_persistence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Temporal persistence of the background spatial pattern",
    )
    max_intensity_mm_dt: float | None = Field(
        None,
        ge=0.0,
        description="Reference maximum intensity of the background component [mm/dt]",
    )


class MeteoScenarioConfig(BaseModel):
    """Meteorological overrides contained in a scenario file."""

    latent_environment: LatentEnvironmentOverrideConfig = Field(default_factory=LatentEnvironmentOverrideConfig)
    storm_birth: StormBirthOverrideConfig = Field(default_factory=StormBirthOverrideConfig)
    background: BackgroundFieldOverrideConfig = Field(default_factory=BackgroundFieldOverrideConfig)


# * Energy module
class PETOverrideConfig(BaseModel):
    """Optional scenario overrides for the simplified PET model."""

    pet_multiplier: float | None = Field(
        None,
        ge=0.0,
        description="Scenario multiplier applied to PET",
    )


class EnergyScenarioConfig(BaseModel):
    """Energy-balance overrides contained in a scenario file."""

    latitude_deg: float | None = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Synthetic latitude used by the solar-geometry model [deg]",
    )
    pet: PETOverrideConfig = Field(default_factory=PETOverrideConfig)


# * Hydrology module
class SoilOverrideConfig(BaseModel):
    """Optional scenario overrides for the simplified soil bucket."""

    capacity_mm: float | None = Field(
        None,
        gt=0.0,
        description="Maximum soil-water storage capacity [mm]",
    )
    initial_relative: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Initial relative filling of the soil bucket [0-1]",
    )
    max_infiltration_mm_dt: float | None = Field(
        None,
        gt=0.0,
        description="Maximum infiltration capacity per simulation step [mm/dt]",
    )
    percolation_rate_mm_dt: float | None = Field(
        None,
        ge=0.0,
        description="Maximum percolation/drainage rate per simulation step [mm/dt]",
    )


class RunoffOverrideConfig(BaseModel):
    """Optional scenario overrides for the simplified runoff partition."""

    subsurface_runoff_fraction: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of soil percolation exposed as subsurface runoff [0-1]",
    )


class HydroScenarioConfig(BaseModel):
    """Hydrology overrides contained in a scenario file."""

    soil: SoilOverrideConfig = Field(default_factory=SoilOverrideConfig)
    runoff: RunoffOverrideConfig = Field(default_factory=RunoffOverrideConfig)


# -------


class ScenarioConfig(BaseModel):
    """Schema of configs/scenarios/*.yaml files."""

    meteo: MeteoScenarioConfig = Field(default_factory=MeteoScenarioConfig)
    energy: EnergyScenarioConfig = Field(default_factory=EnergyScenarioConfig)
    hydro: HydroScenarioConfig = Field(default_factory=HydroScenarioConfig)
