from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class GridConfig(BaseModel):
    """Configuration schema for the spatial grid."""

    nx: int = Field(..., gt=0, description="Number of grid columns")
    ny: int = Field(..., gt=0, description="Number of grid rows")
    dx: float = Field(..., gt=0, description="Grid spacing in x direction [m]")
    dy: float = Field(..., gt=0, description="Grid spacing in y direction [m]")
    x0: float = Field(0.0, description="X origin of the domain [m]")
    y0: float = Field(0.0, description="Y origin of the domain [m]")
    crs: str | None = Field(None, description="Coordinate reference system identifier")


class TimeConfig(BaseModel):
    """Configuration schema for the temporal domain."""

    start: str = Field(..., description="Simulation start datetime in ISO format")
    dt_seconds: int = Field(..., gt=0, description="Simulation time step in seconds")
    n_steps: int = Field(..., gt=0, description="Number of simulation time steps")
    calendar_type: Literal["monthly", "seasonal"] = Field(
        "monthly",
        description="Calendar interpretation used for seasonal indexing",
    )


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


class SimulationConfig(BaseModel):
    """Top-level configuration schema for the simulator architecture base."""

    grid: GridConfig
    time: TimeConfig
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
