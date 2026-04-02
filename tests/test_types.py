from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from simulator.core.time import TimeDefinition
from simulator.core.types import (
    BasinDefinition,
    GridDefinition,
    ReservoirDefinition,
    SensorDefinition,
    SimulationDomain,
    SpatialDomain,
)


def _make_spatial_domain(nx: int = 5, ny: int = 4) -> SpatialDomain:
    grid = GridDefinition(nx=nx, ny=ny, dx=1000.0, dy=1000.0)
    basin = BasinDefinition(mask=np.ones((ny, nx), dtype=bool))
    return SpatialDomain(grid=grid, basin=basin)


def _make_time_definition() -> TimeDefinition:
    return TimeDefinition(
        start=datetime(2025, 1, 1, 0, 0, 0),
        dt_seconds=3600,
        n_steps=10,
    )


def test_simulation_domain_accepts_reservoirs_and_sensors_inside_grid() -> None:
    domain = SimulationDomain(
        spatial=_make_spatial_domain(nx=5, ny=4),
        time=_make_time_definition(),
        reservoirs=(
            ReservoirDefinition(
                name="R1",
                cell_y=2,
                cell_x=3,
                capacity=100.0,
                initial_storage=50.0,
            ),
        ),
        sensors=(
            SensorDefinition(
                name="S1",
                sensor_type="precipitation",
                cell_y=1,
                cell_x=4,
            ),
        ),
    )

    assert domain.shape == (4, 5)
    assert len(domain.reservoirs) == 1
    assert len(domain.sensors) == 1


def test_simulation_domain_rejects_reservoir_outside_grid_in_y() -> None:
    with pytest.raises(ValueError, match="Reservoir 'R1' has cell_y=4"):
        SimulationDomain(
            spatial=_make_spatial_domain(nx=5, ny=4),
            time=_make_time_definition(),
            reservoirs=(
                ReservoirDefinition(
                    name="R1",
                    cell_y=4,
                    cell_x=2,
                    capacity=100.0,
                    initial_storage=50.0,
                ),
            ),
        )


def test_simulation_domain_rejects_sensor_outside_grid_in_x() -> None:
    with pytest.raises(ValueError, match="Sensor 'S1' has cell_x=5"):
        SimulationDomain(
            spatial=_make_spatial_domain(nx=5, ny=4),
            time=_make_time_definition(),
            sensors=(
                SensorDefinition(
                    name="S1",
                    sensor_type="precipitation",
                    cell_y=1,
                    cell_x=5,
                ),
            ),
        )
