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


@pytest.fixture
def small_domain() -> SimulationDomain:
    grid = GridDefinition(
        nx=6,
        ny=5,
        dx=1000.0,
        dy=1000.0,
        x0=0.0,
        y0=0.0,
    )
    basin = BasinDefinition(mask=np.ones(grid.shape, dtype=bool))
    spatial = SpatialDomain(grid=grid, basin=basin)
    time = TimeDefinition(
        start=datetime(2026, 1, 1, 0, 0, 0),
        dt_seconds=3600,
        n_steps=3,
    )
    return SimulationDomain(
        spatial=spatial,
        time=time,
        reservoirs=(),
        sensors=(),
    )


@pytest.fixture
def small_domain_with_reservoir() -> SimulationDomain:
    grid = GridDefinition(
        nx=6,
        ny=5,
        dx=1000.0,
        dy=1000.0,
        x0=0.0,
        y0=0.0,
    )
    basin = BasinDefinition(mask=np.ones(grid.shape, dtype=bool))
    spatial = SpatialDomain(grid=grid, basin=basin)
    time = TimeDefinition(
        start=datetime(2026, 1, 1, 0, 0, 0),
        dt_seconds=3600,
        n_steps=3,
    )
    reservoir = ReservoirDefinition(
        name="test_reservoir",
        cell_y=2,
        cell_x=2,
        capacity=1_000_000.0,
        initial_storage=500_000.0,
    )
    storage_sensor = SensorDefinition(
        name="storage_sensor",
        sensor_type="reservoir_storage",
        cell_y=2,
        cell_x=2,
    )
    return SimulationDomain(
        spatial=spatial,
        time=time,
        reservoirs=(reservoir,),
        sensors=(storage_sensor,),
    )
