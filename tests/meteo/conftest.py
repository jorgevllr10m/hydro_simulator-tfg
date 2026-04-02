from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from simulator.core.contracts import MeteoInput
from simulator.core.time import TimeDefinition
from simulator.core.types import (
    BasinDefinition,
    GridDefinition,
    SimulationDomain,
    SpatialDomain,
)


@pytest.fixture
def domain() -> SimulationDomain:
    grid = GridDefinition(
        nx=6,
        ny=5,
        dx=1_000.0,
        dy=1_000.0,
        x0=0.0,
        y0=0.0,
    )
    basin = BasinDefinition(mask=np.ones(grid.shape, dtype=bool))
    spatial = SpatialDomain(grid=grid, basin=basin)
    time = TimeDefinition(
        start=datetime(2025, 1, 1, 0, 0, 0),
        dt_seconds=3_600,
        n_steps=6,
    )
    return SimulationDomain(spatial=spatial, time=time)


@pytest.fixture
def meteo_input_factory(domain: SimulationDomain):
    def _make(step: int) -> MeteoInput:
        return MeteoInput(
            domain=domain,
            step=step,
            timestamp=domain.time.timestamps[step],
        )

    return _make
