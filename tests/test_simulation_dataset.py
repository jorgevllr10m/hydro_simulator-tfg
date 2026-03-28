from datetime import datetime

import numpy as np

from simulator.core.dataset import create_empty_dataset, write_state_to_dataset
from simulator.core.state import SimulationState
from simulator.core.time import TimeDefinition
from simulator.core.types import (
    BasinDefinition,
    GridDefinition,
    SimulationDomain,
    SpatialDomain,
)


def test_simulation_dataset():
    grid = GridDefinition(nx=4, ny=3, dx=1000.0, dy=1000.0)
    mask = np.ones((3, 4), dtype=bool)
    basin = BasinDefinition(mask=mask)
    spatial = SpatialDomain(grid=grid, basin=basin)

    time = TimeDefinition(
        start=datetime(2026, 1, 1, 0, 0, 0),
        dt_seconds=3600,
        n_steps=5,
    )

    domain = SimulationDomain(spatial=spatial, time=time)

    ds = create_empty_dataset(domain)
    print("\nSIMULATION DATASET TEST")
    print(ds)
    field = np.ones((3, 4), dtype=float)
    state = SimulationState(
        step=0,
        timestamp=datetime(2026, 1, 1, 0, 0, 0),
        precipitation=field * 2.0,
        air_temperature=field * 15.0,
        pet=field * 0.5,
        soil_moisture=field * 100.0,
        surface_runoff=field * 0.2,
        channel_flow=field * 1.5,
    )
    print("\nAFTER MODIFYING STATE")
    ds = write_state_to_dataset(ds, state)
    print(ds["precipitation"].isel(time=0).values)
