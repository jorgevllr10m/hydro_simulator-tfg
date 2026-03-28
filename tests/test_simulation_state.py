from datetime import datetime

import numpy as np

from simulator.core.state import SimulationState


def test_spatial_state():
    field = np.zeros((4, 5), dtype=float)

    state = SimulationState(
        step=0,
        timestamp=datetime(2026, 1, 1, 0, 0, 0),
        precipitation=field,
        air_temperature=field,
        pet=field,
        soil_moisture=field,
        surface_runoff=field,
        channel_flow=field,
    )

    print(state.spatial_shape)
