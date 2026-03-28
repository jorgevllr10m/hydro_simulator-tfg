from datetime import datetime

import numpy as np

from simulator.core.contracts import (
    EnergyOutput,
    HydroOutput,
    MeteoOutput,
)
from simulator.core.engine import merge_module_outputs


def test_merge_module_outputs_without_reservoirs_or_observations():
    field = np.ones((3, 4), dtype=float)

    meteo_out = MeteoOutput(
        precipitation=field * 2.0,
        air_temperature=field * 15.0,
    )

    energy_out = EnergyOutput(
        pet=field * 0.5,
    )

    hydro_out = HydroOutput(
        soil_moisture=field * 100.0,
        infiltration=field * 1.0,
        surface_runoff=field * 0.2,
        channel_flow=field * 3.0,
    )

    state = merge_module_outputs(
        previous_state=None,
        step=0,
        timestamp=datetime(2026, 1, 1, 0, 0, 0),
        meteo_output=meteo_out,
        energy_output=energy_out,
        hydro_output=hydro_out,
    )

    assert state.step == 0
    assert state.precipitation.shape == (3, 4)
    assert np.allclose(state.pet, 0.5)
    assert np.allclose(state.soil_moisture, 100.0)
    assert state.reservoir_storage is None
    assert state.observations is None
