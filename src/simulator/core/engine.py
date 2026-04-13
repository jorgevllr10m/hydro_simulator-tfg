from __future__ import annotations

from datetime import datetime

from simulator.core.contracts import (
    EnergyOutput,
    HydroOutput,
    MeteoOutput,
    RegulatedRoutingOutput,
)
from simulator.core.state import SimulationState


def merge_module_outputs(
    *,
    previous_state: SimulationState
    | None,  # TODO usar previous_state para preservar o propagar variables persistentes cuando algún módulo
    # no produzca todas sus salidas.
    step: int,
    timestamp: datetime,
    meteo_output: MeteoOutput,
    energy_output: EnergyOutput,
    hydro_output: HydroOutput,
    routing_output: RegulatedRoutingOutput,
) -> SimulationState:
    """Merge physical module outputs into a new SimulationState.

    Parameters
    ----------
    previous_state
        Previous simulation state. Included for future transition logic.
    step
        Current simulation step index.
    timestamp
        Current simulation timestamp.
    meteo_output
        Output produced by the meteorology module.
    energy_output
        Output produced by the energy/PET module.
    hydro_output
        Output produced by the local hydrology module.
    routing_output
        Output produced by the regulated routing module.

    Returns
    -------
    SimulationState
        New physical simulation state for the current step.
    """
    return SimulationState(
        step=step,
        timestamp=timestamp,
        precipitation=meteo_output.precipitation,
        air_temperature=meteo_output.air_temperature,
        pet=energy_output.pet,
        background_precipitation=meteo_output.background_precipitation,
        storm_mask=meteo_output.storm_mask,
        aet=hydro_output.aet,
        shortwave_radiation=energy_output.shortwave_radiation,
        net_radiation=energy_output.net_radiation,
        soil_moisture=hydro_output.soil_moisture,
        infiltration=hydro_output.infiltration,
        surface_runoff=hydro_output.surface_runoff,
        subsurface_runoff=hydro_output.subsurface_runoff,
        channel_flow=routing_output.channel_flow_m3s,
        outlet_discharge=routing_output.outlet_discharge_m3s,
        reservoir_inflow=routing_output.reservoir_inflow_m3s,
        reservoir_storage=routing_output.reservoir_storage_m3,
        reservoir_release=routing_output.reservoir_release_m3s,
        reservoir_spill=routing_output.reservoir_spill_m3s,
    )
