from __future__ import annotations

from datetime import datetime

from simulator.core.contracts import (
    EnergyOutput,
    HydroOutput,
    MeteoOutput,
    ObservationOutput,
    ReservoirOutput,
)
from simulator.core.state import SimulationState
from simulator.core.types import FloatArray


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
    reservoir_output: ReservoirOutput | None = None,
    observation_output: ObservationOutput | None = None,
) -> SimulationState:
    """Merge module outputs into a new SimulationState.

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
        Output produced by the hydrology module.
    reservoir_output
        Optional output produced by the reservoir module.
    observation_output
        Optional output produced by the observation module.

    Returns
    -------
    SimulationState
        New simulation state for the current step.
    """
    # TODO rediseñar el manejo de observations para incluir obs_mask y obs_quality_flag o mover observaciones fuera del estado físico.
    # (continue del TODO) (state.py / engine.py)
    # Cuando se implemente de verdad la capa observacional habrá que decidir una de estas dos cosas:
    # o ampliar SimulationState.observations
    # para aceptar bool/int o sacar las observaciones del estado físico y escribirlas directo al dataset observacional.
    observations: dict[str, FloatArray] | None = None

    if observation_output is not None:
        observations = {}

        if observation_output.obs_precipitation is not None:
            observations["obs_precipitation"] = observation_output.obs_precipitation

        if observation_output.obs_discharge is not None:
            observations["obs_discharge"] = observation_output.obs_discharge

        if observation_output.obs_storage is not None:
            observations["obs_storage"] = observation_output.obs_storage

    return SimulationState(
        step=step,
        timestamp=timestamp,
        precipitation=meteo_output.precipitation,
        air_temperature=meteo_output.air_temperature,
        pet=energy_output.pet,
        background_precipitation=meteo_output.background_precipitation,
        storm_mask=meteo_output.storm_mask,
        aet=energy_output.aet,
        shortwave_radiation=energy_output.shortwave_radiation,
        net_radiation=energy_output.net_radiation,
        antecedent_storage=energy_output.antecedent_storage,
        antecedent_relative=energy_output.antecedent_relative,
        antecedent_overflow=energy_output.antecedent_overflow,
        soil_moisture=hydro_output.soil_moisture,
        infiltration=hydro_output.infiltration,
        surface_runoff=hydro_output.surface_runoff,
        subsurface_runoff=hydro_output.subsurface_runoff,
        channel_flow=hydro_output.channel_flow,
        reservoir_inflow=None,  # TODO reservoir_inflow cuando se implemente el acoplamiento real hydro → reservoirs
        # (continue) ese campo tendrá que venir de una salida hidrológica o de una función de enrutamiento/acoplamiento
        reservoir_storage=(reservoir_output.reservoir_storage if reservoir_output is not None else None),
        reservoir_release=(reservoir_output.reservoir_release if reservoir_output is not None else None),
        reservoir_spill=(reservoir_output.reservoir_spill if reservoir_output is not None else None),
        observations=observations,
    )
