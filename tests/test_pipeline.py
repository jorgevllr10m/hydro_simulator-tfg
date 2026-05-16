from __future__ import annotations

from simulator.core.contracts import (
    EnergyInput,
    HydroInput,
    MeteoInput,
    RegulatedRoutingInput,
)
from simulator.energy.model import EnergyBalanceConfig, EnergyBalanceModel
from simulator.hydro.model import HydroConfig, HydroModel
from simulator.meteo.background_field import BackgroundFieldConfig
from simulator.meteo.precipitation_model import (
    StormPrecipitationConfig,
    StormPrecipitationModel,
)
from simulator.meteo.storm_birth import StormBirthConfig
from simulator.meteo.temperature_field import TemperatureFieldConfig
from simulator.routing.model import RegulatedRoutingConfig, RegulatedRoutingModel
from simulator.routing.network import build_simplified_drainage_network


def test_small_pipeline_runs_one_step(small_domain) -> None:
    timestamp = small_domain.time.timestamps[0]

    meteo_model = StormPrecipitationModel(
        StormPrecipitationConfig(
            birth=StormBirthConfig(expected_births_per_step=0.0),
            background=BackgroundFieldConfig(enabled=False),
            temperature=TemperatureFieldConfig(enabled=False),
        )
    )
    energy_model = EnergyBalanceModel(
        EnergyBalanceConfig(),
        shape=small_domain.shape,
    )
    hydro_model = HydroModel(
        HydroConfig(),
        shape=small_domain.shape,
    )
    network = build_simplified_drainage_network(small_domain)
    routing_model = RegulatedRoutingModel(
        RegulatedRoutingConfig(),
        domain=small_domain,
        network=network,
    )

    meteo_output = meteo_model.step(
        MeteoInput(
            domain=small_domain,
            step=0,
            timestamp=timestamp,
        )
    )

    energy_output = energy_model.step(
        EnergyInput(
            domain=small_domain,
            step=0,
            timestamp=timestamp,
            precipitation=meteo_output.precipitation,
            air_temperature=meteo_output.air_temperature,
        )
    )

    hydro_output = hydro_model.step(
        HydroInput(
            domain=small_domain,
            step=0,
            timestamp=timestamp,
            precipitation=meteo_output.precipitation,
            pet=energy_output.pet,
        )
    )

    routing_output = routing_model.step(
        routing_input=RegulatedRoutingInput(
            domain=small_domain,
            step=0,
            timestamp=timestamp,
            surface_runoff=hydro_output.surface_runoff,
            subsurface_runoff=hydro_output.subsurface_runoff,
            pet=energy_output.pet,
        )
    )

    assert meteo_output.precipitation.shape == small_domain.shape
    assert energy_output.pet.shape == small_domain.shape
    assert hydro_output.soil_moisture.shape == small_domain.shape
    assert routing_output.channel_flow_m3s.shape == small_domain.shape
    assert routing_output.outlet_discharge_m3s >= 0.0
