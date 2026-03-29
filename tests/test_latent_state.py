from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from simulator.meteo.latent_state import (
    LatentEnvironmentConfig,
    LatentEnvironmentModel,
    LatentEnvironmentState,
    MoistureScenario,
    StormEnvironmentInput,
    ThermalScenario,
)
from simulator.meteo.regimes import MeteorologicalRegime


def build_default_model(
    *,
    seed: int = 1234,
    thermal_scenario: ThermalScenario = ThermalScenario.NORMAL,
    moisture_scenario: MoistureScenario = MoistureScenario.NORMAL,
) -> LatentEnvironmentModel:
    config = LatentEnvironmentConfig(
        random_seed=seed,
        thermal_scenario=thermal_scenario,
        moisture_scenario=moisture_scenario,
    )
    return LatentEnvironmentModel(config)


def test_latent_environment_state_rejects_invalid_step() -> None:
    model = build_default_model()
    state = model.next_state(step=0, timestamp=datetime(2026, 1, 15))

    with pytest.raises(ValueError):
        LatentEnvironmentState(
            step=-1,
            timestamp=state.timestamp,
            regime=state.regime,
            background_temperature_c=state.background_temperature_c,
            advection=state.advection,
            antecedent_wetness_index=state.antecedent_wetness_index,
            cloudiness_index=state.cloudiness_index,
            convective_potential_index=state.convective_potential_index,
            seasonality_factor=state.seasonality_factor,
            scenario_moisture_factor=state.scenario_moisture_factor,
        )


def test_same_seed_produces_same_sequence() -> None:
    timestamps = [
        datetime(2026, 1, 15),
        datetime(2026, 1, 16),
        datetime(2026, 1, 17),
    ]

    model_a = build_default_model(seed=2024)
    model_b = build_default_model(seed=2024)

    prev_a = None
    prev_b = None
    states_a = []
    states_b = []

    for step, ts in enumerate(timestamps):
        prev_a = model_a.next_state(step=step, timestamp=ts, previous_state=prev_a)
        prev_b = model_b.next_state(step=step, timestamp=ts, previous_state=prev_b)
        states_a.append(prev_a)
        states_b.append(prev_b)

    for state_a, state_b in zip(states_a, states_b, strict=True):
        assert state_a.regime == state_b.regime
        assert state_a.background_temperature_c == pytest.approx(state_b.background_temperature_c)
        assert state_a.advection_speed_mps == pytest.approx(state_b.advection_speed_mps)
        assert state_a.advection_direction_deg == pytest.approx(state_b.advection_direction_deg)
        assert state_a.antecedent_wetness_index == pytest.approx(state_b.antecedent_wetness_index)
        assert state_a.cloudiness_index == pytest.approx(state_b.cloudiness_index)
        assert state_a.convective_potential_index == pytest.approx(state_b.convective_potential_index)
        assert state_a.seasonality_factor == pytest.approx(state_b.seasonality_factor)
        assert state_a.scenario_moisture_factor == pytest.approx(state_b.scenario_moisture_factor)


def test_different_seeds_can_produce_different_sequences() -> None:
    timestamp = datetime(2026, 7, 15)

    state_a = build_default_model(seed=1111).next_state(step=0, timestamp=timestamp)
    state_b = build_default_model(seed=2222).next_state(step=0, timestamp=timestamp)

    different = (
        state_a.background_temperature_c != state_b.background_temperature_c
        or state_a.advection_speed_mps != state_b.advection_speed_mps
        or state_a.advection_direction_deg != state_b.advection_direction_deg
        or state_a.regime != state_b.regime
    )
    assert different


def test_indices_stay_within_bounds_over_multiple_steps() -> None:
    model = build_default_model(seed=2026)
    prev = None

    for step in range(30):
        timestamp = datetime(2026, 1, 1) + timedelta(days=step)
        state = model.next_state(step=step, timestamp=timestamp, previous_state=prev)

        assert 0.0 <= state.antecedent_wetness_index <= 1.0
        assert 0.0 <= state.cloudiness_index <= 1.0
        assert 0.0 <= state.convective_potential_index <= 1.0
        assert 0.0 <= state.seasonality_factor <= 1.0

        prev = state


def test_regime_is_always_valid() -> None:
    model = build_default_model(seed=321)
    prev = None

    for step in range(20):
        state = model.next_state(
            step=step,
            timestamp=datetime(2026, 3, min(step + 1, 28)),
            previous_state=prev,
        )
        assert isinstance(state.regime, MeteorologicalRegime)
        prev = state


def test_warm_scenario_increases_temperature_against_cold_for_same_seed() -> None:
    timestamp = datetime(2026, 7, 15)

    warm_model = build_default_model(seed=77, thermal_scenario=ThermalScenario.WARM)
    cold_model = build_default_model(seed=77, thermal_scenario=ThermalScenario.COLD)

    warm_state = warm_model.next_state(step=0, timestamp=timestamp)
    cold_state = cold_model.next_state(step=0, timestamp=timestamp)

    assert warm_state.background_temperature_c > cold_state.background_temperature_c


def test_wet_scenario_increases_moisture_factor() -> None:
    timestamp = datetime(2026, 4, 10)

    dry_model = build_default_model(seed=55, moisture_scenario=MoistureScenario.DRY)
    wet_model = build_default_model(seed=55, moisture_scenario=MoistureScenario.WET)

    dry_state = dry_model.next_state(step=0, timestamp=timestamp)
    wet_state = wet_model.next_state(step=0, timestamp=timestamp)

    assert wet_state.scenario_moisture_factor > dry_state.scenario_moisture_factor


def test_storm_environment_mapping_is_bounded() -> None:
    model = build_default_model(seed=100)
    state = model.next_state(step=0, timestamp=datetime(2026, 6, 20))
    storm_input = model.build_storm_environment_input(state)

    assert isinstance(storm_input, StormEnvironmentInput)
    assert 0.0 <= storm_input.storm_trigger_factor <= 1.0
    assert 0.0 <= storm_input.storm_organization_factor <= 1.0
    assert 0.0 <= storm_input.moisture_availability <= 1.0
    assert 0.0 <= storm_input.cloudiness_index <= 1.0


def test_storm_environment_mapping_keeps_regime_and_advection() -> None:
    model = build_default_model(seed=42)
    state = model.next_state(step=0, timestamp=datetime(2026, 9, 10))
    storm_input = model.build_storm_environment_input(state)

    assert storm_input.regime == state.regime
    assert storm_input.advection_u_mps == pytest.approx(state.advection_u_mps)
    assert storm_input.advection_v_mps == pytest.approx(state.advection_v_mps)
    assert storm_input.background_temperature_c == pytest.approx(state.background_temperature_c)
