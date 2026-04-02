from __future__ import annotations

import numpy as np
import pytest

from simulator.meteo.latent_state import StormEnvironmentInput
from simulator.meteo.regimes import MeteorologicalRegime
from simulator.meteo.storm_birth import (
    StormBirthConfig,
    compute_expected_births,
    sample_birth_count,
    spawn_storm,
    spawn_storms,
)


def _make_env(
    *,
    trigger: float,
    organization: float,
    moisture: float,
) -> StormEnvironmentInput:
    return StormEnvironmentInput(
        regime=MeteorologicalRegime.CONVECTIVE,
        storm_trigger_factor=trigger,
        storm_organization_factor=organization,
        moisture_availability=moisture,
        advection_u_mps=4.0,
        advection_v_mps=1.0,
        background_temperature_c=20.0,
        cloudiness_index=0.7,
    )


def test_compute_expected_births_increases_with_favorable_environment() -> None:
    config = StormBirthConfig(expected_births_per_step=1.0)

    low_env = _make_env(trigger=0.1, organization=0.1, moisture=0.1)
    high_env = _make_env(trigger=0.9, organization=0.8, moisture=0.9)

    low_births = compute_expected_births(low_env, config)
    high_births = compute_expected_births(high_env, config)

    assert low_births >= 0.0
    assert high_births > low_births


def test_sample_birth_count_is_zero_when_baseline_rate_is_zero() -> None:
    rng = np.random.default_rng(123)
    env = _make_env(trigger=1.0, organization=1.0, moisture=1.0)
    config = StormBirthConfig(expected_births_per_step=0.0)

    assert sample_birth_count(rng, env, config) == 0


def test_spawn_storm_returns_physically_valid_attributes(domain) -> None:
    rng = np.random.default_rng(42)
    env = _make_env(trigger=0.8, organization=0.7, moisture=0.6)
    config = StormBirthConfig()

    storm = spawn_storm(
        rng,
        storm_id=7,
        domain=domain,
        env=env,
        config=config,
    )

    grid = domain.spatial.grid
    margin_x_m = config.birth_margin_cells * grid.dx
    margin_y_m = config.birth_margin_cells * grid.dy

    assert storm.storm_id == 7
    assert (grid.x0 - margin_x_m) <= storm.center_x_m <= (grid.x0 + grid.nx * grid.dx + margin_x_m)
    assert (grid.y0 - margin_y_m) <= storm.center_y_m <= (grid.y0 + grid.ny * grid.dy + margin_y_m)
    assert storm.semi_major_axis_m > 0.0
    assert storm.semi_minor_axis_m > 0.0
    assert storm.semi_major_axis_m >= storm.semi_minor_axis_m
    assert storm.peak_intensity_mmph > 0.0
    assert storm.duration_steps >= 1
    assert storm.age_steps == 0


def test_spawn_storms_is_reproducible_with_same_seed(domain) -> None:
    env = _make_env(trigger=0.9, organization=0.8, moisture=0.8)
    config = StormBirthConfig(expected_births_per_step=2.0, max_new_storms_per_step=10)

    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)

    storms1 = spawn_storms(
        rng1,
        next_storm_id=10,
        domain=domain,
        env=env,
        config=config,
    )
    storms2 = spawn_storms(
        rng2,
        next_storm_id=10,
        domain=domain,
        env=env,
        config=config,
    )

    assert len(storms1) == len(storms2)

    for storm1, storm2 in zip(storms1, storms2):
        assert storm1.storm_id == storm2.storm_id
        assert storm1.center_x_m == pytest.approx(storm2.center_x_m)
        assert storm1.center_y_m == pytest.approx(storm2.center_y_m)
        assert storm1.velocity_u_mps == pytest.approx(storm2.velocity_u_mps)
        assert storm1.velocity_v_mps == pytest.approx(storm2.velocity_v_mps)
        assert storm1.semi_major_axis_m == pytest.approx(storm2.semi_major_axis_m)
        assert storm1.semi_minor_axis_m == pytest.approx(storm2.semi_minor_axis_m)
        assert storm1.orientation_deg == pytest.approx(storm2.orientation_deg)
        assert storm1.peak_intensity_mmph == pytest.approx(storm2.peak_intensity_mmph)
        assert storm1.duration_steps == storm2.duration_steps
        assert storm1.age_steps == storm2.age_steps
