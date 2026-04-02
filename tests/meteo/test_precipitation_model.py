from __future__ import annotations

import numpy as np
import pytest

from simulator.core.contracts import MeteoOutput
from simulator.meteo.latent_state import LatentEnvironmentConfig
from simulator.meteo.precipitation_model import (
    StormPrecipitationConfig,
    StormPrecipitationModel,
)
from simulator.meteo.storm_birth import StormBirthConfig
from simulator.meteo.storm_objects import StormCell


def test_step_returns_valid_meteo_output_and_expected_shapes(
    domain,
    meteo_input_factory,
) -> None:
    config = StormPrecipitationConfig(
        latent_environment=LatentEnvironmentConfig(random_seed=101),
        birth=StormBirthConfig(expected_births_per_step=0.0),
        random_seed=202,
    )
    model = StormPrecipitationModel(config)

    output = model.step(meteo_input_factory(0))

    assert isinstance(output, MeteoOutput)
    assert output.precipitation.shape == domain.shape
    assert output.air_temperature.shape == domain.shape
    assert output.background_precipitation is not None
    assert output.background_precipitation.shape == domain.shape
    assert output.storm_mask is not None
    assert output.storm_mask.shape == domain.shape
    assert np.allclose(output.background_precipitation, 0.0)
    assert np.unique(output.air_temperature).size == 1


def test_reset_reproduces_the_same_sequence(
    meteo_input_factory,
) -> None:
    config = StormPrecipitationConfig(
        latent_environment=LatentEnvironmentConfig(random_seed=11),
        birth=StormBirthConfig(expected_births_per_step=1.5, max_new_storms_per_step=5),
        random_seed=22,
    )
    model = StormPrecipitationModel(config)

    outputs_first_run = [model.step(meteo_input_factory(step)) for step in range(3)]

    stored_first_run = [
        (
            output.precipitation.copy(),
            output.air_temperature.copy(),
            output.background_precipitation.copy(),
            output.storm_mask.copy(),
        )
        for output in outputs_first_run
    ]

    model.reset()

    outputs_second_run = [model.step(meteo_input_factory(step)) for step in range(3)]

    for (prec_1, temp_1, back_1, mask_1), output_2 in zip(stored_first_run, outputs_second_run):
        np.testing.assert_allclose(prec_1, output_2.precipitation)
        np.testing.assert_allclose(temp_1, output_2.air_temperature)
        np.testing.assert_allclose(back_1, output_2.background_precipitation)
        np.testing.assert_allclose(mask_1, output_2.storm_mask)


def test_latest_diagnostics_are_available_after_step(
    meteo_input_factory,
) -> None:
    config = StormPrecipitationConfig(
        latent_environment=LatentEnvironmentConfig(random_seed=33),
        birth=StormBirthConfig(expected_births_per_step=0.5),
        random_seed=44,
    )
    model = StormPrecipitationModel(config)

    model.step(meteo_input_factory(0))

    assert model.latest_latent_state is not None
    assert model.latest_latent_state.step == 0

    assert model.latest_diagnostics is not None
    assert model.latest_diagnostics.latent_state.step == 0
    assert model.latest_diagnostics.n_new_storms >= 0
    assert model.latest_diagnostics.n_active_storms >= 0

    assert isinstance(model.active_storms, tuple)
    assert model.active_storm_count >= 0


def test_step_with_forced_spawn_produces_nonzero_precipitation(
    meteo_input_factory,
    monkeypatch,
) -> None:
    import simulator.meteo.precipitation_model as precipitation_model_module

    def fake_spawn_storms(rng, *, next_storm_id, domain, env, config):
        return [
            StormCell(
                storm_id=next_storm_id,
                center_x_m=2_500.0,
                center_y_m=2_500.0,
                velocity_u_mps=0.0,
                velocity_v_mps=0.0,
                semi_major_axis_m=4_000.0,
                semi_minor_axis_m=2_000.0,
                orientation_deg=0.0,
                peak_intensity_mmph=12.0,
                duration_steps=5,
                age_steps=0,
            )
        ]

    monkeypatch.setattr(precipitation_model_module, "spawn_storms", fake_spawn_storms)

    config = StormPrecipitationConfig(
        latent_environment=LatentEnvironmentConfig(random_seed=55),
        birth=StormBirthConfig(expected_births_per_step=0.0),
        random_seed=66,
    )
    model = precipitation_model_module.StormPrecipitationModel(config)

    output = model.step(meteo_input_factory(0))

    assert output.precipitation.max() > 0.0
    assert output.storm_mask is not None
    assert output.storm_mask.max() == pytest.approx(1.0)
    assert output.background_precipitation is not None
    assert np.allclose(output.background_precipitation, 0.0)
