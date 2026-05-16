from __future__ import annotations

import numpy as np
import pytest

from simulator.core.contracts import MeteoInput
from simulator.meteo.background_field import BackgroundFieldConfig
from simulator.meteo.precipitation_model import (
    StormPrecipitationConfig,
    StormPrecipitationModel,
)
from simulator.meteo.storm_birth import StormBirthConfig
from simulator.meteo.storm_objects import StormCell
from simulator.meteo.temperature_field import TemperatureFieldConfig


def _make_storm(**overrides) -> StormCell:
    params = {
        "storm_id": 1,
        "center_x_m": 1000.0,
        "center_y_m": 2000.0,
        "velocity_u_mps": 3.0,
        "velocity_v_mps": -1.0,
        "semi_major_axis_m": 4000.0,
        "semi_minor_axis_m": 2000.0,
        "orientation_deg": 200.0,
        "peak_intensity_mmph": 12.0,
        "duration_steps": 5,
    }
    params.update(overrides)
    return StormCell(**params)


def test_storm_cell_accepts_valid_axes() -> None:
    storm = _make_storm()

    assert storm.semi_major_axis_m == pytest.approx(4000.0)
    assert storm.semi_minor_axis_m == pytest.approx(2000.0)
    assert storm.orientation_deg == pytest.approx(20.0)


def test_storm_cell_rejects_non_positive_minor_axis() -> None:
    with pytest.raises(ValueError):
        _make_storm(semi_minor_axis_m=0.0)


def test_storm_cell_rejects_minor_axis_larger_than_major_axis() -> None:
    with pytest.raises(ValueError):
        _make_storm(
            semi_major_axis_m=2000.0,
            semi_minor_axis_m=3000.0,
        )


def test_storm_birth_config_rejects_zero_band_minor_axis_factor() -> None:
    with pytest.raises(ValueError):
        StormBirthConfig(band_minor_axis_factor=0.0)


def test_meteo_model_step_returns_valid_fields(small_domain) -> None:
    config = StormPrecipitationConfig(
        birth=StormBirthConfig(expected_births_per_step=0.0),
        background=BackgroundFieldConfig(enabled=False),
        temperature=TemperatureFieldConfig(enabled=False),
    )
    model = StormPrecipitationModel(config)

    output = model.step(
        MeteoInput(
            domain=small_domain,
            step=0,
            timestamp=small_domain.time.timestamps[0],
        )
    )

    assert output.precipitation.shape == small_domain.shape
    assert output.air_temperature.shape == small_domain.shape
    assert output.background_precipitation is not None
    assert output.storm_mask is not None
    assert np.all(output.precipitation >= 0.0)
