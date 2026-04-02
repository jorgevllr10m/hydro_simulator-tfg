from __future__ import annotations

import numpy as np

from simulator.meteo.background_field import (
    build_air_temperature_field,
    build_background_precipitation_field,
)


def test_build_air_temperature_field_returns_uniform_field(domain) -> None:
    field = build_air_temperature_field(domain, background_temperature_c=17.5)

    assert field.shape == domain.shape
    assert np.allclose(field, 17.5)


def test_build_background_precipitation_field_returns_zero_field(domain) -> None:
    field = build_background_precipitation_field(domain)

    assert field.shape == domain.shape
    assert np.allclose(field, 0.0)
