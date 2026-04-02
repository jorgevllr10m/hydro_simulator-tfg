from __future__ import annotations

import numpy as np
import pytest

from simulator.meteo.lifecycle import StormLifecycleConfig
from simulator.meteo.render import (
    StormRenderConfig,
    convert_precipitation_rate_to_step_depth,
    render_storm_mmph,
    render_storms_mmph,
)
from simulator.meteo.storm_objects import StormCell


def _centered_storm(*, peak_intensity_mmph: float) -> StormCell:
    return StormCell(
        storm_id=0,
        center_x_m=2_500.0,
        center_y_m=2_500.0,
        velocity_u_mps=0.0,
        velocity_v_mps=0.0,
        semi_major_axis_m=4_000.0,
        semi_minor_axis_m=2_000.0,
        orientation_deg=0.0,
        peak_intensity_mmph=peak_intensity_mmph,
        duration_steps=5,
        age_steps=3,
    )


def test_render_storm_mmph_creates_positive_precipitation(domain) -> None:
    storm = _centered_storm(peak_intensity_mmph=10.0)
    lifecycle = StormLifecycleConfig()
    render_config = StormRenderConfig()

    precipitation_mmph, storm_mask = render_storm_mmph(
        domain=domain,
        storm=storm,
        lifecycle=lifecycle,
        render_config=render_config,
    )

    assert precipitation_mmph.shape == domain.shape
    assert storm_mask.shape == domain.shape
    assert precipitation_mmph.max() > 0.0
    assert precipitation_mmph[2, 2] == pytest.approx(10.0)
    assert storm_mask[2, 2] == pytest.approx(1.0)


def test_render_storm_mmph_outside_domain_returns_zero_fields(domain) -> None:
    storm = StormCell(
        storm_id=1,
        center_x_m=50_000.0,
        center_y_m=50_000.0,
        velocity_u_mps=0.0,
        velocity_v_mps=0.0,
        semi_major_axis_m=4_000.0,
        semi_minor_axis_m=2_000.0,
        orientation_deg=0.0,
        peak_intensity_mmph=10.0,
        duration_steps=5,
        age_steps=3,
    )

    precipitation_mmph, storm_mask = render_storm_mmph(
        domain=domain,
        storm=storm,
        lifecycle=StormLifecycleConfig(),
        render_config=StormRenderConfig(),
    )

    assert np.allclose(precipitation_mmph, 0.0)
    assert np.allclose(storm_mask, 0.0)


def test_convert_precipitation_rate_to_step_depth() -> None:
    precipitation_mmph = np.array([[12.0]], dtype=float)

    precipitation_mm_dt = convert_precipitation_rate_to_step_depth(
        precipitation_mmph,
        dt_seconds=1_800,
    )

    assert precipitation_mm_dt.shape == (1, 1)
    assert precipitation_mm_dt[0, 0] == pytest.approx(6.0)


def test_render_storms_mmph_combines_contributions(domain) -> None:
    storm_1 = _centered_storm(peak_intensity_mmph=10.0)
    storm_2 = _centered_storm(peak_intensity_mmph=5.0)

    precipitation_mmph, storm_mask = render_storms_mmph(
        domain=domain,
        storms=[storm_1, storm_2],
        lifecycle=StormLifecycleConfig(),
        render_config=StormRenderConfig(),
    )

    assert precipitation_mmph.shape == domain.shape
    assert storm_mask.shape == domain.shape
    assert precipitation_mmph[2, 2] == pytest.approx(15.0)
    assert storm_mask[2, 2] == pytest.approx(1.0)
