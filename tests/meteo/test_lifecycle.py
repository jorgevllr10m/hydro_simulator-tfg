from __future__ import annotations

import pytest

from simulator.meteo.lifecycle import (
    StormLifecycleConfig,
    compute_current_axes_m,
    compute_current_intensity_mmph,
    compute_life_factor,
)
from simulator.meteo.storm_objects import StormCell


def test_compute_life_factor_piecewise_behavior() -> None:
    lifecycle = StormLifecycleConfig(
        growth_fraction=0.30,
        mature_fraction=0.40,
        decay_fraction=0.30,
        minimum_size_factor=0.60,
    )

    assert compute_life_factor(0.15, lifecycle) == pytest.approx(0.5)
    assert compute_life_factor(0.50, lifecycle) == pytest.approx(1.0)
    assert compute_life_factor(0.85, lifecycle) == pytest.approx(0.5)
    assert compute_life_factor(1.00, lifecycle) == pytest.approx(0.0)


def test_compute_current_intensity_mmph_uses_life_factor() -> None:
    storm = StormCell(
        storm_id=0,
        center_x_m=0.0,
        center_y_m=0.0,
        velocity_u_mps=0.0,
        velocity_v_mps=0.0,
        semi_major_axis_m=4_000.0,
        semi_minor_axis_m=2_000.0,
        orientation_deg=0.0,
        peak_intensity_mmph=20.0,
        duration_steps=6,
        age_steps=3,
    )
    lifecycle = StormLifecycleConfig(
        growth_fraction=0.30,
        mature_fraction=0.40,
        decay_fraction=0.30,
        minimum_size_factor=0.60,
    )

    assert storm.life_progress == pytest.approx(0.5)
    assert compute_current_intensity_mmph(storm, lifecycle) == pytest.approx(20.0)


def test_compute_current_axes_m_scales_with_life_factor() -> None:
    storm = StormCell(
        storm_id=0,
        center_x_m=0.0,
        center_y_m=0.0,
        velocity_u_mps=0.0,
        velocity_v_mps=0.0,
        semi_major_axis_m=4_000.0,
        semi_minor_axis_m=2_000.0,
        orientation_deg=0.0,
        peak_intensity_mmph=10.0,
        duration_steps=4,
        age_steps=1,
    )
    lifecycle = StormLifecycleConfig(
        growth_fraction=0.50,
        mature_fraction=0.25,
        decay_fraction=0.25,
        minimum_size_factor=0.60,
    )

    major_axis_m, minor_axis_m = compute_current_axes_m(storm, lifecycle)

    assert storm.life_progress == pytest.approx(0.25)
    assert major_axis_m == pytest.approx(3_200.0)
    assert minor_axis_m == pytest.approx(1_600.0)
