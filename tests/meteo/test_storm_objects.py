from __future__ import annotations

import pytest

from simulator.meteo.storm_objects import StormCell


def test_storm_cell_creation_and_derived_properties() -> None:
    storm = StormCell(
        storm_id=1,
        center_x_m=1_000.0,
        center_y_m=2_000.0,
        velocity_u_mps=3.0,
        velocity_v_mps=-1.0,
        semi_major_axis_m=4_000.0,
        semi_minor_axis_m=2_000.0,
        orientation_deg=200.0,
        peak_intensity_mmph=12.0,
        duration_steps=5,
    )

    assert storm.storm_id == 1
    assert storm.orientation_deg == pytest.approx(20.0)
    assert storm.orientation_rad == pytest.approx(20.0 * 3.141592653589793 / 180.0)
    assert storm.center == (1_000.0, 2_000.0)
    assert storm.velocity == (3.0, -1.0)
    assert storm.speed_mps == pytest.approx((3.0**2 + (-1.0) ** 2) ** 0.5)
    assert storm.life_progress == pytest.approx(0.0)
    assert storm.remaining_steps == 5
    assert storm.is_alive is True


def test_storm_cell_advance_updates_position_and_age() -> None:
    storm = StormCell(
        storm_id=0,
        center_x_m=1_000.0,
        center_y_m=2_000.0,
        velocity_u_mps=3.0,
        velocity_v_mps=-1.0,
        semi_major_axis_m=4_000.0,
        semi_minor_axis_m=2_000.0,
        orientation_deg=30.0,
        peak_intensity_mmph=10.0,
        duration_steps=5,
    )

    storm.advance(dt_seconds=3_600)

    assert storm.center_x_m == pytest.approx(11_800.0)
    assert storm.center_y_m == pytest.approx(-1_600.0)
    assert storm.age_steps == 1
    assert storm.life_progress == pytest.approx(0.2)
    assert storm.remaining_steps == 4
    assert storm.is_alive is True


def test_storm_cell_expire_marks_storm_as_dead() -> None:
    storm = StormCell(
        storm_id=2,
        center_x_m=0.0,
        center_y_m=0.0,
        velocity_u_mps=0.0,
        velocity_v_mps=0.0,
        semi_major_axis_m=2_000.0,
        semi_minor_axis_m=1_000.0,
        orientation_deg=0.0,
        peak_intensity_mmph=8.0,
        duration_steps=4,
    )

    storm.expire()

    assert storm.age_steps == 4
    assert storm.life_progress == pytest.approx(1.0)
    assert storm.remaining_steps == 0
    assert storm.is_alive is False
