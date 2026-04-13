from __future__ import annotations

import math

import pytest

from simulator.meteo.advection import AdvectionField


def test_advection_direction_is_normalized() -> None:
    field = AdvectionField(speed_mps=5.0, direction_deg=370.0)
    assert field.direction_deg == pytest.approx(10.0)


def test_advection_components_match_speed_and_direction() -> None:
    field = AdvectionField(speed_mps=10.0, direction_deg=0.0)

    assert field.u_mps == pytest.approx(10.0)
    assert field.v_mps == pytest.approx(0.0, abs=1e-12)


def test_advection_components_for_northward_direction() -> None:
    field = AdvectionField(speed_mps=4.0, direction_deg=90.0)

    assert field.u_mps == pytest.approx(0.0, abs=1e-12)
    assert field.v_mps == pytest.approx(4.0)


def test_from_uv_reconstructs_speed_and_direction() -> None:
    field = AdvectionField.from_uv(u_mps=3.0, v_mps=4.0)

    assert field.speed_mps == pytest.approx(5.0)
    assert field.direction_deg == pytest.approx(math.degrees(math.atan2(4.0, 3.0)) % 360.0)


def test_negative_speed_is_rejected() -> None:
    with pytest.raises(ValueError):
        AdvectionField(speed_mps=-1.0, direction_deg=45.0)
