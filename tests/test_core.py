from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from simulator.core.contracts import MeteoInput
from simulator.core.time import TimeDefinition
from simulator.core.types import BasinDefinition, GridDefinition, SpatialDomain


def test_time_definition_generates_expected_timestamps() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    time = TimeDefinition(
        start=start,
        dt_seconds=3600,
        n_steps=3,
    )

    timestamps = time.timestamps

    assert list(timestamps) == [
        start,
        start + timedelta(hours=1),
        start + timedelta(hours=2),
    ]
    assert time.total_duration_seconds == 10_800


def test_grid_and_basin_shapes_must_match() -> None:
    grid = GridDefinition(
        nx=4,
        ny=3,
        dx=1000.0,
        dy=1000.0,
    )
    basin = BasinDefinition(mask=np.ones((2, 4), dtype=bool))

    with pytest.raises(ValueError):
        SpatialDomain(grid=grid, basin=basin)


def test_meteo_input_contract_has_no_previous_state_field() -> None:
    field_names = set(MeteoInput.__dataclass_fields__)

    assert field_names == {"domain", "step", "timestamp"}
