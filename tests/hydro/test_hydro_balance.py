from __future__ import annotations

from datetime import datetime

import numpy as np

from simulator.core.contracts import HydroInput
from simulator.core.time import TimeDefinition
from simulator.core.types import (
    BasinDefinition,
    GridDefinition,
    SimulationDomain,
    SpatialDomain,
)
from simulator.hydro.model import HydroConfig, HydroModel
from simulator.hydro.runoff import RunoffConfig
from simulator.hydro.soil import SoilConfig


def _build_test_domain() -> SimulationDomain:
    """Build a minimal fully active simulation domain for hydrology tests."""
    grid = GridDefinition(
        nx=4,
        ny=3,
        dx=1_000.0,
        dy=1_000.0,
    )
    basin = BasinDefinition(mask=np.ones(grid.shape, dtype=bool))
    spatial = SpatialDomain(grid=grid, basin=basin)

    time = TimeDefinition(
        start=datetime(2025, 1, 1, 0, 0, 0),
        dt_seconds=3600,
        n_steps=2,
    )

    return SimulationDomain(
        spatial=spatial,
        time=time,
    )


def _build_test_model(domain: SimulationDomain) -> HydroModel:
    """Build a hydrology model with parameters that generate both runoff and drainage."""
    config = HydroConfig(
        soil=SoilConfig(
            capacity_mm=180.0,
            initial_relative=0.80,
            max_infiltration_mm_dt=10.0,
            infiltration_shape_exponent=1.0,
            et_stress_exponent=1.5,
            percolation_rate_mm_dt=3.0,
            percolation_activation_fraction=0.70,
        ),
        runoff=RunoffConfig(
            subsurface_runoff_fraction=1.0,
        ),
    )
    return HydroModel(
        config,
        shape=domain.shape,
    )


def _build_test_forcing(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return precipitation and PET fields for the tests."""
    precipitation = np.array(
        [
            [0.0, 4.0, 12.0, 25.0],
            [2.0, 8.0, 18.0, 30.0],
            [1.0, 6.0, 15.0, 22.0],
        ],
        dtype=float,
    )
    pet = np.full(shape, 2.0, dtype=float)
    return precipitation, pet


def test_hydro_model_partitions_precipitation_into_infiltration_and_surface_runoff() -> None:
    """Per-cell precipitation must be partitioned into infiltration + surface runoff."""
    domain = _build_test_domain()
    model = _build_test_model(domain)

    precipitation, pet = _build_test_forcing(domain.shape)

    hydro_input = HydroInput(
        domain=domain,
        step=0,
        timestamp=domain.time.timestamps[0],
        precipitation=precipitation,
        pet=pet,
        soil_moisture_prev=model.latest_state.soil_moisture_mm,
    )

    output = model.step(hydro_input)

    np.testing.assert_allclose(
        output.infiltration + output.surface_runoff,
        precipitation,
        rtol=0.0,
        atol=1e-9,
    )


def test_hydro_model_closes_soil_water_balance_per_cell() -> None:
    """Per-cell soil bucket must satisfy the local water-balance equation."""
    domain = _build_test_domain()
    model = _build_test_model(domain)

    precipitation, pet = _build_test_forcing(domain.shape)
    soil_moisture_prev = model.latest_state.soil_moisture_mm.copy()

    hydro_input = HydroInput(
        domain=domain,
        step=0,
        timestamp=domain.time.timestamps[0],
        precipitation=precipitation,
        pet=pet,
        soil_moisture_prev=soil_moisture_prev,
    )

    output = model.step(hydro_input)

    assert output.subsurface_runoff is not None

    expected_soil_moisture = soil_moisture_prev + output.infiltration - output.aet - output.subsurface_runoff

    np.testing.assert_allclose(
        output.soil_moisture,
        expected_soil_moisture,
        rtol=0.0,
        atol=1e-9,
    )
