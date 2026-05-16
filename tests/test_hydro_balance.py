from __future__ import annotations

import numpy as np

from simulator.hydro.soil import SoilConfig, update_soil_bucket


def test_soil_bucket_basic_water_balance() -> None:
    config = SoilConfig(
        capacity_mm=100.0,
        initial_relative=0.5,
        max_infiltration_mm_dt=10.0,
        infiltration_shape_exponent=1.0,
        et_stress_exponent=1.0,
        percolation_rate_mm_dt=2.0,
        percolation_activation_fraction=0.7,
    )

    soil_prev = np.full((2, 2), 50.0, dtype=float)
    precipitation = np.full((2, 2), 5.0, dtype=float)
    pet = np.full((2, 2), 1.0, dtype=float)

    fields = update_soil_bucket(
        soil_moisture_prev_mm=soil_prev,
        precipitation_mm_dt=precipitation,
        pet_mm_dt=pet,
        config=config,
    )

    reconstructed_final = soil_prev + fields.infiltration_mm_dt - fields.aet_mm_dt - fields.percolation_mm_dt

    np.testing.assert_allclose(fields.soil_moisture_mm, reconstructed_final)
    assert np.all(fields.soil_moisture_mm >= 0.0)
    assert np.all(fields.soil_moisture_mm <= config.capacity_mm)
    assert np.all(fields.surface_excess_mm_dt >= 0.0)
