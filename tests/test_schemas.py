import pytest

from simulator.config.schemas import ReservoirConfig, SimulationConfig


def test_simulation_config_valid():
    config = SimulationConfig(
        grid={
            "nx": 64,
            "ny": 64,
            "dx": 1000.0,
            "dy": 1000.0,
        },
        time={
            "start": "2026-01-01T00:00:00",
            "dt_seconds": 3600,
            "n_steps": 24,
            "calendar_type": "monthly",
        },
        reservoirs=[
            {
                "name": "reservoir_a",
                "cell_y": 10,
                "cell_x": 12,
                "capacity": 1_000_000.0,
                "initial_storage": 500_000.0,
            }
        ],
        sensors=[
            {
                "name": "rain_gauge_1",
                "sensor_type": "precipitation",
                "cell_y": 8,
                "cell_x": 9,
            }
        ],
    )

    assert config.grid.nx == 64
    assert config.time.dt_seconds == 3600
    assert config.reservoirs[0].name == "reservoir_a"
    assert config.sensors[0].sensor_type == "precipitation"


def test_reservoir_config_rejects_initial_storage_above_capacity():
    with pytest.raises(ValueError):
        ReservoirConfig(
            name="bad_reservoir",
            cell_y=0,
            cell_x=0,
            capacity=100.0,
            initial_storage=150.0,
        )
