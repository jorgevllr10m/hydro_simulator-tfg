from __future__ import annotations

import pytest

from simulator.config.loader import load_config
from simulator.config.schemas import (
    DomainPresetConfig,
    MasterConfig,
    ReservoirConfig,
    SimulationWindowConfig,
)


def test_master_config_accepts_valid_minimal_configuration() -> None:
    config = MasterConfig(
        run={
            "name": "test_run",
            "output_dir": "outputs/runs",
        },
        simulation={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-02T00:00:00",
            "time_step_hours": 1,
        },
        domain={
            "preset": "tiny",
        },
        scenario={
            "name": "empty",
        },
    )

    assert config.run.name == "test_run"
    assert config.simulation.time_step_hours == 1
    assert config.domain.preset == "tiny"
    assert config.scenario.name == "empty"


def test_simulation_window_rejects_non_divisible_interval() -> None:
    with pytest.raises(ValueError):
        SimulationWindowConfig(
            start_date="2026-01-01T00:00:00",
            end_date="2026-01-01T01:30:00",
            time_step_hours=1,
        )


def test_reservoir_config_rejects_initial_storage_above_capacity() -> None:
    with pytest.raises(ValueError):
        ReservoirConfig(
            name="bad_reservoir",
            cell_y=0,
            cell_x=0,
            capacity=100.0,
            initial_storage=150.0,
        )


def test_domain_preset_rejects_duplicate_sensor_names() -> None:
    with pytest.raises(ValueError):
        DomainPresetConfig(
            grid={
                "nx": 4,
                "ny": 3,
                "dx": 1000.0,
                "dy": 1000.0,
            },
            sensors=[
                {
                    "name": "sensor_a",
                    "sensor_type": "precipitation",
                    "cell_y": 0,
                    "cell_x": 0,
                },
                {
                    "name": "sensor_a",
                    "sensor_type": "discharge",
                    "cell_y": 1,
                    "cell_x": 1,
                },
            ],
        )


def test_load_config_resolves_domain_and_empty_scenario(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    domain_dir = tmp_path / "domain"
    scenario_dir = tmp_path / "scenarios"
    domain_dir.mkdir()
    scenario_dir.mkdir()

    config_path.write_text(
        """
run:
  name: smoke
  output_dir: outputs/runs

simulation:
  start_date: "2026-01-01T00:00:00"
  end_date: "2026-01-01T03:00:00"
  time_step_hours: 1

domain:
  preset: tiny

scenario:
  name: empty
""",
        encoding="utf-8",
    )

    (domain_dir / "tiny.yaml").write_text(
        """
grid:
  nx: 4
  ny: 3
  dx: 1000.0
  dy: 1000.0

reservoirs:
  - name: reservoir_a
    cell_y: 1
    cell_x: 1
    capacity: 1000000.0
    initial_storage: 500000.0

sensors:
  - name: rain_a
    sensor_type: precipitation
    cell_y: 0
    cell_x: 0
""",
        encoding="utf-8",
    )

    (scenario_dir / "empty.yaml").write_text("", encoding="utf-8")

    loaded = load_config(config_path)
    domain = loaded.build_simulation_domain()

    assert loaded.run_name == "smoke"
    assert domain.shape == (3, 4)
    assert domain.n_steps == 3
    assert len(domain.reservoirs) == 1
    assert len(domain.sensors) == 1
