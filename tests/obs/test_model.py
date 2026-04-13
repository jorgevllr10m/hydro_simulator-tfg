from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from simulator.core.contracts import ObservationInput
from simulator.core.time import TimeDefinition
from simulator.core.types import (
    BasinDefinition,
    GridDefinition,
    ReservoirDefinition,
    SensorDefinition,
    SimulationDomain,
    SpatialDomain,
)
from simulator.obs.model import (
    DISCHARGE_SENSOR_TYPE,
    PRECIPITATION_SENSOR_TYPE,
    RESERVOIR_STORAGE_SENSOR_TYPE,
    DischargeObservationConfig,
    ObservationConfig,
    ObservationModel,
    ObservationQualityFlag,
    PrecipitationObservationConfig,
    ReservoirStorageObservationConfig,
)


def _build_domain(
    *,
    sensors: tuple[SensorDefinition, ...] = (),
    reservoirs: tuple[ReservoirDefinition, ...] = (),
    shape: tuple[int, int] = (3, 4),
) -> SimulationDomain:
    """Build a minimal valid simulation domain for observation tests."""
    ny, nx = shape

    grid = GridDefinition(
        nx=nx,
        ny=ny,
        dx=1000.0,
        dy=1000.0,
        x0=0.0,
        y0=0.0,
    )
    basin = BasinDefinition(mask=np.ones(shape, dtype=bool))
    spatial = SpatialDomain(grid=grid, basin=basin)

    time = TimeDefinition(
        start=datetime(2026, 1, 1, 0, 0, 0),
        dt_seconds=3600,
        n_steps=4,
        calendar_type="monthly",
    )

    return SimulationDomain(
        spatial=spatial,
        time=time,
        reservoirs=reservoirs,
        sensors=sensors,
    )


def _build_observation_input(
    *,
    domain: SimulationDomain,
    precipitation: np.ndarray | None = None,
    channel_flow: np.ndarray | None = None,
    reservoir_storage: np.ndarray | None = None,
    step: int = 0,
) -> ObservationInput:
    """Build a valid ObservationInput with convenient defaults."""
    if precipitation is None:
        precipitation = np.zeros(domain.shape, dtype=float)

    if channel_flow is None:
        channel_flow = np.zeros(domain.shape, dtype=float)

    return ObservationInput(
        domain=domain,
        step=step,
        timestamp=domain.time.timestamps[step],
        precipitation=np.asarray(precipitation, dtype=float),
        channel_flow=np.asarray(channel_flow, dtype=float),
        reservoir_storage=None if reservoir_storage is None else np.asarray(reservoir_storage, dtype=float),
    )


def test_observation_model_without_sensors_returns_empty_arrays() -> None:
    """When the domain has no sensors, all observation arrays should be empty."""
    domain = _build_domain()
    observation_input = _build_observation_input(domain=domain)

    model = ObservationModel(ObservationConfig())
    output = model.step(observation_input)

    assert output.obs_precipitation is not None
    assert output.obs_discharge is not None
    assert output.obs_storage is not None
    assert output.obs_mask is not None
    assert output.obs_quality_flag is not None

    assert output.obs_precipitation.shape == (0,)
    assert output.obs_discharge.shape == (0,)
    assert output.obs_storage.shape == (0,)
    assert output.obs_mask.shape == (0,)
    assert output.obs_quality_flag.shape == (0,)

    diagnostics = model.latest_diagnostics
    assert diagnostics is not None
    assert diagnostics.n_sensors == 0
    assert diagnostics.n_available == 0
    assert diagnostics.n_missing == 0
    assert diagnostics.n_censored == 0


def test_precipitation_sensor_reads_correct_cell_without_noise() -> None:
    """A precipitation sensor should read the exact truth value when noise/missing are disabled."""
    sensor = SensorDefinition(
        name="rain_gauge_1",
        sensor_type=PRECIPITATION_SENSOR_TYPE,
        cell_y=1,
        cell_x=2,
    )
    domain = _build_domain(sensors=(sensor,))

    precipitation = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.5, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ],
        dtype=float,
    )
    observation_input = _build_observation_input(
        domain=domain,
        precipitation=precipitation,
    )

    config = ObservationConfig(
        precipitation=PrecipitationObservationConfig(
            enabled=True,
            noise_std_mm_dt=0.0,
            missing_probability=0.0,
            detection_threshold_mm_dt=0.0,
            censor_below_threshold=False,
        )
    )

    model = ObservationModel(config)
    output = model.step(observation_input)

    assert output.obs_precipitation is not None
    assert output.obs_discharge is not None
    assert output.obs_storage is not None
    assert output.obs_mask is not None
    assert output.obs_quality_flag is not None

    assert output.obs_precipitation.shape == (1,)
    assert output.obs_precipitation[0] == pytest.approx(6.5)

    assert np.isnan(output.obs_discharge[0])
    assert np.isnan(output.obs_storage[0])

    assert bool(output.obs_mask[0]) is True
    assert int(output.obs_quality_flag[0]) == int(ObservationQualityFlag.NOMINAL)


def test_discharge_sensor_reads_correct_cell_without_noise() -> None:
    """A discharge sensor should read channel_flow at its cell when noise/missing are disabled."""
    sensor = SensorDefinition(
        name="flow_sensor_1",
        sensor_type=DISCHARGE_SENSOR_TYPE,
        cell_y=2,
        cell_x=1,
    )
    domain = _build_domain(sensors=(sensor,))

    channel_flow = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 12.5, 11.0, 10.0],
        ],
        dtype=float,
    )
    observation_input = _build_observation_input(
        domain=domain,
        channel_flow=channel_flow,
    )

    config = ObservationConfig(
        discharge=DischargeObservationConfig(
            enabled=True,
            relative_noise_std=0.0,
            missing_probability=0.0,
            detection_threshold_m3s=0.0,
            censor_below_threshold=False,
        )
    )

    model = ObservationModel(config)
    output = model.step(observation_input)

    assert output.obs_discharge is not None
    assert output.obs_precipitation is not None
    assert output.obs_storage is not None
    assert output.obs_mask is not None
    assert output.obs_quality_flag is not None

    assert output.obs_discharge.shape == (1,)
    assert output.obs_discharge[0] == pytest.approx(12.5)

    assert np.isnan(output.obs_precipitation[0])
    assert np.isnan(output.obs_storage[0])

    assert bool(output.obs_mask[0]) is True
    assert int(output.obs_quality_flag[0]) == int(ObservationQualityFlag.NOMINAL)


def test_storage_sensor_reads_correct_reservoir_without_noise() -> None:
    """A reservoir-storage sensor should read the matching reservoir storage value."""
    reservoir = ReservoirDefinition(
        name="upper_reservoir",
        cell_y=1,
        cell_x=1,
        capacity=3_000_000.0,
        initial_storage=1_500_000.0,
    )
    sensor = SensorDefinition(
        name="storage_sensor_1",
        sensor_type=RESERVOIR_STORAGE_SENSOR_TYPE,
        cell_y=1,
        cell_x=1,
    )
    domain = _build_domain(
        sensors=(sensor,),
        reservoirs=(reservoir,),
    )

    observation_input = _build_observation_input(
        domain=domain,
        reservoir_storage=np.array([2_345_678.0], dtype=float),
    )

    config = ObservationConfig(
        reservoir_storage=ReservoirStorageObservationConfig(
            enabled=True,
            noise_std_m3=0.0,
            missing_probability=0.0,
        )
    )

    model = ObservationModel(config)
    output = model.step(observation_input)

    assert output.obs_storage is not None
    assert output.obs_precipitation is not None
    assert output.obs_discharge is not None
    assert output.obs_mask is not None
    assert output.obs_quality_flag is not None

    assert output.obs_storage.shape == (1,)
    assert output.obs_storage[0] == pytest.approx(2_345_678.0)

    assert np.isnan(output.obs_precipitation[0])
    assert np.isnan(output.obs_discharge[0])

    assert bool(output.obs_mask[0]) is True
    assert int(output.obs_quality_flag[0]) == int(ObservationQualityFlag.NOMINAL)


def test_missing_probability_one_produces_missing() -> None:
    """If missing_probability is 1, the sensor should always produce a missing observation."""
    sensor = SensorDefinition(
        name="rain_gauge_missing",
        sensor_type=PRECIPITATION_SENSOR_TYPE,
        cell_y=0,
        cell_x=0,
    )
    domain = _build_domain(sensors=(sensor,))
    observation_input = _build_observation_input(
        domain=domain,
        precipitation=np.array(
            [
                [4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )

    config = ObservationConfig(
        precipitation=PrecipitationObservationConfig(
            enabled=True,
            noise_std_mm_dt=0.0,
            missing_probability=1.0,
            detection_threshold_mm_dt=0.0,
            censor_below_threshold=False,
        )
    )

    model = ObservationModel(config)
    output = model.step(observation_input)

    assert output.obs_precipitation is not None
    assert output.obs_mask is not None
    assert output.obs_quality_flag is not None

    assert np.isnan(output.obs_precipitation[0])
    assert bool(output.obs_mask[0]) is False
    assert int(output.obs_quality_flag[0]) == int(ObservationQualityFlag.MISSING)

    diagnostics = model.latest_diagnostics
    assert diagnostics is not None
    assert diagnostics.n_available == 0
    assert diagnostics.n_missing == 1
    assert diagnostics.n_censored == 0


def test_precipitation_censoring_applies_threshold_and_flag() -> None:
    """If censoring is enabled and the observed value is below threshold, the threshold should be reported."""
    sensor = SensorDefinition(
        name="rain_gauge_censored",
        sensor_type=PRECIPITATION_SENSOR_TYPE,
        cell_y=0,
        cell_x=1,
    )
    domain = _build_domain(sensors=(sensor,))
    observation_input = _build_observation_input(
        domain=domain,
        precipitation=np.array(
            [
                [0.0, 0.03, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )

    config = ObservationConfig(
        precipitation=PrecipitationObservationConfig(
            enabled=True,
            noise_std_mm_dt=0.0,
            missing_probability=0.0,
            detection_threshold_mm_dt=0.10,
            censor_below_threshold=True,
        )
    )

    model = ObservationModel(config)
    output = model.step(observation_input)

    assert output.obs_precipitation is not None
    assert output.obs_mask is not None
    assert output.obs_quality_flag is not None

    assert output.obs_precipitation[0] == pytest.approx(0.10)
    assert bool(output.obs_mask[0]) is True
    assert int(output.obs_quality_flag[0]) == int(ObservationQualityFlag.CENSORED)

    diagnostics = model.latest_diagnostics
    assert diagnostics is not None
    assert diagnostics.n_available == 1
    assert diagnostics.n_missing == 0
    assert diagnostics.n_censored == 1


def test_relative_noise_scales_with_signal() -> None:
    """Relative noise should scale the truth multiplicatively, so the same random draw affects larger flows more."""

    # We call the helper directly to isolate the relative-noise behavior itself.
    model = ObservationModel(
        ObservationConfig(
            random_seed=2026,
            discharge=DischargeObservationConfig(
                enabled=True,
                relative_noise_std=0.10,
                missing_probability=0.0,
                detection_threshold_m3s=0.0,
                censor_below_threshold=False,
            ),
        )
    )

    small_truth = 2.0
    large_truth = 20.0

    model.reset()
    observed_small = model._sample_relative_noise(
        truth=small_truth,
        relative_noise_std=0.10,
    )

    model.reset()
    observed_large = model._sample_relative_noise(
        truth=large_truth,
        relative_noise_std=0.10,
    )

    relative_factor_small = observed_small / small_truth
    relative_factor_large = observed_large / large_truth

    assert relative_factor_small == pytest.approx(relative_factor_large)
    assert abs(observed_large - large_truth) > abs(observed_small - small_truth)


def test_storage_sensor_outside_reservoir_cell_raises() -> None:
    """A reservoir-storage sensor must coincide exactly with a reservoir cell."""
    reservoir = ReservoirDefinition(
        name="upper_reservoir",
        cell_y=1,
        cell_x=1,
        capacity=3_000_000.0,
        initial_storage=1_500_000.0,
    )
    sensor = SensorDefinition(
        name="bad_storage_sensor",
        sensor_type=RESERVOIR_STORAGE_SENSOR_TYPE,
        cell_y=0,
        cell_x=0,
    )
    domain = _build_domain(
        sensors=(sensor,),
        reservoirs=(reservoir,),
    )

    observation_input = _build_observation_input(
        domain=domain,
        reservoir_storage=np.array([2_000_000.0], dtype=float),
    )

    model = ObservationModel(ObservationConfig())

    with pytest.raises(ValueError, match="must coincide exactly with a reservoir cell"):
        model.step(observation_input)


def test_unknown_sensor_type_raises() -> None:
    """Unsupported sensor types should fail explicitly."""
    sensor = SensorDefinition(
        name="unknown_sensor",
        sensor_type="soil_moisture",
        cell_y=0,
        cell_x=0,
    )
    domain = _build_domain(sensors=(sensor,))
    observation_input = _build_observation_input(domain=domain)

    model = ObservationModel(ObservationConfig())

    with pytest.raises(ValueError, match="Unsupported sensor_type"):
        model.step(observation_input)


def test_reset_restores_reproducibility() -> None:
    """After reset, the same model and same input should reproduce the same random observations."""
    sensor = SensorDefinition(
        name="rain_gauge_reproducible",
        sensor_type=PRECIPITATION_SENSOR_TYPE,
        cell_y=1,
        cell_x=1,
    )
    domain = _build_domain(sensors=(sensor,))
    observation_input = _build_observation_input(
        domain=domain,
        precipitation=np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )

    config = ObservationConfig(
        random_seed=999,
        precipitation=PrecipitationObservationConfig(
            enabled=True,
            noise_std_mm_dt=0.5,
            missing_probability=0.25,
            detection_threshold_mm_dt=0.1,
            censor_below_threshold=True,
        ),
    )

    model = ObservationModel(config)

    output_first = model.step(observation_input)
    model.reset()
    output_second = model.step(observation_input)

    assert output_first.obs_precipitation is not None
    assert output_second.obs_precipitation is not None
    assert output_first.obs_mask is not None
    assert output_second.obs_mask is not None
    assert output_first.obs_quality_flag is not None
    assert output_second.obs_quality_flag is not None

    np.testing.assert_allclose(
        output_first.obs_precipitation,
        output_second.obs_precipitation,
        equal_nan=True,
    )
    np.testing.assert_array_equal(output_first.obs_mask, output_second.obs_mask)
    np.testing.assert_array_equal(output_first.obs_quality_flag, output_second.obs_quality_flag)
