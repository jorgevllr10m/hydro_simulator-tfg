from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from simulator.common.validation import (
    validate_fraction as _validate_fraction,
)
from simulator.common.validation import (
    validate_non_negative_scalar as _validate_non_negative_scalar,
)
from simulator.common.validation import (
    validate_numeric_scalar as _validate_numeric_scalar,
)
from simulator.common.validation import (
    validate_spatial_float_array as _validate_spatial_float_array,
)
from simulator.common.validation import (
    validate_vector_float_array as _validate_vector_float_array,
)
from simulator.core.contracts import ObservationInput, ObservationOutput
from simulator.core.types import SimulationDomain

BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]
VectorFloatArray = NDArray[np.float64]

PRECIPITATION_SENSOR_TYPE = "precipitation"
DISCHARGE_SENSOR_TYPE = "discharge"
RESERVOIR_STORAGE_SENSOR_TYPE = "reservoir_storage"

SUPPORTED_SENSOR_TYPES = {
    PRECIPITATION_SENSOR_TYPE,
    DISCHARGE_SENSOR_TYPE,
    RESERVOIR_STORAGE_SENSOR_TYPE,
}


def _normalize_sensor_type(sensor_type: str) -> str:
    """Normalize a sensor type string for internal matching."""
    if not isinstance(sensor_type, str):
        raise TypeError(f"'sensor_type' must be a string, got {type(sensor_type).__name__}")
    normalized = sensor_type.strip().lower()
    if not normalized:
        raise ValueError("'sensor_type' must be a non-empty string")
    return normalized


class ObservationQualityFlag(IntEnum):
    """Quality/status code of one observed value."""

    MISSING = 0
    """
    No usable observation.
    """

    NOMINAL = 1
    """
    There was normal observation.
    """

    CENSORED = 2
    """
    There was observation, but it was limited by the sensor threshold.
    """


@dataclass(frozen=True)
class PrecipitationObservationConfig:
    """Configuration of synthetic precipitation observations."""

    enabled: bool = True
    """
    Activate or deactivate precipitation observation.
    """

    noise_std_mm_dt: float = 0.15
    """
    Standard deviation of the absolute additive noise in mm/dt.
    """

    missing_probability: float = 0.02
    """
    Probability that the observation is missing.
    """

    detection_threshold_mm_dt: float = 0.05
    """
    Minimum detectable threshold of the sensor.
    """

    censor_below_threshold: bool = True
    """
    If it's enabled and the observed value falls below the threshold, the returned value is the threshold
    """

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"'enabled' must be bool, got {type(self.enabled).__name__}")

        _validate_non_negative_scalar("noise_std_mm_dt", self.noise_std_mm_dt)
        _validate_fraction("missing_probability", self.missing_probability)
        _validate_non_negative_scalar("detection_threshold_mm_dt", self.detection_threshold_mm_dt)

        if not isinstance(self.censor_below_threshold, bool):
            raise TypeError(f"'censor_below_threshold' must be bool, got {type(self.censor_below_threshold).__name__}")


@dataclass(frozen=True)
class DischargeObservationConfig:
    """Configuration of synthetic discharge observations."""

    enabled: bool = True
    """
    Activate or deactivate discharge observation.
    """

    relative_noise_std: float = 0.08
    """
    Standard deviation of the relative noise in [0, 1].
    """

    missing_probability: float = 0.02
    """
    Probability that the observation is missing.
    """
    detection_threshold_m3s: float = 0.10
    """
    Minimum detectable threshold of the sensor.
    """

    censor_below_threshold: bool = True
    """
    If it's enabled and the observed value falls below the threshold, the returned value is the threshold
    """

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"'enabled' must be bool, got {type(self.enabled).__name__}")

        _validate_non_negative_scalar("relative_noise_std", self.relative_noise_std)
        _validate_fraction("missing_probability", self.missing_probability)
        _validate_non_negative_scalar("detection_threshold_m3s", self.detection_threshold_m3s)

        if not isinstance(self.censor_below_threshold, bool):
            raise TypeError(f"'censor_below_threshold' must be bool, got {type(self.censor_below_threshold).__name__}")


@dataclass(frozen=True)
class ReservoirStorageObservationConfig:
    """Configuration of synthetic reservoir-storage observations."""

    enabled: bool = True
    """
    Activate or deactivate precipitation observation.
    """

    noise_std_m3: float = 15_000.0
    """
    Standard deviation of the absolute additive noise in m3.
    """

    missing_probability: float = 0.01
    """
    Probability that the observation is missing.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"'enabled' must be bool, got {type(self.enabled).__name__}")

        _validate_non_negative_scalar("noise_std_m3", self.noise_std_m3)
        _validate_fraction("missing_probability", self.missing_probability)


@dataclass(frozen=True)
class ObservationConfig:
    """Top-level configuration of the synthetic observation module."""

    random_seed: int = 9876

    precipitation: PrecipitationObservationConfig = field(default_factory=PrecipitationObservationConfig)
    discharge: DischargeObservationConfig = field(default_factory=DischargeObservationConfig)
    reservoir_storage: ReservoirStorageObservationConfig = field(default_factory=ReservoirStorageObservationConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.random_seed, int):
            raise TypeError(f"'random_seed' must be int, got {type(self.random_seed).__name__}")

        if not isinstance(self.precipitation, PrecipitationObservationConfig):
            raise TypeError(f"'precipitation' must be a PrecipitationObservationConfig, got {type(self.precipitation).__name__}")

        if not isinstance(self.discharge, DischargeObservationConfig):
            raise TypeError(f"'discharge' must be a DischargeObservationConfig, got {type(self.discharge).__name__}")

        if not isinstance(self.reservoir_storage, ReservoirStorageObservationConfig):
            raise TypeError(
                f"'reservoir_storage' must be a ReservoirStorageObservationConfig, got {type(self.reservoir_storage).__name__}"
            )


@dataclass(frozen=True)
class ObservationStepDiagnostics:
    """Lightweight diagnostics of the latest observation step."""

    step: int
    timestamp: datetime
    n_sensors: int
    n_available: int
    n_missing: int
    n_censored: int

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError(f"'step' must be a non-negative integer, got {self.step!r}")

        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"'timestamp' must be a datetime, got {type(self.timestamp).__name__}")

        integer_fields = {
            "n_sensors": self.n_sensors,
            "n_available": self.n_available,
            "n_missing": self.n_missing,
            "n_censored": self.n_censored,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int):
                raise TypeError(f"'{name}' must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"'{name}' must be >= 0, got {value}")

        if self.n_available + self.n_missing != self.n_sensors:
            raise ValueError(
                "'n_available' + 'n_missing' must equal 'n_sensors', "
                f"got available={self.n_available}, missing={self.n_missing}, sensors={self.n_sensors}"
            )

        if self.n_censored > self.n_available:
            raise ValueError(f"'n_censored' cannot exceed 'n_available', got censored={self.n_censored}, available={self.n_available}")


class ObservationModel:
    """Stateful synthetic observation operator.

    This model converts the simulator truth at sensor locations into observed
    values with optional:
    - additive or relative noise
    - missing values
    - left-censoring / detection thresholds

    Conventions
    -----------
    - each sensor measures exactly one variable
    - outputs are always 1D arrays of length n_sensors
    - non-applicable variables for a sensor are left as NaN
    - `obs_mask[i]` is True when the sensor produced a usable reported value
    - `obs_quality_flag[i]` uses ObservationQualityFlag
    """

    def __init__(self, config: ObservationConfig) -> None:
        if not isinstance(config, ObservationConfig):
            raise TypeError(f"'config' must be an ObservationConfig, got {type(config).__name__}")

        self.config = config
        self._rng = np.random.default_rng(config.random_seed)
        self._latest_diagnostics: ObservationStepDiagnostics | None = None

    @property
    def latest_diagnostics(self) -> ObservationStepDiagnostics | None:
        """Return diagnostics from the most recent completed step."""
        return self._latest_diagnostics

    def reset(self) -> None:
        """Reset the RNG and diagnostics to the initial reproducible state."""
        self._rng = np.random.default_rng(self.config.random_seed)
        self._latest_diagnostics = None

    def _observe_precipitation_sensor(
        self,
        *,
        truth: float,
    ) -> tuple[float, bool, ObservationQualityFlag]:
        """Observe one precipitation sensor."""
        config = self.config.precipitation
        truth = max(0.0, float(truth))

        if not config.enabled:
            return (float("nan"), False, ObservationQualityFlag.MISSING)

        if self._sample_missing(config.missing_probability):
            return (float("nan"), False, ObservationQualityFlag.MISSING)

        # Preserve exact physical zero.
        practical_zero_mm_dt = 1e-6
        if truth <= practical_zero_mm_dt:
            return (0.0, True, ObservationQualityFlag.NOMINAL)

        observed = self._sample_additive_noise(
            truth=truth,
            noise_std=config.noise_std_mm_dt,
        )
        observed = max(0.0, observed)

        observed, censored = self._apply_left_censoring(
            value=observed,
            threshold=config.detection_threshold_mm_dt,
            enabled=config.censor_below_threshold,
        )

        flag = ObservationQualityFlag.CENSORED if censored else ObservationQualityFlag.NOMINAL
        return (observed, True, flag)

    def _observe_discharge_sensor(
        self,
        *,
        truth: float,
    ) -> tuple[float, bool, ObservationQualityFlag]:
        """Observe one discharge sensor."""
        config = self.config.discharge
        truth = max(0.0, float(truth))

        if not config.enabled:
            return (float("nan"), False, ObservationQualityFlag.MISSING)

        if self._sample_missing(config.missing_probability):
            return (float("nan"), False, ObservationQualityFlag.MISSING)

        # Preserve exact physical zero.
        practical_zero_m3s = 1e-3
        if truth <= practical_zero_m3s:
            return (0.0, True, ObservationQualityFlag.NOMINAL)

        observed = self._sample_relative_noise(
            truth=truth,
            relative_noise_std=config.relative_noise_std,
        )
        observed = max(0.0, observed)

        observed, censored = self._apply_left_censoring(
            value=observed,
            threshold=config.detection_threshold_m3s,
            enabled=config.censor_below_threshold,
        )

        flag = ObservationQualityFlag.CENSORED if censored else ObservationQualityFlag.NOMINAL
        return (observed, True, flag)

    def _observe_reservoir_storage_sensor(
        self,
        *,
        domain: SimulationDomain,
        sensor_index: int,
        observation_input: ObservationInput,
    ) -> tuple[float, bool, ObservationQualityFlag]:
        """Observe one reservoir-storage sensor."""
        config = self.config.reservoir_storage

        if not config.enabled:
            return (float("nan"), False, ObservationQualityFlag.MISSING)

        if observation_input.reservoir_storage is None:
            raise ValueError("A 'reservoir_storage' sensor requires 'observation_input.reservoir_storage', but it was None.")

        sensor = domain.sensors[sensor_index]
        reservoir_id = self._resolve_reservoir_id_for_sensor(
            domain=domain,
            sensor_index=sensor_index,
        )
        reservoir = domain.reservoirs[reservoir_id]

        truth = float(observation_input.reservoir_storage[reservoir_id])

        if not np.isfinite(truth):
            return (float("nan"), False, ObservationQualityFlag.MISSING)

        truth = float(np.clip(truth, 0.0, float(reservoir.capacity)))

        if self._sample_missing(config.missing_probability):
            return (float("nan"), False, ObservationQualityFlag.MISSING)

        # Preserve exact physical zero.
        if truth <= 0.0:
            return (0.0, True, ObservationQualityFlag.NOMINAL)

        observed = self._sample_additive_noise(
            truth=truth,
            noise_std=config.noise_std_m3,
        )
        observed = float(np.clip(observed, 0.0, float(reservoir.capacity)))

        if not np.isfinite(observed):
            raise ValueError(f"Observed reservoir storage for sensor '{sensor.name}' became non-finite: {observed!r}")

        return (observed, True, ObservationQualityFlag.NOMINAL)

    def _resolve_reservoir_id_for_sensor(
        self,
        *,
        domain: SimulationDomain,
        sensor_index: int,
    ) -> int:
        """Return the reservoir index matched exactly by one storage sensor."""
        sensor = domain.sensors[sensor_index]

        matches = [
            reservoir_id
            for reservoir_id, reservoir in enumerate(domain.reservoirs)
            if reservoir.cell_y == sensor.cell_y and reservoir.cell_x == sensor.cell_x
        ]

        if not matches:
            raise ValueError(
                "A 'reservoir_storage' sensor must coincide exactly with a reservoir cell. "
                f"Sensor '{sensor.name}' is at {(sensor.cell_y, sensor.cell_x)} and matches no reservoir."
            )

        if len(matches) > 1:
            raise ValueError(
                f"Sensor '{sensor.name}' at {(sensor.cell_y, sensor.cell_x)} matches multiple reservoirs, "
                "which is not allowed in this MVP."
            )

        return int(matches[0])

    def _sample_additive_noise(
        self,
        *,
        truth: float,
        noise_std: float,
    ) -> float:
        """Return truth plus additive absolute noise."""
        truth = _validate_numeric_scalar("truth", truth)
        noise_std = _validate_non_negative_scalar("noise_std", noise_std)

        if noise_std == 0.0:
            return truth

        return float(truth + self._rng.normal(0.0, noise_std))

    def _sample_relative_noise(
        self,
        *,
        truth: float,
        relative_noise_std: float,
    ) -> float:
        """Return truth perturbed by multiplicative relative noise."""
        truth = _validate_numeric_scalar("truth", truth)
        relative_noise_std = _validate_non_negative_scalar("relative_noise_std", relative_noise_std)

        if relative_noise_std == 0.0:
            return truth

        relative_error = float(self._rng.normal(0.0, relative_noise_std))
        return float(truth * (1.0 + relative_error))

    def _sample_missing(
        self,
        missing_probability: float,
    ) -> bool:
        """Return whether the current observation becomes missing."""
        missing_probability = _validate_fraction("missing_probability", missing_probability)

        if missing_probability <= 0.0:
            return False
        if missing_probability >= 1.0:
            return True

        return bool(self._rng.random() < missing_probability)

    @staticmethod
    def _apply_left_censoring(
        *,
        value: float,
        threshold: float,
        enabled: bool,
    ) -> tuple[float, bool]:
        """Apply simple left-censoring below a detection threshold."""
        value = _validate_numeric_scalar("value", value)
        threshold = _validate_non_negative_scalar("threshold", threshold)

        if not isinstance(enabled, bool):
            raise TypeError(f"'enabled' must be bool, got {type(enabled).__name__}")

        if not enabled:
            return (value, False)

        if 0.0 < value < threshold:
            return (threshold, True)

        return (value, False)

    def step(
        self,
        observation_input: ObservationInput,
    ) -> ObservationOutput:
        """Generate synthetic observations for one simulation step."""
        if not isinstance(observation_input, ObservationInput):
            raise TypeError(f"'observation_input' must be an ObservationInput, got {type(observation_input).__name__}")

        domain = observation_input.domain
        if not isinstance(domain, SimulationDomain):
            raise TypeError(f"'observation_input.domain' must be a SimulationDomain, got {type(domain).__name__}")

        if not isinstance(observation_input.step, int) or observation_input.step < 0:
            raise ValueError(f"'observation_input.step' must be a non-negative integer, got {observation_input.step!r}")

        if not isinstance(observation_input.timestamp, datetime):
            raise TypeError(f"'observation_input.timestamp' must be a datetime, got {type(observation_input.timestamp).__name__}")

        _validate_spatial_float_array("observation_input.precipitation", observation_input.precipitation)
        _validate_spatial_float_array("observation_input.channel_flow", observation_input.channel_flow)

        if observation_input.precipitation.shape != domain.shape:
            raise ValueError(
                f"'observation_input.precipitation' must have shape {domain.shape}, got {observation_input.precipitation.shape}"
            )

        if observation_input.channel_flow.shape != domain.shape:
            raise ValueError(
                f"'observation_input.channel_flow' must have shape {domain.shape}, got {observation_input.channel_flow.shape}"
            )

        if observation_input.reservoir_storage is not None:
            _validate_vector_float_array("observation_input.reservoir_storage", observation_input.reservoir_storage)

            if observation_input.reservoir_storage.shape[0] != len(domain.reservoirs):
                raise ValueError(
                    "'observation_input.reservoir_storage' must have length "
                    f"{len(domain.reservoirs)}, got {observation_input.reservoir_storage.shape[0]}"
                )

        n_sensors = len(domain.sensors)

        obs_precipitation = np.full(n_sensors, np.nan, dtype=float)
        obs_discharge = np.full(n_sensors, np.nan, dtype=float)
        obs_storage = np.full(n_sensors, np.nan, dtype=float)
        obs_mask = np.zeros(n_sensors, dtype=bool)
        obs_quality_flag = np.full(n_sensors, ObservationQualityFlag.MISSING, dtype=int)

        for sensor_index, sensor in enumerate(domain.sensors):
            sensor_type = _normalize_sensor_type(sensor.sensor_type)

            if sensor_type not in SUPPORTED_SENSOR_TYPES:
                raise ValueError(
                    f"Unsupported sensor_type {sensor.sensor_type!r} for sensor '{sensor.name}'. "
                    f"Supported types: {sorted(SUPPORTED_SENSOR_TYPES)}"
                )

            if sensor_type == PRECIPITATION_SENSOR_TYPE:
                value, mask, flag = self._observe_precipitation_sensor(
                    truth=float(observation_input.precipitation[sensor.cell_y, sensor.cell_x]),
                )
                if mask:
                    obs_precipitation[sensor_index] = value
                obs_mask[sensor_index] = mask
                obs_quality_flag[sensor_index] = int(flag)
                continue

            if sensor_type == DISCHARGE_SENSOR_TYPE:
                value, mask, flag = self._observe_discharge_sensor(
                    truth=float(observation_input.channel_flow[sensor.cell_y, sensor.cell_x]),
                )
                if mask:
                    obs_discharge[sensor_index] = value
                obs_mask[sensor_index] = mask
                obs_quality_flag[sensor_index] = int(flag)
                continue

            value, mask, flag = self._observe_reservoir_storage_sensor(
                domain=domain,
                sensor_index=sensor_index,
                observation_input=observation_input,
            )
            if mask:
                obs_storage[sensor_index] = value
            obs_mask[sensor_index] = mask
            obs_quality_flag[sensor_index] = int(flag)

        n_available = int(np.count_nonzero(obs_mask))
        n_missing = int(n_sensors - n_available)
        n_censored = int(np.count_nonzero(obs_quality_flag == int(ObservationQualityFlag.CENSORED)))

        self._latest_diagnostics = ObservationStepDiagnostics(
            step=observation_input.step,
            timestamp=observation_input.timestamp,
            n_sensors=n_sensors,
            n_available=n_available,
            n_missing=n_missing,
            n_censored=n_censored,
        )

        return ObservationOutput(
            obs_precipitation=obs_precipitation,
            obs_discharge=obs_discharge,
            obs_storage=obs_storage,
            obs_mask=obs_mask,
            obs_quality_flag=obs_quality_flag,
        )
