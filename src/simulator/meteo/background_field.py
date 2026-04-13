from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.common.validation import (
    clamp01 as _clamp01,
)
from simulator.common.validation import (
    validate_fraction as _validate_fraction,
)
from simulator.common.validation import (
    validate_numeric_scalar as _validate_numeric_scalar,
)
from simulator.core.types import FloatArray, SimulationDomain
from simulator.meteo.latent_state import LatentEnvironmentState
from simulator.meteo.regimes import MeteorologicalRegime


def build_uniform_spatial_field(
    domain: SimulationDomain,
    value: int | float,
) -> FloatArray:
    """Return a 2D float field filled with a uniform scalar value."""
    scalar_value = _validate_numeric_scalar("value", value)
    return np.full(domain.shape, scalar_value, dtype=float)


def build_air_temperature_field(
    domain: SimulationDomain,
    background_temperature_c: int | float,
) -> FloatArray:
    """Build the 2D air-temperature field for the current step."""
    background_temperature_c = _validate_numeric_scalar(
        "background_temperature_c",
        background_temperature_c,
    )
    return build_uniform_spatial_field(domain, background_temperature_c)


@dataclass(frozen=True)
class BackgroundFieldConfig:
    """Configuration of the correlated background precipitation field."""

    enabled: bool = True
    random_seed: int = 2468

    temporal_persistence: float = 0.85
    spatial_smoothing_passes: int = 4

    max_intensity_mm_dt: float = 2.0

    dry_activation_threshold: float = 0.88
    wet_activation_threshold: float = 0.55

    # Temporal intermittency / persistence of the background component
    activity_memory: float = 0.80
    activity_noise_std: float = 0.08
    dry_activity_target: float = 0.00
    wet_activity_target: float = 0.95

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"'enabled' must be bool, got {type(self.enabled).__name__}")

        if not isinstance(self.random_seed, int):
            raise TypeError(f"'random_seed' must be int, got {type(self.random_seed).__name__}")

        _validate_fraction("temporal_persistence", self.temporal_persistence)

        if not isinstance(self.spatial_smoothing_passes, int):
            raise TypeError(f"'spatial_smoothing_passes' must be int, got {type(self.spatial_smoothing_passes).__name__}")
        if self.spatial_smoothing_passes < 0:
            raise ValueError(f"'spatial_smoothing_passes' must be >= 0, got {self.spatial_smoothing_passes}")

        max_intensity_mm_dt = _validate_numeric_scalar(
            "max_intensity_mm_dt",
            self.max_intensity_mm_dt,
        )
        if max_intensity_mm_dt < 0.0:
            raise ValueError(f"'max_intensity_mm_dt' must be >= 0, got {max_intensity_mm_dt}")

        dry_activation_threshold = _validate_fraction(
            "dry_activation_threshold",
            self.dry_activation_threshold,
        )
        wet_activation_threshold = _validate_fraction(
            "wet_activation_threshold",
            self.wet_activation_threshold,
        )
        if wet_activation_threshold > dry_activation_threshold:
            raise ValueError(
                "'wet_activation_threshold' must be <= 'dry_activation_threshold', "
                f"got wet={wet_activation_threshold}, dry={dry_activation_threshold}"
            )

        _validate_fraction("activity_memory", self.activity_memory)

        activity_noise_std = _validate_numeric_scalar(
            "activity_noise_std",
            self.activity_noise_std,
        )
        if activity_noise_std < 0.0:
            raise ValueError(f"'activity_noise_std' must be >= 0, got {activity_noise_std}")

        _validate_fraction("dry_activity_target", self.dry_activity_target)
        _validate_fraction("wet_activity_target", self.wet_activity_target)

        if self.dry_activity_target > self.wet_activity_target:
            raise ValueError(
                "'dry_activity_target' must be <= 'wet_activity_target', "
                f"got dry={self.dry_activity_target}, wet={self.wet_activity_target}"
            )


@dataclass(frozen=True)
class BackgroundFieldState:
    """Persistent internal state of the background-field generator."""

    normalized_field: FloatArray
    activity_factor: float

    def __post_init__(self) -> None:
        if not isinstance(self.normalized_field, np.ndarray):
            raise TypeError(f"'normalized_field' must be a numpy.ndarray, got {type(self.normalized_field).__name__}")
        if self.normalized_field.ndim != 2:
            raise ValueError(f"'normalized_field' must be a 2D array with shape (ny, nx), got ndim={self.normalized_field.ndim}")
        if not np.issubdtype(self.normalized_field.dtype, np.floating):
            raise TypeError(f"'normalized_field' must have a floating dtype, got {self.normalized_field.dtype}")

        if not isinstance(self.activity_factor, (int, float)):
            raise TypeError(f"'activity_factor' must be numeric, got {type(self.activity_factor).__name__}")
        if not 0.0 <= float(self.activity_factor) <= 1.0:
            raise ValueError(f"'activity_factor' must be within [0, 1], got {self.activity_factor}")


class BackgroundFieldModel:
    """Stateful generator of correlated background precipitation fields."""

    def __init__(self, config: BackgroundFieldConfig) -> None:
        if not isinstance(config, BackgroundFieldConfig):
            raise TypeError(f"'config' must be a BackgroundFieldConfig, got {type(config).__name__}")

        self.config = config
        self._rng = np.random.default_rng(config.random_seed)
        self._latest_state: BackgroundFieldState | None = None

    @property
    def latest_state(self) -> BackgroundFieldState | None:
        """Return the latest internal state of the background field."""
        return self._latest_state

    def reset(self) -> None:
        """Reset RNG and persistent state to their initial reproducible state."""
        self._rng = np.random.default_rng(self.config.random_seed)
        self._latest_state = None

    def step(
        self,
        domain: SimulationDomain,
        latent_state: LatentEnvironmentState,
    ) -> FloatArray:
        """Generate the background precipitation field for one simulation step."""
        if not isinstance(domain, SimulationDomain):
            raise TypeError(f"'domain' must be a SimulationDomain, got {type(domain).__name__}")

        if not isinstance(latent_state, LatentEnvironmentState):
            raise TypeError(f"'latent_state' must be a LatentEnvironmentState, got {type(latent_state).__name__}")

        if not self.config.enabled:
            zero_field = np.zeros(domain.shape, dtype=float)
            self._latest_state = BackgroundFieldState(
                normalized_field=zero_field.copy(),
                activity_factor=0.0,
            )
            return zero_field

        raw_candidate = self._rng.random(domain.shape, dtype=float)
        candidate = self._smooth_and_normalize(
            raw_candidate,
            passes=self.config.spatial_smoothing_passes,
        )

        if self._latest_state is None:
            blended = candidate
        else:
            blended = (
                self.config.temporal_persistence * self._latest_state.normalized_field
                + (1.0 - self.config.temporal_persistence) * candidate
            )
            blended = self._normalize_field(blended)

        support = self._compute_background_support(latent_state)
        activation_threshold = self._compute_activation_threshold(support)
        activated = self._activate_field(blended, activation_threshold)

        target_activity = self._compute_activity_target(support)
        activity_factor = self._update_activity_factor(target_activity)

        background_precipitation = (self.config.max_intensity_mm_dt * support * activity_factor * activated).astype(float, copy=False)

        self._latest_state = BackgroundFieldState(
            normalized_field=blended.astype(float, copy=False),
            activity_factor=activity_factor,
        )
        return background_precipitation

    def _compute_background_support(
        self,
        latent_state: LatentEnvironmentState,
    ) -> float:
        """Return the large-scale support factor for background precipitation."""
        regime_weights = {
            MeteorologicalRegime.STABLE_DRY: 0.05,
            MeteorologicalRegime.TRANSITIONAL: 0.30,
            MeteorologicalRegime.CONVECTIVE: 0.20,
            MeteorologicalRegime.FRONTAL_PERSISTENT: 0.85,
        }

        regime_term = regime_weights[latent_state.regime]
        wetness_term = 0.55 + 0.45 * latent_state.antecedent_wetness_index
        cloudiness_term = 0.55 + 0.45 * latent_state.cloudiness_index
        spell_term = 0.60 + 0.40 * latent_state.precipitation_spell_index

        support = regime_term * wetness_term * cloudiness_term * spell_term
        return _clamp01(support)

    def _compute_activation_threshold(self, support: float) -> float:
        """Return the spatial threshold used to activate the smoothed field."""
        support = _clamp01(support)

        return self.config.dry_activation_threshold - support * (
            self.config.dry_activation_threshold - self.config.wet_activation_threshold
        )

    def _compute_activity_target(self, support: float) -> float:
        """Return the target temporal activity level for the background field."""
        support = _clamp01(support)

        return self.config.dry_activity_target + support * (self.config.wet_activity_target - self.config.dry_activity_target)

    def _update_activity_factor(self, target_activity: float) -> float:
        """Update the temporal activity factor with memory and noise."""
        target_activity = _clamp01(target_activity)

        noisy_target = _clamp01(target_activity + float(self._rng.normal(0.0, self.config.activity_noise_std)))

        if self._latest_state is None:
            blended_activity = noisy_target
        else:
            blended_activity = (
                self.config.activity_memory * self._latest_state.activity_factor + (1.0 - self.config.activity_memory) * noisy_target
            )

        activity_deadzone = 0.05

        if blended_activity <= activity_deadzone:
            return 0.0

        return _clamp01((blended_activity - activity_deadzone) / (1.0 - activity_deadzone))

    @staticmethod
    def _activate_field(
        normalized_field: FloatArray,
        threshold: float,
    ) -> FloatArray:
        """Threshold and rescale a normalized field back into [0, 1]."""
        threshold = _clamp01(threshold)

        if threshold >= 1.0:
            return np.zeros_like(normalized_field, dtype=float)

        activated = (normalized_field - threshold) / max(1e-12, 1.0 - threshold)
        return np.clip(activated, 0.0, 1.0)

    @staticmethod
    def _normalize_field(field: FloatArray) -> FloatArray:
        """Normalize a 2D field into [0, 1]."""
        min_value = float(np.min(field))
        max_value = float(np.max(field))

        if max_value <= min_value:
            return np.zeros_like(field, dtype=float)

        return (field - min_value) / (max_value - min_value)

    @classmethod
    def _smooth_and_normalize(
        cls,
        field: FloatArray,
        *,
        passes: int,
    ) -> FloatArray:
        """Apply repeated box smoothing and normalize to [0, 1]."""
        smoothed = field.astype(float, copy=True)

        for _ in range(passes):
            smoothed = cls._box_smooth_once(smoothed)

        return cls._normalize_field(smoothed)

    @staticmethod
    def _box_smooth_once(field: FloatArray) -> FloatArray:
        """Apply one 3x3 box filter using edge padding."""
        padded = np.pad(field, pad_width=1, mode="edge")

        smoothed = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0

        return smoothed.astype(float, copy=False)


def build_background_precipitation_field(
    domain: SimulationDomain,
    latent_state: LatentEnvironmentState | None = None,
    model: BackgroundFieldModel | None = None,
) -> FloatArray:
    """Compatibility helper for the background precipitation field."""
    if latent_state is None or model is None:
        return np.zeros(domain.shape, dtype=float)

    return model.step(domain=domain, latent_state=latent_state)
