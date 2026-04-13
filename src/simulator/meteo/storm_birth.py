from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from simulator.common.validation import (
    validate_non_negative_int as _validate_non_negative_int,
)
from simulator.common.validation import (
    validate_non_negative_scalar as _validate_non_negative_float,
)
from simulator.common.validation import (
    validate_positive_int as _validate_positive_int,
)
from simulator.common.validation import (
    validate_positive_scalar as _validate_positive_float,
)
from simulator.core.types import SimulationDomain
from simulator.meteo.latent_state import StormEnvironmentInput
from simulator.meteo.storm_objects import StormCell


@dataclass(frozen=True)
class StormBirthConfig:
    """Configuration controlling storm spawning.

    Notes
    -----
    - `expected_births_per_step` is the baseline expected number of new storms
      before modulation by the meteorological environment.
    - Axis lengths are expressed as semi-axes in meters.
    - `size_cv`, `intensity_cv`, `duration_cv` and `velocity_perturbation_cv`
      are coefficient-of-variation style controls.
    - `birth_margin_cells` expands the spawning box beyond the domain limits.
    """

    expected_births_per_step: float = 1.0
    max_new_storms_per_step: int = 8

    mean_semi_major_axis_m: float = 4_000.0
    mean_semi_minor_axis_m: float = 2_000.0
    size_cv: float = 0.25

    mean_peak_intensity_mmph: float = 12.0
    intensity_cv: float = 0.35

    mean_duration_steps: int = 6
    duration_cv: float = 0.30

    velocity_perturbation_cv: float = 0.20
    orientation_jitter_deg: float = 20.0  # How much can the orientation deviate from
    # the direction of advection when the environment is organized?

    birth_margin_cells: int = 2

    # Band / line organization (phase 4)
    band_cluster_probability: float = 0.35
    band_spacing_m: float = 6_000.0
    band_position_jitter_m: float = 1_200.0
    band_minor_axis_factor: float = 0.80
    band_velocity_shared_weight: float = 0.70

    def __post_init__(self) -> None:
        _validate_non_negative_float("expected_births_per_step", self.expected_births_per_step)
        _validate_non_negative_int("max_new_storms_per_step", self.max_new_storms_per_step)

        _validate_positive_float("mean_semi_major_axis_m", self.mean_semi_major_axis_m)
        _validate_positive_float("mean_semi_minor_axis_m", self.mean_semi_minor_axis_m)
        if self.mean_semi_minor_axis_m > self.mean_semi_major_axis_m:
            raise ValueError("'mean_semi_minor_axis_m' cannot exceed 'mean_semi_major_axis_m'")

        _validate_non_negative_float("size_cv", self.size_cv)

        _validate_positive_float("mean_peak_intensity_mmph", self.mean_peak_intensity_mmph)
        _validate_non_negative_float("intensity_cv", self.intensity_cv)

        _validate_positive_int("mean_duration_steps", self.mean_duration_steps)
        _validate_non_negative_float("duration_cv", self.duration_cv)

        _validate_non_negative_float("velocity_perturbation_cv", self.velocity_perturbation_cv)
        _validate_non_negative_float("orientation_jitter_deg", self.orientation_jitter_deg)

        _validate_non_negative_int("birth_margin_cells", self.birth_margin_cells)

        _validate_non_negative_float("band_cluster_probability", self.band_cluster_probability)
        if self.band_cluster_probability > 1.0:
            raise ValueError(f"'band_cluster_probability' must be <= 1, got {self.band_cluster_probability}")

        _validate_positive_float("band_spacing_m", self.band_spacing_m)
        _validate_non_negative_float("band_position_jitter_m", self.band_position_jitter_m)

        _validate_non_negative_float("band_minor_axis_factor", self.band_minor_axis_factor)
        if self.band_minor_axis_factor > 1.0:
            raise ValueError(f"'band_minor_axis_factor' must be <= 1, got {self.band_minor_axis_factor}")

        _validate_non_negative_float("band_velocity_shared_weight", self.band_velocity_shared_weight)
        if self.band_velocity_shared_weight > 1.0:
            raise ValueError(f"'band_velocity_shared_weight' must be <= 1, got {self.band_velocity_shared_weight}")


@dataclass(frozen=True)
class StormBirthResult:
    """Result of storm spawning for one simulation step."""

    storms: list[StormCell]
    band_reorganization_applied: bool
    band_births_count: int
    band_probability: float


def compute_expected_births(
    env: StormEnvironmentInput,
    config: StormBirthConfig,
) -> float:
    """Return the Poisson mean for the number of new storms in the current step."""
    trigger_term = 0.15 + 0.85 * env.storm_trigger_factor
    moisture_term = 0.50 + 0.50 * env.moisture_availability
    organization_term = 0.75 + 0.25 * env.storm_organization_factor

    expected_births = config.expected_births_per_step * trigger_term * moisture_term * organization_term

    return max(0.0, expected_births)


def sample_birth_count(
    rng: np.random.Generator,
    env: StormEnvironmentInput,
    config: StormBirthConfig,
) -> int:
    """Sample the number of new storms spawned at the current step."""
    expected_births = compute_expected_births(env, config)

    if expected_births <= 0.0:
        return 0

    sampled_count = int(rng.poisson(expected_births))
    return min(sampled_count, config.max_new_storms_per_step)


def _sample_lognormal_from_mean_cv(
    rng: np.random.Generator,
    *,
    mean: float,
    cv: float,
) -> float:
    """Sample a positive value from a lognormal distribution using mean and CV."""
    if cv == 0.0:
        return mean

    sigma2 = math.log(1.0 + cv**2)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - 0.5 * sigma2

    return float(rng.lognormal(mean=mu, sigma=sigma))


def _sample_initial_position_m(
    rng: np.random.Generator,
    domain: SimulationDomain,
    config: StormBirthConfig,
) -> tuple[float, float]:
    """Sample a storm center position, optionally including an outer margin."""
    grid = domain.spatial.grid

    margin_x_m = config.birth_margin_cells * grid.dx
    margin_y_m = config.birth_margin_cells * grid.dy

    min_x_m = grid.x0 - margin_x_m
    max_x_m = grid.x0 + grid.nx * grid.dx + margin_x_m

    min_y_m = grid.y0 - margin_y_m
    max_y_m = grid.y0 + grid.ny * grid.dy + margin_y_m

    center_x_m = float(rng.uniform(min_x_m, max_x_m))
    center_y_m = float(rng.uniform(min_y_m, max_y_m))

    return (center_x_m, center_y_m)


def _sample_axes_m(
    rng: np.random.Generator,
    env: StormEnvironmentInput,
    config: StormBirthConfig,
) -> tuple[float, float]:
    """Sample storm semi-axes in meters."""
    organization_scale = 0.85 + 0.50 * env.storm_organization_factor

    major_mean_m = config.mean_semi_major_axis_m * organization_scale
    minor_mean_m = config.mean_semi_minor_axis_m * (0.90 + 0.20 * env.moisture_availability)

    semi_major_axis_m = _sample_lognormal_from_mean_cv(
        rng,
        mean=major_mean_m,
        cv=config.size_cv,
    )
    semi_minor_axis_m = _sample_lognormal_from_mean_cv(
        rng,
        mean=minor_mean_m,
        cv=config.size_cv,
    )

    semi_minor_axis_m = min(semi_minor_axis_m, semi_major_axis_m)

    return (semi_major_axis_m, semi_minor_axis_m)


def _sample_orientation_deg(
    rng: np.random.Generator,
    env: StormEnvironmentInput,
    config: StormBirthConfig,
) -> float:
    """Sample storm orientation in degrees.

    Low organization -> almost free orientation.
    High organization -> closer alignment to the background advection vector.
    """
    advection_angle_deg = math.degrees(math.atan2(env.advection_v_mps, env.advection_u_mps)) % 180.0

    free_orientation_deg = float(rng.uniform(0.0, 180.0))
    aligned_orientation_deg = float(rng.normal(advection_angle_deg, config.orientation_jitter_deg)) % 180.0

    organization_weight = env.storm_organization_factor
    orientation_deg = ((1.0 - organization_weight) * free_orientation_deg + organization_weight * aligned_orientation_deg) % 180.0

    return orientation_deg


def _sample_peak_intensity_mmph(
    rng: np.random.Generator,
    env: StormEnvironmentInput,
    config: StormBirthConfig,
) -> float:
    """Sample storm mature-stage peak intensity in mm/h."""
    intensity_scale = 0.60 + 0.80 * env.storm_trigger_factor + 0.40 * env.moisture_availability + 0.20 * env.storm_organization_factor

    mean_peak_intensity_mmph = config.mean_peak_intensity_mmph * intensity_scale

    return _sample_lognormal_from_mean_cv(
        rng,
        mean=mean_peak_intensity_mmph,
        cv=config.intensity_cv,
    )


def _sample_duration_steps(
    rng: np.random.Generator,
    env: StormEnvironmentInput,
    config: StormBirthConfig,
) -> int:
    """Sample storm lifetime in simulation steps."""
    duration_scale = 0.75 + 0.50 * env.storm_organization_factor + 0.25 * env.moisture_availability

    mean_duration_steps = max(1.0, config.mean_duration_steps * duration_scale)

    sampled_duration = _sample_lognormal_from_mean_cv(
        rng,
        mean=mean_duration_steps,
        cv=config.duration_cv,
    )

    return max(1, int(round(sampled_duration)))


def _sample_velocity_components_mps(
    rng: np.random.Generator,
    env: StormEnvironmentInput,
    config: StormBirthConfig,
) -> tuple[float, float]:
    """Sample storm advection velocity around the environmental background flow."""
    base_u_mps = env.advection_u_mps
    base_v_mps = env.advection_v_mps

    base_speed_mps = math.hypot(base_u_mps, base_v_mps)

    if base_speed_mps == 0.0:
        angle_rad = float(rng.uniform(0.0, 2.0 * math.pi))
        speed_mps = float(rng.uniform(0.0, 1.0))
        return (speed_mps * math.cos(angle_rad), speed_mps * math.sin(angle_rad))

    perturbation_std_mps = config.velocity_perturbation_cv * base_speed_mps

    velocity_u_mps = float(rng.normal(base_u_mps, perturbation_std_mps))
    velocity_v_mps = float(rng.normal(base_v_mps, perturbation_std_mps))

    return (velocity_u_mps, velocity_v_mps)


def _sample_band_angle_deg(
    rng: np.random.Generator,
    env: StormEnvironmentInput,
    config: StormBirthConfig,
) -> float:
    """Sample a common band orientation, usually close to advection."""
    advection_angle_deg = math.degrees(math.atan2(env.advection_v_mps, env.advection_u_mps)) % 180.0

    # Strong organization -> tighter alignment around the advection direction.
    band_jitter_deg = max(5.0, 0.5 * config.orientation_jitter_deg)
    return float(rng.normal(advection_angle_deg, band_jitter_deg)) % 180.0


def _reorganize_storms_as_band(
    rng: np.random.Generator,
    storms: list[StormCell],
    domain: SimulationDomain,
    env: StormEnvironmentInput,
    config: StormBirthConfig,
) -> list[StormCell]:
    """Reposition and align a group of storms so they form a band-like structure."""
    if len(storms) < 2:
        return storms

    anchor_x_m, anchor_y_m = _sample_initial_position_m(rng, domain, config)
    band_angle_deg = _sample_band_angle_deg(rng, env, config)
    band_angle_rad = math.radians(band_angle_deg)

    dir_x = math.cos(band_angle_rad)
    dir_y = math.sin(band_angle_rad)

    # Perpendicular direction for cross-band jitter
    normal_x = -dir_y
    normal_y = dir_x

    center_index = 0.5 * (len(storms) - 1)

    shared_u_mps, shared_v_mps = _sample_velocity_components_mps(rng, env, config)

    for idx, storm in enumerate(storms):
        along_offset_m = (idx - center_index) * config.band_spacing_m
        along_jitter_m = float(rng.normal(0.0, config.band_position_jitter_m))
        cross_jitter_m = float(rng.normal(0.0, 0.5 * config.band_position_jitter_m))  # 0.5 x ... so that it
        # still looks like a band.

        storm.center_x_m = anchor_x_m + (along_offset_m + along_jitter_m) * dir_x + cross_jitter_m * normal_x
        storm.center_y_m = anchor_y_m + (along_offset_m + along_jitter_m) * dir_y + cross_jitter_m * normal_y

        storm.orientation_deg = float(rng.normal(band_angle_deg, max(3.0, 0.35 * config.orientation_jitter_deg))) % 180.0

        storm.velocity_u_mps = (
            config.band_velocity_shared_weight * shared_u_mps + (1.0 - config.band_velocity_shared_weight) * storm.velocity_u_mps
        )
        storm.velocity_v_mps = (
            config.band_velocity_shared_weight * shared_v_mps + (1.0 - config.band_velocity_shared_weight) * storm.velocity_v_mps
        )

        storm.semi_minor_axis_m = min(
            storm.semi_major_axis_m,
            storm.semi_minor_axis_m * config.band_minor_axis_factor,
        )

    return storms


def _maybe_reorganize_storms_as_band(
    rng: np.random.Generator,
    storms: list[StormCell],
    domain: SimulationDomain,
    env: StormEnvironmentInput,
    config: StormBirthConfig,
) -> StormBirthResult:
    """Convert a same-step storm group into a band when organization is high enough."""
    if len(storms) < 2:
        return StormBirthResult(
            storms=storms,
            band_reorganization_applied=False,
            band_births_count=0,
            band_probability=0.0,
        )

    band_probability = config.band_cluster_probability * env.storm_organization_factor

    if rng.random() >= band_probability:
        return StormBirthResult(
            storms=storms,
            band_reorganization_applied=False,
            band_births_count=0,
            band_probability=band_probability,
        )

    reorganized_storms = _reorganize_storms_as_band(
        rng=rng,
        storms=storms,
        domain=domain,
        env=env,
        config=config,
    )

    return StormBirthResult(
        storms=reorganized_storms,
        band_reorganization_applied=True,
        band_births_count=len(reorganized_storms),
        band_probability=band_probability,
    )


def spawn_storm(
    rng: np.random.Generator,
    *,
    storm_id: int,
    domain: SimulationDomain,
    env: StormEnvironmentInput,
    config: StormBirthConfig,
) -> StormCell:
    """Create one new storm cell sampled from the current environment."""
    center_x_m, center_y_m = _sample_initial_position_m(rng, domain, config)
    semi_major_axis_m, semi_minor_axis_m = _sample_axes_m(rng, env, config)
    orientation_deg = _sample_orientation_deg(rng, env, config)
    peak_intensity_mmph = _sample_peak_intensity_mmph(rng, env, config)
    duration_steps = _sample_duration_steps(rng, env, config)
    velocity_u_mps, velocity_v_mps = _sample_velocity_components_mps(rng, env, config)

    return StormCell(
        storm_id=storm_id,
        center_x_m=center_x_m,
        center_y_m=center_y_m,
        velocity_u_mps=velocity_u_mps,
        velocity_v_mps=velocity_v_mps,
        semi_major_axis_m=semi_major_axis_m,
        semi_minor_axis_m=semi_minor_axis_m,
        orientation_deg=orientation_deg,
        peak_intensity_mmph=peak_intensity_mmph,
        duration_steps=duration_steps,
        age_steps=0,
    )


def spawn_storms(
    rng: np.random.Generator,
    *,
    next_storm_id: int,
    domain: SimulationDomain,
    env: StormEnvironmentInput,
    config: StormBirthConfig,
) -> StormBirthResult:
    """Sample and return all new storms born at the current step."""
    n_births = sample_birth_count(rng, env, config)

    storms: list[StormCell] = []
    for local_index in range(n_births):
        storms.append(
            spawn_storm(
                rng,
                storm_id=next_storm_id + local_index,
                domain=domain,
                env=env,
                config=config,
            )
        )

    return _maybe_reorganize_storms_as_band(
        rng=rng,
        storms=storms,
        domain=domain,
        env=env,
        config=config,
    )
