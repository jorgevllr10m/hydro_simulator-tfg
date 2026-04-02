from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from simulator.core.types import SimulationDomain
from simulator.meteo.latent_state import StormEnvironmentInput
from simulator.meteo.storm_objects import StormCell


def _validate_positive_float(name: str, value: int | float) -> float:
    """Validate a strictly positive numeric scalar."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"'{name}' must be numeric, got {type(value).__name__}")
    numeric_value = float(value)
    if numeric_value <= 0.0:
        raise ValueError(f"'{name}' must be > 0, got {numeric_value}")
    return numeric_value


def _validate_non_negative_float(name: str, value: int | float) -> float:
    """Validate a non-negative numeric scalar."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"'{name}' must be numeric, got {type(value).__name__}")
    numeric_value = float(value)
    if numeric_value < 0.0:
        raise ValueError(f"'{name}' must be >= 0, got {numeric_value}")
    return numeric_value


def _validate_non_negative_int(name: str, value: int) -> int:
    """Validate a non-negative integer."""
    if not isinstance(value, int):
        raise TypeError(f"'{name}' must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"'{name}' must be >= 0, got {value}")
    return value


def _validate_positive_int(name: str, value: int) -> int:
    """Validate a strictly positive integer."""
    if not isinstance(value, int):
        raise TypeError(f"'{name}' must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"'{name}' must be > 0, got {value}")
    return value


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
) -> list[StormCell]:
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

    return storms
