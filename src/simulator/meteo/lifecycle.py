from __future__ import annotations

import math
from dataclasses import dataclass

from simulator.common.validation import validate_fraction as _validate_fraction
from simulator.meteo.storm_objects import StormCell


@dataclass(frozen=True)
class StormLifecycleConfig:
    """Configuration of the storm life-cycle shape."""

    growth_fraction: float = 0.30
    mature_fraction: float = 0.40
    decay_fraction: float = 0.30
    minimum_size_factor: float = 0.60

    def __post_init__(self) -> None:
        growth_fraction = _validate_fraction("growth_fraction", self.growth_fraction)
        mature_fraction = _validate_fraction("mature_fraction", self.mature_fraction)
        decay_fraction = _validate_fraction("decay_fraction", self.decay_fraction)
        minimum_size_factor = _validate_fraction("minimum_size_factor", self.minimum_size_factor)

        if minimum_size_factor <= 0.0:
            raise ValueError(f"'minimum_size_factor' must be > 0, got {minimum_size_factor}")

        total = growth_fraction + mature_fraction + decay_fraction
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "Life-cycle fractions must sum to 1.0, got "
                f"{total} from growth={growth_fraction}, "
                f"mature={mature_fraction}, decay={decay_fraction}"
            )


def compute_life_factor(
    progress: float,
    lifecycle: StormLifecycleConfig,
) -> float:
    """Return a normalized life factor in [0, 1] from normalized progress.

    Parameters
    ----------
    progress
        Normalized storm age in [0, 1].
    lifecycle
        Life-cycle configuration.

    Returns
    -------
    float
        Relative storm strength:
        - rises during growth
        - stays at 1 during maturity
        - decays to 0 during decay
    """
    progress = _validate_fraction("progress", progress)

    if progress >= 1.0:
        return 0.0

    growth_end = lifecycle.growth_fraction
    mature_end = lifecycle.growth_fraction + lifecycle.mature_fraction

    if growth_end > 0.0 and progress < growth_end:
        return progress / growth_end

    if progress < mature_end:
        return 1.0

    if lifecycle.decay_fraction > 0.0:
        return max(0.0, (1.0 - progress) / lifecycle.decay_fraction)

    return 0.0


def compute_storm_life_factor(
    storm: StormCell,
    lifecycle: StormLifecycleConfig,
) -> float:
    """Return the current life factor of a storm."""
    return compute_life_factor(storm.life_progress, lifecycle)


def compute_current_intensity_mmph(
    storm: StormCell,
    lifecycle: StormLifecycleConfig,
) -> float:
    """Return the current storm rainfall intensity in mm/h."""
    life_factor = compute_storm_life_factor(storm, lifecycle)
    return storm.peak_intensity_mmph * life_factor


def compute_current_axes_m(
    storm: StormCell,
    lifecycle: StormLifecycleConfig,
) -> tuple[float, float]:
    """Return the current effective semi-axes in meters.

    The storm footprint is allowed to shrink/grow with life-cycle phase,
    but it never collapses completely while the storm is still tracked.
    """
    life_factor = compute_storm_life_factor(storm, lifecycle)

    size_factor = lifecycle.minimum_size_factor + (1.0 - lifecycle.minimum_size_factor) * life_factor

    current_major_axis_m = storm.semi_major_axis_m * size_factor
    current_minor_axis_m = storm.semi_minor_axis_m * size_factor

    return (current_major_axis_m, current_minor_axis_m)
