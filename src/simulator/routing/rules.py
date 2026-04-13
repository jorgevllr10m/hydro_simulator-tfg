from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from simulator.common.validation import (
    validate_fraction as _validate_fraction,
)
from simulator.common.validation import (
    validate_non_negative_scalar as _validate_non_negative_scalar,
)
from simulator.common.validation import (
    validate_numeric_scalar as _validate_numeric_scalar,
)
from simulator.routing.storage import compute_storage_fraction


def _interpolate_linearly(
    x: int | float,
    *,
    x0: int | float,
    x1: int | float,
    y0: int | float,
    y1: int | float,
) -> float:
    """Return linear interpolation between two points."""
    x = _validate_numeric_scalar("x", x)
    x0 = _validate_numeric_scalar("x0", x0)
    x1 = _validate_numeric_scalar("x1", x1)
    y0 = _validate_numeric_scalar("y0", y0)
    y1 = _validate_numeric_scalar("y1", y1)

    if x1 <= x0:
        raise ValueError(f"'x1' must be > 'x0', got x0={x0}, x1={x1}")

    if x <= x0:
        return y0
    if x >= x1:
        return y1

    weight = (x - x0) / (x1 - x0)
    return float(y0 + weight * (y1 - y0))


class ReservoirOperationZone(str, Enum):
    """Simple operating zones of the reservoir."""

    CONSERVATION = "conservation"
    NORMAL = "normal"
    FLOOD_CONTROL = "flood_control"


@dataclass(frozen=True)
class ReservoirRulesConfig:
    """Simple operating rules for one reservoir.

    Notes
    -----
    - Below `conservation_fraction`, the reservoir tries to preserve water and
      release only `min_release_m3s`.
    - Between `conservation_fraction` and `flood_fraction`, release increases
      smoothly from `min_release_m3s` to `target_release_m3s`.
    - Above `flood_fraction`, release increases smoothly from
      `target_release_m3s` to `max_controlled_release_m3s`.
    - Spill is not decided here; spill is handled later by the storage balance
      if storage still exceeds capacity.
    """

    min_release_m3s: float = 0.2
    """
    Minimum you try to release when the reservoir is low.
    """

    target_release_m3s: float = 1.5
    """
    Typical release when in normal zone.
    """

    max_controlled_release_m3s: float = 6.0
    """
    Maximum controlled release that the reservoir attempts to release before the spill comes into play.
    """

    conservation_fraction: float = 0.5
    """
    It is the threshold below which the reservoir is in conservation mode.
    """

    flood_fraction: float = 0.90
    """
    It is the threshold from which it enters avenue control mode.
    """

    def __post_init__(self) -> None:
        min_release_m3s = _validate_non_negative_scalar(
            "min_release_m3s",
            self.min_release_m3s,
        )
        target_release_m3s = _validate_non_negative_scalar(
            "target_release_m3s",
            self.target_release_m3s,
        )
        max_controlled_release_m3s = _validate_non_negative_scalar(
            "max_controlled_release_m3s",
            self.max_controlled_release_m3s,
        )

        conservation_fraction = _validate_fraction(
            "conservation_fraction",
            self.conservation_fraction,
        )
        flood_fraction = _validate_fraction(
            "flood_fraction",
            self.flood_fraction,
        )

        if target_release_m3s < min_release_m3s:
            raise ValueError(
                f"'target_release_m3s' must be >= 'min_release_m3s', got target={target_release_m3s}, min={min_release_m3s}"
            )

        if max_controlled_release_m3s < target_release_m3s:
            raise ValueError(
                "'max_controlled_release_m3s' must be >= 'target_release_m3s', "
                f"got max={max_controlled_release_m3s}, target={target_release_m3s}"
            )

        if flood_fraction <= conservation_fraction:
            raise ValueError(
                f"'flood_fraction' must be > 'conservation_fraction', got conservation={conservation_fraction}, flood={flood_fraction}"
            )


@dataclass(frozen=True)
class ReservoirRuleDecision:
    """Detailed decision returned by the simplified operating rules."""

    storage_fraction: float
    zone: ReservoirOperationZone
    requested_release_m3s: float

    def __post_init__(self) -> None:
        _validate_fraction("storage_fraction", self.storage_fraction)

        if not isinstance(self.zone, ReservoirOperationZone):
            raise TypeError(f"'zone' must be a ReservoirOperationZone, got {type(self.zone).__name__}")

        _validate_non_negative_scalar("requested_release_m3s", self.requested_release_m3s)


def determine_reservoir_operation_zone(
    storage_m3: int | float,
    *,
    capacity_m3: int | float,
    config: ReservoirRulesConfig,
) -> ReservoirOperationZone:
    """Return the current operating zone from normalized storage."""
    if not isinstance(config, ReservoirRulesConfig):
        raise TypeError(f"'config' must be a ReservoirRulesConfig, got {type(config).__name__}")

    storage_fraction = compute_storage_fraction(
        storage_m3,
        capacity_m3=capacity_m3,
    )

    if storage_fraction < config.conservation_fraction:
        return ReservoirOperationZone.CONSERVATION

    if storage_fraction < config.flood_fraction:
        return ReservoirOperationZone.NORMAL

    return ReservoirOperationZone.FLOOD_CONTROL


def compute_requested_release_m3s(
    storage_m3: int | float,
    *,
    capacity_m3: int | float,
    config: ReservoirRulesConfig,
) -> float:
    """Return the requested controlled release from simple operating rules."""
    if not isinstance(config, ReservoirRulesConfig):
        raise TypeError(f"'config' must be a ReservoirRulesConfig, got {type(config).__name__}")

    storage_fraction = compute_storage_fraction(
        storage_m3,
        capacity_m3=capacity_m3,
    )

    if storage_fraction < config.conservation_fraction:
        return float(config.min_release_m3s)

    if storage_fraction < config.flood_fraction:
        return _interpolate_linearly(
            storage_fraction,
            x0=config.conservation_fraction,
            x1=config.flood_fraction,
            y0=config.min_release_m3s,
            y1=config.target_release_m3s,
        )

    return _interpolate_linearly(
        storage_fraction,
        x0=config.flood_fraction,
        x1=1.0,
        y0=config.target_release_m3s,
        y1=config.max_controlled_release_m3s,
    )


def apply_reservoir_operating_rules(
    storage_m3: int | float,
    *,
    capacity_m3: int | float,
    config: ReservoirRulesConfig,
) -> ReservoirRuleDecision:
    """Return the simplified operating decision for one reservoir."""
    if not isinstance(config, ReservoirRulesConfig):
        raise TypeError(f"'config' must be a ReservoirRulesConfig, got {type(config).__name__}")

    storage_fraction = compute_storage_fraction(
        storage_m3,
        capacity_m3=capacity_m3,
    )
    zone = determine_reservoir_operation_zone(
        storage_m3,
        capacity_m3=capacity_m3,
        config=config,
    )
    requested_release_m3s = compute_requested_release_m3s(
        storage_m3,
        capacity_m3=capacity_m3,
        config=config,
    )

    return ReservoirRuleDecision(
        storage_fraction=storage_fraction,
        zone=zone,
        requested_release_m3s=requested_release_m3s,
    )
