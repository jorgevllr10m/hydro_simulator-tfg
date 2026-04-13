from __future__ import annotations

from dataclasses import dataclass

from simulator.common.validation import (
    validate_non_negative_scalar as _validate_non_negative_scalar,
)
from simulator.common.validation import (
    validate_positive_scalar as _validate_positive_scalar,
)


@dataclass(frozen=True)
class ReservoirStorageConfig:
    """Physical configuration used by the simplified reservoir storage model.

    Notes
    -----
    - `surface_area_max_m2` is the reservoir area when storage reaches capacity.
    - `area_exponent` controls the curvature of the area-storage relationship.
    - `evaporation_factor` scales PET to represent open-water evaporation in
      a simple way.
    """

    surface_area_max_m2: float = 2_000_000.0
    """
    Maximum surface area of ​​the reservoir when it is at maximum capacity
    """

    area_exponent: float = 0.8
    """
    It controls how the area grows as storage increases.
    """

    evaporation_factor: float = 1.0
    """
    It is a simple factor to convert PET into evaporation over free water.
    """

    def __post_init__(self) -> None:
        _validate_positive_scalar("surface_area_max_m2", self.surface_area_max_m2)
        _validate_positive_scalar("area_exponent", self.area_exponent)
        _validate_non_negative_scalar("evaporation_factor", self.evaporation_factor)


@dataclass(frozen=True)
class ReservoirStorageUpdateResult:
    """Detailed storage-balance result for one reservoir and one time step."""

    storage_before_step_m3: float
    inflow_m3s: float
    inflow_volume_m3: float

    storage_after_inflow_m3: float

    surface_area_m2: float
    evaporation_loss_m3: float
    storage_after_evaporation_m3: float

    controlled_release_m3s: float
    controlled_release_volume_m3: float
    storage_after_controlled_release_m3: float

    spill_m3s: float
    spill_volume_m3: float

    storage_m3: float
    total_outflow_m3s: float
    total_outflow_volume_m3: float

    def __post_init__(self) -> None:
        scalar_fields = {
            "storage_before_step_m3": self.storage_before_step_m3,
            "inflow_m3s": self.inflow_m3s,
            "inflow_volume_m3": self.inflow_volume_m3,
            "storage_after_inflow_m3": self.storage_after_inflow_m3,
            "surface_area_m2": self.surface_area_m2,
            "evaporation_loss_m3": self.evaporation_loss_m3,
            "storage_after_evaporation_m3": self.storage_after_evaporation_m3,
            "controlled_release_m3s": self.controlled_release_m3s,
            "controlled_release_volume_m3": self.controlled_release_volume_m3,
            "storage_after_controlled_release_m3": self.storage_after_controlled_release_m3,
            "spill_m3s": self.spill_m3s,
            "spill_volume_m3": self.spill_volume_m3,
            "storage_m3": self.storage_m3,
            "total_outflow_m3s": self.total_outflow_m3s,
            "total_outflow_volume_m3": self.total_outflow_volume_m3,
        }

        for name, value in scalar_fields.items():
            _validate_non_negative_scalar(name, value)


def compute_storage_fraction(
    storage_m3: int | float,
    *,
    capacity_m3: int | float,
) -> float:
    """Return normalized storage in the [0, 1] interval."""
    storage_m3 = _validate_non_negative_scalar("storage_m3", storage_m3)
    capacity_m3 = _validate_positive_scalar("capacity_m3", capacity_m3)

    return min(1.0, max(0.0, storage_m3 / capacity_m3))


def compute_reservoir_surface_area_m2(
    storage_m3: int | float,
    *,
    capacity_m3: int | float,
    config: ReservoirStorageConfig,
) -> float:
    """Return reservoir surface area from storage using a simple power law.

    Formula
    -------
    area = area_max * (storage / capacity) ** area_exponent

    Behavior
    --------
    - storage = 0 -> area = 0
    - storage = capacity -> area = area_max
    """
    if not isinstance(config, ReservoirStorageConfig):
        raise TypeError(f"'config' must be a ReservoirStorageConfig, got {type(config).__name__}")

    storage_fraction = compute_storage_fraction(
        storage_m3,
        capacity_m3=capacity_m3,
    )

    if storage_fraction == 0.0:
        return 0.0

    surface_area_m2 = config.surface_area_max_m2 * (storage_fraction**config.area_exponent)
    return max(0.0, float(surface_area_m2))


def convert_depth_mm_to_volume_m3(
    depth_mm: int | float,
    *,
    surface_area_m2: int | float,
) -> float:
    """Convert a water depth in mm over a surface into a volume in m3."""
    depth_mm = _validate_non_negative_scalar("depth_mm", depth_mm)
    surface_area_m2 = _validate_non_negative_scalar("surface_area_m2", surface_area_m2)

    depth_m = depth_mm / 1000.0
    volume_m3 = depth_m * surface_area_m2

    return max(0.0, float(volume_m3))


def compute_reservoir_evaporation_loss_m3(
    storage_m3: int | float,
    *,
    capacity_m3: int | float,
    pet_mm_dt: int | float,
    config: ReservoirStorageConfig,
) -> float:
    """Return evaporation loss volume from PET and current reservoir area."""
    if not isinstance(config, ReservoirStorageConfig):
        raise TypeError(f"'config' must be a ReservoirStorageConfig, got {type(config).__name__}")

    pet_mm_dt = _validate_non_negative_scalar("pet_mm_dt", pet_mm_dt)

    surface_area_m2 = compute_reservoir_surface_area_m2(
        storage_m3,
        capacity_m3=capacity_m3,
        config=config,
    )

    effective_evaporation_depth_mm = config.evaporation_factor * pet_mm_dt

    evaporation_loss_m3 = convert_depth_mm_to_volume_m3(
        effective_evaporation_depth_mm,
        surface_area_m2=surface_area_m2,
    )

    storage_m3 = _validate_non_negative_scalar("storage_m3", storage_m3)

    return min(storage_m3, evaporation_loss_m3)


def convert_discharge_m3s_to_volume_m3(
    discharge_m3s: int | float,
    *,
    dt_seconds: int | float,
) -> float:
    """Convert discharge [m3/s] into step volume [m3]."""
    discharge_m3s = _validate_non_negative_scalar("discharge_m3s", discharge_m3s)
    dt_seconds = _validate_positive_scalar("dt_seconds", dt_seconds)

    return max(0.0, float(discharge_m3s * dt_seconds))


def convert_volume_m3_to_discharge_m3s(
    volume_m3: int | float,
    *,
    dt_seconds: int | float,
) -> float:
    """Convert step volume [m3] into average discharge [m3/s]."""
    volume_m3 = _validate_non_negative_scalar("volume_m3", volume_m3)
    dt_seconds = _validate_positive_scalar("dt_seconds", dt_seconds)

    return max(0.0, float(volume_m3 / dt_seconds))


def update_reservoir_storage(
    *,
    storage_prev_m3: int | float,
    inflow_m3s: int | float,
    controlled_release_m3s: int | float,
    pet_mm_dt: int | float,
    capacity_m3: int | float,
    dt_seconds: int | float,
    config: ReservoirStorageConfig,
) -> ReservoirStorageUpdateResult:
    """Update reservoir storage using a simple mass balance.

    Update order
    ------------
    1. Add inflow to the previous storage.
    2. Compute evaporation loss from the reservoir surface.
    3. Apply controlled release decided by operating rules.
    4. Spill any volume remaining above capacity.

    Simplified balance
    ------------------
    storage_final =
        storage_prev
        + inflow_volume
        - evaporation_loss
        - controlled_release_volume
        - spill_volume
    """
    if not isinstance(config, ReservoirStorageConfig):
        raise TypeError(f"'config' must be a ReservoirStorageConfig, got {type(config).__name__}")

    storage_prev_m3 = _validate_non_negative_scalar("storage_prev_m3", storage_prev_m3)
    inflow_m3s = _validate_non_negative_scalar("inflow_m3s", inflow_m3s)
    controlled_release_m3s = _validate_non_negative_scalar(
        "controlled_release_m3s",
        controlled_release_m3s,
    )
    pet_mm_dt = _validate_non_negative_scalar("pet_mm_dt", pet_mm_dt)
    capacity_m3 = _validate_positive_scalar("capacity_m3", capacity_m3)
    dt_seconds = _validate_positive_scalar("dt_seconds", dt_seconds)

    inflow_volume_m3 = convert_discharge_m3s_to_volume_m3(
        inflow_m3s,
        dt_seconds=dt_seconds,
    )
    storage_after_inflow_m3 = storage_prev_m3 + inflow_volume_m3

    surface_area_m2 = compute_reservoir_surface_area_m2(
        storage_after_inflow_m3,
        capacity_m3=capacity_m3,
        config=config,
    )

    evaporation_loss_m3 = convert_depth_mm_to_volume_m3(
        config.evaporation_factor * pet_mm_dt,
        surface_area_m2=surface_area_m2,
    )
    evaporation_loss_m3 = min(storage_after_inflow_m3, evaporation_loss_m3)

    storage_after_evaporation_m3 = max(0.0, storage_after_inflow_m3 - evaporation_loss_m3)

    requested_release_volume_m3 = convert_discharge_m3s_to_volume_m3(
        controlled_release_m3s,
        dt_seconds=dt_seconds,
    )
    controlled_release_volume_m3 = min(storage_after_evaporation_m3, requested_release_volume_m3)
    controlled_release_m3s_effective = convert_volume_m3_to_discharge_m3s(
        controlled_release_volume_m3,
        dt_seconds=dt_seconds,
    )

    storage_after_controlled_release_m3 = max(
        0.0,
        storage_after_evaporation_m3 - controlled_release_volume_m3,
    )

    spill_volume_m3 = max(0.0, storage_after_controlled_release_m3 - capacity_m3)
    spill_m3s = convert_volume_m3_to_discharge_m3s(
        spill_volume_m3,
        dt_seconds=dt_seconds,
    )

    storage_final_m3 = max(0.0, storage_after_controlled_release_m3 - spill_volume_m3)

    total_outflow_volume_m3 = controlled_release_volume_m3 + spill_volume_m3
    total_outflow_m3s = convert_volume_m3_to_discharge_m3s(
        total_outflow_volume_m3,
        dt_seconds=dt_seconds,
    )

    return ReservoirStorageUpdateResult(
        storage_before_step_m3=storage_prev_m3,
        inflow_m3s=inflow_m3s,
        inflow_volume_m3=inflow_volume_m3,
        storage_after_inflow_m3=storage_after_inflow_m3,
        surface_area_m2=surface_area_m2,
        evaporation_loss_m3=evaporation_loss_m3,
        storage_after_evaporation_m3=storage_after_evaporation_m3,
        controlled_release_m3s=controlled_release_m3s_effective,
        controlled_release_volume_m3=controlled_release_volume_m3,
        storage_after_controlled_release_m3=storage_after_controlled_release_m3,
        spill_m3s=spill_m3s,
        spill_volume_m3=spill_volume_m3,
        storage_m3=storage_final_m3,
        total_outflow_m3s=total_outflow_m3s,
        total_outflow_volume_m3=total_outflow_volume_m3,
    )
