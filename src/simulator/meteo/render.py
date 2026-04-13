from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from simulator.core.types import BoolArray, FloatArray, SimulationDomain
from simulator.meteo.lifecycle import (
    StormLifecycleConfig,
    compute_current_axes_m,
    compute_current_intensity_mmph,
)
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


@dataclass(frozen=True)
class StormRenderConfig:
    """Configuration of storm rasterization.

    Parameters
    ----------
    footprint_sigma_cutoff
        Maximum rendered distance in Gaussian-sigma units. Larger values
        capture more of the storm tail but increase computation.
    storm_mask_threshold_mmph
        Minimum storm rainfall rate used to flag a cell as storm-affected.
    """

    # * Outside of 3σ the contribution is already very small
    footprint_sigma_cutoff: float = 3.0
    storm_mask_threshold_mmph: float = 0.1  # minimum rainfall to be taken into account

    def __post_init__(self) -> None:
        _validate_positive_float("footprint_sigma_cutoff", self.footprint_sigma_cutoff)
        _validate_non_negative_float("storm_mask_threshold_mmph", self.storm_mask_threshold_mmph)


def _compute_storm_bbox_m(
    storm: StormCell,
    *,
    current_major_axis_m: float,
    current_minor_axis_m: float,
    sigma_cutoff: float,
) -> tuple[float, float, float, float]:
    """Return a conservative axis-aligned bounding box in world coordinates.

    The storm is rendered as a rotated elliptical Gaussian. This function
    computes a Cartesian bounding box large enough to contain the ellipse
    up to `sigma_cutoff`.
    """
    theta = storm.orientation_rad  # angle
    cos_theta = math.cos(theta)  # x axis
    sin_theta = math.sin(theta)  # y axis

    half_width_x_m = sigma_cutoff * math.sqrt((current_major_axis_m * cos_theta) ** 2 + (current_minor_axis_m * sin_theta) ** 2)
    half_width_y_m = sigma_cutoff * math.sqrt((current_major_axis_m * sin_theta) ** 2 + (current_minor_axis_m * cos_theta) ** 2)

    # * box size
    min_x_m = storm.center_x_m - half_width_x_m
    max_x_m = storm.center_x_m + half_width_x_m
    min_y_m = storm.center_y_m - half_width_y_m
    max_y_m = storm.center_y_m + half_width_y_m

    return (min_x_m, max_x_m, min_y_m, max_y_m)


def _compute_bbox_indices(
    domain: SimulationDomain,
    *,
    min_x_m: float,
    max_x_m: float,
    min_y_m: float,
    max_y_m: float,
) -> tuple[int, int, int, int]:
    """Convert a world-coordinate bounding box into grid index ranges.

    Returned indices follow Python slicing convention:
    [y_start:y_end, x_start:x_end]
    """
    x_coords = domain.spatial.x_coords
    y_coords = domain.spatial.y_coords

    x_start = int(np.searchsorted(x_coords, min_x_m, side="left"))
    x_end = int(np.searchsorted(x_coords, max_x_m, side="right"))
    y_start = int(np.searchsorted(y_coords, min_y_m, side="left"))
    y_end = int(np.searchsorted(y_coords, max_y_m, side="right"))

    ny, nx = domain.shape

    x_start = max(0, min(nx, x_start))
    x_end = max(0, min(nx, x_end))
    y_start = max(0, min(ny, y_start))
    y_end = max(0, min(ny, y_end))

    return (y_start, y_end, x_start, x_end)


def render_storm_mmph(
    domain: SimulationDomain,
    storm: StormCell,
    lifecycle: StormLifecycleConfig,
    render_config: StormRenderConfig,
) -> tuple[FloatArray, BoolArray]:
    """Render one storm as a 2D rainfall-rate field and storm mask.

    Returns
    -------
    precipitation_mmph
        2D rainfall rate contributed by this single storm in mm/h.
    storm_mask
        2D boolean mask (0/1) marking cells significantly affected by the storm.
    """
    ny, nx = domain.shape
    precipitation_mmph = np.zeros((ny, nx), dtype=float)
    storm_mask = np.zeros((ny, nx), dtype=bool)

    if not storm.is_alive:
        return precipitation_mmph, storm_mask

    current_intensity_mmph = compute_current_intensity_mmph(storm, lifecycle)
    if current_intensity_mmph <= 0.0:
        return precipitation_mmph, storm_mask

    current_major_axis_m, current_minor_axis_m = compute_current_axes_m(storm, lifecycle)

    min_x_m, max_x_m, min_y_m, max_y_m = _compute_storm_bbox_m(
        storm,
        current_major_axis_m=current_major_axis_m,
        current_minor_axis_m=current_minor_axis_m,
        sigma_cutoff=render_config.footprint_sigma_cutoff,
    )

    y_start, y_end, x_start, x_end = _compute_bbox_indices(
        domain,
        min_x_m=min_x_m,
        max_x_m=max_x_m,
        min_y_m=min_y_m,
        max_y_m=max_y_m,
    )

    if x_start >= x_end or y_start >= y_end:
        return precipitation_mmph, storm_mask

    x_local_m = domain.spatial.x_coords[x_start:x_end]
    y_local_m = domain.spatial.y_coords[y_start:y_end]

    x_grid_m, y_grid_m = np.meshgrid(x_local_m, y_local_m)

    dx_m = x_grid_m - storm.center_x_m
    dy_m = y_grid_m - storm.center_y_m

    cos_theta = math.cos(storm.orientation_rad)
    sin_theta = math.sin(storm.orientation_rad)

    x_rot_m = cos_theta * dx_m + sin_theta * dy_m
    y_rot_m = -sin_theta * dx_m + cos_theta * dy_m

    exponent = -0.5 * ((x_rot_m / current_major_axis_m) ** 2 + (y_rot_m / current_minor_axis_m) ** 2)

    local_precipitation_mmph = current_intensity_mmph * np.exp(exponent)

    precipitation_mmph[y_start:y_end, x_start:x_end] = local_precipitation_mmph
    storm_mask[y_start:y_end, x_start:x_end] = local_precipitation_mmph >= render_config.storm_mask_threshold_mmph

    return precipitation_mmph, storm_mask


def render_storms_mmph(
    domain: SimulationDomain,
    storms: list[StormCell],
    lifecycle: StormLifecycleConfig,
    render_config: StormRenderConfig,
) -> tuple[FloatArray, BoolArray]:
    """Render all active storms into a total rainfall-rate field and storm mask."""
    ny, nx = domain.shape
    total_precipitation_mmph = np.zeros((ny, nx), dtype=float)
    total_storm_mask = np.zeros((ny, nx), dtype=bool)

    for storm in storms:
        storm_precipitation_mmph, storm_mask = render_storm_mmph(
            domain=domain,
            storm=storm,
            lifecycle=lifecycle,
            render_config=render_config,
        )
        total_precipitation_mmph += storm_precipitation_mmph
        total_storm_mask = np.maximum(total_storm_mask, storm_mask)

    return total_precipitation_mmph, total_storm_mask


def convert_precipitation_rate_to_step_depth(
    precipitation_mmph: FloatArray,
    *,
    dt_seconds: int | float,
) -> FloatArray:
    """Convert rainfall rate from mm/h to step depth mm/dt."""
    dt_seconds = _validate_positive_float("dt_seconds", dt_seconds)
    return precipitation_mmph * (dt_seconds / 3600.0)


def render_storms_to_step_fields(
    domain: SimulationDomain,
    storms: list[StormCell],
    lifecycle: StormLifecycleConfig,
    render_config: StormRenderConfig,
) -> tuple[FloatArray, BoolArray]:
    """Render all storms and return the official step outputs.

    Returns
    -------
    precipitation_mm_dt
        Total storm precipitation depth for the current simulation step.
    storm_mask
        Bool 2D mask (0/1) of storm-affected cells.
    """
    precipitation_mmph, storm_mask = render_storms_mmph(
        domain=domain,
        storms=storms,
        lifecycle=lifecycle,
        render_config=render_config,
    )

    precipitation_mm_dt = convert_precipitation_rate_to_step_depth(
        precipitation_mmph,
        dt_seconds=domain.time.dt_seconds,
    )

    return precipitation_mm_dt, storm_mask
