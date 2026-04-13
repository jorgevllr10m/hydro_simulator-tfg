from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.common.validation import (
    validate_fraction as _validate_fraction,
)
from simulator.common.validation import (
    validate_non_negative_scalar as _validate_non_negative_scalar,
)
from simulator.common.validation import (
    validate_spatial_float_array as _validate_spatial_float_array,
)
from simulator.core.types import FloatArray


@dataclass(frozen=True)
class SoilConfig:
    """Configuration of the simplified per-cell soil bucket.

    Notes
    -----
    - Soil storage is expressed as equivalent water depth [mm].
    - Fluxes are expressed per simulation time step [mm/dt].
    - This module computes the official soil water balance and AET.
    """

    capacity_mm: float = 180.0
    """
    Maximum soil-water storage capacity per cell.
    """

    initial_relative: float = 0.50
    """
    Initial relative filling of the soil bucket in [0, 1].
    """

    max_infiltration_mm_dt: float = 18.0
    """
    Maximum infiltration capacity per time step under dry conditions.
    """

    infiltration_shape_exponent: float = 1.0
    """
    Controls how fast infiltration capacity decreases as soil saturation increases.
    - 1.0 -> linear decrease
    - >1.0 -> slower decrease at intermediate wetness
    - <1.0 -> earlier decrease
    """

    et_stress_exponent: float = 1.5
    """
    Controls how strongly AET decreases when the soil dries.
    """

    percolation_rate_mm_dt: float = 2.5
    """
    Maximum percolation/drainage rate per time step under very wet conditions.
    """

    percolation_activation_fraction: float = 0.70
    """
    Relative saturation threshold above which percolation begins.
    """

    def __post_init__(self) -> None:
        capacity_mm = _validate_non_negative_scalar("capacity_mm", self.capacity_mm)
        if capacity_mm <= 0.0:
            raise ValueError(f"'capacity_mm' must be > 0, got {capacity_mm}")

        _validate_fraction("initial_relative", self.initial_relative)

        max_infiltration_mm_dt = _validate_non_negative_scalar(
            "max_infiltration_mm_dt",
            self.max_infiltration_mm_dt,
        )
        if max_infiltration_mm_dt <= 0.0:
            raise ValueError(f"'max_infiltration_mm_dt' must be > 0, got {max_infiltration_mm_dt}")

        infiltration_shape_exponent = _validate_non_negative_scalar(
            "infiltration_shape_exponent",
            self.infiltration_shape_exponent,
        )
        if infiltration_shape_exponent <= 0.0:
            raise ValueError(f"'infiltration_shape_exponent' must be > 0, got {infiltration_shape_exponent}")

        et_stress_exponent = _validate_non_negative_scalar(
            "et_stress_exponent",
            self.et_stress_exponent,
        )
        if et_stress_exponent <= 0.0:
            raise ValueError(f"'et_stress_exponent' must be > 0, got {et_stress_exponent}")

        _validate_non_negative_scalar("percolation_rate_mm_dt", self.percolation_rate_mm_dt)
        _validate_fraction(
            "percolation_activation_fraction",
            self.percolation_activation_fraction,
        )


@dataclass(frozen=True)
class SoilStepFields:
    """Detailed soil-bucket diagnostics for one simulation step."""

    infiltration_capacity_mm_dt: FloatArray
    infiltration_mm_dt: FloatArray
    surface_excess_mm_dt: FloatArray

    available_water_before_et_mm: FloatArray
    soil_relative_saturation_before_et: FloatArray
    soil_water_stress_factor: FloatArray
    aet_mm_dt: FloatArray

    storage_after_et_mm: FloatArray
    percolation_mm_dt: FloatArray

    soil_moisture_mm: FloatArray
    soil_relative_saturation: FloatArray

    def __post_init__(self) -> None:
        array_fields = {
            "infiltration_capacity_mm_dt": self.infiltration_capacity_mm_dt,
            "infiltration_mm_dt": self.infiltration_mm_dt,
            "surface_excess_mm_dt": self.surface_excess_mm_dt,
            "available_water_before_et_mm": self.available_water_before_et_mm,
            "soil_relative_saturation_before_et": self.soil_relative_saturation_before_et,
            "soil_water_stress_factor": self.soil_water_stress_factor,
            "aet_mm_dt": self.aet_mm_dt,
            "storage_after_et_mm": self.storage_after_et_mm,
            "percolation_mm_dt": self.percolation_mm_dt,
            "soil_moisture_mm": self.soil_moisture_mm,
            "soil_relative_saturation": self.soil_relative_saturation,
        }

        for name, value in array_fields.items():
            _validate_spatial_float_array(name, value)

        spatial_shape = self.soil_moisture_mm.shape
        for name, value in array_fields.items():
            if value.shape != spatial_shape:
                raise ValueError(f"'{name}' must have shape {spatial_shape}, got {value.shape}")

        bounded_fields = {
            "soil_relative_saturation_before_et": self.soil_relative_saturation_before_et,
            "soil_water_stress_factor": self.soil_water_stress_factor,
            "soil_relative_saturation": self.soil_relative_saturation,
        }
        for name, value in bounded_fields.items():
            if np.any((value < 0.0) | (value > 1.0)):
                raise ValueError(f"'{name}' must remain within [0, 1]")

        non_negative_fields = {
            "infiltration_capacity_mm_dt": self.infiltration_capacity_mm_dt,
            "infiltration_mm_dt": self.infiltration_mm_dt,
            "surface_excess_mm_dt": self.surface_excess_mm_dt,
            "available_water_before_et_mm": self.available_water_before_et_mm,
            "aet_mm_dt": self.aet_mm_dt,
            "storage_after_et_mm": self.storage_after_et_mm,
            "percolation_mm_dt": self.percolation_mm_dt,
            "soil_moisture_mm": self.soil_moisture_mm,
        }
        for name, value in non_negative_fields.items():
            if np.any(value < 0.0):
                raise ValueError(f"'{name}' must be >= 0 everywhere")


def build_initial_soil_moisture_mm(
    shape: tuple[int, int],
    *,
    config: SoilConfig,
) -> FloatArray:
    """Return the initial soil-moisture field in mm."""
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError(f"'shape' must be a tuple[int, int], got {shape!r}")

    ny, nx = shape
    if not isinstance(ny, int) or not isinstance(nx, int) or ny <= 0 or nx <= 0:
        raise ValueError(f"'shape' must contain positive integers, got {shape!r}")

    if not isinstance(config, SoilConfig):
        raise TypeError(f"'config' must be a SoilConfig, got {type(config).__name__}")

    initial_soil_moisture_mm = config.capacity_mm * config.initial_relative
    return np.full(shape, initial_soil_moisture_mm, dtype=float)


def compute_soil_relative_saturation(
    soil_moisture_mm: FloatArray,
    *,
    capacity_mm: int | float,
) -> FloatArray:
    """Return soil moisture normalized to [0, 1]."""
    _validate_spatial_float_array("soil_moisture_mm", soil_moisture_mm)

    capacity_mm = _validate_non_negative_scalar("capacity_mm", capacity_mm)
    if capacity_mm <= 0.0:
        raise ValueError(f"'capacity_mm' must be > 0, got {capacity_mm}")

    soil_moisture_mm = np.clip(soil_moisture_mm.astype(float, copy=False), 0.0, capacity_mm)
    soil_relative_saturation = soil_moisture_mm / capacity_mm

    return np.clip(soil_relative_saturation, 0.0, 1.0).astype(float, copy=False)


def compute_infiltration_capacity_mm_dt(
    soil_relative_saturation: FloatArray,
    *,
    max_infiltration_mm_dt: int | float,
    infiltration_shape_exponent: int | float,
) -> FloatArray:
    """Return infiltration capacity per step as a function of soil wetness.

    Behavior
    --------
    - dry soil -> infiltration capacity close to max_infiltration_mm_dt
    - saturated soil -> infiltration capacity close to 0
    """
    _validate_spatial_float_array("soil_relative_saturation", soil_relative_saturation)

    if np.any((soil_relative_saturation < 0.0) | (soil_relative_saturation > 1.0)):
        raise ValueError("'soil_relative_saturation' must remain within [0, 1]")

    max_infiltration_mm_dt = _validate_non_negative_scalar(
        "max_infiltration_mm_dt",
        max_infiltration_mm_dt,
    )
    infiltration_shape_exponent = _validate_non_negative_scalar(
        "infiltration_shape_exponent",
        infiltration_shape_exponent,
    )
    if infiltration_shape_exponent <= 0.0:
        raise ValueError(f"'infiltration_shape_exponent' must be > 0, got {infiltration_shape_exponent}")

    dryness_factor = np.power(
        1.0 - soil_relative_saturation.astype(float, copy=False),
        infiltration_shape_exponent,
    )
    infiltration_capacity_mm_dt = max_infiltration_mm_dt * dryness_factor

    return np.clip(infiltration_capacity_mm_dt, 0.0, None).astype(float, copy=False)


def compute_soil_water_stress_factor(
    soil_relative_saturation: FloatArray,
    *,
    et_stress_exponent: int | float,
) -> FloatArray:
    """Return an evapotranspiration stress factor in [0, 1].

    Behavior
    --------
    - dry soil -> lower factor
    - wet soil -> factor closer to 1
    """
    _validate_spatial_float_array("soil_relative_saturation", soil_relative_saturation)

    if np.any((soil_relative_saturation < 0.0) | (soil_relative_saturation > 1.0)):
        raise ValueError("'soil_relative_saturation' must remain within [0, 1]")

    et_stress_exponent = _validate_non_negative_scalar(
        "et_stress_exponent",
        et_stress_exponent,
    )
    if et_stress_exponent <= 0.0:
        raise ValueError(f"'et_stress_exponent' must be > 0, got {et_stress_exponent}")

    soil_water_stress_factor = np.power(
        soil_relative_saturation.astype(float, copy=False),
        et_stress_exponent,
    )
    return np.clip(soil_water_stress_factor, 0.0, 1.0).astype(float, copy=False)


def compute_aet_mm_dt(
    pet_mm_dt: FloatArray,
    *,
    soil_water_stress_factor: FloatArray,
) -> FloatArray:
    """Return official soil AET limited by soil-water stress."""
    _validate_spatial_float_array("pet_mm_dt", pet_mm_dt)
    _validate_spatial_float_array("soil_water_stress_factor", soil_water_stress_factor)

    if pet_mm_dt.shape != soil_water_stress_factor.shape:
        raise ValueError(
            "'pet_mm_dt' and 'soil_water_stress_factor' must have the same shape, "
            f"got {pet_mm_dt.shape} and {soil_water_stress_factor.shape}"
        )

    if np.any((soil_water_stress_factor < 0.0) | (soil_water_stress_factor > 1.0)):
        raise ValueError("'soil_water_stress_factor' must remain within [0, 1]")

    pet_mm_dt = np.clip(pet_mm_dt.astype(float, copy=False), 0.0, None)
    aet_mm_dt = pet_mm_dt * soil_water_stress_factor

    return np.clip(aet_mm_dt, 0.0, None).astype(float, copy=False)


def compute_percolation_mm_dt(
    soil_moisture_mm: FloatArray,
    *,
    capacity_mm: int | float,
    percolation_rate_mm_dt: int | float,
    percolation_activation_fraction: int | float,
) -> FloatArray:
    """Return percolation/drainage per step from wet soil states.

    Behavior
    --------
    - below activation threshold -> no percolation
    - above threshold -> percolation increases smoothly with saturation
    - at full saturation -> percolation approaches percolation_rate_mm_dt
    """
    _validate_spatial_float_array("soil_moisture_mm", soil_moisture_mm)

    capacity_mm = _validate_non_negative_scalar("capacity_mm", capacity_mm)
    if capacity_mm <= 0.0:
        raise ValueError(f"'capacity_mm' must be > 0, got {capacity_mm}")

    percolation_rate_mm_dt = _validate_non_negative_scalar(
        "percolation_rate_mm_dt",
        percolation_rate_mm_dt,
    )
    percolation_activation_fraction = _validate_fraction(
        "percolation_activation_fraction",
        percolation_activation_fraction,
    )

    soil_relative_saturation = compute_soil_relative_saturation(
        soil_moisture_mm,
        capacity_mm=capacity_mm,
    )

    denominator = max(1e-12, 1.0 - percolation_activation_fraction)
    wetness_excess_fraction = np.clip(
        (soil_relative_saturation - percolation_activation_fraction) / denominator,
        0.0,
        1.0,
    )

    percolation_mm_dt = percolation_rate_mm_dt * wetness_excess_fraction
    percolation_mm_dt = np.minimum(percolation_mm_dt, soil_moisture_mm)

    return np.clip(percolation_mm_dt, 0.0, None).astype(float, copy=False)


def update_soil_bucket(
    *,
    soil_moisture_prev_mm: FloatArray,
    precipitation_mm_dt: FloatArray,
    pet_mm_dt: FloatArray,
    config: SoilConfig,
) -> SoilStepFields:
    """Update the per-cell soil bucket for one simulation step.

    Update order
    ------------
    1. Compute infiltration capacity from previous soil wetness.
    2. Partition precipitation into infiltration and direct surface excess.
    3. Update temporary soil storage with infiltrated water.
    4. Compute AET from PET limited by soil-water stress.
    5. Apply percolation/drainage from wet soil states.
    6. Return final soil moisture and diagnostics.

    Balance
    -------
    soil_final = soil_prev + infiltration - aet - percolation
    """
    _validate_spatial_float_array("soil_moisture_prev_mm", soil_moisture_prev_mm)
    _validate_spatial_float_array("precipitation_mm_dt", precipitation_mm_dt)
    _validate_spatial_float_array("pet_mm_dt", pet_mm_dt)

    if soil_moisture_prev_mm.shape != precipitation_mm_dt.shape:
        raise ValueError(
            "'soil_moisture_prev_mm' and 'precipitation_mm_dt' must have the same shape, "
            f"got {soil_moisture_prev_mm.shape} and {precipitation_mm_dt.shape}"
        )
    if soil_moisture_prev_mm.shape != pet_mm_dt.shape:
        raise ValueError(
            "'soil_moisture_prev_mm' and 'pet_mm_dt' must have the same shape, "
            f"got {soil_moisture_prev_mm.shape} and {pet_mm_dt.shape}"
        )

    if not isinstance(config, SoilConfig):
        raise TypeError(f"'config' must be a SoilConfig, got {type(config).__name__}")

    soil_moisture_prev_mm = np.clip(
        soil_moisture_prev_mm.astype(float, copy=False),
        0.0,
        config.capacity_mm,
    )
    precipitation_mm_dt = np.clip(precipitation_mm_dt.astype(float, copy=False), 0.0, None)
    pet_mm_dt = np.clip(pet_mm_dt.astype(float, copy=False), 0.0, None)

    soil_relative_saturation_prev = compute_soil_relative_saturation(
        soil_moisture_prev_mm,
        capacity_mm=config.capacity_mm,
    )

    infiltration_capacity_mm_dt = compute_infiltration_capacity_mm_dt(
        soil_relative_saturation_prev,
        max_infiltration_mm_dt=config.max_infiltration_mm_dt,
        infiltration_shape_exponent=config.infiltration_shape_exponent,
    )

    storage_room_before_input_mm = np.clip(
        config.capacity_mm - soil_moisture_prev_mm,
        0.0,
        None,
    )

    infiltration_mm_dt = np.minimum(
        precipitation_mm_dt,
        np.minimum(infiltration_capacity_mm_dt, storage_room_before_input_mm),
    )

    surface_excess_mm_dt = np.clip(
        precipitation_mm_dt - infiltration_mm_dt,
        0.0,
        None,
    )

    available_water_before_et_mm = soil_moisture_prev_mm + infiltration_mm_dt

    soil_relative_saturation_before_et = compute_soil_relative_saturation(
        available_water_before_et_mm,
        capacity_mm=config.capacity_mm,
    )

    soil_water_stress_factor = compute_soil_water_stress_factor(
        soil_relative_saturation_before_et,
        et_stress_exponent=config.et_stress_exponent,
    )

    aet_mm_dt = compute_aet_mm_dt(
        pet_mm_dt,
        soil_water_stress_factor=soil_water_stress_factor,
    )
    aet_mm_dt = np.minimum(aet_mm_dt, available_water_before_et_mm)

    storage_after_et_mm = np.clip(
        available_water_before_et_mm - aet_mm_dt,
        0.0,
        config.capacity_mm,
    )

    percolation_mm_dt = compute_percolation_mm_dt(
        storage_after_et_mm,
        capacity_mm=config.capacity_mm,
        percolation_rate_mm_dt=config.percolation_rate_mm_dt,
        percolation_activation_fraction=config.percolation_activation_fraction,
    )
    percolation_mm_dt = np.minimum(percolation_mm_dt, storage_after_et_mm)

    soil_moisture_mm = np.clip(
        storage_after_et_mm - percolation_mm_dt,
        0.0,
        config.capacity_mm,
    )

    soil_relative_saturation = compute_soil_relative_saturation(
        soil_moisture_mm,
        capacity_mm=config.capacity_mm,
    )

    return SoilStepFields(
        infiltration_capacity_mm_dt=infiltration_capacity_mm_dt.astype(float, copy=False),
        infiltration_mm_dt=infiltration_mm_dt.astype(float, copy=False),
        surface_excess_mm_dt=surface_excess_mm_dt.astype(float, copy=False),
        available_water_before_et_mm=available_water_before_et_mm.astype(float, copy=False),
        soil_relative_saturation_before_et=soil_relative_saturation_before_et.astype(
            float,
            copy=False,
        ),
        soil_water_stress_factor=soil_water_stress_factor.astype(float, copy=False),
        aet_mm_dt=aet_mm_dt.astype(float, copy=False),
        storage_after_et_mm=storage_after_et_mm.astype(float, copy=False),
        percolation_mm_dt=percolation_mm_dt.astype(float, copy=False),
        soil_moisture_mm=soil_moisture_mm.astype(float, copy=False),
        soil_relative_saturation=soil_relative_saturation.astype(float, copy=False),
    )
