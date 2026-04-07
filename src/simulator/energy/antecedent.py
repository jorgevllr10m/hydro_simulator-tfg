from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.core.types import FloatArray


def _validate_numeric_scalar(name: str, value: int | float) -> float:
    """Validate a numeric scalar and return it as float."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"'{name}' must be numeric, got {type(value).__name__}")
    return float(value)


def _validate_non_negative_scalar(name: str, value: int | float) -> float:
    """Validate a non-negative numeric scalar and return it as float."""
    numeric_value = _validate_numeric_scalar(name, value)
    if numeric_value < 0.0:
        raise ValueError(f"'{name}' must be >= 0, got {numeric_value}")
    return numeric_value


def _validate_fraction(name: str, value: int | float) -> float:
    """Validate a scalar fraction in [0, 1]."""
    numeric_value = _validate_numeric_scalar(name, value)
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"'{name}' must be within [0, 1], got {numeric_value}")
    return numeric_value


def _validate_spatial_float_array(name: str, value: FloatArray) -> None:
    """Validate a 2D NumPy float array."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"'{name}' must be a numpy.ndarray, got {type(value).__name__}")
    if value.ndim != 2:
        raise ValueError(f"'{name}' must be a 2D array with shape (ny, nx), got ndim={value.ndim}")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"'{name}' must have a floating dtype, got {value.dtype}")


@dataclass(frozen=True)
class AntecedentConfig:
    """Configuration of the simplified antecedent-water store.

    Notes
    -----
    - This is not yet a full hydrological soil model.
    - It is a lightweight water-availability memory used to limit ET.
    - Storage is expressed as an equivalent water depth in mm.
    """

    capacity_mm: float = 120.0
    """
    Maximum reserve capacity per cell.
    """

    initial_relative: float = 0.50
    """
    Initial state, but expressed as a fraction of the capacity. 0.50 means half capacity
    """

    wetting_efficiency: float = 0.85
    """
    Fraction of the rainfall that actually enters the reservoir
    """

    stress_exponent: float = 1.5
    """
    It is used to decide how the actual ET falls when the antecedent reserve dries up.
    large --> the system “suffers” from dryness sooner
    """

    def __post_init__(self) -> None:
        capacity_mm = _validate_non_negative_scalar("capacity_mm", self.capacity_mm)
        if capacity_mm <= 0.0:
            raise ValueError(f"'capacity_mm' must be > 0, got {capacity_mm}")

        _validate_fraction("initial_relative", self.initial_relative)
        _validate_fraction("wetting_efficiency", self.wetting_efficiency)

        stress_exponent = _validate_non_negative_scalar("stress_exponent", self.stress_exponent)
        if stress_exponent <= 0.0:
            raise ValueError(f"'stress_exponent' must be > 0, got {stress_exponent}")


@dataclass(frozen=True)
class AntecedentFields:
    """Antecedent-water diagnostics and ET losses for one simulation step."""

    effective_wetting_input_mm_dt: FloatArray
    """
    Effective rainfall that actually enters the warehouse.
    """

    available_water_before_et_mm: FloatArray
    """
    Water available just before applying evapotranspiration.
    """

    water_stress_factor: FloatArray
    """
    Factor between 0 and 1 that limits PET.
    - 1 --> no stress, AET ≈ PET
    - 0 --> total dryness, AET ≈ 0
    """

    aet_mm_dt: FloatArray
    """
    Effective step's evapotranspiration
    """

    antecedent_storage_mm: FloatArray
    """
    Final storage after rain and ET.
    """

    antecedent_relative: FloatArray
    """
    Final storage standardized between 0 and 1.
    """

    antecedent_overflow_mm_dt: FloatArray
    """
    Excess water that does not fit in the reservoir.
    """

    def __post_init__(self) -> None:
        array_fields = {
            "effective_wetting_input_mm_dt": self.effective_wetting_input_mm_dt,
            "available_water_before_et_mm": self.available_water_before_et_mm,
            "water_stress_factor": self.water_stress_factor,
            "aet_mm_dt": self.aet_mm_dt,
            "antecedent_storage_mm": self.antecedent_storage_mm,
            "antecedent_relative": self.antecedent_relative,
            "antecedent_overflow_mm_dt": self.antecedent_overflow_mm_dt,
        }
        for name, value in array_fields.items():
            _validate_spatial_float_array(name, value)

        spatial_shape = self.antecedent_storage_mm.shape
        for name, value in array_fields.items():
            if value.shape != spatial_shape:
                raise ValueError(f"'{name}' must have shape {spatial_shape}, got {value.shape}")

        bounded_fields = {
            "water_stress_factor": self.water_stress_factor,
            "antecedent_relative": self.antecedent_relative,
        }
        for name, value in bounded_fields.items():
            if np.any((value < 0.0) | (value > 1.0)):
                raise ValueError(f"'{name}' must remain within [0, 1]")

        non_negative_fields = {
            "effective_wetting_input_mm_dt": self.effective_wetting_input_mm_dt,
            "available_water_before_et_mm": self.available_water_before_et_mm,
            "aet_mm_dt": self.aet_mm_dt,
            "antecedent_storage_mm": self.antecedent_storage_mm,
            "antecedent_overflow_mm_dt": self.antecedent_overflow_mm_dt,
        }
        for name, value in non_negative_fields.items():
            if np.any(value < 0.0):
                raise ValueError(f"'{name}' must be >= 0 everywhere")


def build_initial_antecedent_storage_mm(
    shape: tuple[int, int],
    *,
    config: AntecedentConfig,
) -> FloatArray:
    """Return the initial antecedent-water storage field in mm."""
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError(f"'shape' must be a tuple[int, int], got {shape!r}")

    ny, nx = shape
    if not isinstance(ny, int) or not isinstance(nx, int) or ny <= 0 or nx <= 0:
        raise ValueError(f"'shape' must contain positive integers, got {shape!r}")

    if not isinstance(config, AntecedentConfig):
        raise TypeError(f"'config' must be an AntecedentConfig, got {type(config).__name__}")

    initial_storage_mm = config.capacity_mm * config.initial_relative
    return np.full(shape, initial_storage_mm, dtype=float)


def compute_effective_wetting_input_mm_dt(
    precipitation_mm_dt: FloatArray,
    *,
    wetting_efficiency: int | float,
) -> FloatArray:
    """Return the fraction of precipitation that recharges the antecedent store."""
    _validate_spatial_float_array("precipitation_mm_dt", precipitation_mm_dt)
    wetting_efficiency = _validate_fraction("wetting_efficiency", wetting_efficiency)

    precipitation_mm_dt = np.clip(precipitation_mm_dt.astype(float, copy=False), 0.0, None)
    effective_wetting_input_mm_dt = wetting_efficiency * precipitation_mm_dt

    return effective_wetting_input_mm_dt.astype(float, copy=False)


def compute_antecedent_relative(
    antecedent_storage_mm: FloatArray,
    *,
    capacity_mm: int | float,
) -> FloatArray:
    """Return antecedent storage normalized to [0, 1]."""
    _validate_spatial_float_array("antecedent_storage_mm", antecedent_storage_mm)

    capacity_mm = _validate_non_negative_scalar("capacity_mm", capacity_mm)
    if capacity_mm <= 0.0:
        raise ValueError(f"'capacity_mm' must be > 0, got {capacity_mm}")

    antecedent_storage_mm = np.clip(antecedent_storage_mm.astype(float, copy=False), 0.0, capacity_mm)
    antecedent_relative = antecedent_storage_mm / capacity_mm

    return np.clip(antecedent_relative, 0.0, 1.0).astype(float, copy=False)


def compute_water_stress_factor(
    antecedent_relative: FloatArray,
    *,
    stress_exponent: int | float,
) -> FloatArray:
    """Return an ET stress factor in [0, 1] from relative antecedent storage.

    Behavior
    --------
    - drier conditions -> lower factor
    - wetter conditions -> factor closer to 1
    """
    _validate_spatial_float_array("antecedent_relative", antecedent_relative)

    if np.any((antecedent_relative < 0.0) | (antecedent_relative > 1.0)):
        raise ValueError("'antecedent_relative' must remain within [0, 1]")

    stress_exponent = _validate_non_negative_scalar("stress_exponent", stress_exponent)
    if stress_exponent <= 0.0:
        raise ValueError(f"'stress_exponent' must be > 0, got {stress_exponent}")

    water_stress_factor = np.power(antecedent_relative.astype(float, copy=False), stress_exponent)
    return np.clip(water_stress_factor, 0.0, 1.0).astype(float, copy=False)


def compute_actual_evapotranspiration_mm_dt(
    pet_mm_dt: FloatArray,
    *,
    water_stress_factor: FloatArray,
) -> FloatArray:
    """Return actual evapotranspiration limited by water availability."""
    _validate_spatial_float_array("pet_mm_dt", pet_mm_dt)
    _validate_spatial_float_array("water_stress_factor", water_stress_factor)

    if pet_mm_dt.shape != water_stress_factor.shape:
        raise ValueError(
            f"'pet_mm_dt' and 'water_stress_factor' must have the same shape, got {pet_mm_dt.shape} and {water_stress_factor.shape}"
        )

    if np.any((water_stress_factor < 0.0) | (water_stress_factor > 1.0)):
        raise ValueError("'water_stress_factor' must remain within [0, 1]")

    pet_mm_dt = np.clip(pet_mm_dt.astype(float, copy=False), 0.0, None)
    aet_mm_dt = pet_mm_dt * water_stress_factor

    return np.clip(aet_mm_dt, 0.0, None).astype(float, copy=False)


def update_antecedent_store(
    *,
    antecedent_storage_prev_mm: FloatArray,
    precipitation_mm_dt: FloatArray,
    pet_mm_dt: FloatArray,
    config: AntecedentConfig,
) -> AntecedentFields:
    """Update the antecedent-water store and compute AET for one step.

    Update order
    ------------
    1. A fraction of current precipitation wets/recharges the store.
    2. ET stress is computed from the temporarily wetted storage.
    3. Actual ET is limited by that stress factor.
    4. Final storage is updated and clipped to [0, capacity].
    """
    _validate_spatial_float_array("antecedent_storage_prev_mm", antecedent_storage_prev_mm)
    _validate_spatial_float_array("precipitation_mm_dt", precipitation_mm_dt)
    _validate_spatial_float_array("pet_mm_dt", pet_mm_dt)

    if antecedent_storage_prev_mm.shape != precipitation_mm_dt.shape:
        raise ValueError(
            "'antecedent_storage_prev_mm' and 'precipitation_mm_dt' must have the same shape, "
            f"got {antecedent_storage_prev_mm.shape} and {precipitation_mm_dt.shape}"
        )
    if antecedent_storage_prev_mm.shape != pet_mm_dt.shape:
        raise ValueError(
            "'antecedent_storage_prev_mm' and 'pet_mm_dt' must have the same shape, "
            f"got {antecedent_storage_prev_mm.shape} and {pet_mm_dt.shape}"
        )

    if not isinstance(config, AntecedentConfig):
        raise TypeError(f"'config' must be an AntecedentConfig, got {type(config).__name__}")

    antecedent_storage_prev_mm = np.clip(
        antecedent_storage_prev_mm.astype(float, copy=False),
        0.0,
        config.capacity_mm,
    )
    precipitation_mm_dt = np.clip(precipitation_mm_dt.astype(float, copy=False), 0.0, None)
    pet_mm_dt = np.clip(pet_mm_dt.astype(float, copy=False), 0.0, None)

    effective_wetting_input_mm_dt = compute_effective_wetting_input_mm_dt(
        precipitation_mm_dt,
        wetting_efficiency=config.wetting_efficiency,
    )

    available_water_before_et_mm = antecedent_storage_prev_mm + effective_wetting_input_mm_dt

    antecedent_relative_before_et = compute_antecedent_relative(
        available_water_before_et_mm,
        capacity_mm=config.capacity_mm,
    )
    water_stress_factor = compute_water_stress_factor(
        antecedent_relative_before_et,
        stress_exponent=config.stress_exponent,
    )

    aet_mm_dt = compute_actual_evapotranspiration_mm_dt(
        pet_mm_dt,
        water_stress_factor=water_stress_factor,
    )
    aet_mm_dt = np.minimum(aet_mm_dt, available_water_before_et_mm)

    storage_after_et_before_clip_mm = available_water_before_et_mm - aet_mm_dt
    antecedent_overflow_mm_dt = np.clip(
        storage_after_et_before_clip_mm - config.capacity_mm,
        0.0,
        None,
    )
    antecedent_storage_mm = np.clip(
        storage_after_et_before_clip_mm,
        0.0,
        config.capacity_mm,
    )
    antecedent_relative = compute_antecedent_relative(
        antecedent_storage_mm,
        capacity_mm=config.capacity_mm,
    )

    return AntecedentFields(
        effective_wetting_input_mm_dt=effective_wetting_input_mm_dt,
        available_water_before_et_mm=available_water_before_et_mm,
        water_stress_factor=water_stress_factor,
        aet_mm_dt=aet_mm_dt.astype(float, copy=False),
        antecedent_storage_mm=antecedent_storage_mm.astype(float, copy=False),
        antecedent_relative=antecedent_relative.astype(float, copy=False),
        antecedent_overflow_mm_dt=antecedent_overflow_mm_dt.astype(float, copy=False),
    )
