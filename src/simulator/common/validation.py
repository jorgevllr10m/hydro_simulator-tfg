from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


def validate_numeric_scalar(name: str, value: int | float) -> float:
    """Validate a numeric scalar and return it as float."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"'{name}' must be numeric, got {type(value).__name__}")
    return float(value)


def validate_non_negative_scalar(name: str, value: int | float) -> float:
    """Validate a non-negative numeric scalar and return it as float."""
    numeric_value = validate_numeric_scalar(name, value)
    if numeric_value < 0.0:
        raise ValueError(f"'{name}' must be >= 0, got {numeric_value}")
    return numeric_value


def validate_positive_scalar(name: str, value: int | float) -> float:
    """Validate a strictly positive numeric scalar and return it as float."""
    numeric_value = validate_numeric_scalar(name, value)
    if numeric_value <= 0.0:
        raise ValueError(f"'{name}' must be > 0, got {numeric_value}")
    return numeric_value


def validate_fraction(name: str, value: int | float) -> float:
    """Validate a scalar fraction in [0, 1]."""
    numeric_value = validate_numeric_scalar(name, value)
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"'{name}' must be within [0, 1], got {numeric_value}")
    return numeric_value


def validate_int(name: str, value: int) -> int:
    """Validate that a value is an int."""
    if not isinstance(value, int):
        raise TypeError(f"'{name}' must be an int, got {type(value).__name__}")
    return value


def validate_non_negative_int(name: str, value: int) -> int:
    """Validate a non-negative integer."""
    value = validate_int(name, value)
    if value < 0:
        raise ValueError(f"'{name}' must be >= 0, got {value}")
    return value


def validate_positive_int(name: str, value: int) -> int:
    """Validate a strictly positive integer."""
    value = validate_int(name, value)
    if value <= 0:
        raise ValueError(f"'{name}' must be > 0, got {value}")
    return value


def validate_shape_2d(shape: tuple[int, int]) -> tuple[int, int]:
    """Validate a canonical spatial shape (ny, nx)."""
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError(f"'shape' must be a tuple[int, int], got {shape!r}")

    ny, nx = shape
    if not isinstance(ny, int) or not isinstance(nx, int):
        raise TypeError(f"'shape' must contain integers, got {shape!r}")
    if ny <= 0 or nx <= 0:
        raise ValueError(f"'shape' must contain positive integers, got {shape!r}")

    return (ny, nx)


def validate_float_array(name: str, value: FloatArray) -> None:
    """Validate that a value is a NumPy array with floating dtype."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"'{name}' must be a numpy.ndarray, got {type(value).__name__}")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"'{name}' must have a floating dtype, got {value.dtype}")


def validate_spatial_float_array(name: str, value: FloatArray) -> None:
    """Validate a 2D NumPy float array."""
    validate_float_array(name, value)
    if value.ndim != 2:
        raise ValueError(f"'{name}' must be a 2D array with shape (ny, nx), got ndim={value.ndim}")


def validate_vector_float_array(name: str, value: FloatArray) -> None:
    """Validate a 1D NumPy float array."""
    validate_float_array(name, value)
    if value.ndim != 1:
        raise ValueError(f"'{name}' must be a 1D array, got ndim={value.ndim}")


def validate_bool_array(name: str, value: BoolArray) -> None:
    """Validate that a value is a NumPy boolean array."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"'{name}' must be a numpy.ndarray, got {type(value).__name__}")
    if value.dtype != np.bool_:
        raise TypeError(f"'{name}' must have dtype bool, got {value.dtype}")


def validate_spatial_bool_array(name: str, value: BoolArray) -> None:
    """Validate a 2D NumPy boolean array."""
    validate_bool_array(name, value)
    if value.ndim != 2:
        raise ValueError(f"'{name}' must be a 2D boolean array with shape (ny, nx), got ndim={value.ndim}")


def validate_int_array(name: str, value: IntArray, *, ndim: int) -> None:
    """Validate a NumPy integer array with a fixed number of dimensions."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"'{name}' must be a numpy.ndarray, got {type(value).__name__}")
    if value.ndim != ndim:
        raise ValueError(f"'{name}' must be a {ndim}D array, got ndim={value.ndim}")
    if not np.issubdtype(value.dtype, np.integer):
        raise TypeError(f"'{name}' must have integer dtype, got {value.dtype}")


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value to the [lower, upper] interval."""
    return max(lower, min(upper, float(value)))


def clamp01(value: float) -> float:
    """Clamp a numeric value to the [0, 1] interval."""
    return clamp(value, 0.0, 1.0)


def validate_latitude_deg(latitude_deg: int | float) -> float:
    """Validate latitude in decimal degrees."""
    latitude_deg = validate_numeric_scalar("latitude_deg", latitude_deg)
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError(f"'latitude_deg' must be within [-90, 90], got {latitude_deg}")
    return latitude_deg
