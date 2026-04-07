from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.core.types import FloatArray
from simulator.energy.solar import SolarGeometry


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


@dataclass(frozen=True)
class SolarRadiationConfig:
    """Configuration of the simplified shortwave-radiation model.

    Notes
    -----
    - This phase uses only shortwave radiation.
    - Net radiation is approximated as net shortwave radiation.
    - Cloud attenuation is represented with a precipitation-based proxy.
    """

    # * FAO parameters
    solar_constant_w_m2: float = 1361.0
    """
    It represents the average irradiance at the top of the atmosphere when the rays arrive perpendicularly.
    """
    clear_sky_transmissivity: float = 0.72
    """
    It represents what fraction of the radiation from above the atmosphere
    manages to pass through the atmosphere on a clear day.
    """

    albedo: float = 0.23
    """Fraction reflected by the surface."""

    min_cloud_factor: float = 0.25
    """
    It represents what fraction of the radiation from above the atmosphere
    manages to pass through the atmosphere on a cloudy day.
    """
    # * -----------

    precip_cloud_sensitivity: float = 0.18
    """
    To control how much radiation decreases as precipitation increases
    """

    def __post_init__(self) -> None:
        solar_constant_w_m2 = _validate_non_negative_scalar(
            "solar_constant_w_m2",
            self.solar_constant_w_m2,
        )
        if solar_constant_w_m2 <= 0.0:
            raise ValueError(f"'solar_constant_w_m2' must be > 0, got {solar_constant_w_m2}")

        _validate_fraction("clear_sky_transmissivity", self.clear_sky_transmissivity)
        _validate_fraction("albedo", self.albedo)
        _validate_fraction("min_cloud_factor", self.min_cloud_factor)
        _validate_non_negative_scalar("precip_cloud_sensitivity", self.precip_cloud_sensitivity)


@dataclass(frozen=True)
class RadiationFields:
    """Shortwave-radiation diagnostics for one simulation step."""

    toa_shortwave_w_m2: float
    """
    Shortwave radiation in the upper atmosphere above a horizontal surface.
    """
    clear_sky_shortwave_w_m2: float
    """
    What would reach the surface under clear skies, after a fixed transmittance.
    """

    cloud_factor: FloatArray
    """
    2D field between 0 and 1 that attenuates radiation due to cloudiness.
    """
    shortwave_in_w_m2: FloatArray
    """
    Incident shortwave radiation actually received at the surface.
    """
    net_shortwave_w_m2: FloatArray
    """
    Part absorbed by the surface after discounting reflection.
    """
    net_radiation_mj_m2_dt: FloatArray
    """
    Same as net_shortwave_w_m2 but integrated over time and in units useful for PET.
    """

    def __post_init__(self) -> None:
        scalar_fields = {
            "toa_shortwave_w_m2": self.toa_shortwave_w_m2,
            "clear_sky_shortwave_w_m2": self.clear_sky_shortwave_w_m2,
        }
        for name, value in scalar_fields.items():
            _validate_non_negative_scalar(name, value)

        array_fields = {
            "cloud_factor": self.cloud_factor,
            "shortwave_in_w_m2": self.shortwave_in_w_m2,
            "net_shortwave_w_m2": self.net_shortwave_w_m2,
            "net_radiation_mj_m2_dt": self.net_radiation_mj_m2_dt,
        }
        for name, value in array_fields.items():
            _validate_spatial_float_array(name, value)

        spatial_shape = self.cloud_factor.shape
        for name, value in array_fields.items():
            if value.shape != spatial_shape:
                raise ValueError(f"'{name}' must have shape {spatial_shape}, got {value.shape}")

        if np.any((self.cloud_factor < 0.0) | (self.cloud_factor > 1.0)):
            raise ValueError("'cloud_factor' must remain within [0, 1]")

        if np.any(self.shortwave_in_w_m2 < 0.0):
            raise ValueError("'shortwave_in_w_m2' must be >= 0 everywhere")

        if np.any(self.net_shortwave_w_m2 < 0.0):
            raise ValueError("'net_shortwave_w_m2' must be >= 0 everywhere")

        if np.any(self.net_radiation_mj_m2_dt < 0.0):
            raise ValueError("'net_radiation_mj_m2_dt' must be >= 0 everywhere")


def _validate_spatial_float_array(name: str, value: FloatArray) -> None:
    """Validate a 2D NumPy float array."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"'{name}' must be a numpy.ndarray, got {type(value).__name__}")
    if value.ndim != 2:
        raise ValueError(f"'{name}' must be a 2D array with shape (ny, nx), got ndim={value.ndim}")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"'{name}' must have a floating dtype, got {value.dtype}")


def _broadcast_scalar_to_shape(
    value: int | float,
    shape: tuple[int, int],
) -> FloatArray:
    """Return a 2D float field filled with a scalar value."""
    scalar_value = _validate_numeric_scalar("value", value)
    return np.full(shape, scalar_value, dtype=float)


def compute_toa_shortwave_w_m2(
    solar_geometry: SolarGeometry,
    *,
    solar_constant_w_m2: int | float,
) -> float:
    """Return top-of-atmosphere shortwave radiation on a horizontal surface.

    Instantaneous simplified form:
    G_toa = G_sc * d_r * cos(theta_z)
    """
    if not isinstance(solar_geometry, SolarGeometry):
        raise TypeError(f"'solar_geometry' must be a SolarGeometry, got {type(solar_geometry).__name__}")

    solar_constant_w_m2 = _validate_non_negative_scalar(
        "solar_constant_w_m2",
        solar_constant_w_m2,
    )
    if solar_constant_w_m2 <= 0.0:
        raise ValueError(f"'solar_constant_w_m2' must be > 0, got {solar_constant_w_m2}")

    return solar_constant_w_m2 * solar_geometry.inverse_earth_sun_distance_factor * solar_geometry.cos_zenith


def compute_clear_sky_shortwave_w_m2(
    toa_shortwave_w_m2: int | float,
    *,
    clear_sky_transmissivity: int | float,
) -> float:
    """Return simplified clear-sky shortwave radiation at the surface."""
    toa_shortwave_w_m2 = _validate_non_negative_scalar(
        "toa_shortwave_w_m2",
        toa_shortwave_w_m2,
    )
    clear_sky_transmissivity = _validate_fraction(
        "clear_sky_transmissivity",
        clear_sky_transmissivity,
    )

    return toa_shortwave_w_m2 * clear_sky_transmissivity


def compute_cloud_factor(
    precipitation_mm_dt: FloatArray,
    *,
    min_cloud_factor: int | float,
    precip_cloud_sensitivity: int | float,
) -> FloatArray:
    """Return a cloud-attenuation factor in [min_cloud_factor, 1].

    The proxy is intentionally simple:
    - no precipitation -> factor near 1
    - stronger precipitation -> lower incoming shortwave radiation
    """
    _validate_spatial_float_array("precipitation_mm_dt", precipitation_mm_dt)

    min_cloud_factor = _validate_fraction("min_cloud_factor", min_cloud_factor)
    precip_cloud_sensitivity = _validate_non_negative_scalar(
        "precip_cloud_sensitivity",
        precip_cloud_sensitivity,
    )

    precipitation_mm_dt = np.clip(precipitation_mm_dt.astype(float, copy=False), 0.0, None)

    cloud_factor = np.exp(-precip_cloud_sensitivity * precipitation_mm_dt)
    cloud_factor = np.clip(cloud_factor, min_cloud_factor, 1.0)

    return cloud_factor.astype(float, copy=False)


def compute_shortwave_in_w_m2(
    clear_sky_shortwave_w_m2: int | float,
    cloud_factor: FloatArray,
) -> FloatArray:
    """Return incident shortwave radiation at the surface."""
    clear_sky_shortwave_w_m2 = _validate_non_negative_scalar(
        "clear_sky_shortwave_w_m2",
        clear_sky_shortwave_w_m2,
    )
    _validate_spatial_float_array("cloud_factor", cloud_factor)

    if np.any((cloud_factor < 0.0) | (cloud_factor > 1.0)):
        raise ValueError("'cloud_factor' must remain within [0, 1]")

    return clear_sky_shortwave_w_m2 * cloud_factor


def compute_net_shortwave_w_m2(
    shortwave_in_w_m2: FloatArray,
    *,
    albedo: int | float,
) -> FloatArray:
    """Return net shortwave radiation after surface reflection."""
    _validate_spatial_float_array("shortwave_in_w_m2", shortwave_in_w_m2)
    albedo = _validate_fraction("albedo", albedo)

    net_shortwave = (1.0 - albedo) * shortwave_in_w_m2
    return np.clip(net_shortwave, 0.0, None).astype(float, copy=False)


def convert_radiation_w_m2_to_mj_m2_dt(
    radiation_w_m2: FloatArray,
    *,
    dt_seconds: int | float,
) -> FloatArray:
    """Convert radiation from W/m² to MJ/m² over one simulation step."""
    _validate_spatial_float_array("radiation_w_m2", radiation_w_m2)

    dt_seconds = _validate_non_negative_scalar("dt_seconds", dt_seconds)
    if dt_seconds <= 0.0:
        raise ValueError(f"'dt_seconds' must be > 0, got {dt_seconds}")

    radiation_mj_m2_dt = radiation_w_m2 * dt_seconds / 1_000_000.0
    return np.clip(radiation_mj_m2_dt, 0.0, None).astype(float, copy=False)


def compute_radiation_fields(
    *,
    precipitation_mm_dt: FloatArray,
    solar_geometry: SolarGeometry,
    dt_seconds: int | float,
    config: SolarRadiationConfig,
) -> RadiationFields:
    """Compute simplified shortwave-radiation fields for one step."""
    if not isinstance(solar_geometry, SolarGeometry):
        raise TypeError(f"'solar_geometry' must be a SolarGeometry, got {type(solar_geometry).__name__}")
    if not isinstance(config, SolarRadiationConfig):
        raise TypeError(f"'config' must be a SolarRadiationConfig, got {type(config).__name__}")

    _validate_spatial_float_array("precipitation_mm_dt", precipitation_mm_dt)

    toa_shortwave_w_m2 = compute_toa_shortwave_w_m2(
        solar_geometry,
        solar_constant_w_m2=config.solar_constant_w_m2,
    )
    clear_sky_shortwave_w_m2 = compute_clear_sky_shortwave_w_m2(
        toa_shortwave_w_m2,
        clear_sky_transmissivity=config.clear_sky_transmissivity,
    )

    cloud_factor = compute_cloud_factor(
        precipitation_mm_dt,
        min_cloud_factor=config.min_cloud_factor,
        precip_cloud_sensitivity=config.precip_cloud_sensitivity,
    )
    shortwave_in_w_m2 = compute_shortwave_in_w_m2(
        clear_sky_shortwave_w_m2=clear_sky_shortwave_w_m2,
        cloud_factor=cloud_factor,
    )
    net_shortwave_w_m2 = compute_net_shortwave_w_m2(
        shortwave_in_w_m2,
        albedo=config.albedo,
    )
    net_radiation_mj_m2_dt = convert_radiation_w_m2_to_mj_m2_dt(
        net_shortwave_w_m2,
        dt_seconds=dt_seconds,
    )

    return RadiationFields(
        toa_shortwave_w_m2=toa_shortwave_w_m2,
        clear_sky_shortwave_w_m2=clear_sky_shortwave_w_m2,
        cloud_factor=cloud_factor,
        shortwave_in_w_m2=shortwave_in_w_m2,
        net_shortwave_w_m2=net_shortwave_w_m2,
        net_radiation_mj_m2_dt=net_radiation_mj_m2_dt,
    )
