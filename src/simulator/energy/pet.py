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


def _validate_positive_scalar(name: str, value: int | float) -> float:
    """Validate a strictly positive numeric scalar and return it as float."""
    numeric_value = _validate_numeric_scalar(name, value)
    if numeric_value <= 0.0:
        raise ValueError(f"'{name}' must be > 0, got {numeric_value}")
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
class PETConfig:
    """Configuration of the simplified Priestley-Taylor PET model.

    Notes
    -----
    - PET is computed from net radiation and air temperature.
    - This phase uses a simplified Priestley-Taylor formulation.
    - The aerodynamic term is not modeled explicitly.
    """

    priestley_taylor_alpha: float = 1.26
    """
    Priestley-Taylor factor. (alpha)
    """
    # Equilibrium evaporation is a purely energetic estimate.
    # Priestley-Taylor multiplies it by alpha to better approximate the actual PET
    # under broadly humid conditions.

    psychrometric_constant_kpa_c: float = 0.066
    """
    (Gamma)
    If a greater fraction of the energy goes to heat the air, less is left to evaporate water;
    If a greater fraction goes to evaporate water, evapotranspiration increases.
    """

    latent_heat_mj_kg: float = 2.45
    """
    Latent vaporization heat. It represents how much energy is needed to evaporate a certain mass of water. (Lambda)
    """

    pet_multiplier: float = 1.0
    """
    It's a scenario-free multiplier.
    - 1.0 doesn't change anything.
    - 1.2, PET increases by 20%.
    """
    # it alloswsmore evaporative scenarios; less evaporative scenarios...

    def __post_init__(self) -> None:
        _validate_positive_scalar("priestley_taylor_alpha", self.priestley_taylor_alpha)
        _validate_positive_scalar("psychrometric_constant_kpa_c", self.psychrometric_constant_kpa_c)
        _validate_positive_scalar("latent_heat_mj_kg", self.latent_heat_mj_kg)
        _validate_non_negative_scalar("pet_multiplier", self.pet_multiplier)


@dataclass(frozen=True)
class PETFields:
    """PET diagnostics for one simulation step."""

    saturation_vapor_pressure_kpa: FloatArray
    """
    Saturation vapor pressure, it is maximum vapor pressure that the air could sustain at that temperature.
    It increases with temperature.
    """

    slope_saturation_vapor_pressure_curve_kpa_c: FloatArray
    """
    Slope of the saturation vapor pressure curve.
    How quickly that maximum capacity changes when the temperature rises.
    """
    # It's a way of quantifying "how much the temperature pushes".

    equilibrium_evaporation_mm_dt: FloatArray
    """
    Evaporation that would come out only from available energy, before the Priestley-Taylor adjustment.
    """

    pet_mm_dt: FloatArray
    """
    Step's PET
    """

    def __post_init__(self) -> None:
        array_fields = {
            "saturation_vapor_pressure_kpa": self.saturation_vapor_pressure_kpa,
            "slope_saturation_vapor_pressure_curve_kpa_c": self.slope_saturation_vapor_pressure_curve_kpa_c,
            "equilibrium_evaporation_mm_dt": self.equilibrium_evaporation_mm_dt,
            "pet_mm_dt": self.pet_mm_dt,
        }
        for name, value in array_fields.items():
            _validate_spatial_float_array(name, value)

        spatial_shape = self.pet_mm_dt.shape
        for name, value in array_fields.items():
            if value.shape != spatial_shape:
                raise ValueError(f"'{name}' must have shape {spatial_shape}, got {value.shape}")

        if np.any(self.saturation_vapor_pressure_kpa < 0.0):
            raise ValueError("'saturation_vapor_pressure_kpa' must be >= 0 everywhere")

        if np.any(self.slope_saturation_vapor_pressure_curve_kpa_c < 0.0):
            raise ValueError("'slope_saturation_vapor_pressure_curve_kpa_c' must be >= 0 everywhere")

        if np.any(self.equilibrium_evaporation_mm_dt < 0.0):
            raise ValueError("'equilibrium_evaporation_mm_dt' must be >= 0 everywhere")

        if np.any(self.pet_mm_dt < 0.0):
            raise ValueError("'pet_mm_dt' must be >= 0 everywhere")


def compute_saturation_vapor_pressure_kpa(
    air_temperature_c: FloatArray,
) -> FloatArray:
    """Return saturation vapor pressure in kPa.

    Uses the standard FAO-style approximation (chapter 3 eq11):
    es(T) = 0.6108 * exp(17.27*T / (T + 237.3))
    """
    _validate_spatial_float_array("air_temperature_c", air_temperature_c)

    air_temperature_c = air_temperature_c.astype(float, copy=False)

    saturation_vapor_pressure_kpa = 0.6108 * np.exp(17.27 * air_temperature_c / (air_temperature_c + 237.3))
    return np.clip(saturation_vapor_pressure_kpa, 0.0, None).astype(float, copy=False)


def compute_slope_saturation_vapor_pressure_curve_kpa_c(
    air_temperature_c: FloatArray,
) -> FloatArray:
    """Return slope of the saturation vapor pressure curve in kPa/°C.

    Uses the standard derivative form (chapter 3 eq13):
    Delta = 4098 * es(T) / (T + 237.3)^2
    """
    _validate_spatial_float_array("air_temperature_c", air_temperature_c)

    air_temperature_c = air_temperature_c.astype(float, copy=False)
    saturation_vapor_pressure_kpa = compute_saturation_vapor_pressure_kpa(air_temperature_c)

    slope_kpa_c = 4098.0 * saturation_vapor_pressure_kpa / np.square(air_temperature_c + 237.3)
    return np.clip(slope_kpa_c, 0.0, None).astype(float, copy=False)


def compute_equilibrium_evaporation_mm_dt(
    net_radiation_mj_m2_dt: FloatArray,
    slope_saturation_vapor_pressure_curve_kpa_c: FloatArray,
    psychrometric_constant_kpa_c: int | float,
    latent_heat_mj_kg: int | float,
) -> FloatArray:
    """Return equilibrium evaporation in mm/dt.

    Energy-only equilibrium form:
    E_eq = (Delta / (Delta + gamma)) * (Rn / lambda)
    """
    _validate_spatial_float_array("net_radiation_mj_m2_dt", net_radiation_mj_m2_dt)
    _validate_spatial_float_array(
        "slope_saturation_vapor_pressure_curve_kpa_c",
        slope_saturation_vapor_pressure_curve_kpa_c,
    )

    if net_radiation_mj_m2_dt.shape != slope_saturation_vapor_pressure_curve_kpa_c.shape:
        raise ValueError(
            "'net_radiation_mj_m2_dt' and "
            "'slope_saturation_vapor_pressure_curve_kpa_c' must have the same shape, "
            f"got {net_radiation_mj_m2_dt.shape} and {slope_saturation_vapor_pressure_curve_kpa_c.shape}"
        )

    psychrometric_constant_kpa_c = _validate_positive_scalar(
        "psychrometric_constant_kpa_c",
        psychrometric_constant_kpa_c,
    )
    latent_heat_mj_kg = _validate_positive_scalar(
        "latent_heat_mj_kg",
        latent_heat_mj_kg,
    )

    net_radiation_mj_m2_dt = np.clip(net_radiation_mj_m2_dt.astype(float, copy=False), 0.0, None)
    slope_saturation_vapor_pressure_curve_kpa_c = np.clip(
        slope_saturation_vapor_pressure_curve_kpa_c.astype(float, copy=False),
        0.0,
        None,
    )

    # In this simplified scheme, what fraction of the available energy
    # potential ends up being expressed as evaporation?
    # - If Δ is large compared to γ, the weight approaches 1;
    # - If γ weighs more, the value decreases.
    evaporative_weight = slope_saturation_vapor_pressure_curve_kpa_c / (
        slope_saturation_vapor_pressure_curve_kpa_c + psychrometric_constant_kpa_c
    )
    equilibrium_evaporation_mm_dt = evaporative_weight * (net_radiation_mj_m2_dt / latent_heat_mj_kg)

    return np.clip(equilibrium_evaporation_mm_dt, 0.0, None).astype(float, copy=False)


def compute_priestley_taylor_pet_mm_dt(
    net_radiation_mj_m2_dt: FloatArray,
    air_temperature_c: FloatArray,
    config: PETConfig,
) -> PETFields:
    """Return PET fields using a simplified Priestley-Taylor formulation."""
    _validate_spatial_float_array("net_radiation_mj_m2_dt", net_radiation_mj_m2_dt)
    _validate_spatial_float_array("air_temperature_c", air_temperature_c)

    if net_radiation_mj_m2_dt.shape != air_temperature_c.shape:
        raise ValueError(
            "'net_radiation_mj_m2_dt' and 'air_temperature_c' must have the same shape, "
            f"got {net_radiation_mj_m2_dt.shape} and {air_temperature_c.shape}"
        )

    if not isinstance(config, PETConfig):
        raise TypeError(f"'config' must be a PETConfig, got {type(config).__name__}")

    saturation_vapor_pressure_kpa = compute_saturation_vapor_pressure_kpa(air_temperature_c)
    slope_saturation_vapor_pressure_curve_kpa_c = compute_slope_saturation_vapor_pressure_curve_kpa_c(air_temperature_c)

    equilibrium_evaporation_mm_dt = compute_equilibrium_evaporation_mm_dt(
        net_radiation_mj_m2_dt=net_radiation_mj_m2_dt,
        slope_saturation_vapor_pressure_curve_kpa_c=slope_saturation_vapor_pressure_curve_kpa_c,
        psychrometric_constant_kpa_c=config.psychrometric_constant_kpa_c,
        latent_heat_mj_kg=config.latent_heat_mj_kg,
    )

    pet_mm_dt = config.priestley_taylor_alpha * equilibrium_evaporation_mm_dt * config.pet_multiplier
    pet_mm_dt = np.clip(pet_mm_dt, 0.0, None).astype(float, copy=False)

    return PETFields(
        saturation_vapor_pressure_kpa=saturation_vapor_pressure_kpa,
        slope_saturation_vapor_pressure_curve_kpa_c=slope_saturation_vapor_pressure_curve_kpa_c,
        equilibrium_evaporation_mm_dt=equilibrium_evaporation_mm_dt,
        pet_mm_dt=pet_mm_dt,
    )
