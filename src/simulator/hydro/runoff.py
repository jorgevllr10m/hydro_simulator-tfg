from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.common.validation import (
    validate_fraction as _validate_fraction,
)
from simulator.common.validation import (
    validate_spatial_float_array as _validate_spatial_float_array,
)
from simulator.core.types import FloatArray
from simulator.hydro.soil import SoilStepFields


@dataclass(frozen=True)
class RunoffConfig:
    """Configuration of the simplified runoff partition.

    Notes
    -----
    - Surface runoff is taken directly from the soil surface excess.
    - Subsurface runoff is derived from soil percolation/drainage.
    - No routing is performed in this phase.
    """

    subsurface_runoff_fraction: float = 1.0
    """
    Fraction of soil percolation that is exposed as subsurface runoff.
    - 1.0 -> all percolation becomes subsurface runoff
    - <1.0 -> part of percolation is treated as deeper loss outside this MVP
    """

    def __post_init__(self) -> None:
        _validate_fraction("subsurface_runoff_fraction", self.subsurface_runoff_fraction)


@dataclass(frozen=True)
class RunoffFields:
    """Local runoff products derived from one soil-bucket step."""

    infiltration_mm_dt: FloatArray
    surface_runoff_mm_dt: FloatArray
    subsurface_runoff_mm_dt: FloatArray

    def __post_init__(self) -> None:
        array_fields = {
            "infiltration_mm_dt": self.infiltration_mm_dt,
            "surface_runoff_mm_dt": self.surface_runoff_mm_dt,
            "subsurface_runoff_mm_dt": self.subsurface_runoff_mm_dt,
        }

        for name, value in array_fields.items():
            _validate_spatial_float_array(name, value)

        spatial_shape = self.infiltration_mm_dt.shape
        for name, value in array_fields.items():
            if value.shape != spatial_shape:
                raise ValueError(f"'{name}' must have shape {spatial_shape}, got {value.shape}")

        for name, value in array_fields.items():
            if np.any(value < 0.0):
                raise ValueError(f"'{name}' must be >= 0 everywhere")


def compute_surface_runoff_mm_dt(
    surface_excess_mm_dt: FloatArray,
) -> FloatArray:
    """Return surface runoff from direct soil surface excess."""
    _validate_spatial_float_array("surface_excess_mm_dt", surface_excess_mm_dt)

    surface_runoff_mm_dt = np.clip(surface_excess_mm_dt.astype(float, copy=False), 0.0, None)
    return surface_runoff_mm_dt.astype(float, copy=False)


def compute_subsurface_runoff_mm_dt(
    percolation_mm_dt: FloatArray,
    *,
    subsurface_runoff_fraction: int | float,
) -> FloatArray:
    """Return subsurface runoff from soil percolation/drainage."""
    _validate_spatial_float_array("percolation_mm_dt", percolation_mm_dt)
    subsurface_runoff_fraction = _validate_fraction(
        "subsurface_runoff_fraction",
        subsurface_runoff_fraction,
    )

    percolation_mm_dt = np.clip(percolation_mm_dt.astype(float, copy=False), 0.0, None)
    subsurface_runoff_mm_dt = subsurface_runoff_fraction * percolation_mm_dt

    return np.clip(subsurface_runoff_mm_dt, 0.0, None).astype(float, copy=False)


def derive_runoff_fields(
    soil_fields: SoilStepFields,
    *,
    config: RunoffConfig,
) -> RunoffFields:
    """Derive local runoff products from one soil-bucket update."""
    if not isinstance(soil_fields, SoilStepFields):
        raise TypeError(f"'soil_fields' must be a SoilStepFields, got {type(soil_fields).__name__}")

    if not isinstance(config, RunoffConfig):
        raise TypeError(f"'config' must be a RunoffConfig, got {type(config).__name__}")

    infiltration_mm_dt = np.clip(
        soil_fields.infiltration_mm_dt.astype(float, copy=False),
        0.0,
        None,
    )

    surface_runoff_mm_dt = compute_surface_runoff_mm_dt(
        soil_fields.surface_excess_mm_dt,
    )

    subsurface_runoff_mm_dt = compute_subsurface_runoff_mm_dt(
        soil_fields.percolation_mm_dt,
        subsurface_runoff_fraction=config.subsurface_runoff_fraction,
    )

    return RunoffFields(
        infiltration_mm_dt=infiltration_mm_dt.astype(float, copy=False),
        surface_runoff_mm_dt=surface_runoff_mm_dt.astype(float, copy=False),
        subsurface_runoff_mm_dt=subsurface_runoff_mm_dt.astype(float, copy=False),
    )
