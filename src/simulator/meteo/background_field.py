from __future__ import annotations

import numpy as np

from simulator.core.types import FloatArray, SimulationDomain


def _validate_numeric_scalar(name: str, value: int | float) -> float:
    """Validate a numeric scalar and return it as float."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"'{name}' must be numeric, got {type(value).__name__}")
    return float(value)


def build_uniform_spatial_field(
    domain: SimulationDomain,
    value: int | float,
) -> FloatArray:
    """Return a 2D float field filled with a uniform scalar value."""
    scalar_value = _validate_numeric_scalar("value", value)
    return np.full(domain.shape, scalar_value, dtype=float)


def build_air_temperature_field(
    domain: SimulationDomain,
    background_temperature_c: int | float,
) -> FloatArray:
    """Build the 2D air-temperature field for the current step.

    Phase-3 MVP choice:
    - temperature is spatially uniform
    - its scalar value comes from the latent meteorological environment
    """
    background_temperature_c = _validate_numeric_scalar(
        "background_temperature_c",
        background_temperature_c,
    )
    return build_uniform_spatial_field(domain, background_temperature_c)


# TODO(phase3->phase4): replace the zero background field with a simple
# stratiform/spatially correlated precipitation component when the refined
# background field enters the next meteorological phase.
def build_background_precipitation_field(
    domain: SimulationDomain,
) -> FloatArray:
    """Build the 2D background precipitation field for the current step.

    Phase-3 MVP choice:
    - no refined stratiform/background component yet
    - the returned field is therefore zero everywhere
    - units are mm/dt, consistent with the historical dataset convention
    """
    return np.zeros(domain.shape, dtype=float)
