from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MeteorologicalRegime(str, Enum):
    """Discrete weather regimes used by the simplified meteorological environment."""

    STABLE_DRY = "stable_dry"
    TRANSITIONAL = "transitional"
    CONVECTIVE = "convective"
    FRONTAL_PERSISTENT = "frontal_persistent"


@dataclass(frozen=True)
class RegimeProfile:
    """Baseline tendencies associated with a meteorological regime.

    Notes
    -----
    - cloudiness_index, convective_potential_index and wetness_equilibrium
      are normalized to the [0, 1] range.
    - temperature_anomaly_c is an additive thermal offset in degrees Celsius.
    """

    cloudiness_index: float
    convective_potential_index: float
    wetness_equilibrium: float
    temperature_anomaly_c: float

    def __post_init__(self) -> None:
        bounded_fields = {
            "cloudiness_index": self.cloudiness_index,
            "convective_potential_index": self.convective_potential_index,
            "wetness_equilibrium": self.wetness_equilibrium,
        }

        for name, value in bounded_fields.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"'{name}' must be numeric, got {type(value).__name__}")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"'{name}' must be within [0, 1], got {value}")

        if not isinstance(self.temperature_anomaly_c, (int, float)):
            raise TypeError(f"'temperature_anomaly_c' must be numeric, got {type(self.temperature_anomaly_c).__name__}")


# TODO(phase2): calibrate default regime profile values after storm generator integration.
REGIME_PROFILES: dict[MeteorologicalRegime, RegimeProfile] = {
    MeteorologicalRegime.STABLE_DRY: RegimeProfile(
        cloudiness_index=0.15,  # * This regime tends to have a base cloudiness of 0.15
        convective_potential_index=0.10,
        wetness_equilibrium=0.20,
        temperature_anomaly_c=1.0,
    ),
    MeteorologicalRegime.TRANSITIONAL: RegimeProfile(
        cloudiness_index=0.45,
        convective_potential_index=0.40,
        wetness_equilibrium=0.50,
        temperature_anomaly_c=0.0,
    ),
    MeteorologicalRegime.CONVECTIVE: RegimeProfile(
        cloudiness_index=0.50,
        convective_potential_index=0.85,
        wetness_equilibrium=0.55,
        temperature_anomaly_c=2.0,
    ),
    MeteorologicalRegime.FRONTAL_PERSISTENT: RegimeProfile(
        cloudiness_index=0.85,
        convective_potential_index=0.25,
        wetness_equilibrium=0.80,
        temperature_anomaly_c=-0.5,
    ),
}


def get_regime_profile(regime: MeteorologicalRegime) -> RegimeProfile:
    """Return the baseline profile associated with a regime."""
    try:
        return REGIME_PROFILES[regime]
    except KeyError as exc:
        raise ValueError(f"Unsupported meteorological regime: {regime!r}") from exc
