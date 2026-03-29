from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AdvectionField:
    """Simplified effective advection for the meteorological environment.

    Notes
    -----
    - speed_mps is the effective advection speed in meters per second.
    - direction_deg follows the mathematical convention:
      0 deg = +x (east), 90 deg = +y (north), increasing counter-clockwise.
    - u_mps and v_mps are derived Cartesian components.
    """

    speed_mps: float
    direction_deg: float

    def __post_init__(self) -> None:
        if not isinstance(self.speed_mps, (int, float)):
            raise TypeError(f"'speed_mps' must be numeric, got {type(self.speed_mps).__name__}")
        if self.speed_mps < 0.0:
            raise ValueError(f"'speed_mps' must be >= 0, got {self.speed_mps}")

        if not isinstance(self.direction_deg, (int, float)):
            raise TypeError(f"'direction_deg' must be numeric, got {type(self.direction_deg).__name__}")

        normalized_direction = float(self.direction_deg) % 360.0
        object.__setattr__(self, "direction_deg", normalized_direction)
        object.__setattr__(self, "speed_mps", float(self.speed_mps))

    @property
    def direction_rad(self) -> float:
        """Return the advection direction in radians."""
        return math.radians(self.direction_deg)

    @property
    def u_mps(self) -> float:
        """Return the x component of the advection velocity."""
        return self.speed_mps * math.cos(self.direction_rad)

    @property
    def v_mps(self) -> float:
        """Return the y component of the advection velocity."""
        return self.speed_mps * math.sin(self.direction_rad)

    @classmethod
    def from_uv(cls, u_mps: float, v_mps: float) -> AdvectionField:
        """Build an advection field from Cartesian velocity components."""
        if not isinstance(u_mps, (int, float)):
            raise TypeError(f"'u_mps' must be numeric, got {type(u_mps).__name__}")
        if not isinstance(v_mps, (int, float)):
            raise TypeError(f"'v_mps' must be numeric, got {type(v_mps).__name__}")

        u_mps = float(u_mps)
        v_mps = float(v_mps)

        speed_mps = math.hypot(u_mps, v_mps)
        direction_deg = math.degrees(math.atan2(v_mps, u_mps)) % 360.0
        return cls(speed_mps=speed_mps, direction_deg=direction_deg)
