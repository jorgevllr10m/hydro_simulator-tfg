from __future__ import annotations

import math
from dataclasses import dataclass

from simulator.common.validation import (
    validate_numeric_scalar,
    validate_positive_scalar,
)


@dataclass
class StormCell:
    """Mutable storm object tracked across simulation steps.

    - Center coordinates and semi-axes are expressed in meters.
    - Velocity components are expressed in m/s.
    - peak_intensity_mmph is the mature-stage maximum rainfall intensity.
    - duration_steps and age_steps are expressed in simulation steps.
    - orientation_deg is normalized to [0, 180) because an ellipse is
      invariant under a 180-degree rotation.
    """

    storm_id: int

    center_x_m: float
    center_y_m: float

    velocity_u_mps: float
    velocity_v_mps: float

    semi_major_axis_m: float  # half of the main axis.
    semi_minor_axis_m: float  # half of the secondary axis
    orientation_deg: float

    peak_intensity_mmph: float
    duration_steps: int
    age_steps: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.storm_id, int):
            raise TypeError(f"'storm_id' must be an int, got {type(self.storm_id).__name__}")
        if self.storm_id < 0:
            raise ValueError(f"'storm_id' must be >= 0, got {self.storm_id}")

        self.center_x_m = validate_numeric_scalar("center_x_m", self.center_x_m)
        self.center_y_m = validate_numeric_scalar("center_y_m", self.center_y_m)

        # Velocity components may be negative, so they are validated separately.
        if not isinstance(self.velocity_u_mps, (int, float)):
            raise TypeError(f"'velocity_u_mps' must be numeric, got {type(self.velocity_u_mps).__name__}")
        if not isinstance(self.velocity_v_mps, (int, float)):
            raise TypeError(f"'velocity_v_mps' must be numeric, got {type(self.velocity_v_mps).__name__}")
        self.velocity_u_mps = float(self.velocity_u_mps)
        self.velocity_v_mps = float(self.velocity_v_mps)

        self.semi_major_axis_m = validate_positive_scalar("semi_major_axis_m", self.semi_major_axis_m)
        self.semi_minor_axis_m = validate_positive_scalar("semi_minor_axis_m", self.semi_minor_axis_m)

        if self.semi_minor_axis_m > self.semi_major_axis_m:
            raise ValueError(
                "'semi_minor_axis_m' cannot exceed 'semi_major_axis_m', "
                f"got minor={self.semi_minor_axis_m}, major={self.semi_major_axis_m}"
            )

        if not isinstance(self.orientation_deg, (int, float)):
            raise TypeError(f"'orientation_deg' must be numeric, got {type(self.orientation_deg).__name__}")
        self.orientation_deg = float(self.orientation_deg) % 180.0

        self.peak_intensity_mmph = validate_positive_scalar(
            "peak_intensity_mmph",
            self.peak_intensity_mmph,
        )

        if not isinstance(self.duration_steps, int):
            raise TypeError(f"'duration_steps' must be an int, got {type(self.duration_steps).__name__}")
        if self.duration_steps <= 0:
            raise ValueError(f"'duration_steps' must be > 0, got {self.duration_steps}")

        if not isinstance(self.age_steps, int):
            raise TypeError(f"'age_steps' must be an int, got {type(self.age_steps).__name__}")
        if self.age_steps < 0:
            raise ValueError(f"'age_steps' must be >= 0, got {self.age_steps}")
        if self.age_steps > self.duration_steps:
            raise ValueError(f"'age_steps' ({self.age_steps}) cannot exceed 'duration_steps' ({self.duration_steps})")

    @property
    def orientation_rad(self) -> float:
        """Return the storm orientation in radians."""
        return math.radians(self.orientation_deg)

    @property
    def center(self) -> tuple[float, float]:
        """Return storm center coordinates as (x, y) in meters."""
        return (self.center_x_m, self.center_y_m)

    @property
    def velocity(self) -> tuple[float, float]:
        """Return advection velocity as (u, v) in m/s."""
        return (self.velocity_u_mps, self.velocity_v_mps)

    @property
    def speed_mps(self) -> float:
        """Return the scalar advection speed in m/s."""
        return math.hypot(self.velocity_u_mps, self.velocity_v_mps)

    @property
    def life_progress(self) -> float:
        """Return normalized life progress in the [0, 1] interval."""  # at the beggining 0.0, half will be 0.5, ...
        return min(1.0, self.age_steps / self.duration_steps)

    @property
    def remaining_steps(self) -> int:
        """Return the number of remaining alive steps."""
        return max(0, self.duration_steps - self.age_steps)

    @property
    def is_alive(self) -> bool:
        """Return whether the storm is still active."""
        return self.age_steps < self.duration_steps

    def advance(self, dt_seconds: int | float) -> None:
        """Advance the storm one simulation step.

        This updates:
        - center position using constant advection over dt_seconds
        - age_steps by exactly one discrete simulation step
        """
        dt_seconds = validate_positive_scalar("dt_seconds", dt_seconds)

        self.center_x_m += self.velocity_u_mps * dt_seconds
        self.center_y_m += self.velocity_v_mps * dt_seconds
        self.age_steps += 1

        if self.age_steps > self.duration_steps:
            self.age_steps = self.duration_steps

    def expire(self) -> None:
        """Force the storm to an expired state."""
        self.age_steps = self.duration_steps
