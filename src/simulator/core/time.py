from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np


@dataclass(frozen=True)
class TimeDefinition:
    """Temporal domain for the simulation.

    Attributes
    ----------
    start : datetime
        The start time of the simulation.
    dt_seconds : int
        Time step duration in seconds.
    n_steps : int
        Number of simulation steps.
    """

    start: datetime
    dt_seconds: int
    n_steps: int

    def __post_init__(self):
        """Validate time parameters after initialization."""
        if self.dt_seconds <= 0:
            raise ValueError(f"Time step 'dt_seconds' must be a positive integer, got {self.dt_seconds}")
        if self.n_steps <= 0:
            raise ValueError(f"Number of steps 'n_steps' must be a positive integer, got {self.n_steps}")

    @property
    def timestamps(self) -> np.ndarray:
        """Generate an array of timestamps based on start time and time step."""
        return np.array([self.start + timedelta(seconds=self.dt_seconds * i) for i in range(self.n_steps)])

    @property
    def step_index(self) -> np.ndarray:
        """Return an array of step indices for the simulation."""
        return np.arange(self.n_steps)

    @property
    def total_duration_seconds(self) -> int:
        """Return total simulated duration in seconds."""
        return self.dt_seconds * self.n_steps

    @property
    def total_duration(self) -> timedelta:
        """Return total simulated duration as a timedelta."""
        return timedelta(seconds=self.total_duration_seconds)
