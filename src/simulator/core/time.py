from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

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
    calendar_type: str
        'monthly' or 'seasonal'
    calendar : Optional[str]
        Optional calendar string for managing seasonality.
    """

    start: datetime
    dt_seconds: int
    n_steps: int
    calendar_type: str = "monthly"
    calendar: Optional[str] = None

    def __post_init__(self):
        """Validate time parameters after initialization."""
        if self.dt_seconds <= 0:
            raise ValueError(f"Time step 'dt_seconds' must be a positive integer, got {self.dt_seconds}")
        if self.n_steps <= 0:
            raise ValueError(f"Number of steps 'n_steps' must be a positive integer, got {self.n_steps}")
        if self.calendar_type not in ["monthly", "seasonal"]:
            raise ValueError(f"Invalid calendar type {self.calendar_type}. Use 'monthly' or 'seasonal'.")

    @property
    def timestamps(self) -> np.ndarray:
        """Generate an array of timestamps based on start time and time step."""
        return np.array([self.start + timedelta(seconds=self.dt_seconds * i) for i in range(self.n_steps)])

    @property
    def step_index(self) -> np.ndarray:
        """Return an array of step indices for the simulation."""
        return np.arange(self.n_steps)

    @property
    def months(self) -> np.ndarray:
        """Generate an array of months for each time step."""
        months = []
        current_time = self.start
        for i in range(self.n_steps):
            month = current_time.month
            months.append(month)
            current_time += timedelta(seconds=self.dt_seconds)
        return np.array(months)

    @property
    def seasons(self) -> np.ndarray:
        """Generate an array of seasons for each time step."""
        seasons = []
        for month in self.months:
            if month in [12, 1, 2]:
                seasons.append("Winter")
            elif month in [3, 4, 5]:
                seasons.append("Spring")
            elif month in [6, 7, 8]:
                seasons.append("Summer")
            else:
                seasons.append("Fall")
        return np.array(seasons)

    # TODO reemplazar los factores fijos de apply_seasonality por parámetros de configuración o escenario.
    def apply_seasonality(self, values: np.ndarray) -> np.ndarray:
        """Apply seasonality adjustments to a values array based on the season."""
        seasonality_factors = {
            "Winter": 1.2,  # For example, winter might have 20% higher precipitation
            "Spring": 1.0,
            "Summer": 0.8,
            "Fall": 1.0,
        }

        adjusted_values = np.copy(values)
        for i, season in enumerate(self.seasons):
            adjusted_values[i] *= seasonality_factors.get(season, 1.0)

        return adjusted_values

    @property
    def total_duration_seconds(self) -> int:
        """Return total simulated duration in seconds."""
        return self.dt_seconds * self.n_steps

    @property
    def total_duration(self) -> timedelta:
        """Return total simulated duration as a timedelta."""
        return timedelta(seconds=self.total_duration_seconds)
