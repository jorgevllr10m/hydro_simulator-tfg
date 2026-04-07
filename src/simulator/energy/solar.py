from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


def _validate_numeric_scalar(name: str, value: int | float) -> float:
    """Validate a numeric scalar and return it as float."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"'{name}' must be numeric, got {type(value).__name__}")
    return float(value)


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value to the [lower, upper] interval."""
    return max(lower, min(upper, float(value)))


def _validate_latitude_deg(latitude_deg: int | float) -> float:
    """Validate latitude in decimal degrees."""
    latitude_deg = _validate_numeric_scalar("latitude_deg", latitude_deg)

    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError(f"'latitude_deg' must be within [-90, 90], got {latitude_deg}")

    return latitude_deg


@dataclass(frozen=True)
class SolarGeometry:
    """Solar geometry diagnostics for one simulation time step.

    Notes
    -----
    - Local clock time is used as a proxy for local solar time.
    - `cos_zenith` is clipped to [0, 1], so it is 0 during night.
    - `solar_elevation_deg` may be negative when the sun is below the horizon.
    """

    timestamp: datetime
    latitude_deg: float
    day_of_year: int
    fractional_hour_local: float
    """Local time continues for the day. example: 13:30 --> 13.5"""

    solar_declination_rad: float
    """ It is the angle that describes the position of the Sun with respect to the
    plane of the Earth's equator due to the tilt of the Earth's axis.
    - Positive: The sun "favors" the northern hemisphere.
    - Negative: It favors the southern hemisphere."""
    hour_angle_rad: float
    """ This represents how far the Sun has moved from solar noon.
    - 0 around noon
    - Negative in the morning
    - Positive in the afternoon
    """
    inverse_earth_sun_distance_factor: float
    """
    Correction factor for the annual variation of the Earth-Sun distance.
    (Earth's orbit is not perfectly circular).
    """

    cos_zenith: float
    """
    Same as solar_elevation_deg but more useful for radiation
    - 1: Sun very high, almost overhead
    - 0: Sun on or below the horizon
    """
    solar_elevation_deg: float
    """
    It's the angle of the sun above the horizon.
    - 90°: Sun directly overhead
    - 0°: Horizon
    - Negative: Sun below the horizon
    """

    daylight_duration_hours: float
    """
    Total day length, in hours, for that latitude and day of the year.
    - Winter: fewer hours
    - Summer: more hours
    """
    daylight_fraction: float
    """daylight_duration_hours normalized to [0, 1]."""

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"'timestamp' must be a datetime, got {type(self.timestamp).__name__}")

        latitude_deg = _validate_latitude_deg(self.latitude_deg)
        object.__setattr__(self, "latitude_deg", latitude_deg)

        if not isinstance(self.day_of_year, int):
            raise TypeError(f"'day_of_year' must be an int, got {type(self.day_of_year).__name__}")
        if not 1 <= self.day_of_year <= 366:
            raise ValueError(f"'day_of_year' must be within [1, 366], got {self.day_of_year}")

        fractional_hour_local = _validate_numeric_scalar("fractional_hour_local", self.fractional_hour_local)
        if not 0.0 <= fractional_hour_local < 24.0:
            raise ValueError(f"'fractional_hour_local' must be within [0, 24), got {fractional_hour_local}")
        object.__setattr__(self, "fractional_hour_local", fractional_hour_local)

        numeric_fields = {
            "solar_declination_rad": self.solar_declination_rad,
            "hour_angle_rad": self.hour_angle_rad,
            "inverse_earth_sun_distance_factor": self.inverse_earth_sun_distance_factor,
            "cos_zenith": self.cos_zenith,
            "solar_elevation_deg": self.solar_elevation_deg,
            "daylight_duration_hours": self.daylight_duration_hours,
            "daylight_fraction": self.daylight_fraction,
        }
        for name, value in numeric_fields.items():
            _validate_numeric_scalar(name, value)

        if self.inverse_earth_sun_distance_factor <= 0.0:
            raise ValueError(f"'inverse_earth_sun_distance_factor' must be > 0, got {self.inverse_earth_sun_distance_factor}")

        if not 0.0 <= self.cos_zenith <= 1.0:
            raise ValueError(f"'cos_zenith' must be within [0, 1], got {self.cos_zenith}")

        if not -90.0 <= self.solar_elevation_deg <= 90.0:
            raise ValueError(f"'solar_elevation_deg' must be within [-90, 90], got {self.solar_elevation_deg}")

        if not 0.0 <= self.daylight_duration_hours <= 24.0:
            raise ValueError(f"'daylight_duration_hours' must be within [0, 24], got {self.daylight_duration_hours}")

        if not 0.0 <= self.daylight_fraction <= 1.0:
            raise ValueError(f"'daylight_fraction' must be within [0, 1], got {self.daylight_fraction}")

    @property
    def latitude_rad(self) -> float:
        """Return latitude in radians."""
        return math.radians(self.latitude_deg)

    @property
    def solar_declination_deg(self) -> float:
        """Return solar declination in degrees."""
        return math.degrees(self.solar_declination_rad)

    @property
    def hour_angle_deg(self) -> float:
        """Return solar hour angle in degrees."""
        return math.degrees(self.hour_angle_rad)

    @property
    def is_daylight(self) -> bool:
        """Return whether the sun is above the horizon."""
        return self.cos_zenith > 0.0


def day_of_year(timestamp: datetime) -> int:
    """Return day of year in [1, 366]."""
    if not isinstance(timestamp, datetime):
        raise TypeError(f"'timestamp' must be a datetime, got {type(timestamp).__name__}")

    return int(timestamp.timetuple().tm_yday)


def fractional_hour_local(timestamp: datetime) -> float:
    """Return local-clock hour as a continuous value in [0, 24)."""
    if not isinstance(timestamp, datetime):
        raise TypeError(f"'timestamp' must be a datetime, got {type(timestamp).__name__}")

    return timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0 + timestamp.microsecond / 3_600_000_000.0


def solar_declination_rad(day_of_year_value: int) -> float:
    """Return solar declination in radians.

    Uses the FAO-56 style approximation:
    delta = 0.409 * sin(2*pi*J/365 - 1.39) where J is the number day of the year.
    """
    if not isinstance(day_of_year_value, int):
        raise TypeError(f"'day_of_year_value' must be an int, got {type(day_of_year_value).__name__}")
    if not 1 <= day_of_year_value <= 366:
        raise ValueError(f"'day_of_year_value' must be within [1, 366], got {day_of_year_value}")

    return 0.409 * math.sin((2.0 * math.pi * day_of_year_value / 365.0) - 1.39)


def inverse_earth_sun_distance_factor(day_of_year_value: int) -> float:
    """Return the inverse relative Earth-Sun distance factor.

    The Earth is not always exactly the same distance from the Sun. This causes extraterrestrial
    radiation to change slightly throughout the year.

    This factor corrects for precisely that.

    If dr is slightly higher, a little more potential energy arrives.
    If dr is slightly lower, a little less.

    Uses the FAO-56 style approximation:
    dr = 1 + 0.033 * cos(2*pi*J/365) where J is the number day of the year.
    """
    if not isinstance(day_of_year_value, int):
        raise TypeError(f"'day_of_year_value' must be an int, got {type(day_of_year_value).__name__}")
    if not 1 <= day_of_year_value <= 366:
        raise ValueError(f"'day_of_year_value' must be within [1, 366], got {day_of_year_value}")

    return 1.0 + 0.033 * math.cos(2.0 * math.pi * day_of_year_value / 365.0)


def solar_hour_angle_rad(fractional_hour_local_value: int | float) -> float:
    """Return the solar hour angle in radians.

    Local clock time is approximated as local solar time:
    - noon -> 0
    - morning -> negative angles
    - afternoon -> positive angles
    """
    fractional_hour_local_value = _validate_numeric_scalar(
        "fractional_hour_local_value",
        fractional_hour_local_value,
    )
    if not 0.0 <= fractional_hour_local_value < 24.0:
        raise ValueError(f"'fractional_hour_local_value' must be within [0, 24), got {fractional_hour_local_value}")

    return math.pi / 12.0 * (fractional_hour_local_value - 12.0)


def sunset_hour_angle_rad(
    latitude_deg: int | float,
    declination_rad: int | float,
) -> float:
    """Return sunset hour angle in radians.

    Handles polar-day / polar-night limits by clipping the arccos argument.
    """
    latitude_deg = _validate_latitude_deg(latitude_deg)
    declination_rad = _validate_numeric_scalar("declination_rad", declination_rad)

    latitude_rad = math.radians(latitude_deg)
    arccos_argument = -math.tan(latitude_rad) * math.tan(declination_rad)

    if arccos_argument <= -1.0:
        return math.pi  # 24 hours of light

    if arccos_argument >= 1.0:
        return 0.0  # 0 hours of light

    return math.acos(arccos_argument)


def daylight_duration_hours(
    latitude_deg: int | float,
    declination_rad: int | float,
) -> float:
    """Return daylight duration in hours."""
    sunset_angle_rad = sunset_hour_angle_rad(
        latitude_deg=latitude_deg,
        declination_rad=declination_rad,
    )
    return 24.0 / math.pi * sunset_angle_rad


def compute_solar_cos_zenith(
    *,
    latitude_deg: int | float,
    declination_rad: int | float,
    hour_angle_rad: int | float,
) -> float:
    """Return clipped cos(zenith) in [0, 1].
    - day --> positive value
    - night --> 0
    """
    latitude_deg = _validate_latitude_deg(latitude_deg)
    declination_rad = _validate_numeric_scalar("declination_rad", declination_rad)
    hour_angle_rad = _validate_numeric_scalar("hour_angle_rad", hour_angle_rad)

    latitude_rad = math.radians(latitude_deg)

    raw_cos_zenith = math.sin(latitude_rad) * math.sin(declination_rad) + math.cos(latitude_rad) * math.cos(
        declination_rad
    ) * math.cos(hour_angle_rad)

    return _clamp(raw_cos_zenith, 0.0, 1.0)


def compute_solar_elevation_deg(
    latitude_deg: int | float,
    declination_rad: int | float,
    hour_angle_rad: int | float,
) -> float:
    """Return solar elevation angle in degrees.

    Negative values indicate that the sun is below the horizon.
    """
    latitude_deg = _validate_latitude_deg(latitude_deg)
    declination_rad = _validate_numeric_scalar("declination_rad", declination_rad)
    hour_angle_rad = _validate_numeric_scalar("hour_angle_rad", hour_angle_rad)

    latitude_rad = math.radians(latitude_deg)

    raw_cos_zenith = math.sin(latitude_rad) * math.sin(declination_rad) + math.cos(latitude_rad) * math.cos(
        declination_rad
    ) * math.cos(hour_angle_rad)
    raw_cos_zenith = _clamp(raw_cos_zenith, -1.0, 1.0)

    return math.degrees(math.asin(raw_cos_zenith))


def compute_solar_geometry(
    *,
    timestamp: datetime,
    latitude_deg: int | float,
) -> SolarGeometry:
    """Compute simplified solar geometry for one simulation step."""
    if not isinstance(timestamp, datetime):
        raise TypeError(f"'timestamp' must be a datetime, got {type(timestamp).__name__}")

    latitude_deg = _validate_latitude_deg(latitude_deg)

    day = day_of_year(timestamp)
    hour_local = fractional_hour_local(timestamp)
    declination_rad = solar_declination_rad(day)
    hour_angle = solar_hour_angle_rad(hour_local)
    distance_factor = inverse_earth_sun_distance_factor(day)

    cos_zenith = compute_solar_cos_zenith(
        latitude_deg=latitude_deg,
        declination_rad=declination_rad,
        hour_angle_rad=hour_angle,
    )
    solar_elevation = compute_solar_elevation_deg(
        latitude_deg=latitude_deg,
        declination_rad=declination_rad,
        hour_angle_rad=hour_angle,
    )

    daylight_hours = daylight_duration_hours(
        latitude_deg=latitude_deg,
        declination_rad=declination_rad,
    )
    daylight_fraction_value = daylight_hours / 24.0

    return SolarGeometry(
        timestamp=timestamp,
        latitude_deg=latitude_deg,
        day_of_year=day,
        fractional_hour_local=hour_local,
        solar_declination_rad=declination_rad,
        hour_angle_rad=hour_angle,
        inverse_earth_sun_distance_factor=distance_factor,
        cos_zenith=cos_zenith,
        solar_elevation_deg=solar_elevation,
        daylight_duration_hours=daylight_hours,
        daylight_fraction=daylight_fraction_value,
    )
