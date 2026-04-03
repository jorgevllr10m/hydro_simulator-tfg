from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from simulator.meteo.advection import AdvectionField
from simulator.meteo.regimes import MeteorologicalRegime, get_regime_profile


def _clamp01(value: float) -> float:
    """Clamp a numeric value to the [0, 1] interval."""
    return max(0.0, min(1.0, float(value)))


class ThermalScenario(str, Enum):
    """User-facing thermal scenario for the latent environment."""

    COLD = "cold"
    NORMAL = "normal"
    WARM = "warm"


class MoistureScenario(str, Enum):
    """User-facing moisture tendency scenario for the latent environment."""

    DRY = "dry"
    NORMAL = "normal"
    WET = "wet"


@dataclass(frozen=True)
class LatentEnvironmentConfig:
    """Configuration of the simplified latent meteorological environment."""

    # TODO(phase2): revisit latent environment default parameters using validation metrics.

    # * meteorology rules. They can be adjusted based on the observed results.
    random_seed: int = 1234

    initial_regime: MeteorologicalRegime = MeteorologicalRegime.TRANSITIONAL

    # * temperature parameters
    mean_temperature_c: float = 15.0
    annual_temperature_amplitude_c: float = 10.0
    peak_temperature_day_of_year: int = 200
    temperature_noise_std_c: float = 0.75

    # * advection parameters
    prevailing_advection_speed_mps: float = 4.0  # mean
    advection_speed_std_mps: float = 1.0  # variability
    prevailing_advection_direction_deg: float = 225.0  # mean
    advection_direction_std_deg: float = 20.0  # variability

    # * persistence/memory parameters
    # How much does the previous parameter weigh against the new target
    regime_persistence: float = 0.85
    temperature_memory: float = 0.80
    wetness_memory: float = 0.90
    cloudiness_memory: float = 0.75
    convective_memory: float = 0.70
    advection_memory: float = 0.70

    # * Wet/dry spell memory
    spell_memory: float = 0.88
    spell_noise_std: float = 0.06
    dry_spell_target: float = 0.5
    wet_spell_target: float = 0.95

    dry_drift_per_step: float = 0.01

    thermal_scenario: ThermalScenario = ThermalScenario.NORMAL
    moisture_scenario: MoistureScenario = MoistureScenario.NORMAL

    def __post_init__(self) -> None:
        if not isinstance(self.random_seed, int):
            raise TypeError(f"'random_seed' must be an int, got {type(self.random_seed).__name__}")

        if not isinstance(self.mean_temperature_c, (int, float)):
            raise TypeError(f"'mean_temperature_c' must be numeric, got {type(self.mean_temperature_c).__name__}")

        if not isinstance(self.annual_temperature_amplitude_c, (int, float)):
            raise TypeError(
                f"'annual_temperature_amplitude_c' must be numeric, got {type(self.annual_temperature_amplitude_c).__name__}"
            )
        if self.annual_temperature_amplitude_c < 0.0:
            raise ValueError(f"'annual_temperature_amplitude_c' must be >= 0, got {self.annual_temperature_amplitude_c}")

        if not isinstance(self.peak_temperature_day_of_year, int):
            raise TypeError(f"'peak_temperature_day_of_year' must be an int, got {type(self.peak_temperature_day_of_year).__name__}")
        if not 1 <= self.peak_temperature_day_of_year <= 366:
            raise ValueError(f"'peak_temperature_day_of_year' must be within [1, 366], got {self.peak_temperature_day_of_year}")

        non_negative_fields = {
            "temperature_noise_std_c": self.temperature_noise_std_c,
            "prevailing_advection_speed_mps": self.prevailing_advection_speed_mps,
            "advection_speed_std_mps": self.advection_speed_std_mps,
            "advection_direction_std_deg": self.advection_direction_std_deg,
            "dry_drift_per_step": self.dry_drift_per_step,
            "spell_noise_std": self.spell_noise_std,
        }
        for name, value in non_negative_fields.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"'{name}' must be numeric, got {type(value).__name__}")
            if value < 0.0:
                raise ValueError(f"'{name}' must be >= 0, got {value}")

        normalized_fields = {
            "regime_persistence": self.regime_persistence,
            "temperature_memory": self.temperature_memory,
            "wetness_memory": self.wetness_memory,
            "cloudiness_memory": self.cloudiness_memory,
            "convective_memory": self.convective_memory,
            "advection_memory": self.advection_memory,
            "spell_memory": self.spell_memory,
            "dry_spell_target": self.dry_spell_target,
            "wet_spell_target": self.wet_spell_target,
        }
        for name, value in normalized_fields.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"'{name}' must be numeric, got {type(value).__name__}")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"'{name}' must be within [0, 1], got {value}")

        if self.dry_spell_target > self.wet_spell_target:
            raise ValueError(
                f"'dry_spell_target' must be <= 'wet_spell_target', got dry={self.dry_spell_target}, wet={self.wet_spell_target}"
            )

        if not isinstance(self.prevailing_advection_direction_deg, (int, float)):
            raise TypeError(
                f"'prevailing_advection_direction_deg' must be numeric, got {type(self.prevailing_advection_direction_deg).__name__}"
            )


@dataclass(frozen=True)
class LatentEnvironmentState:
    """Latent meteorological environment at a single simulation step."""

    step: int
    timestamp: datetime

    regime: MeteorologicalRegime
    background_temperature_c: float
    advection: AdvectionField

    antecedent_wetness_index: float
    precipitation_spell_index: float
    cloudiness_index: float
    convective_potential_index: float

    seasonality_factor: float

    # Diagnostic-only field:
    # the moisture scenario already affects the latent environment through
    # wetness and regime biases, so this factor is kept for inspection and
    # traceability rather than as an additional forcing term.
    scenario_moisture_factor: float

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError(f"'step' must be a non-negative integer, got {self.step!r}")

        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"'timestamp' must be a datetime, got {type(self.timestamp).__name__}")

        if not isinstance(self.regime, MeteorologicalRegime):
            raise TypeError(f"'regime' must be a MeteorologicalRegime, got {type(self.regime).__name__}")

        if not isinstance(self.background_temperature_c, (int, float)):
            raise TypeError(f"'background_temperature_c' must be numeric, got {type(self.background_temperature_c).__name__}")

        if not isinstance(self.advection, AdvectionField):
            raise TypeError(f"'advection' must be an AdvectionField, got {type(self.advection).__name__}")

        bounded_fields = {
            "antecedent_wetness_index": self.antecedent_wetness_index,
            "cloudiness_index": self.cloudiness_index,
            "convective_potential_index": self.convective_potential_index,
            "seasonality_factor": self.seasonality_factor,
            "precipitation_spell_index": self.precipitation_spell_index,
        }
        for name, value in bounded_fields.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"'{name}' must be numeric, got {type(value).__name__}")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"'{name}' must be within [0, 1], got {value}")

        if not isinstance(self.scenario_moisture_factor, (int, float)):
            raise TypeError(f"'scenario_moisture_factor' must be numeric, got {type(self.scenario_moisture_factor).__name__}")

    @property
    def advection_speed_mps(self) -> float:
        """Return effective advection speed."""
        return self.advection.speed_mps

    @property
    def advection_direction_deg(self) -> float:
        """Return effective advection direction."""
        return self.advection.direction_deg

    @property
    def advection_u_mps(self) -> float:
        """Return effective advection x component."""
        return self.advection.u_mps

    @property
    def advection_v_mps(self) -> float:
        """Return effective advection y component."""
        return self.advection.v_mps


@dataclass(frozen=True)
class StormEnvironmentInput:
    """Internal interface exposed to the future storm / precipitation generator."""

    regime: MeteorologicalRegime
    storm_trigger_factor: float
    storm_organization_factor: float
    moisture_availability: float
    advection_u_mps: float
    advection_v_mps: float
    background_temperature_c: float
    cloudiness_index: float

    def __post_init__(self) -> None:
        bounded_fields = {
            "storm_trigger_factor": self.storm_trigger_factor,
            "storm_organization_factor": self.storm_organization_factor,
            "moisture_availability": self.moisture_availability,
            "cloudiness_index": self.cloudiness_index,
        }
        for name, value in bounded_fields.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"'{name}' must be numeric, got {type(value).__name__}")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"'{name}' must be within [0, 1], got {value}")


class LatentEnvironmentModel:
    """Stateful generator of latent meteorological states."""

    def __init__(self, config: LatentEnvironmentConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)

    def next_state(
        self,
        step: int,
        timestamp: datetime,
        previous_state: LatentEnvironmentState | None = None,
    ) -> LatentEnvironmentState:
        """Generate the next latent meteorological state."""
        if not isinstance(step, int) or step < 0:
            raise ValueError(f"'step' must be a non-negative integer, got {step!r}")
        if not isinstance(timestamp, datetime):
            raise TypeError(f"'timestamp' must be a datetime, got {type(timestamp).__name__}")

        seasonality_factor = self._seasonality_factor(timestamp)

        if previous_state is None:
            regime = self.config.initial_regime
        else:
            regime = self._sample_next_regime(previous_state.regime, seasonality_factor)

        regime_profile = get_regime_profile(regime)

        background_temperature_c = self._compute_temperature(
            regime_profile.temperature_anomaly_c,
            seasonality_factor,
            previous_state.background_temperature_c if previous_state is not None else None,
        )

        advection = self._compute_advection(regime, previous_state.advection if previous_state is not None else None)

        antecedent_wetness_index = self._compute_wetness(
            regime_profile.wetness_equilibrium,
            seasonality_factor,
            previous_state.antecedent_wetness_index if previous_state is not None else None,
        )

        precipitation_spell_index = self._compute_precipitation_spell_index(
            regime=regime,
            wetness_index=antecedent_wetness_index,
            previous_spell_index=(previous_state.precipitation_spell_index if previous_state is not None else None),
        )

        cloudiness_index = self._compute_cloudiness(
            regime_profile.cloudiness_index,
            antecedent_wetness_index,
            previous_state.cloudiness_index if previous_state is not None else None,
        )

        convective_potential_index = self._compute_convective_potential(
            regime_profile.convective_potential_index,
            background_temperature_c,
            antecedent_wetness_index,
            seasonality_factor,
            previous_state.convective_potential_index if previous_state is not None else None,
        )

        return LatentEnvironmentState(
            step=step,
            timestamp=timestamp,
            regime=regime,
            background_temperature_c=background_temperature_c,
            advection=advection,
            antecedent_wetness_index=antecedent_wetness_index,
            precipitation_spell_index=precipitation_spell_index,
            cloudiness_index=cloudiness_index,
            convective_potential_index=convective_potential_index,
            seasonality_factor=seasonality_factor,
            scenario_moisture_factor=self._scenario_moisture_factor(),
        )

    def build_storm_environment_input(
        self,
        state: LatentEnvironmentState,
    ) -> StormEnvironmentInput:
        """Convert a latent state into an internal forcing view for storm generation."""
        # TODO(phase2): tune storm_trigger_factor and storm_organization_factor after phase 3.
        # * factors that affect the generation of a storm
        spell_weighted_moisture = _clamp01(0.70 * state.antecedent_wetness_index + 0.30 * state.precipitation_spell_index)

        storm_trigger_factor = _clamp01(
            0.50 * state.convective_potential_index
            + 0.25 * state.antecedent_wetness_index
            + 0.10 * state.precipitation_spell_index
            + 0.15 * state.seasonality_factor
        )

        storm_organization_factor = _clamp01(
            0.50 * state.cloudiness_index
            + 0.30 * state.antecedent_wetness_index
            + 0.10 * state.precipitation_spell_index
            + 0.10 * self._regime_organization_bonus(state.regime)
        )

        return StormEnvironmentInput(
            regime=state.regime,
            storm_trigger_factor=storm_trigger_factor,
            storm_organization_factor=storm_organization_factor,
            moisture_availability=spell_weighted_moisture,
            advection_u_mps=state.advection_u_mps,
            advection_v_mps=state.advection_v_mps,
            background_temperature_c=state.background_temperature_c,
            cloudiness_index=state.cloudiness_index,
        )

    def _seasonality_factor(self, timestamp: datetime) -> float:
        """Return a normalized seasonality factor in [0, 1]."""
        day_of_year = timestamp.timetuple().tm_yday
        angular_phase = 2.0 * math.pi * ((day_of_year - self.config.peak_temperature_day_of_year) / 365.25)
        return _clamp01(0.5 * (1.0 + math.cos(angular_phase)))

    def _sample_next_regime(
        self,
        previous_regime: MeteorologicalRegime,
        seasonality_factor: float,
    ) -> MeteorologicalRegime:
        """Sample the next weather regime with simple persistence and seasonal bias."""
        if self._rng.random() < self.config.regime_persistence:
            return previous_regime

        # TODO(phase2): review regime transition weights once seasonal scenarios are validated.
        weights = {
            MeteorologicalRegime.STABLE_DRY: 1.0 + 0.35 * (1.0 - seasonality_factor) + self._stable_dry_scenario_bonus(),
            MeteorologicalRegime.TRANSITIONAL: 1.0,
            MeteorologicalRegime.CONVECTIVE: 1.0 + 0.60 * seasonality_factor + self._convective_scenario_bonus(),
            MeteorologicalRegime.FRONTAL_PERSISTENT: 1.0 + 0.60 * (1.0 - seasonality_factor) + self._frontal_scenario_bonus(),
        }

        regimes = tuple(weights.keys())
        probabilities = np.array([max(1e-6, weights[regime]) for regime in regimes], dtype=float)
        probabilities /= probabilities.sum()

        choice_index = int(self._rng.choice(len(regimes), p=probabilities))
        return regimes[choice_index]

    def _compute_temperature(
        self,
        regime_temperature_anomaly_c: float,
        seasonality_factor: float,
        previous_temperature_c: float | None,
    ) -> float:
        """Compute background temperature with seasonal forcing and temporal memory."""
        seasonal_anomaly_c = self.config.annual_temperature_amplitude_c * (2.0 * seasonality_factor - 1.0)
        thermal_offset_c = self._thermal_scenario_offset_c()

        target_temperature_c = self.config.mean_temperature_c + seasonal_anomaly_c + thermal_offset_c + regime_temperature_anomaly_c

        noise_c = float(self._rng.normal(0.0, self.config.temperature_noise_std_c))

        if previous_temperature_c is None:
            return float(target_temperature_c + noise_c)

        return float(
            self.config.temperature_memory * previous_temperature_c
            + (1.0 - self.config.temperature_memory) * target_temperature_c
            + noise_c
        )

    def _compute_advection(
        self,
        regime: MeteorologicalRegime,
        previous_advection: AdvectionField | None,
    ) -> AdvectionField:
        """Compute effective advection with regime-dependent perturbations and memory."""
        base_speed_mps = self.config.prevailing_advection_speed_mps + self._regime_speed_adjustment_mps(regime)
        candidate_speed_mps = max(
            0.0,
            float(self._rng.normal(base_speed_mps, self.config.advection_speed_std_mps)),
        )
        candidate_direction_deg = self.config.prevailing_advection_direction_deg + float(
            self._rng.normal(0.0, self.config.advection_direction_std_deg)
        )
        candidate = AdvectionField(
            speed_mps=candidate_speed_mps,
            direction_deg=candidate_direction_deg,
        )

        if previous_advection is None:
            return candidate

        blended_u = self.config.advection_memory * previous_advection.u_mps + (1.0 - self.config.advection_memory) * candidate.u_mps
        blended_v = self.config.advection_memory * previous_advection.v_mps + (1.0 - self.config.advection_memory) * candidate.v_mps
        return AdvectionField.from_uv(blended_u, blended_v)

    def _compute_wetness(
        self,
        regime_wetness_equilibrium: float,
        seasonality_factor: float,
        previous_wetness: float | None,
    ) -> float:
        """Compute antecedent wetness with memory and scenario drift."""
        season_wetness_adjustment = 0.10 * (1.0 - seasonality_factor)
        target_wetness = _clamp01(regime_wetness_equilibrium + self._scenario_moisture_shift() + season_wetness_adjustment)

        if previous_wetness is None:
            return target_wetness

        updated_wetness = (
            self.config.wetness_memory * previous_wetness
            + (1.0 - self.config.wetness_memory) * target_wetness
            - self.config.dry_drift_per_step
        )
        return _clamp01(updated_wetness)

    def _compute_precipitation_spell_index(
        self,
        *,
        regime: MeteorologicalRegime,
        wetness_index: float,
        previous_spell_index: float | None,
    ) -> float:
        """Compute a smooth wet/dry spell index in [0, 1].

        Interpretation
        --------------
        - values near 0 -> dry spell
        - values near 1 -> wet spell

        The target spell depends on:
        - the current regime
        - the current antecedent wetness
        - the user-facing moisture scenario
        and evolves with temporal memory.
        """
        regime_target = self._regime_spell_target(regime)

        wetness_target = self.config.dry_spell_target + wetness_index * (self.config.wet_spell_target - self.config.dry_spell_target)

        target_spell_index = _clamp01(0.60 * regime_target + 0.40 * wetness_target + self._scenario_spell_shift())

        if previous_spell_index is None:
            return target_spell_index

        noisy_target = _clamp01(target_spell_index + float(self._rng.normal(0.0, self.config.spell_noise_std)))

        return _clamp01(self.config.spell_memory * previous_spell_index + (1.0 - self.config.spell_memory) * noisy_target)

    def _regime_spell_target(self, regime: MeteorologicalRegime) -> float:
        """Return the baseline wet/dry spell tendency associated with a regime."""
        targets = {
            MeteorologicalRegime.STABLE_DRY: 0.10,
            MeteorologicalRegime.TRANSITIONAL: 0.45,
            MeteorologicalRegime.CONVECTIVE: 0.55,
            MeteorologicalRegime.FRONTAL_PERSISTENT: 0.85,
        }
        return targets[regime]

    def _scenario_spell_shift(self) -> float:
        """Return a small additive shift for the wet/dry spell tendency."""
        shifts = {
            MoistureScenario.DRY: -0.10,
            MoistureScenario.NORMAL: 0.0,
            MoistureScenario.WET: 0.10,
        }
        return shifts[self.config.moisture_scenario]

    def _compute_cloudiness(
        self,
        regime_cloudiness_index: float,
        wetness_index: float,
        previous_cloudiness: float | None,
    ) -> float:
        """Compute cloudiness as a smooth function of regime and wetness."""
        target_cloudiness = _clamp01(regime_cloudiness_index + 0.25 * (wetness_index - 0.5))

        if previous_cloudiness is None:
            return target_cloudiness

        return _clamp01(
            self.config.cloudiness_memory * previous_cloudiness + (1.0 - self.config.cloudiness_memory) * target_cloudiness
        )

    def _compute_convective_potential(
        self,
        regime_convective_index: float,
        background_temperature_c: float,
        wetness_index: float,
        seasonality_factor: float,
        previous_convective_potential: float | None,
    ) -> float:
        """Compute a simplified convective potential index."""
        # * Temperature support
        # If the temperature is above average:
        #       increase the thermal support.
        # If it's below average:
        #       lower it.
        temperature_support = _clamp01(
            0.5
            + 0.5
            * (
                (background_temperature_c - self.config.mean_temperature_c)
                / max(1.0, self.config.annual_temperature_amplitude_c + 4.0)
            )
        )

        target_convective_potential = _clamp01(
            regime_convective_index
            + 0.20 * (seasonality_factor - 0.5)
            + 0.20 * (wetness_index - 0.5)
            + 0.15 * (temperature_support - 0.5)
        )

        if previous_convective_potential is None:
            return target_convective_potential

        return _clamp01(
            self.config.convective_memory * previous_convective_potential
            + (1.0 - self.config.convective_memory) * target_convective_potential
        )

    def _thermal_scenario_offset_c(self) -> float:
        """Return the thermal offset associated with the selected scenario."""
        offsets = {
            ThermalScenario.COLD: -2.0,
            ThermalScenario.NORMAL: 0.0,
            ThermalScenario.WARM: 2.0,
        }
        return offsets[self.config.thermal_scenario]

    def _scenario_moisture_shift(self) -> float:
        """Return the wetness shift associated with the selected moisture scenario."""
        shifts = {
            MoistureScenario.DRY: -0.15,
            MoistureScenario.NORMAL: 0.0,
            MoistureScenario.WET: 0.15,
        }
        return shifts[self.config.moisture_scenario]

    def _scenario_moisture_factor(self) -> float:
        """Return a diagnostic moisture factor associated with the selected scenario."""
        factors = {
            MoistureScenario.DRY: 0.85,
            MoistureScenario.NORMAL: 1.00,
            MoistureScenario.WET: 1.15,
        }
        return factors[self.config.moisture_scenario]

    def _stable_dry_scenario_bonus(self) -> float:
        """Return extra sampling weight for the stable-dry regime."""
        if self.config.moisture_scenario is MoistureScenario.DRY:
            return 0.35
        return 0.0

    def _convective_scenario_bonus(self) -> float:
        """Return extra sampling weight for the convective regime."""
        if self.config.thermal_scenario is ThermalScenario.WARM:
            return 0.25
        return 0.0

    def _frontal_scenario_bonus(self) -> float:
        """Return extra sampling weight for the frontal-persistent regime."""
        if self.config.moisture_scenario is MoistureScenario.WET:
            return 0.35
        return 0.0

    def _regime_speed_adjustment_mps(self, regime: MeteorologicalRegime) -> float:
        """Return a simple regime-dependent correction for advection speed."""
        adjustments = {
            MeteorologicalRegime.STABLE_DRY: -0.5,
            MeteorologicalRegime.TRANSITIONAL: 0.0,
            MeteorologicalRegime.CONVECTIVE: 0.5,
            MeteorologicalRegime.FRONTAL_PERSISTENT: 1.0,
        }
        return adjustments[regime]

    def _regime_organization_bonus(self, regime: MeteorologicalRegime) -> float:
        """Return a simple organization bonus used by the future storm generator."""
        bonuses = {
            MeteorologicalRegime.STABLE_DRY: 0.05,
            MeteorologicalRegime.TRANSITIONAL: 0.40,
            MeteorologicalRegime.CONVECTIVE: 0.60,
            MeteorologicalRegime.FRONTAL_PERSISTENT: 0.85,
        }
        return bonuses[regime]
