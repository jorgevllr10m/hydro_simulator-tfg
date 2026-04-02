from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulator.core.contracts import MeteoInput, MeteoOutput
from simulator.meteo.background_field import (
    build_air_temperature_field,
    build_background_precipitation_field,
)
from simulator.meteo.latent_state import (
    LatentEnvironmentConfig,
    LatentEnvironmentModel,
    LatentEnvironmentState,
    StormEnvironmentInput,
)
from simulator.meteo.lifecycle import StormLifecycleConfig
from simulator.meteo.render import StormRenderConfig, render_storms_to_step_fields
from simulator.meteo.storm_birth import StormBirthConfig, spawn_storms
from simulator.meteo.storm_objects import StormCell


@dataclass(frozen=True)
class StormPrecipitationConfig:
    """Top-level configuration of the phase-3 meteorology model."""

    latent_environment: LatentEnvironmentConfig = field(default_factory=LatentEnvironmentConfig)
    birth: StormBirthConfig = field(default_factory=StormBirthConfig)
    lifecycle: StormLifecycleConfig = field(default_factory=StormLifecycleConfig)
    render: StormRenderConfig = field(default_factory=StormRenderConfig)

    random_seed: int = 4321

    def __post_init__(self) -> None:
        if not isinstance(self.latent_environment, LatentEnvironmentConfig):
            raise TypeError(f"'latent_environment' must be a LatentEnvironmentConfig, got {type(self.latent_environment).__name__}")

        if not isinstance(self.birth, StormBirthConfig):
            raise TypeError(f"'birth' must be a StormBirthConfig, got {type(self.birth).__name__}")

        if not isinstance(self.lifecycle, StormLifecycleConfig):
            raise TypeError(f"'lifecycle' must be a StormLifecycleConfig, got {type(self.lifecycle).__name__}")

        if not isinstance(self.render, StormRenderConfig):
            raise TypeError(f"'render' must be a StormRenderConfig, got {type(self.render).__name__}")

        if not isinstance(self.random_seed, int):
            raise TypeError(f"'random_seed' must be an int, got {type(self.random_seed).__name__}")


@dataclass(frozen=True)
class StormStepDiagnostics:
    """Lightweight diagnostics of the latest meteorological step."""

    latent_state: LatentEnvironmentState
    storm_environment: StormEnvironmentInput
    n_new_storms: int
    n_active_storms: int


class StormPrecipitationModel:
    """Stateful meteorology model for phase 3.

    This model couples:
    - latent meteorological environment
    - storm spawning
    - storm life cycle and advection
    - storm rasterization
    - simple background fields

    The public step method consumes a MeteoInput and returns a MeteoOutput,
    matching the project module contracts.
    """

    def __init__(self, config: StormPrecipitationConfig) -> None:
        if not isinstance(config, StormPrecipitationConfig):
            raise TypeError(f"'config' must be a StormPrecipitationConfig, got {type(config).__name__}")

        self.config = config

        self._storm_rng = np.random.default_rng(config.random_seed)
        self._latent_model = LatentEnvironmentModel(config.latent_environment)

        self._latest_latent_state: LatentEnvironmentState | None = None
        self._active_storms: list[StormCell] = []
        self._next_storm_id: int = 0
        self._latest_diagnostics: StormStepDiagnostics | None = None

    @property
    def latest_latent_state(self) -> LatentEnvironmentState | None:
        """Return the latest latent meteorological state."""
        return self._latest_latent_state

    @property
    def latest_diagnostics(self) -> StormStepDiagnostics | None:
        """Return diagnostics from the most recent completed step."""
        return self._latest_diagnostics

    @property
    def active_storm_count(self) -> int:
        """Return the number of currently tracked active storms."""
        return len(self._active_storms)

    @property
    def active_storms(self) -> tuple[StormCell, ...]:
        """Return the currently tracked storms as a read-only tuple view."""
        return tuple(self._active_storms)

    def reset(self) -> None:
        """Reset internal state and RNGs to their initial reproducible state."""
        self._storm_rng = np.random.default_rng(self.config.random_seed)
        self._latent_model = LatentEnvironmentModel(self.config.latent_environment)

        self._latest_latent_state = None
        self._active_storms = []
        self._next_storm_id = 0
        self._latest_diagnostics = None

    def step(self, meteo_input: MeteoInput) -> MeteoOutput:
        """Advance the meteorology model by one simulation step.

        Temporal convention used here
        -----------------------------
        1. Build latent environment for the current step
        2. Remove expired storms from previous steps
        3. Spawn new storms from current forcing
        4. Initialize newborn storms for immediate rendering
        5. Render current-step precipitation and mask
        6. Advance storms to prepare the next step
        7. Remove storms that expire after the advance
        """
        if not isinstance(meteo_input, MeteoInput):
            raise TypeError(f"'meteo_input' must be a MeteoInput, got {type(meteo_input).__name__}")

        latent_state = self._latent_model.next_state(
            step=meteo_input.step,
            timestamp=meteo_input.timestamp,
            previous_state=self._latest_latent_state,
        )
        self._latest_latent_state = latent_state

        storm_environment = self._latent_model.build_storm_environment_input(latent_state)

        self._drop_expired_storms()

        new_storms = spawn_storms(
            rng=self._storm_rng,
            next_storm_id=self._next_storm_id,
            domain=meteo_input.domain,
            env=storm_environment,
            config=self.config.birth,
        )
        self._next_storm_id += len(new_storms)

        self._initialize_newborn_storms(new_storms)
        self._active_storms.extend(new_storms)

        storm_precipitation_mm_dt, storm_mask = render_storms_to_step_fields(
            domain=meteo_input.domain,
            storms=self._active_storms,
            lifecycle=self.config.lifecycle,
            render_config=self.config.render,
        )

        background_precipitation = build_background_precipitation_field(meteo_input.domain)

        precipitation = background_precipitation + storm_precipitation_mm_dt

        air_temperature = build_air_temperature_field(
            meteo_input.domain,
            latent_state.background_temperature_c,
        )

        meteo_output = MeteoOutput(
            precipitation=precipitation,
            air_temperature=air_temperature,
            background_precipitation=background_precipitation,
            storm_mask=storm_mask,
        )

        self._latest_diagnostics = StormStepDiagnostics(
            latent_state=latent_state,
            storm_environment=storm_environment,
            n_new_storms=len(new_storms),
            n_active_storms=len(self._active_storms),
        )

        self._advance_active_storms(meteo_input.domain.time.dt_seconds)
        self._drop_expired_storms()

        return meteo_output

    def _initialize_newborn_storms(self, storms: list[StormCell]) -> None:
        """Shift newborn storms into the first renderable discrete-time state.

        Why this is needed:
        - newborn storms are spawned with age_steps = 0
        - under the current lifecycle convention, age_steps = 0 implies
          zero rainfall intensity at render time
        - to let newborn storms contribute immediately while preserving
          the intended number of rendered active steps, we:
              * increase duration_steps by 1
              * set age_steps to 1
        """
        for storm in storms:
            if storm.age_steps != 0:
                continue

            storm.duration_steps += 1
            storm.age_steps = 1

    def _advance_active_storms(self, dt_seconds: int | float) -> None:
        """Advance all currently active storms by one model time step."""
        for storm in self._active_storms:
            storm.advance(dt_seconds)

    def _drop_expired_storms(self) -> None:
        """Remove storms that are no longer alive."""
        self._active_storms = [storm for storm in self._active_storms if storm.is_alive]
