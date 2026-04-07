from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from simulator.config.schemas import (
    DomainPresetConfig,
    MasterConfig,
    ScenarioConfig,
)
from simulator.core.time import TimeDefinition
from simulator.core.types import (
    BasinDefinition,
    GridDefinition,
    ReservoirDefinition,
    SensorDefinition,
    SimulationDomain,
    SpatialDomain,
)
from simulator.energy.antecedent import AntecedentConfig
from simulator.energy.model import EnergyBalanceConfig
from simulator.energy.pet import PETConfig
from simulator.meteo.background_field import BackgroundFieldConfig
from simulator.meteo.latent_state import LatentEnvironmentConfig
from simulator.meteo.precipitation_model import StormPrecipitationConfig
from simulator.meteo.storm_birth import StormBirthConfig


@dataclass(frozen=True)
class LoadedConfig:
    """Resolved external configuration of one simulation run.

    This object contains:
    - validated master config.yaml
    - validated domain preset
    - validated scenario file

    It also exposes convenience helpers to build the internal runtime
    objects actually used by the simulator.
    """

    config_path: Path
    master: MasterConfig
    domain_preset: DomainPresetConfig
    scenario: ScenarioConfig

    @property
    def project_root(self) -> Path:
        """Return the project root inferred from config.yaml location."""
        return self.config_path.resolve().parent.parent

    @property
    def run_name(self) -> str:
        """Return configured run name."""
        return self.master.run.name

    @property
    def run_output_dir(self) -> Path:
        """Return the concrete output directory for this run.

        The final path is:
        <project_root>/<run.output_dir>/<run.name>
        if output_dir is relative, or:
        <absolute_output_dir>/<run.name>
        if output_dir is absolute.
        """
        base_output_dir = Path(self.master.run.output_dir)

        if base_output_dir.is_absolute():
            return base_output_dir / self.master.run.name

        return self.project_root / base_output_dir / self.master.run.name

    @property
    def domain_preset_name(self) -> str:
        """Return selected domain preset name."""
        return self.master.domain.preset

    @property
    def scenario_name(self) -> str:
        """Return selected scenario name."""
        return self.master.scenario.name

    def build_time_definition(self) -> TimeDefinition:
        """Build internal TimeDefinition from the user-facing simulation window.

        External configuration uses:
        - start_date
        - end_date
        - time_step_hours

        Internal runtime uses:
        - start
        - dt_seconds
        - n_steps

        Convention:
        the simulation interval is interpreted as [start_date, end_date),
        i.e. start included, end excluded.
        """
        simulation = self.master.simulation

        start_dt = datetime.fromisoformat(simulation.start_date)
        end_dt = datetime.fromisoformat(simulation.end_date)
        dt_seconds = simulation.time_step_hours * 3600

        total_seconds = int((end_dt - start_dt).total_seconds())
        n_steps = total_seconds // dt_seconds

        return TimeDefinition(
            start=start_dt,
            dt_seconds=dt_seconds,
            n_steps=n_steps,
            calendar_type=simulation.calendar_type,
        )

    def build_simulation_domain(self) -> SimulationDomain:
        """Build the internal SimulationDomain from validated external config.

        Current MVP choice:
        - basin mask is fully active over the whole grid
        - grid, reservoirs and sensors come from the selected domain preset
        - time comes from the master config simulation window
        """
        grid = GridDefinition(**self.domain_preset.grid.model_dump())

        basin = BasinDefinition(mask=np.ones(grid.shape, dtype=bool))
        spatial = SpatialDomain(grid=grid, basin=basin)

        time = self.build_time_definition()

        reservoirs = tuple(ReservoirDefinition(**reservoir.model_dump()) for reservoir in self.domain_preset.reservoirs)

        sensors = tuple(SensorDefinition(**sensor.model_dump()) for sensor in self.domain_preset.sensors)

        return SimulationDomain(
            spatial=spatial,
            time=time,
            reservoirs=reservoirs,
            sensors=sensors,
        )

    # * Build meteo module config
    def build_storm_precipitation_config(self) -> StormPrecipitationConfig:
        """Build internal meteorology config from scenario overrides.

        Defaults are defined in the runtime dataclasses themselves.
        The scenario file only provides optional overrides.

        This means:
        - omitted YAML fields do not need to be written
        - omitted YAML fields fall back to runtime defaults
        """
        latent_environment_overrides = self.scenario.meteo.latent_environment.model_dump(exclude_none=True)
        storm_birth_overrides = self.scenario.meteo.storm_birth.model_dump(exclude_none=True)
        background_overrides = self.scenario.meteo.background.model_dump(exclude_none=True)

        latent_environment = LatentEnvironmentConfig(**latent_environment_overrides)
        storm_birth = StormBirthConfig(**storm_birth_overrides)
        background = BackgroundFieldConfig(**background_overrides)

        return StormPrecipitationConfig(
            latent_environment=latent_environment,
            birth=storm_birth,
            background=background,
        )

    # * Build energy module config
    def build_energy_balance_config(self) -> EnergyBalanceConfig:
        """Build internal energy config from scenario overrides.

        Defaults are defined in the runtime dataclasses themselves.
        The scenario file only provides optional overrides.

        This means:
        - omitted YAML fields do not need to be written
        - omitted YAML fields fall back to runtime defaults
        """
        energy = self.scenario.energy

        pet_overrides = energy.pet.model_dump(exclude_none=True)
        antecedent_overrides = energy.antecedent.model_dump(exclude_none=True)

        pet = PETConfig(**pet_overrides)
        antecedent = AntecedentConfig(**antecedent_overrides)

        energy_overrides: dict[str, Any] = {}
        if energy.latitude_deg is not None:
            energy_overrides["latitude_deg"] = energy.latitude_deg

        return EnergyBalanceConfig(
            pet=pet,
            antecedent=antecedent,
            **energy_overrides,
        )


def _read_yaml_mapping(
    path: str | Path,
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Read a YAML file and return its root mapping."""
    resolved_path = Path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

    if resolved_path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Configuration file must be a YAML file: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if content is None:
        if allow_empty:
            return {}
        raise ValueError(f"Configuration file is empty: {resolved_path}")

    if not isinstance(content, dict):
        raise ValueError(f"Configuration root must be a mapping/dictionary: {resolved_path}")

    return content


def _domain_preset_path(config_path: Path, preset_name: str) -> Path:
    """Return path to the selected domain preset file."""
    return config_path.resolve().parent / "domain" / f"{preset_name}.yaml"


def _scenario_path(config_path: Path, scenario_name: str) -> Path:
    """Return path to the selected scenario file."""
    return config_path.resolve().parent / "scenarios" / f"{scenario_name}.yaml"


def load_master_config(config_path: str | Path) -> MasterConfig:
    """Load and validate the master config.yaml file."""
    path = Path(config_path)
    mapping = _read_yaml_mapping(path)
    return MasterConfig.model_validate(mapping)


def load_domain_preset(
    *,
    config_path: str | Path,
    preset_name: str,
) -> DomainPresetConfig:
    """Load and validate one domain preset selected from the master config."""
    path = _domain_preset_path(Path(config_path), preset_name)
    mapping = _read_yaml_mapping(path)
    return DomainPresetConfig.model_validate(mapping)


def load_scenario(
    *,
    config_path: str | Path,
    scenario_name: str,
) -> ScenarioConfig:
    """Load and validate one scenario file selected from the master config.

    Empty scenario files are allowed and interpreted as no overrides.
    """
    path = _scenario_path(Path(config_path), scenario_name)
    mapping = _read_yaml_mapping(path, allow_empty=True)
    return ScenarioConfig.model_validate(mapping)


def load_config(config_path: str | Path) -> LoadedConfig:
    """Load and resolve the full external configuration stack.

    Resolution order:
    1. master config.yaml
    2. selected domain preset from configs/domain/
    3. selected scenario file from configs/scenarios/
    """
    path = Path(config_path).resolve()

    master = load_master_config(path)
    domain_preset = load_domain_preset(
        config_path=path,
        preset_name=master.domain.preset,
    )
    scenario = load_scenario(
        config_path=path,
        scenario_name=master.scenario.name,
    )

    return LoadedConfig(
        config_path=path,
        master=master,
        domain_preset=domain_preset,
        scenario=scenario,
    )
