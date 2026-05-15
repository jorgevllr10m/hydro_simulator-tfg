# Configuration architecture

## Configuration layers

The simulator resolves configuration from three YAML sources.

## 1. Master configuration

Path example:

```text
configs/config.yaml
```

Main responsibilities:

- define the run name
- define the output directory root
- define the simulation window
- select the domain preset
- select the scenario

Typical structure:

```yaml
run:
  name: "wet"
  output_dir: "outputs/runs"

simulation:
  start_date: "2026-06-01"
  end_date: "2026-08-10"
  time_step_hours: 1

domain:
  preset: "medium"

scenario:
  name: "wet"
```

Interpretation of the simulation window:

- `start_date` is inclusive
- `end_date` is exclusive
- `time_step_hours` must divide the interval exactly

Internally, this is converted to:

- start datetime
- `dt_seconds`
- `n_steps`

## 2. Domain preset

Path pattern:

```text
configs/domain/<preset_name>.yaml
```

Responsibilities:

- define the grid
- define the reservoirs
- define the sensors

Typical structure:

```yaml
grid:
  nx: 80
  ny: 60
  dx: 1000.0
  dy: 1000.0
  x0: 0.0
  y0: 0.0

reservoirs:
  - name: upper_reservoir
    cell_y: 15
    cell_x: 20
    capacity: 3000000.0
    initial_storage: 1800000.0

sensors:
  - name: rain_gauge_upper
    sensor_type: precipitation
    cell_y: 10
    cell_x: 15

  - name: flow_sensor_outlet
    sensor_type: discharge
    cell_y: 59
    cell_x: 79

  - name: storage_sensor_upper_reservoir
    sensor_type: reservoir_storage
    cell_y: 15
    cell_x: 20
```

The current MVP builds a basin mask that is fully active over the whole grid.

## 3. Scenario file

Path pattern:

```text
configs/scenarios/<scenario_name>.yaml
```

Responsibilities:

- override selected meteorology parameters
- override energy settings such as latitude or PET multiplier
- override hydrology parameters
- override routing and reservoir rules
- override observation settings

Scenario files are intentionally sparse. Any omitted field falls back to the default runtime dataclass value.

## Scenario structure

A scenario may contain the following top-level sections:

```yaml
meteo:
  latent_environment:
    random_seed: 1701
    initial_regime: frontal_persistent
    thermal_scenario: normal
    moisture_scenario: wet
    regime_persistence: 0.82
    spell_memory: 0.86

  storm_birth:
    expected_births_per_step: 0.72
    mean_peak_intensity_mmph: 10.8
    mean_duration_steps: 6
    band_cluster_probability: 0.45

  background:
    enabled: true
    random_seed: 2701
    temporal_persistence: 0.86
    max_intensity_mm_dt: 3.2

  temperature:
    enabled: true
    random_seed: 4701
    gradient_amplitude_c: 0.70

energy:
  latitude_deg: 40.0
  pet:
    pet_multiplier: 0.95

hydro:
  soil:
    capacity_mm: 180.0
    initial_relative: 0.65
    max_infiltration_mm_dt: 17.0
    percolation_rate_mm_dt: 3.0

  runoff:
    subsurface_runoff_fraction: 1.0

routing:
  enable_reservoirs: true
  channel:
    channel_time_constant_hours: 1.0
  reservoir_rules:
    min_release_m3s: 0.2
    target_release_m3s: 1.5

obs:
  random_seed: 3701
  precipitation:
    enabled: true
    noise_std_mm_dt: 0.15
    missing_probability: 0.02
    detection_threshold_mm_dt: 0.05
  discharge:
    enabled: true
    relative_noise_std: 0.08
    missing_probability: 0.02
    detection_threshold_m3s: 0.10
  reservoir_storage:
    enabled: true
    noise_std_m3: 15000.0
```

Only the fields exposed in `simulator.config.schemas` are accepted in the YAML layer. Runtime dataclasses contain additional defaults that are not necessarily exposed as user-facing overrides.

## Exposed scenario controls

### Meteorology

The YAML schema exposes selected controls for:

- latent environment: random seed, initial regime, thermal scenario, moisture scenario, regime persistence, spell memory
- storm birth: expected births, mean peak intensity, mean duration, band-cluster probability
- background precipitation: enable flag, random seed, temporal persistence, maximum intensity
- temperature field: enable flag, random seed, gradient amplitude

### Energy

The YAML schema exposes:

- `latitude_deg`
- `pet.pet_multiplier`

Solar-radiation coefficients and Priestley-Taylor physical constants remain runtime defaults in the current user-facing schema.

### Hydrology

The YAML schema exposes:

- soil capacity
- initial relative saturation
- maximum infiltration
- percolation rate
- subsurface runoff fraction

### Routing

The YAML schema exposes:

- reservoir enable/disable switch
- channel routing time constant
- minimum release
- target release

Other reservoir storage and rule-curve settings remain runtime defaults in the current user-facing schema.

### Observations

The YAML schema exposes:

- observation random seed
- enable flags by sensor family
- precipitation absolute noise, missing probability, detection threshold
- discharge relative noise, missing probability, detection threshold
- reservoir-storage absolute noise

When a precipitation or discharge detection threshold is explicitly provided and is greater than zero, the loader enables left-censoring internally.

## Validation path

YAML files are validated in two stages:

1. **Schema validation**
   - done with Pydantic models in `simulator.config.schemas`
   - catches invalid structure, bounds, duplicated names, and unsupported sensor types

2. **Runtime-object construction**
   - done in `simulator.config.loader`
   - converts validated YAML data into internal dataclass objects used by the simulator

## Runtime builders

The loader builds the following internal objects:

- `TimeDefinition`
- `SimulationDomain`
- `StormPrecipitationConfig`
- `EnergyBalanceConfig`
- `HydroConfig`
- `RegulatedRoutingConfig`
- `ObservationConfig`

This separation is useful because the external YAML schema is user-facing, while the runtime dataclasses are simulator-facing.

## Important conventions

### Relative paths

If `run.output_dir` is relative, the final run directory is:

```text
<project_root>/<run.output_dir>/<run.name>
```

If `run.output_dir` is absolute, the final run directory is:

```text
<absolute_output_dir>/<run.name>
```

### Domain selection

The selected preset is resolved from:

```text
configs/domain/<preset>.yaml
```

### Scenario selection

The selected scenario is resolved from:

```text
configs/scenarios/<scenario>.yaml
```

### Empty scenario files

Scenario files may be empty. In that case they are interpreted as no overrides.

## What is configured where

### Master config

Use it for:

- simulation period
- time step
- choosing which domain to run
- choosing which scenario to run
- selecting the output run name

### Domain preset

Use it for:

- spatial size and resolution
- number and location of reservoirs
- reservoir capacity and initial storage
- number, location, and type of sensors

### Scenario file

Use it for:

- behavior changes without editing code
- dry, wet, warm, cold, persistent-storm, and extreme-event cases
- changes in sensor quality
- changes in routing speed or reservoir releases
- changes in PET demand and initial soil wetness

## Why this design is useful

This layout separates three concerns cleanly:

- **what world exists** → domain preset
- **when it is simulated** → master config
- **how the world behaves in this run** → scenario file

That makes runs easy to compare and keeps experiment management simple.
