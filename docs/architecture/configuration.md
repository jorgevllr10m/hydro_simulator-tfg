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
  name: "base_run"
  output_dir: "outputs/runs"

simulation:
  start_date: "2026-04-01"
  end_date: "2026-06-10"
  time_step_hours: 1

domain:
  preset: "medium"

scenario:
  name: "baseline"
```

Interpretation of the simulation window:

- `start_date` is inclusive
- `end_date` is exclusive
- `time_step_hours` must divide the interval exactly

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

## Validation path

YAML files are validated in two stages:

1. **schema validation**
   - done with Pydantic models in `simulator.config.schemas`
   - catches invalid structure, bounds, and unsupported values

2. **runtime-object construction**
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
Scenario files may be empty. In that case they are interpreted as “no overrides”.

## What is configured where

### Master config
Use it for:

- simulation period
- time step
- choosing which domain to run
- choosing which scenario to run

### Domain preset
Use it for:

- spatial size and resolution
- number and location of reservoirs
- number and location of sensors

### Scenario file
Use it for:

- behavior changes without editing code
- dry/wet/warm/cold cases
- changes in sensor quality
- changes in routing speed or reservoir releases

## Why this design is useful

This layout separates three concerns cleanly:

- **what world exists** → domain preset
- **when it is simulated** → master config
- **how the world behaves in this run** → scenario file

That makes runs easy to compare and keeps experiment management simple.
