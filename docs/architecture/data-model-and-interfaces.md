# Data model and interfaces

## Static domain objects

The simulator distinguishes between static definitions and dynamic state.

### Grid definition
The grid is regular and rectangular, with:

- `nx`, `ny`
- `dx`, `dy`
- optional origin and CRS

The canonical array shape is always:

```text
(ny, nx)
```

### Basin definition
The basin is a boolean mask over the grid:

- `True` means active basin cell
- `False` means outside the basin

In the current MVP, the loader builds a fully active basin mask.

### Reservoir definition
Each reservoir stores:

- name
- grid cell `(cell_y, cell_x)`
- capacity
- initial storage

### Sensor definition
Each sensor stores:

- name
- sensor type
- grid cell `(cell_y, cell_x)`

Supported sensor types are:

- `precipitation`
- `discharge`
- `reservoir_storage`

### Simulation domain
`SimulationDomain` groups:

- spatial domain
- temporal domain
- reservoirs
- sensors

This is the static world passed to all modules.

## Time model

The internal time model stores:

- simulation start
- step duration in seconds
- number of steps
- calendar interpretation

It also exposes:

- timestamp array
- month array
- season array
- total duration

## Dynamic simulation state

`SimulationState` is the merged physical state for one time step. It contains the main products from all physical modules:

- precipitation
- air temperature
- PET
- soil moisture
- runoff
- channel flow
- outlet discharge

Optional fields include:

- background precipitation
- storm mask
- AET
- shortwave and net radiation
- infiltration
- subsurface runoff
- reservoir inflow, storage, release, and spill

This object is not the internal state of any one module. It is the **integrated physical snapshot** written to the truth dataset.

## Module contracts

Each module communicates through typed dataclasses.

### Meteorology
Input:
- domain
- step
- timestamp
- optional previous state

Output:
- precipitation
- air temperature
- optional background precipitation
- optional storm mask

### Energy
Input:
- precipitation
- air temperature

Output:
- PET
- shortwave radiation
- net radiation

### Hydrology
Input:
- precipitation
- PET

Output:
- soil moisture
- infiltration
- surface runoff
- AET
- optional subsurface runoff

### Routing
Input:
- surface runoff
- PET
- optional subsurface runoff

Output:
- lateral inflow
- total cell inflow
- routed channel flow
- outlet discharge
- reservoir diagnostics and state variables

### Observation
Input:
- precipitation
- channel flow
- optional reservoir storage

Output:
- observation vectors
- availability mask
- quality flags

## Dataset model

The project writes two xarray datasets.

## Truth dataset
Contains simulated physical variables across time, space, and reservoirs.

Main variable groups:

- meteorology
- energy
- hydrology
- routing
- reservoirs

Coordinates include:

- `time`
- `y`
- `x`
- `reservoir`

## Observation dataset
Contains synthetic observation products across time and sensors.

Main variable groups:

- `obs_precipitation`
- `obs_discharge`
- `obs_storage`
- `obs_mask`
- `obs_quality_flag`

Coordinates include:

- `time`
- `sensor`

It also stores sensor metadata:

- sensor name
- sensor type
- sensor cell indices

## Dataset writing workflow

The runner:

1. creates empty truth and observation datasets
2. executes one step of the simulator
3. merges physical outputs into `SimulationState`
4. writes that state into the truth dataset
5. writes the observation product into the observation dataset

This avoids ad hoc file writing inside each module and keeps persistence centralized.

## Why this model works well

The project separates information into three levels:

- **static world definition**
- **dynamic physical state**
- **synthetic observations**

That separation makes the codebase easier to test, extend, and document.
