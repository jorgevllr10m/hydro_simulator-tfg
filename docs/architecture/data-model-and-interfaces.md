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

This means array indexing follows:

```text
field[cell_y, cell_x]
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
- capacity in `m3`
- initial storage in `m3`

Capacity and initial storage are reservoir-specific. Other storage and operating-rule parameters are shared through the routing configuration in the current MVP.

### Sensor definition

Each sensor stores:

- name
- sensor type
- grid cell `(cell_y, cell_x)`

Supported sensor types are:

- `precipitation`
- `discharge`
- `reservoir_storage`

A `reservoir_storage` sensor must coincide exactly with one reservoir cell.

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
- step index array
- month array
- season array
- total duration

The external YAML uses `start_date`, `end_date`, and `time_step_hours`; the loader converts these to the internal representation.

## Dynamic simulation state

`SimulationState` is the merged physical state for one time step. It contains the main products from all physical modules:

- precipitation
- air temperature
- PET
- soil moisture
- surface runoff
- channel flow
- outlet discharge

Optional fields include:

- background precipitation
- storm mask
- AET
- shortwave radiation
- net radiation
- infiltration
- subsurface runoff
- reservoir inflow
- reservoir storage
- reservoir release
- reservoir spill

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

- domain
- step
- timestamp
- precipitation
- air temperature

Output:

- PET
- shortwave radiation
- net radiation

### Hydrology

Input:

- domain
- step
- timestamp
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

- domain
- step
- timestamp
- surface runoff
- PET
- optional subsurface runoff

Output:

- lateral inflow field
- total cell inflow field
- routed channel-flow field
- outlet discharge
- reservoir inflow
- requested reservoir release
- reservoir storage fraction
- reservoir surface area
- reservoir evaporation loss
- reservoir storage
- reservoir release
- reservoir spill
- reservoir total outflow
- reservoir operation zones

Only a subset of reservoir diagnostics is currently persisted in the truth dataset.

### Observation

Input:

- domain
- step
- timestamp
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

### Coordinates

- `time`
- `y`
- `x`
- `reservoir`
- `sensor`

The `sensor` coordinate may be present because the shared coordinate builder is used, but observation values are stored in the observation dataset.

### Static fields

- `basin_mask`

### Meteorology

- `precipitation`
- `background_precipitation`
- `storm_mask`
- `air_temperature`

### Energy

- `pet`
- `shortwave_radiation`
- `net_radiation`

### Hydrology

- `aet`
- `soil_moisture`
- `infiltration`
- `surface_runoff`
- `subsurface_runoff`

### Routing

- `channel_flow`
- `outlet_discharge`

### Reservoirs

- `reservoir_inflow`
- `reservoir_storage`
- `reservoir_release`
- `reservoir_spill`

## Observation dataset

Contains synthetic observation products across time and sensors.

Main variables:

- `obs_precipitation`
- `obs_discharge`
- `obs_storage`
- `obs_mask`
- `obs_quality_flag`

Coordinates include:

- `time`
- `sensor`

It also stores sensor metadata:

- `sensor_name`
- `sensor_type`
- `sensor_cell_y`
- `sensor_cell_x`

## Dataset writing workflow

The runner:

1. creates empty truth and observation datasets
2. executes one step of the simulator
3. merges physical outputs into `SimulationState`
4. writes that state into the truth dataset
5. writes the observation product into the observation dataset
6. appends one row to the step-summary CSV buffer
7. appends one row per sensor to the observation CSV buffer

This avoids ad hoc file writing inside each module and keeps persistence centralized.

## CSV model

The runner also writes two families of CSV files:

- step summary rows
- per-sensor observation rows

Both are written in English column names and also in Spanish translated column names with `_es` suffix.

CSV formatting is spreadsheet-friendly:

- semicolon separator: `;`
- comma decimal separator: `,`

## Why this model works well

The project separates information into three levels:

- **static world definition**
- **dynamic physical state**
- **synthetic observations**

That separation makes the codebase easier to test, extend, and document.
