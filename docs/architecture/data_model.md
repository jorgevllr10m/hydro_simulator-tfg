# Data Model

## 1. Purpose

This document defines the internal data model of the synthetic basin simulator. Its goal is to establish a common representation for spatial domain, temporal domain, dynamic state, historical outputs, dimensions, variables, units, and validation rules.

The main design objective is to ensure that all simulator modules operate on a shared and explicit data contract. This avoids implicit dependencies, reduces coupling between modules, and makes the system easier to test, extend, and document.

---

## 2. Architectural data layers

The simulator data model is structured into three complementary layers:

### 2.1. `SimulationDomain`

`SimulationDomain` contains the static definition of the simulation world. It describes what exists before the simulation starts and what does not change during execution.

It includes:

- spatial domain
- temporal domain
- reservoir definitions
- sensor definitions

This layer is used to describe the simulation context and to validate the consistency of all dynamic data.

### 2.2. `SimulationState`

`SimulationState` contains the dynamic state of the simulator at a single time step. It represents the current snapshot of the system.

It includes the physical fields and dynamic variables required by the simulation engine, such as precipitation, soil moisture, runoff, channel flow, and reservoir state.

This layer is the main object exchanged and reconstructed during each simulation step.

### 2.3. `SimulationDataset`

The historical output of the simulator is stored as an `xarray.Dataset`. This dataset contains time series of the main simulator variables and acts as the official persistent representation of the simulation results.

This layer is used for:

- storing full simulation outputs
- traceability
- diagnostics
- plotting and analysis
- future export to NetCDF or similar formats

The three-layer design can be summarized as follows:

- `SimulationDomain`: what exists
- `SimulationState`: how the system is now
- `SimulationDataset`: how the system evolved over time

---

## 3. Spatial domain

### 3.1. Grid model

The simulator uses a regular 2D rectangular grid as its spatial domain. This choice was made because it is simple, explicit, compatible with NumPy and xarray, and sufficient for a synthetic basin simulator MVP.

The spatial grid is defined by:

- `nx`: number of columns
- `ny`: number of rows
- `dx`: grid spacing in x direction, in meters
- `dy`: grid spacing in y direction, in meters
- `x0`: x origin of the domain
- `y0`: y origin of the domain
- `crs`: optional coordinate reference system identifier

### 3.2. Spatial conventions

The simulator adopts the following spatial conventions:

- 2D spatial arrays use shape `(ny, nx)`
- dimension order is always `(y, x)`
- `x` and `y` coordinates refer to **cell centers**
- coordinates are stored as 1D arrays:
  - `x_coords`
  - `y_coords`

This convention is used consistently across state fields, contracts, and dataset variables.

### 3.3. Basin representation

The basin is represented as a boolean mask over the full rectangular grid.

- `True` means the cell belongs to the basin
- `False` means the cell is outside the basin

The basin mask is stored as:

- `BasinDefinition.mask`
- dataset variable `basin_mask(y, x)`

This design separates the computational grid from the active basin area and allows the simulator to represent non-rectangular basins while retaining a simple rectangular numerical domain.

### 3.4. Spatial domain object

The full spatial domain is represented by `SpatialDomain`, which groups:

- `GridDefinition`
- `BasinDefinition`

`SpatialDomain` validates that grid shape and basin mask shape are consistent.

---

## 4. Temporal domain

### 4.1. Time model

The simulator uses a fixed-step temporal discretization. All modules operate on the same temporal base.

The temporal domain is defined by:

- `start`: simulation start datetime
- `dt_seconds`: time step duration in seconds
- `n_steps`: number of simulation steps
- `calendar_type`: calendar interpretation for seasonality
- `calendar`: optional future extension field

### 4.2. Temporal conventions

The simulator adopts the following temporal conventions:

- one global clock is used by all modules
- each time step has:
  - an integer index: `step`
  - a real timestamp: `timestamp`
- timestamps are generated from:
  - `start`
  - `dt_seconds`
  - `n_steps`

This dual representation allows efficient indexing while keeping the simulation traceable in real time.

### 4.3. Derived temporal attributes

The temporal model currently provides:

- `timestamps`
- `step_index`
- `months`
- `seasons`
- `total_duration_seconds`
- `total_duration`

These derived properties support seasonality, diagnostics, and future scenario logic.

### 4.4. Shared simulation clock

All modules must use the same simulation clock. This means that, at a given step `t`:

- meteorology generates forcing fields for `t`
- energy/PET computes evaporative demand for `t`
- hydrology consumes forcing fields for `t`
- reservoirs update storage and release using inflows at `t`
- observation generates synthetic measurements from the simulated truth at `t`

No module maintains an independent or inconsistent time axis.

---

## 5. Static domain entities

### 5.1. Reservoir definition

Reservoirs are defined as static domain entities. A reservoir definition includes:

- `name`
- `cell_y`
- `cell_x`
- `capacity`
- `initial_storage`

This information identifies the reservoir location in the grid and provides the minimum information required to initialize its dynamic state.

### 5.2. Sensor definition

Sensors are also defined as static domain entities. A sensor definition includes:

- `name`
- `sensor_type`
- `cell_y`
- `cell_x`

This is sufficient for the architectural phase and will later support the observation layer.

---

## 6. Standard dimensions

The simulator uses a fixed set of standard dimension names. These dimensions are part of the global data contract and must remain stable across modules and outputs.

### 6.1. Standard dimension names

- `time`
- `y`
- `x`
- `reservoir`
- `sensor`
- `link`
- `storm`

### 6.2. Standard shapes

Typical variable shapes are:

- spatial field: `(y, x)`
- spatiotemporal field: `(time, y, x)`
- reservoir time series: `(time, reservoir)`
- sensor time series: `(time, sensor)`
- network time series: `(time, link)`
- storm-object variables: `(time, storm)` or `(storm,)`, depending on the variable

Even if not all dimensions are fully used yet, they are defined now to keep the architecture consistent and extensible.

---

## 7. Dynamic state representation

### 7.1. Purpose of `SimulationState`

`SimulationState` represents the dynamic simulator state at a single time step. It is the object used by the engine to pass the current state of the system between module executions and to assemble the result of a simulation step.

### 7.2. Mandatory fields

The current state includes the following required fields:

- `step`
- `timestamp`
- `precipitation`
- `air_temperature`
- `pet`
- `soil_moisture`
- `surface_runoff`
- `channel_flow`

These fields are always expected to exist in a valid state.

### 7.3. Optional fields

The state also supports optional dynamic fields for modular extensibility:

- `background_precipitation`
- `storm_mask`
- `infiltration`
- `subsurface_runoff`
- `reservoir_inflow`
- `reservoir_storage`
- `reservoir_release`
- `reservoir_spill`
- `observations`

These fields may be absent in some simulation configurations or implementation stages.

### 7.4. Shape rules

The state follows these rules:

- spatial fields must be 2D arrays with shape `(ny, nx)`
- reservoir fields must be 1D arrays with shape `(n_reservoirs,)`
- all spatial fields in the same state must share the same shape

This is validated inside `SimulationState`.

### 7.5. State mutability policy

The simulator adopts a **semi-immutable** state strategy:

- modules do not directly mutate the shared state
- modules return typed outputs
- a central merge function builds the new `SimulationState`

This design reduces side effects and improves testability.

---

## 8. Historical dataset representation

### 8.1. Official historical format

The full simulation history is stored in an `xarray.Dataset`. This dataset is the official persistent representation of the simulator output.

### 8.2. Dataset coordinates

The dataset uses the following coordinates:

- `time`
- `y`
- `x`
- `reservoir`

Future extensions may also introduce:

- `sensor`
- `link`
- `storm`

### 8.3. Dataset variables

The current variable catalog includes:

#### Static variable
- `basin_mask`

#### Meteorology
- `precipitation`
- `background_precipitation`
- `storm_mask`
- `air_temperature`
- `pet`

#### Hydrology
- `soil_moisture`
- `infiltration`
- `surface_runoff`
- `subsurface_runoff`
- `channel_flow`

#### Reservoirs
- `reservoir_inflow`
- `reservoir_storage`
- `reservoir_release`
- `reservoir_spill`

#### Observation layer
- `obs_precipitation`
- `obs_discharge`
- `obs_storage`
- `obs_mask`
- `obs_quality_flag`

Not all variables are fully instantiated in the current stage, but all have already been formally defined as part of the global data contract.

### 8.4. Empty dataset initialization

The dataset is initialized with:

- explicit coordinates
- predefined variable names
- predefined dimensions
- variable metadata
- missing values represented with `NaN`

Using `NaN` instead of zero makes it easier to distinguish between “not computed yet” and “physical zero”.

### 8.5. Writing state to dataset

The current state is written into the historical dataset one time step at a time using a dedicated writer function. This function maps the dynamic fields of `SimulationState` into the corresponding temporal slice of the dataset.

This keeps state evolution and persistence clearly separated.

---

## 9. Naming conventions

The simulator uses the following naming rules:

- English names
- `snake_case`
- explicit physical meaning
- stable names across modules, state, and dataset

Examples:

- `precipitation`
- `soil_moisture`
- `surface_runoff`
- `channel_flow`
- `reservoir_storage`

These conventions are intended to reduce ambiguity and to keep documentation and implementation aligned.

---

## 10. Units conventions

The simulator uses one canonical unit per variable. Units should remain stable throughout the project.

Current conventions are:

- precipitation: `mm/dt`
- background precipitation: `mm/dt`
- PET: `mm/dt`
- infiltration: `mm/dt`
- surface runoff: `mm/dt`
- subsurface runoff: `mm/dt`
- soil moisture: `mm`
- air temperature: `degC`
- channel flow: `m3/s`
- reservoir inflow: `m3/s`
- reservoir release: `m3/s`
- reservoir spill: `m3/s`
- reservoir storage: `m3`
- masks and flags: `1`

This convention was chosen because areal water variables are naturally represented as water depth per simulation time step, while routed flows and reservoir volumes are better represented as discharge and storage units.

---

## 11. Validation strategy

The simulator uses validation at multiple levels.

### 11.1. Configuration validation

External configuration is validated using Pydantic schemas. This ensures that YAML or dictionary-based configuration files have the expected structure and value constraints.

Examples:

- positive grid sizes
- positive time step
- valid calendar type
- unique sensor and reservoir names
- initial reservoir storage not exceeding capacity

### 11.2. Internal object validation

Internal domain and state objects validate their own consistency using dataclass validation logic.

Examples:

- valid grid sizes
- valid mask type and shape
- matching grid and basin shapes
- correct field dimensionality
- floating spatial fields
- floating or vector reservoir fields

This layered strategy provides early failure for bad inputs and robust validation for runtime objects.

---

## 12. Current design choices

The following architectural decisions are part of the current design:

- regular 2D rectangular grid
- spatial order `(y, x)`
- fixed-step temporal simulation
- common clock for all modules
- static domain separated from dynamic state
- state separated from historical dataset
- typed contracts between modules
- semi-immutable state update
- explicit variable catalog and dimension naming

These decisions provide a stable and extensible foundation for later scientific implementation.

---

## 13. Planned extensions

The current data model is intentionally minimal but extensible. Several future extensions are already anticipated:

- connect `reservoir_inflow` to real hydro-to-reservoir coupling
- decide whether `storm_mask` should be strictly boolean
- integrate sensor coordinates and observation variables into the dataset
- redesign observation storage if `obs_mask` and `obs_quality_flag` are promoted to first-class state or dataset variables
- connect seasonality factors to scenario or configuration files
- validate that sensors and reservoirs lie inside the spatial grid
- add explicit network and storm-object representations when those modules are implemented

These extensions do not invalidate the current model; they are natural next steps built on top of it.

---

## 14. Summary

The simulator data model is based on a clear separation between static domain, dynamic state, and historical outputs. It uses:

- dataclasses for internal domain and state representation
- NumPy arrays for numerical fields
- xarray for historical storage and labeled dimensions
- Pydantic for configuration validation

This architecture provides a robust base for implementing the scientific logic of the simulator in subsequent phases.
