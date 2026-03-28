# Module Diagram and Internal Interfaces

## 1. Purpose

This document describes the internal module architecture of the synthetic basin simulator. Its goal is to define how the main simulator modules interact, what each module consumes, what each module produces, and how the simulation engine coordinates them at each time step.

The design principle adopted in this project is that modules communicate through explicit typed contracts. Each module receives a well-defined input object and returns a well-defined output object. Modules do not directly mutate the global historical dataset and do not rely on hidden shared variables.

---

## 2. Main modules

The simulator is organized around the following conceptual modules:

- meteorology
- energy / PET
- hydrology
- reservoirs
- observation
- recorder / dataset writer

Each module has a clear responsibility and only exchanges the information required by its contract.

### 2.1. Meteorology

The meteorology module generates atmospheric forcing fields for the current simulation step.

Typical outputs include:

- precipitation
- air temperature
- optional background precipitation
- optional storm mask

This module is responsible for producing the physical forcing fields that drive the rest of the system.

### 2.2. Energy / PET

The energy or PET module computes evaporative demand for the current step.

Its main output is:

- potential evapotranspiration (`pet`)

This module is kept conceptually separate from meteorology, even if future implementations decide to place both under a broader atmospheric component.

### 2.3. Hydrology

The hydrology module updates the terrestrial water balance and routed flow variables.

Typical outputs include:

- updated soil moisture
- infiltration
- surface runoff
- optional subsurface runoff
- channel flow
- optional outlet discharge

This module consumes atmospheric forcing but does not generate it.

### 2.4. Reservoirs

The reservoir module updates reservoir storage and regulated outflows.

Typical outputs include:

- reservoir storage
- reservoir release
- reservoir spill

This module acts on inflows coming from upstream hydrology or routing logic.

### 2.5. Observation

The observation module generates synthetic measurements from the simulated truth.

Typical outputs include:

- observed precipitation
- observed discharge
- observed storage
- observation availability mask
- observation quality flags

This module does not modify the physical state of the system. It only produces an observed representation of that state.

### 2.6. Recorder / dataset writer

The recorder writes the current simulator state into the historical `xarray.Dataset`.

This component is intentionally separated from the physical modules so that numerical simulation and persistence remain decoupled.

---

## 3. Design principle: explicit contracts

The simulator does **not** use the strategy where each module reads and writes directly into a shared global dataset.

Instead, the adopted strategy is:

- each module receives a typed input
- each module returns a typed output
- a central engine merges outputs into the new state
- a separate writer persists that state into the historical dataset

This design provides:

- modularity
- easier unit testing
- clearer dependencies
- lower coupling
- easier debugging

---

## 4. Static and dynamic layers

The module architecture relies on three internal data layers:

- `SimulationDomain`
- `SimulationState`
- `SimulationDataset`

### 4.1. `SimulationDomain`

Contains the static simulation context:

- spatial domain
- temporal domain
- reservoirs
- sensors

Modules may read from the domain, but they do not modify it.

### 4.2. `SimulationState`

Contains the dynamic state at the current step.

Modules do not mutate the shared state directly. Instead, they use the required parts of the current state as input context and return outputs for the next state.

### 4.3. `SimulationDataset`

Contains the historical record of the simulation. It is updated after the new state has been assembled.

---

## 5. Input-output contracts by module

The current architecture defines the following typed contracts.

---

## 5.1. Meteorology contracts

### Input: `MeteoInput`

The meteorology module receives:

- `domain`
- `step`
- `timestamp`
- optional `previous_state`

The optional previous state is included to allow future atmospheric persistence or latent-state evolution.

### Output: `MeteoOutput`

The meteorology module returns:

- `precipitation`
- `air_temperature`
- optional `background_precipitation`
- optional `storm_mask`

### Interpretation

Meteorology is responsible only for atmospheric forcing. It should not know anything about reservoirs or observation logic.

---

## 5.2. Energy / PET contracts

### Input: `EnergyInput`

The energy module receives:

- `domain`
- `step`
- `timestamp`
- `air_temperature`

### Output: `EnergyOutput`

The energy module returns:

- `pet`

### Interpretation

This module converts meteorological information into evaporative demand. It remains separate to keep the architecture explicit.

---

## 5.3. Hydrology contracts

### Input: `HydroInput`

The hydrology module receives:

- `domain`
- `step`
- `timestamp`
- `precipitation`
- `pet`
- `soil_moisture_prev`

### Output: `HydroOutput`

The hydrology module returns:

- `soil_moisture`
- `infiltration`
- `surface_runoff`
- `channel_flow`
- optional `subsurface_runoff`
- optional `outlet_discharge`

### Interpretation

Hydrology consumes atmospheric forcing and updates the terrestrial and routed hydrological state. It does not generate meteorological variables.

---

## 5.4. Reservoir contracts

### Input: `ReservoirInput`

The reservoir module receives:

- `domain`
- `step`
- `timestamp`
- `reservoir_inflow`
- `reservoir_storage_prev`

### Output: `ReservoirOutput`

The reservoir module returns:

- `reservoir_storage`
- `reservoir_release`
- `reservoir_spill`

### Interpretation

This module operates on reservoir dynamics only. It assumes that inflows are provided from outside, typically by hydrology or routing logic.

---

## 5.5. Observation contracts

### Input: `ObservationInput`

The observation module receives:

- `domain`
- `step`
- `timestamp`
- `precipitation`
- `channel_flow`
- optional `reservoir_storage`

### Output: `ObservationOutput`

The observation module returns:

- optional `obs_precipitation`
- optional `obs_discharge`
- optional `obs_storage`
- optional `obs_mask`
- optional `obs_quality_flag`

### Interpretation

The observation module creates synthetic measurements from the simulated truth. It is not part of the physical evolution of the system.

---

## 6. Execution order at each time step

The recommended execution order for each simulation step `t` is:

1. meteorology generates forcing fields
2. energy / PET computes evaporative demand
3. hydrology updates soil moisture, runoff, and flow
4. reservoirs update storage and releases
5. observation generates synthetic measurements
6. recorder writes the final state of step `t` into the historical dataset

This order is consistent with physical causality:

- forcing fields must exist before hydrology
- PET must exist before hydrology
- hydrology must exist before reservoir inflow can be interpreted
- the observed system must exist before synthetic observations can be generated
- the final merged state must exist before persistence

---

## 7. Internal engine flow

The simulator engine follows a stepwise orchestration strategy.

At a high level, the internal flow is:

1. build typed module inputs for the current step
2. execute each module in the agreed order
3. collect typed outputs
4. merge outputs into a new `SimulationState`
5. write the state to the historical dataset

Conceptually, the execution loop can be expressed as:

```python
for step, timestamp in enumerate(domain.time.timestamps):
    meteo_out = meteo.step(...)
    energy_out = energy.step(...)
    hydro_out = hydro.step(...)
    reservoir_out = reservoirs.step(...)
    obs_out = observation.step(...)
    state = merge_module_outputs(...)
    dataset = write_state_to_dataset(dataset, state)


8. State update policy

The simulator uses a semi-immutable state update policy.

This means:

modules receive input data as read-only information
modules return typed outputs
a central merge function constructs the next SimulationState

Modules do not directly modify the shared state object or the dataset.

Advantages
fewer side effects
simpler unit testing
clearer ownership of outputs
easier debugging
easier extension of module contracts

9. Merge function

The engine uses a dedicated merge function, merge_module_outputs, to combine module outputs into a new SimulationState.

Responsibilities of the merge function
assign current step
assign current timestamp
insert meteorological outputs into the new state
insert PET output into the new state
insert hydrological outputs into the new state
insert reservoir outputs if present
insert observation products into the state representation when appropriate
Current architectural note

At the current stage:

previous_state is already part of the merge signature but is not yet fully used
reservoir_inflow is still a future coupling point and is not yet derived from hydrology
observation quality flags and masks are not yet fully integrated into the physical state or historical dataset

These are known future extensions, not current design errors.

10. Dependency structure between modules

The dependency chain between modules is the following:

meteorology depends only on domain, time, and optionally previous state
energy depends on meteorological outputs
hydrology depends on meteorology and energy
reservoirs depend on hydrology
observation depends on the already simulated truth
recorder depends on the final assembled state

This can be summarized as:

Meteorology -> Energy/PET -> Hydrology -> Reservoirs -> Observation -> Recorder

The engine coordinates this chain explicitly.

11. Why modules do not write directly to the dataset

A design alternative would be to let each module directly modify the global xarray.Dataset. This option was rejected for the following reasons:

high coupling between modules and storage layout
reduced clarity about which module produces which variables
harder testing of isolated module behavior
more difficult debugging
more fragile future refactoring

Instead, modules return typed outputs and the engine handles persistence separately.

12. Why typed contracts were adopted

Typed dataclass-based contracts were adopted because they provide:

explicit module interfaces
early validation of array shape and type
better testability
easier reasoning about data flow
reduced ambiguity in future development

This is especially valuable in a modular simulator where several scientific components interact.

13. Current limitations and future coupling points

The current interface design is intentionally minimal but extensible. The following points are already identified for future work:

connect hydrological routing outputs to reservoir_inflow
decide whether storm_mask should become a strict boolean field
define richer observation persistence in dataset space
integrate sensor coordinate and observation variables into the historical dataset
extend network support using the link dimension
extend storm-object support using the storm dimension
make selective use of previous_state for persistent hidden or latent variables
add initialization helpers for the first state of the simulator

These future extensions fit naturally within the current architecture.

14. Summary

The internal simulator architecture is based on a clear module chain, typed contracts, a semi-immutable state update policy, and explicit separation between simulation logic and historical storage.

The main architectural principles are:

explicit module interfaces
shared simulation clock
ordered step execution
central state assembly
separated dataset persistence

This design provides a robust foundation for implementing the scientific logic of meteorology, hydrology, reservoirs, and synthetic observation in later phases.
