# Architecture overview

## Purpose

The simulator is organized as a modular hydrometeorological pipeline that produces:

- a **truth dataset** containing the simulated physical state
- an **observation dataset** containing synthetic sensor measurements
- summary CSV files and quick-look plots for inspection
- validation CSV files for post-run analysis

The architecture favors:

- clear module boundaries
- explicit typed contracts between modules
- reproducibility through deterministic configuration and random seeds
- explainable synthetic processes rather than black-box generation

## End-to-end pipeline

For each simulation step, the runner executes the modules in this order:

1. **Meteorology**
   - Updates the latent weather state.
   - Spawns and evolves storm objects.
   - Renders total precipitation and air temperature.

2. **Energy**
   - Computes solar geometry from timestamp and latitude.
   - Builds shortwave and net radiation.
   - Computes PET.

3. **Hydrology**
   - Updates soil moisture.
   - Computes AET, infiltration, surface runoff, and subsurface runoff.

4. **Routing**
   - Converts runoff depth to discharge.
   - Routes water through the synthetic drainage network.
   - Updates reservoir storage, releases, and spill.

5. **Observation**
   - Samples truth values at configured sensor locations.
   - Produces noisy, possibly missing or censored observations.

6. **Persistence**
   - Merges module outputs into a unified simulation state.
   - Writes truth and observation products into xarray datasets.
   - Accumulates CSV summaries and run diagnostics.

## Module boundaries

### `simulator.config`
Responsible for loading YAML files and validating them with Pydantic schemas.

### `simulator.core`
Contains repository-wide foundational structures:

- spatial and temporal domain definitions
- shared array aliases and dataclasses
- module input/output contracts
- runtime state aggregation
- dataset creation and writing helpers

### `simulator.meteo`
Implements the rainfall-generation system:

- latent environment
- weather regimes
- advection
- storm birth
- storm lifecycle
- raster rendering
- correlated background precipitation

### `simulator.energy`
Implements the atmospheric-energy part of the pipeline:

- solar geometry
- shortwave radiation
- net radiation
- PET

### `simulator.hydro`
Implements local hydrology at cell scale:

- soil bucket storage
- AET limitation by soil moisture
- infiltration and direct surface excess
- percolation and runoff partition

### `simulator.routing`
Implements downstream propagation and regulated storage:

- synthetic static drainage network
- linear channel routing
- reservoir operating rules
- reservoir storage balance

### `simulator.obs`
Implements the synthetic observation operator:

- precipitation gauges
- discharge sensors
- reservoir-storage sensors
- noise, missingness, and detection-threshold censoring

### `simulator.cli`
Exposes the user-facing commands:

- `hydro-sim`
- `hydro-sim-plot`
- `hydro-sim-validate`

## Design principles

### 1. Typed interfaces between modules
Each module consumes a typed input dataclass and returns a typed output dataclass. This keeps dependencies explicit and reduces hidden coupling.

### 2. Persistent state only where needed
Some modules are purely diagnostic within a step, while others maintain internal state across time:

- meteorology keeps active storms and latent state
- hydrology keeps soil moisture
- routing keeps previous channel outflow and reservoir storage
- observations keep only RNG state and last diagnostics

### 3. Separation between truth and observations
The project distinguishes clearly between:

- **physical truth** produced by the simulator
- **observed products** produced by the synthetic sensing layer

This separation is reflected in the dataset writers and output files.

### 4. Reproducible stochasticity
Random processes are used extensively, but each configurable subsystem uses its own seeded random generator so repeated runs remain reproducible.

## Typical run flow

At startup, the runner:

1. loads the master config
2. resolves the selected domain preset and scenario
3. builds runtime configuration objects
4. constructs the simulation domain
5. instantiates the module models
6. allocates empty output datasets
7. iterates over all time steps
8. writes final artifacts to the run directory

## Output philosophy

The repository stores several levels of information simultaneously:

- raw gridded truth fields
- per-sensor observation products
- aggregated step summaries
- quick-look plots
- validation summaries

This makes the simulator useful both as a data generator and as an inspectable experimental environment.
