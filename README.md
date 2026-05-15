# Hydro Simulator TFG

Spatiotemporal hydrometeorological simulator for generating synthetic truth and synthetic observations over a regulated basin.

## What this repository does

This project simulates an hourly hydrometeorological pipeline over a synthetic 2D basin:

1. **Meteorology**
   - A latent weather environment evolves through discrete regimes.
   - Storm cells are born, advected, rendered to raster precipitation fields, and combined with a correlated background precipitation field.
   - A spatial air-temperature field is generated for every step from the latent background temperature plus a smooth zero-mean anomaly.

2. **Energy balance**
   - Solar geometry is computed from timestamp and synthetic latitude.
   - Shortwave radiation and net radiation are estimated.
   - Potential evapotranspiration (PET) is computed with a simplified Priestley-Taylor formulation.

3. **Hydrology**
   - Each cell runs a soil-bucket model.
   - Precipitation is partitioned into infiltration, surface runoff, subsurface runoff, and actual evapotranspiration (AET).

4. **Routing and reservoirs**
   - Runoff is converted to discharge and propagated through a simplified drainage network.
   - Reservoirs store inflow, evaporate water, release flow according to simple operating rules, and spill when they exceed capacity.

5. **Observation layer**
   - Synthetic sensors sample truth fields.
   - Observations may include noise, missing values, and detection-threshold censoring.

6. **Outputs and validation**
   - Full truth and observation datasets are saved.
   - Step summaries and per-sensor observation tables are written as CSV.
   - A separate plotting command generates figures from an existing run.
   - A separate validation command generates internal validation summaries.

The goal is not to reproduce a real basin exactly, but to create **controlled, explainable, and reproducible synthetic datasets** for experimentation, testing, and validation.

## Repository structure

```text
configs/
  config.yaml                # master run configuration
  domain/                    # domain presets: grid, reservoirs, sensors
  scenarios/                 # scenario overrides

src/simulator/
  cli/                       # run, plot, validate commands
  common/                    # shared validation helpers
  config/                    # YAML loading and Pydantic schema validation
  core/                      # shared types, contracts, state, datasets, engine
  meteo/                     # latent environment, storm rainfall, background fields, temperature field
  energy/                    # solar geometry, radiation, PET
  hydro/                     # soil bucket and runoff partition
  routing/                   # drainage network, channel routing, reservoirs
  obs/                       # synthetic observation operator

outputs/
  runs/<run_name>/           # generated datasets, CSVs, figures, validation
```

## Installation

The project requires Python 3.13 or newer.

Install the project in editable mode:

```bash
pip install -e .
```

Install project with development dependencies:

```bash
pip install -e ".[dev]"
```

## Main commands

Run one simulation:

```bash
hydro-sim --config configs/config.yaml
```

Generate figures from an existing run:

```bash
hydro-sim-plot --run-dir outputs/runs/<run_name>
```

Generate validation CSVs from an existing run:

```bash
hydro-sim-validate --run-dir outputs/runs/<run_name>
```

## Output naming

The runner writes the main artifacts using the selected scenario name as suffix:

```text
simulation_truth_<scenario>.nc
simulation_observations_<scenario>.nc
simulation_summary_<scenario>.csv
simulation_summary_<scenario>_es.csv
simulation_observations_<scenario>.csv
simulation_observations_<scenario>_es.csv
```

If NetCDF writing is not available, datasets fall back to `.pkl`.

The plotting and validation commands resolve scenario-suffixed outputs directly from `--run-dir`. Legacy unsuffixed names are still supported for compatibility.

`hydro-sim-plot` reads the truth dataset, uses its `scenario_name` metadata when available, and selects the matching summary and observation CSV. `hydro-sim-validate` selects a matching summary/observation CSV pair with the same suffix. Both commands ignore `_es.csv` input files because their parsers use the internal English column names.

Generated plot titles, axis labels, legends, and plotting console messages are in Spanish. Validation CSV columns, run-level summary text, and validation console messages are also in Spanish. Output filenames remain stable.

## Configuration overview

The simulator uses three configuration layers:

- **Master config**: selects run name, simulation window, domain preset, and scenario.
- **Domain preset**: defines the grid, reservoirs, and sensors.
- **Scenario file**: overrides selected module parameters for meteorology, energy, hydrology, routing, and observations.

See:

- [`docs/architecture/configuration.md`](docs/architecture/configuration.md)
- [`docs/usage/cli.md`](docs/usage/cli.md)

## Documentation map

Start here:

- [`docs/index.md`](docs/index.md)
- [`docs/architecture/overview.md`](docs/architecture/overview.md)
- [`docs/methodology/meteorology.md`](docs/methodology/meteorology.md)
- [`docs/methodology/energy.md`](docs/methodology/energy.md)
- [`docs/methodology/hydrology.md`](docs/methodology/hydrology.md)
- [`docs/methodology/routing-and-reservoirs.md`](docs/methodology/routing-and-reservoirs.md)
- [`docs/methodology/observation-layer.md`](docs/methodology/observation-layer.md)
- [`docs/usage/outputs.md`](docs/usage/outputs.md)
- [`docs/validation/validation-workflow.md`](docs/validation/validation-workflow.md)

## Current scope and assumptions

This repository currently implements an MVP-style simulator with the following design choices:

- the basin mask is fully active over the selected rectangular grid
- the drainage network is synthetic and deterministic
- the outlet is chosen automatically on the basin boundary
- the meteorology is stochastic but reproducible through seeds
- scenario files expose only a controlled subset of runtime parameters
- reservoir storage/rule parameters are shared at model-config level, while capacity and initial storage are domain-specific
- observations are synthetic products derived from the simulated truth

## Development notes

Development dependencies are defined in `pyproject.toml`. The repository uses:

- `pytest`
- `ruff`
- `pre-commit`

Enable hooks with:

```bash
pre-commit install
```

Run checks manually with:

```bash
pre-commit run --all-files
```
