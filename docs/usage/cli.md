# CLI usage

## Available commands

The repository defines three command-line entry points.

## 1. Run a simulation

Command:

```bash
hydro-sim --config configs/config.yaml
```

### What it does

This command:

1. loads and validates YAML configuration
2. builds the runtime domain and module configs
3. instantiates all module models
4. runs the full time loop
5. writes datasets and CSV files

Figure generation is not part of the runner. Use `hydro-sim-plot` after a completed run.

### Important argument

```bash
--config <path>
```

Default:

```text
configs/config.yaml
```

### Output naming

The runner writes scenario-suffixed outputs:

```text
simulation_truth_<scenario>.nc
simulation_observations_<scenario>.nc
simulation_summary_<scenario>.csv
simulation_summary_<scenario>_es.csv
simulation_observations_<scenario>.csv
simulation_observations_<scenario>_es.csv
```

If NetCDF writing fails because the required backend is unavailable, the dataset is saved as `.pkl` instead.

## 2. Generate plots from an existing run

Command:

```bash
hydro-sim-plot --run-dir outputs/runs/<run_name>
```

Optional:

```bash
hydro-sim-plot --run-dir outputs/runs/<run_name> --output-dir outputs/runs/<run_name>/figures
```

### What it does

This command reads the outputs of a finished simulation and creates figure files such as:

- accumulated field maps
- peak-step snapshots
- summary time series
- truth-vs-observed sensor plots

It writes figures to:

```text
<run-dir>/figures/
```

unless `--output-dir` is provided.

### Input file resolution

The plotting command is designed to be launched only with `--run-dir`. It resolves the required artifacts from that directory.

It supports both legacy unsuffixed names:

```text
simulation_truth.nc
simulation_truth.pkl
simulation_summary.csv
simulation_observations.csv
```

and current scenario-suffixed names:

```text
simulation_truth_<scenario>.nc
simulation_truth_<scenario>.pkl
simulation_summary_<scenario>.csv
simulation_observations_<scenario>.csv
```

When a scenario-suffixed truth dataset is found, the command reads the `scenario_name` attribute from the dataset and then opens the matching summary and observation CSV for that scenario.

Spanish-column CSVs ending in `_es.csv` are ignored by the plotting command because the parser expects internal English column names.

If several valid candidates are present and the scenario cannot be resolved unambiguously, the command fails explicitly instead of selecting one arbitrarily.

### Plot language

The generated figure titles, axis labels, legends, colorbar labels where applicable, and console status messages are written in Spanish.

The output PNG filenames remain stable and are not translated, so downstream scripts can continue using the same file paths.

## 3. Generate validation summaries

Command:

```bash
hydro-sim-validate --run-dir outputs/runs/<run_name>
```

Optional:

```bash
hydro-sim-validate --run-dir outputs/runs/<run_name> --output-dir outputs/runs/<run_name>/validation
```

### What it does

This command reads generated CSV outputs and computes:

- validation by sensor
- validation by sensor type
- a run-level validation summary

It writes outputs to:

```text
<run-dir>/validation/
```

unless `--output-dir` is provided.

### Input file resolution

The validation command is also launched only with `--run-dir`. It resolves the required English-column CSV pair from that directory.

It supports both legacy unsuffixed names:

```text
simulation_summary.csv
simulation_observations.csv
```

and current scenario-suffixed names:

```text
simulation_summary_<scenario>.csv
simulation_observations_<scenario>.csv
```

The selected summary and observation CSV must share the same suffix. Spanish-column CSVs ending in `_es.csv` are ignored because the validation parser expects internal English column names.

If several valid scenario pairs are found in the same run directory, the command fails explicitly and asks the user to separate the results or use a directory with a single scenario.

### Validation output language

Validation CSV column names are Spanish. The run-level summary description and console status messages are also written in Spanish.

The output CSV filenames remain unchanged:

```text
validacion_observaciones_por_sensor.csv
validacion_observaciones_por_tipo.csv
resumen_validacion_sistema.csv
```

## Typical workflow

For the current implementation:

```bash
hydro-sim --config configs/config.yaml
hydro-sim-plot --run-dir outputs/runs/<run_name>
hydro-sim-validate --run-dir outputs/runs/<run_name>
```

No copy or symlink step is required when the run directory contains exactly one scenario-suffixed output set.

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

Optional development dependencies:

```bash
pip install -e ".[dev]"
```

## Development tools

### Run tests

```bash
pytest
```

### Run linting and formatting

```bash
ruff check .
ruff format .
```

### Enable pre-commit hooks

```bash
pre-commit install
```

### Run hooks manually

```bash
pre-commit run --all-files
```

## Notes on reproducibility

Most stochastic subsystems use explicit random seeds in configuration dataclasses. Re-running the same configuration should reproduce the same outputs unless code, configuration, package versions, or random-generation logic change.
