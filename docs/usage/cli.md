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
5. writes datasets, CSV files, and quick-look plots

### Important argument

```bash
--config <path>
```

Default:

```text
configs/config.yaml
```

## 2. Generate plots from an existing run

Command:

```bash
hydro-sim-plot --run-dir outputs/runs/base_run
```

Optional:

```bash
hydro-sim-plot --run-dir outputs/runs/base_run --output-dir outputs/runs/base_run/figures
```

### What it does

This command reads the outputs of a finished simulation and creates figure files such as:

- accumulated field maps
- summary time series
- truth-vs-observed sensor plots

It expects the run directory to contain:

- `simulation_truth.nc` or `simulation_truth.pkl`
- `simulation_summary.csv`
- `simulation_observations.csv`

## 3. Generate validation summaries

Command:

```bash
hydro-sim-validate --run-dir outputs/runs/base_run
```

Optional:

```bash
hydro-sim-validate --run-dir outputs/runs/base_run --output-dir outputs/runs/base_run/validation
```

### What it does

This command reads the generated CSV outputs and computes:

- validation by sensor
- validation by sensor type
- a run-level validation summary

## Typical workflow

A normal workflow is:

```bash
hydro-sim --config configs/config.yaml
hydro-sim-plot --run-dir outputs/runs/base_run
hydro-sim-validate --run-dir outputs/runs/base_run
```

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

### Run linting

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

Most stochastic subsystems use explicit random seeds in configuration dataclasses. Re-running the same configuration should therefore reproduce the same outputs unless code or configuration changes.
