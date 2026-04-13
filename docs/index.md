# Documentation index

This directory contains the technical documentation for the simulator repository.

## Architecture

- [`architecture/overview.md`](architecture/overview.md)
  High-level view of the simulator pipeline and module responsibilities.

- [`architecture/configuration.md`](architecture/configuration.md)
  External YAML configuration structure, schema validation, and runtime object construction.

- [`architecture/data-model-and-interfaces.md`](architecture/data-model-and-interfaces.md)
  Static domain definitions, runtime state, module contracts, and dataset structure.

## Methodology

- [`methodology/meteorology.md`](methodology/meteorology.md)
  Latent environment, weather regimes, storm birth, advection, rendering, and background precipitation.

- [`methodology/energy.md`](methodology/energy.md)
  Solar geometry, radiation, and PET computation.

- [`methodology/hydrology.md`](methodology/hydrology.md)
  Soil bucket update, AET, infiltration, surface runoff, and subsurface runoff.

- [`methodology/routing-and-reservoirs.md`](methodology/routing-and-reservoirs.md)
  Synthetic drainage network, channel routing, reservoir rules, and storage balance.

- [`methodology/observation-layer.md`](methodology/observation-layer.md)
  Synthetic sensing, missing values, noise, censoring, and observation products.

## Usage

- [`usage/cli.md`](usage/cli.md)
  How to run simulations, generate plots, and compute validation summaries.

- [`usage/outputs.md`](usage/outputs.md)
  Description of datasets, CSV files, plots, and run-directory contents.

## Validation

- [`validation/validation-workflow.md`](validation/validation-workflow.md)
  What the validation command computes and how to interpret the outputs.
