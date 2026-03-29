# Phase 2 — Simplified Latent Meteorological Environment

## Purpose

This phase implements a simplified latent meteorological environment that provides the background conditions for future storm and precipitation generation.

The goal is not to reproduce the full atmosphere, but to create a plausible, explainable and reproducible environmental state that modulates rainfall occurrence, organization and movement.

---

## Scope of this phase

This phase includes:

- seasonal forcing
- simplified meteorological regimes
- background temperature
- effective advection
- antecedent wetness memory
- cloudiness proxy
- convective potential proxy
- internal interface towards the future storm generator

This phase does **not** include:

- explicit storm object generation
- precipitation field rendering
- hydrology
- reservoirs
- observation products

---

## Main design idea

The meteorological environment is represented as a **latent global state** for each simulation step.

This state is intentionally low-dimensional and uses proxy variables instead of full atmospheric physics. The latent environment is designed to be:

- simple enough for a TFG/MVP
- physically plausible in a broad sense
- easy to configure and reproduce
- directly usable by later precipitation modules

---

## Core concepts

### Regime

A **meteorological regime** is a discrete category describing the dominant weather mode at a time step.

The current implementation uses four regimes:

- `stable_dry`
- `transitional`
- `convective`
- `frontal_persistent`

Each regime provides baseline tendencies for:

- cloudiness
- convective potential
- wetness equilibrium
- temperature anomaly

### Scenario

A **scenario** is a fixed experiment-level bias defined by configuration.

Scenarios do not describe the instantaneous weather state. Instead, they modify the overall behavior of the latent environment.

The current implementation includes:

- thermal scenario: `cold`, `normal`, `warm`
- moisture scenario: `dry`, `normal`, `wet`

### Effective advection

**Effective advection** is a simplified representation of the background horizontal motion that will later be used to move storm objects or organize precipitation fields.

It is not intended to reproduce full atmospheric wind dynamics. It is a compact proxy consisting of:

- speed in m/s
- direction in degrees
- derived Cartesian components `u` and `v`

### Antecedent wetness

The **antecedent wetness index** represents the recent wet/dry memory of the meteorological environment.

It is not soil moisture and does not belong to the hydrology module. It is a simplified atmospheric-environmental memory term used to influence cloudiness, convective potential and future precipitation behavior.

### Convective potential

The **convective potential index** is a simplified proxy describing how favorable the current environment is for convective activity.

It is not CAPE and should not be interpreted as a physically rigorous atmospheric stability variable. It is a synthetic control variable built from:

- regime baseline
- seasonality
- temperature support
- antecedent wetness
- temporal memory

---

## Main classes

### `LatentEnvironmentConfig`

Defines the fixed parameters controlling the latent environment model, including:

- random seed
- initial regime
- temperature seasonality parameters
- advection parameters
- persistence and memory coefficients
- thermal and moisture scenarios

This object represents the **rules of the experiment**, not the instantaneous state.

### `LatentEnvironmentState`

Represents the latent meteorological environment at a single simulation step.

Fields include:

- `regime`
- `background_temperature_c`
- `advection`
- `antecedent_wetness_index`
- `cloudiness_index`
- `convective_potential_index`
- `seasonality_factor`
- `scenario_moisture_factor`

This object represents the **current background environment** used by the meteorological module.

### `LatentEnvironmentModel`

Stateful generator of latent meteorological states.

It applies:

- seasonal forcing
- regime persistence and resampling
- temperature update
- advection update
- wetness update
- cloudiness update
- convective potential update

It also provides a translation from latent state to a future storm-oriented forcing view.

### `StormEnvironmentInput`

Internal interface exposed to the future storm/precipitation generator.

It summarizes the latent state into operational variables such as:

- `storm_trigger_factor`
- `storm_organization_factor`
- `moisture_availability`
- advection components
- background temperature
- cloudiness

This is an internal meteorology-layer contract and should not be confused with the global `MeteoOutput` contract defined in `core/contracts.py`.

---

## Execution flow

1. A `LatentEnvironmentConfig` is created.
2. A `LatentEnvironmentModel` is instantiated with that configuration.
3. For each simulation step, `next_state(...)` is called.
4. The model:
   - computes a continuous seasonality factor from the timestamp
   - keeps or resamples the regime
   - retrieves the regime profile
   - computes temperature
   - computes effective advection
   - computes antecedent wetness
   - computes cloudiness
   - computes convective potential
5. A `LatentEnvironmentState` is returned.
6. When needed, the state is transformed into `StormEnvironmentInput` for later storm generation.

---

## Stochastic but reproducible behavior

The latent environment is stochastic because it includes random sampling for:

- regime transitions
- temperature noise
- advection perturbations

However, it is reproducible because the model uses a deterministic NumPy random generator initialized from a fixed `random_seed`.

This means:

- same configuration + same seed -> same generated sequence
- same configuration + different seed -> different sequence

This is essential for controlled synthetic experiments.

---

## Relationship with the global simulator architecture

This latent environment is an **internal sublayer of the `meteo` module**.

It does not replace the global simulator contracts defined in `core/contracts.py`.

At this phase:

- the latent environment remains internal to `meteo`
- the global `MeteoOutput` contract is not yet fully produced by this phase
- future storm and precipitation generation modules will consume the latent environment and eventually build the full meteorological output

---

## Current limitations

This phase intentionally does not model:

- pressure
- humidity as a full physical variable
- radiative transfer
- explicit atmospheric dynamics
- explicit storm cells
- 2D precipitation rendering

These omissions are deliberate and consistent with the simplified, proxy-based design of the project.

---

## Expected future adjustments

The current parameter values are reasonable defaults, but some parts are expected to be recalibrated later, especially after storm generation is implemented.

Likely adjustments include:

- regime profile values
- regime transition weights
- scenario offsets and shifts
- advection regime adjustments
- storm trigger weighting
- storm organization weighting

These are calibration-level changes and should not require architectural redesign.
