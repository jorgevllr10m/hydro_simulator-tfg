# Observation-layer methodology

## Overview

The observation layer converts simulated truth into synthetic measurements at configured sensor locations.

Its role is to approximate what an imperfect monitoring network would report, rather than exposing the simulator truth directly.

The observation operator currently supports three sensor types:

- `precipitation`
- `discharge`
- `reservoir_storage`

## Key idea

For each sensor and time step, the model:

1. extracts the relevant truth value from the physical simulation
2. decides whether the observation is missing
3. applies noise if the observation exists
4. applies threshold-based censoring when relevant
5. stores the result and its quality metadata

## Sensor types

### Precipitation sensor

Samples the precipitation field at the sensor cell.

Possible effects:

- additive absolute noise
- missing observations
- detection threshold
- left-censoring below threshold

Exact physical zero is preserved as `0.0` and marked as nominal rather than censored.

### Discharge sensor

Samples the routed channel-flow field at the sensor cell.

Possible effects:

- multiplicative relative noise
- missing observations
- detection threshold
- left-censoring below threshold

Very small physical zero-like values are preserved as `0.0` and marked as nominal rather than censored.

### Reservoir-storage sensor

Samples the storage of the reservoir located exactly at the same cell as the sensor.

Possible effects:

- additive absolute noise
- missing observations

This sensor type requires an exact one-to-one match between the sensor cell and a reservoir cell. Observed storage is clipped to the physical interval `[0, capacity]`.

## Quality flags

Each observed value is associated with a quality/status code:

- `MISSING = 0`
- `NOMINAL = 1`
- `CENSORED = 2`

Interpretation:

- **MISSING** → no usable reported value
- **NOMINAL** → normal observation
- **CENSORED** → the value existed but was raised to the detection threshold

## Availability mask

In addition to the value arrays, the model outputs a boolean mask per sensor:

- `True` means a usable reported value exists
- `False` means the observation is missing

This is useful because the observation vectors contain one slot per sensor, but only one physical variable is applicable to each sensor type.

## Vector output convention

The observation outputs are stored as vectors of length `n_sensors`:

- `obs_precipitation`
- `obs_discharge`
- `obs_storage`
- `obs_mask`
- `obs_quality_flag`

Non-applicable variables remain `NaN` for a given sensor.

Example:

- a precipitation gauge has a value in `obs_precipitation`
- the same sensor has `NaN` in `obs_discharge` and `obs_storage`

## Missing data model

Missingness is sampled independently using configured probabilities for each sensor class.

This allows scenarios with:

- sparse missing data
- degraded monitoring quality
- more realistic synthetic datasets for testing downstream models

If a sensor family is disabled, matching sensors return missing values.

## Noise model

### Precipitation

Uses additive noise with standard deviation in physical units (`mm/dt`).

### Discharge

Uses relative multiplicative noise around the truth.

### Reservoir storage

Uses additive noise in storage units (`m3`).

All observed values are clipped when needed to keep them physically meaningful.

## Censoring model

Precipitation and discharge sensors can apply left-censoring below a configured detection threshold.

This represents sensors that cannot reliably report very small non-zero values.

Effect:

- if the noisy observed value is between 0 and the threshold
- and censoring is enabled
- the returned value becomes exactly the threshold
- and the quality flag is set to `CENSORED`

The loader enables censoring automatically when a precipitation or discharge detection threshold is explicitly provided and is greater than zero.

## Diagnostics

For each step, the observation module summarizes:

- number of sensors
- number of available observations
- number of missing observations
- number of censored observations

These summaries are later exported into the run-level step CSV.

## Output files

Observation products are saved in two forms:

1. an xarray observation dataset with vectors over `time` and `sensor`
2. a flattened per-sensor CSV with one row per sensor and time step

The runner writes both English and Spanish-column CSV variants.

## Why this layer matters

The simulator is not just a truth generator. It is also a **data-generation environment for partially observed systems**.

This makes the repository useful for tasks such as:

- testing data pipelines
- validating inference methods
- studying robustness to missing data
- comparing truth and observation products
