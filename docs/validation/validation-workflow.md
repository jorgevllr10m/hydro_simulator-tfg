# Validation workflow

## Purpose

The validation command does not compare the simulator against external real data. Instead, it validates the **internal consistency and observation behavior** of one generated run.

Its focus is:

- observation availability
- censoring rates
- truth-vs-observed error metrics
- compact run-level diagnostics

## Inputs

The validation command reads:

- `simulation_observations.csv`
- `simulation_summary.csv`

from a finished run directory.

## Output files

The validation workflow generates three CSV files.

## 1. Validation by sensor

Filename:

```text
validacion_observaciones_por_sensor.csv
```

Each row summarizes one sensor over the whole run.

Typical metrics include:

- total observations
- available observations
- missing observations
- censored observations
- missing rate
- censoring rate
- mean truth value
- bias
- MAE
- RMSE
- relative bias
- relative MAE
- relative RMSE

This file is the most useful one for checking whether individual sensors behave as expected.

## 2. Validation by sensor type

Filename:

```text
validacion_observaciones_por_tipo.csv
```

This aggregates the same idea by sensor class:

- precipitation
- discharge
- reservoir_storage

It is useful to understand whether one observation family is systematically noisier or more censored than another.

## 3. System validation summary

Filename:

```text
resumen_validacion_sistema.csv
```

This is a one-row aggregated summary of the whole run. It includes metrics such as:

- number of steps
- accumulated precipitation
- peak precipitation
- accumulated PET
- accumulated AET
- accumulated surface runoff
- accumulated subsurface runoff
- peak outlet discharge
- final total reservoir storage
- mean available observations per step
- mean missing observations per step
- mean censored observations per step

## Error metrics used

For valid truth/observation pairs, the validation code computes:

- **Bias** = mean(observed - truth)
- **MAE** = mean absolute error
- **RMSE** = root mean square error

Relative versions are also computed by dividing by the mean truth value when that mean is finite and positive.

## How to interpret the outputs

### Missing rate
High values indicate sparse monitoring or aggressive missing-data settings.

### Censored rate
High values usually mean the detection threshold is large relative to typical signal magnitude.

### Bias
- positive bias → observations tend to overestimate
- negative bias → observations tend to underestimate

### MAE and RMSE
- MAE gives a typical absolute error scale
- RMSE penalizes large deviations more strongly

### Peak outlet discharge
Useful as a compact hydrological signature for the run.

## Recommended use

Use validation outputs to answer questions such as:

- Are my synthetic sensors too noisy?
- Are my missing-value probabilities too high?
- Are censoring thresholds hiding too much low flow or light rainfall?
- Are some sensor types much less reliable than others?
- Does the run exhibit roughly the hydrological behavior expected for the chosen scenario?

## Important scope note

This validation workflow is **internal to the synthetic simulation environment**. It does not claim physical realism by itself. Its function is to help assess:

- simulator behavior
- observation degradation
- run diagnostics
- quality of generated synthetic data for downstream experiments
