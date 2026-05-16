# Meteorology methodology

## Overview

The meteorological module produces two gridded outputs for every step:

- total precipitation
- air temperature

It combines five ideas:

1. a **latent meteorological environment**
2. **storm objects** born from that environment
3. **storm rendering** to raster precipitation fields
4. a **correlated background precipitation field**
5. a **smooth spatial air-temperature anomaly field**

The final precipitation field is:

```text
total precipitation = storm precipitation + background precipitation
```

The final air-temperature field is:

```text
air temperature = latent background temperature + spatial zero-mean anomaly
```

when the temperature anomaly model is enabled.

## 1. Latent environment

The latent environment is a compact description of the current weather state. It is not itself a raster field. Instead, it controls the processes that later generate fields.

Its state includes:

- current meteorological regime
- background temperature
- advection vector
- antecedent wetness index
- precipitation spell index
- cloudiness index
- convective potential index
- seasonality factor
- diagnostic scenario moisture factor

### Meteorological regimes

The model uses four discrete regimes:

- `stable_dry`
- `transitional`
- `convective`
- `frontal_persistent`

Each regime has a baseline profile with:

- cloudiness tendency
- convective tendency
- wetness equilibrium
- temperature anomaly

### Thermal and moisture scenarios

The latent environment also supports user-facing scenario tendencies:

- thermal: `cold`, `normal`, `warm`
- moisture: `dry`, `normal`, `wet`

These scenario values influence temperature, regime weights, wetness, and wet/dry spell tendency.

### Temporal evolution

At every step, the latent model updates:

- regime
- temperature
- advection
- wetness
- spell index
- cloudiness
- convective potential

The update uses:

- persistence terms
- seasonal forcing
- scenario-dependent shifts
- random perturbations

This gives a stochastic but reproducible sequence of weather states.

## 2. Storm-generation forcing

The latent state is converted into a compact storm-forcing object used by the birth model. It includes:

- storm trigger factor
- storm organization factor
- moisture availability
- advection components
- background temperature
- cloudiness

These factors summarize whether storm birth is likely, how organized storms should be, and how they should move.

## 3. Storm birth

New storm cells are sampled at each step from a Poisson process.

The expected number of births depends on:

- baseline expected births per step
- storm trigger factor
- moisture availability
- organization factor

For each new storm, the model samples:

- initial center position
- semi-major axis
- semi-minor axis
- orientation
- peak intensity
- duration
- velocity components

All these properties are modulated by the current latent environment.

### Storm geometry constraints

Storm cells are represented as rotated ellipses. To keep the geometry valid:

- the semi-major axis must be strictly positive
- the semi-minor axis must be strictly positive
- the semi-minor axis cannot exceed the semi-major axis

These constraints prevent degenerate storm footprints, such as zero-width ellipses, and protect the raster renderer from invalid divisions.

### Band organization

When several storms are born in the same step, the model may reorganize them into a line or band. This is more likely when the environment is organized.

The band reorganization can:

- align storm orientation
- share part of the velocity
- place births along a preferred axis
- reduce cross-band dispersion
- reduce minor-axis width to sharpen band structure

The band minor-axis factor must be strictly positive and no greater than one. This allows bands to become sharper while preventing the minor axis from collapsing to zero.

This helps generate structured rain features instead of purely isolated cells.

## 4. Storm lifecycle

Each storm is tracked over multiple simulation steps.

A normalized lifecycle shape controls:

- growth phase
- mature phase
- decay phase

From the storm age, the module computes:

- current life factor
- current intensity
- current effective axes

Newborn storms are shifted into a renderable state so they can already contribute in the first step where they appear.

## 5. Advection

Each active storm stores:

- center position
- velocity components

After rendering a step, the model advances all storms using:

```text
position_next = position_current + velocity * dt
```

Expired storms are removed.

## 6. Raster rendering

Storms are rendered as rotated elliptical Gaussian rain cells.

For each storm, the renderer:

1. computes its current intensity and footprint size
2. builds a bounding box around the effective footprint
3. evaluates the rotated Gaussian on the local subgrid
4. accumulates contributions into the precipitation field
5. marks cells above a rainfall threshold in `storm_mask`

The output of storm rendering is:

- storm precipitation depth in `mm/dt`
- storm mask as a boolean 2D field

## 7. Background precipitation field

Besides explicit storm objects, the module also generates a correlated large-scale background field.

This field is useful for representing:

- stratiform precipitation
- weak widespread rain
- persistent rainy backgrounds

The background model uses:

- a smoothed random field
- temporal persistence
- support from the latent environment
- activation thresholds
- an activity factor with temporal memory

Its intensity depends on regime, wetness, cloudiness, and precipitation-spell state.

## 8. Air-temperature field

The current meteorological module no longer uses only a uniform temperature field when the temperature field model is enabled.

The temperature model:

- starts from the latent background temperature
- adds a smooth spatial anomaly
- keeps the anomaly at zero spatial mean
- combines an oriented gradient component with a smoothed random component
- can transition smoothly to a new anomaly orientation over several steps

The anomaly redistributes temperature within the domain without changing the domain-mean temperature generated by the latent environment.

If the temperature-field configuration is disabled, the module returns a uniform field equal to the latent background temperature.

## Step sequence inside the meteorology model

For each time step:

1. update latent environment
2. convert latent state to storm-forcing input
3. remove expired storms from previous steps
4. spawn new storms
5. initialize newborn storms for immediate rendering
6. render storm precipitation and storm mask
7. build background precipitation
8. sum storm and background precipitation
9. build air-temperature field
10. store step diagnostics
11. advance active storms for the next time step
12. remove storms that expire after the advance

The meteorology model owns its temporal state internally. The step input provides the domain, step index, and timestamp; previous meteorological state is not passed through the public input contract.

## Diagnostics

The meteorology model stores lightweight diagnostics such as:

- latent state
- storm-forcing summary
- number of new storms
- number of active storms
- background activity factor
- precipitation spell index
- whether band reorganization happened
- number of band births
- band probability

These diagnostics are later exported in the run summary CSV.

## Scope and simplifications

This is a synthetic weather generator, not a full atmospheric model. Important simplifications include:

- discrete weather regimes instead of fluid dynamics
- object-based storm cells instead of cloud microphysics
- Gaussian storm footprints
- synthetic, not terrain-driven advection
- heuristic background precipitation support
- spatial temperature anomalies without explicit topography

These simplifications are intentional. They keep the simulator transparent and computationally light while still generating structured and seasonally modulated rainfall and temperature forcing.
