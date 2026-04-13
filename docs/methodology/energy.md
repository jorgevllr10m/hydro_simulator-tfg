# Energy methodology

## Overview

The energy module computes three fields for every simulation step:

- shortwave radiation
- net radiation
- potential evapotranspiration (PET)

Its purpose is to transform time, latitude, and cloudiness proxy information into an evaporative forcing that can later be consumed by the hydrology module.

## Inputs

The module receives:

- simulation timestamp
- grid shape
- precipitation field
- air-temperature field

The timestamp controls solar geometry. The precipitation field is used as a simple proxy for cloud attenuation.

## 1. Solar geometry

The first stage computes a simplified solar-geometry object from:

- timestamp
- synthetic latitude

Derived quantities include:

- day of year
- local fractional hour
- solar declination
- hour angle
- inverse Earth–Sun distance factor
- cosine of the solar zenith angle
- solar elevation angle
- daylight duration
- daylight fraction

This gives the module a physically interpretable day/night and seasonal cycle.

## 2. Top-of-atmosphere radiation

Using solar geometry and a solar constant, the model estimates top-of-atmosphere shortwave radiation on a horizontal surface.

This stage captures:

- the annual Earth–Sun distance modulation
- the diurnal solar-angle modulation

## 3. Clear-sky radiation

The top-of-atmosphere radiation is attenuated by a fixed clear-sky transmissivity to estimate what would reach the surface under clear conditions.

## 4. Cloud attenuation

The model uses precipitation as a simple cloudiness proxy.

Higher precipitation implies stronger attenuation of shortwave radiation. The resulting cloud factor is bounded between:

- a minimum cloud factor
- 1.0

This keeps the method simple while still coupling rainy periods with lower incoming radiation.

## 5. Net shortwave and net radiation

Incident shortwave radiation is reduced by surface albedo, producing absorbed net shortwave radiation.

That energy is then converted from:

```text
W/m²
```

to:

```text
MJ/m²/dt
```

over the simulation step duration.

In the current MVP, net radiation is approximated from net shortwave only. Longwave components are not modeled explicitly.

## 6. PET with simplified Priestley–Taylor

PET is computed from:

- net radiation
- air temperature
- Priestley–Taylor parameters

Intermediate diagnostics include:

- saturation vapor pressure
- slope of the saturation vapor pressure curve
- equilibrium evaporation

The final PET is:

```text
PET = alpha * equilibrium evaporation * pet_multiplier
```

where:

- `alpha` is the Priestley–Taylor coefficient
- `pet_multiplier` is a scenario-level scaling factor

## Why PET is computed here

The hydrology module needs an evaporative demand signal, but actual evapotranspiration depends on soil-water availability. Therefore the separation is:

- **energy module** → PET
- **hydrology module** → AET

This is a clean modeling boundary: atmosphere proposes demand, soil decides how much can actually happen.

## Diagnostics

The module stores detailed step diagnostics that include:

- solar geometry
- radiation fields
- PET intermediate fields

These are useful for debugging and for explaining how PET changed through time.

## Main simplifications

The current energy module intentionally omits:

- explicit humidity fields
- wind-driven aerodynamic terms
- longwave radiation balance
- terrain shading
- spatially varying latitude or topography

Despite that, it still captures the core drivers needed for this simulator:

- day/night cycle
- seasonal cycle
- cloud attenuation
- temperature-sensitive evaporative demand
