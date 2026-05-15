# Energy methodology

## Overview

The energy module computes three fields for every simulation step:

- shortwave radiation
- net radiation
- potential evapotranspiration (PET)

Its purpose is to transform time, latitude, precipitation, and air temperature into an evaporative forcing that can later be consumed by the hydrology module.

## Inputs

The module receives:

- simulation domain
- simulation step
- simulation timestamp
- precipitation field
- air-temperature field

The timestamp and latitude control solar geometry. The precipitation field is used as a simple proxy for cloud attenuation. The air-temperature field is used by the PET calculation.

## Configuration

The top-level runtime config is `EnergyBalanceConfig`.

It contains:

- `latitude_deg`
- `solar`, a `SolarRadiationConfig`
- `pet`, a `PETConfig`

The current YAML schema exposes:

- `energy.latitude_deg`
- `energy.pet.pet_multiplier`

The remaining solar and Priestley-Taylor parameters use runtime dataclass defaults unless the code is extended to expose them in the schema.

## 1. Solar geometry

The first stage computes a simplified solar-geometry object from:

- timestamp
- synthetic latitude

Derived quantities include:

- day of year
- local fractional hour
- solar declination
- hour angle
- inverse Earth-Sun distance factor
- cosine of the solar zenith angle
- solar elevation angle
- daylight duration
- daylight fraction

This gives the module a physically interpretable day/night and seasonal cycle.

## 2. Top-of-atmosphere radiation

Using solar geometry and a solar constant, the model estimates top-of-atmosphere shortwave radiation on a horizontal surface.

This stage captures:

- the annual Earth-Sun distance modulation
- the diurnal solar-angle modulation

During night, the clipped cosine of the zenith angle is zero, so incoming shortwave radiation is zero.

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

## 6. PET with simplified Priestley-Taylor

PET is computed from:

- net radiation
- air temperature
- Priestley-Taylor parameters

Intermediate diagnostics include:

- saturation vapor pressure
- slope of the saturation vapor pressure curve
- equilibrium evaporation

The final PET is:

```text
PET = alpha * equilibrium evaporation * pet_multiplier
```

where:

- `alpha` is the Priestley-Taylor coefficient
- `pet_multiplier` is a scenario-level scaling factor

All negative radiation and PET values are clipped to physically meaningful non-negative values.

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

## Outputs

The public `EnergyOutput` contains:

- `pet`
- `shortwave_radiation`
- `net_radiation`

These fields are merged into the physical simulation state and written to the truth dataset.

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
