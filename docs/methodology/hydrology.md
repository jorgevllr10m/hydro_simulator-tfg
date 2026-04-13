# Hydrology methodology

## Overview

The hydrology module performs **local cell-scale water balance**. It does not route water between cells. Its outputs are later passed to the routing module.

For every step, it computes:

- soil moisture
- infiltration
- surface runoff
- subsurface runoff
- actual evapotranspiration (AET)

## Persistent state

The hydrology model keeps one persistent field through time:

- `soil_moisture_mm`

This represents the current water stored in the soil bucket for each grid cell.

## Soil-bucket concept

Each cell is represented by a simplified soil bucket with:

- a maximum storage capacity
- a current storage
- an infiltration capacity that decreases as the soil wets up
- an AET stress function
- a percolation rule that activates in wet conditions

All fluxes are expressed per simulation step in `mm/dt`, while soil storage is expressed in `mm`.

## Update order

The soil update follows this sequence:

1. compute relative saturation from previous soil moisture
2. compute infiltration capacity from wetness
3. partition precipitation into infiltration and surface excess
4. update temporary storage with infiltrated water
5. compute AET from PET and soil-water stress
6. remove AET from storage
7. compute percolation from wet soil states
8. remove percolation from storage
9. return final soil moisture and diagnostics

The final balance is conceptually:

```text
soil_final = soil_prev + infiltration - aet - percolation
```

## 1. Relative saturation

Soil moisture is normalized by storage capacity to obtain relative saturation in the range `[0, 1]`.

This normalized value is then reused by several later calculations.

## 2. Infiltration capacity

The infiltration capacity decreases as the soil becomes wetter.

Interpretation:

- dry soil → high infiltration capacity
- saturated soil → low infiltration capacity

This is controlled by:

- `max_infiltration_mm_dt`
- `infiltration_shape_exponent`

## 3. Infiltration and surface excess

Actual infiltration is limited by three things:

- incoming precipitation
- infiltration capacity
- storage room remaining in the bucket

Any rainfall that cannot infiltrate becomes direct surface excess, which later becomes surface runoff.

## 4. AET from PET and soil stress

The hydrology module turns PET into AET using a soil-water stress factor.

Interpretation:

- wet soil → AET close to PET
- dry soil → AET reduced

The stress factor depends on relative saturation and an exponent parameter controlling how strongly dryness reduces ET.

AET is also limited so it cannot exceed the water actually available in the soil after infiltration.

## 5. Percolation

Percolation is activated only above a configurable relative saturation threshold.

Interpretation:

- below threshold → no percolation
- above threshold → percolation increases smoothly
- near saturation → percolation approaches the configured maximum rate

Percolation cannot exceed the water stored after AET.

## 6. Runoff partition

After the soil update, the hydrology module derives runoff products:

- **surface runoff** = direct surface excess
- **subsurface runoff** = a configurable fraction of percolation

This keeps the hydrology module simple while still distinguishing rapid and slower runoff components.

## Outputs

The final hydrology output contains:

- `soil_moisture`
- `infiltration`
- `surface_runoff`
- `subsurface_runoff`
- `aet`

These are still local cell-level products. No downstream propagation has happened yet.

## Diagnostics

The module stores detailed step diagnostics, including:

- infiltration capacity
- infiltration
- surface excess
- water available before ET
- soil relative saturation before ET
- soil-water stress factor
- AET
- storage after ET
- percolation
- final soil moisture
- final relative saturation

These diagnostics are useful when analyzing why a given rain event produced more infiltration or more runoff.

## Scope and simplifications

This is an intentionally lightweight hydrology model. It does not include:

- multilayer soil physics
- groundwater tables
- lateral soil redistribution
- infiltration fronts
- snow
- channel processes

Its role in the simulator is narrower and very clear:

- translate precipitation and PET into local hydrologic response
- preserve temporal soil-memory effects
- produce runoff forcing for the routing module
