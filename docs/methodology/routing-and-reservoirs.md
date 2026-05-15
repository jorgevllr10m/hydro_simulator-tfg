# Routing and reservoirs methodology

## Overview

The routing module transforms local runoff products into downstream discharge while optionally regulating flow through reservoirs.

It combines three elements:

1. a static synthetic drainage network
2. a simple channel-routing operator for non-reservoir cells
3. a reservoir model with operating rules and storage balance

## 1. Synthetic drainage network

The project builds a deterministic drainage network directly from the basin mask.

### Current assumptions

- the active basin is one connected component
- each active cell drains to exactly one active 4-neighbour
- the outlet is chosen automatically as the most south-eastern active boundary cell

### How the network is built

1. choose the outlet cell
2. compute graph distance from every active cell to the outlet
3. assign each active cell a downstream neighbour with smaller distance
4. build upstream adjacency lists
5. build an upstream-to-downstream traversal order
6. mark reservoir cells in the network

This network is synthetic. It is not extracted from DEM topography. Its main purpose is to provide a consistent downstream structure for routing experiments.

## 2. Converting runoff to discharge

The routing module receives runoff depths in `mm/dt` and converts them to lateral inflow in `m3/s` using:

- cell area
- step duration

The conversion is:

```text
volume_per_step_m3 = runoff_mm_dt / 1000 * cell_area_m2
discharge_m3s = volume_per_step_m3 / dt_seconds
```

By default, lateral inflow may include:

- surface runoff
- subsurface runoff

This is controlled by `include_subsurface_runoff`.

## 3. Channel routing in non-reservoir cells

For normal cells, routed outflow is computed with a linear-reservoir response.

Behavior:

- `channel_time_constant_hours = 0` → instantaneous routing
- larger time constant → stronger lag and attenuation

The discrete update is:

```text
outflow_t = outflow_(t-1) + alpha * (inflow_t - outflow_(t-1))
alpha = dt / (K + dt)
```

The module keeps the previous routed outflow for every cell, which introduces temporal memory.

## 4. Reservoir cells

If a cell contains a reservoir and reservoir regulation is enabled, that cell bypasses the local channel operator.

For reservoir cells, the step logic is:

1. sum lateral inflow and upstream propagated inflow
2. treat the total as reservoir inflow
3. compute the requested controlled release from operating rules
4. update reservoir storage with inflow, evaporation, release, and spill
5. propagate total reservoir outflow downstream

If reservoir regulation is disabled, reservoir cells behave like ordinary channel cells for routing purposes.

## 5. Reservoir operating rules

Reservoir operating rules are based on normalized storage fraction.

The model defines three zones:

- `conservation`
- `normal`
- `flood_control`

### Conservation zone

When storage is low, the reservoir tries to preserve water and releases only the minimum configured flow.

### Normal zone

Between two storage fractions, requested release increases linearly from minimum release to target release.

### Flood-control zone

When storage is high, requested release increases further, from target release toward maximum controlled release.

These rules decide only the **requested controlled release**. Spill is handled later by the storage balance if capacity is still exceeded.

## 6. Reservoir storage balance

The simplified reservoir storage update uses the following order:

1. add inflow volume to storage
2. compute evaporation loss from reservoir surface area
3. remove controlled release volume
4. spill any excess volume above capacity

Conceptually:

```text
storage_final = storage_prev + inflow_volume - evaporation_loss - controlled_release_volume - spill_volume
```

### Reservoir area

Surface area grows with storage using a power-law relationship:

```text
area = area_max * (storage / capacity) ** area_exponent
```

### Reservoir evaporation

PET at the reservoir cell is converted to open-water evaporation using:

- current reservoir surface area
- `evaporation_factor`
- PET depth for the current step

Evaporation loss cannot exceed available storage.

### Controlled release

Requested release is converted to a step volume and limited by the water actually available after inflow and evaporation.

### Spill

Any remaining volume above capacity is spilled.

## 7. Output variables

The routing output includes:

- lateral inflow field
- total cell inflow field
- routed channel-flow field
- outlet discharge
- reservoir inflow
- requested release
- storage fraction
- reservoir surface area
- evaporation loss
- storage
- release
- spill
- total reservoir outflow
- current reservoir operation zone

The truth dataset currently persists the core routing and reservoir state variables:

- `channel_flow`
- `outlet_discharge`
- `reservoir_inflow`
- `reservoir_storage`
- `reservoir_release`
- `reservoir_spill`

Additional reservoir diagnostics are available in the in-memory routing output and can be added to the dataset later if needed.

## 8. Persistent state

The routing model keeps:

- previous routed outflow for each grid cell
- current storage for each reservoir

This makes the routing response dynamic over time.

## Scope and simplifications

The current routing system intentionally uses a lightweight synthetic network and simplified storage rules. It does not include:

- DEM-derived flow directions
- travel-time distributions by reach geometry
- Muskingum or Saint-Venant routing
- cascaded rule curves by reservoir
- reservoir-specific operating policies beyond capacity and initial storage

The current design is sufficient for the project goal: producing structured, interpretable discharge and storage dynamics from synthetic meteorological forcing.
