from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from simulator.core.state import SimulationState
from simulator.core.types import SimulationDomain

TIME_DIM = "time"
Y_DIM = "y"
X_DIM = "x"
RESERVOIR_DIM = "reservoir"
SENSOR_DIM = "sensor"
LINK_DIM = "link"
STORM_DIM = "storm"


@dataclass(frozen=True)
class VariableSpec:
    """Specification of a dataset variable."""

    name: str
    dims: tuple[str, ...]
    units: str
    description: str


VARIABLE_SPECS: dict[str, VariableSpec] = {
    "basin_mask": VariableSpec(
        name="basin_mask",
        dims=(Y_DIM, X_DIM),
        units="1",
        description="Boolean mask of active basin cells",
    ),
    # Meteorology
    "precipitation": VariableSpec(
        name="precipitation",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="mm/dt",
        description="Precipitation depth per simulation time step",
    ),
    "background_precipitation": VariableSpec(
        name="background_precipitation",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="mm/dt",
        description="Background or stratiform precipitation component",
    ),
    # TODO en dataset su unidad es "1" y conceptualmente es una máscara, pero en state.py y contracts.py se valida como campo flotante
    # 2D, no booleano. Cuando se implementen tormentas habrá que decidir si es estrictamente BoolArray.
    "storm_mask": VariableSpec(
        name="storm_mask",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="1",
        description="Mask of cells affected by storm objects",
    ),
    "air_temperature": VariableSpec(
        name="air_temperature",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="degC",
        description="Near-surface air temperature",
    ),
    # Energy
    "pet": VariableSpec(
        name="pet",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="mm/dt",
        description="Potential evapotranspiration per simulation time step",
    ),
    "aet": VariableSpec(
        name="aet",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="mm/dt",
        description="Actual evapotranspiration per simulation time step",
    ),
    "shortwave_radiation": VariableSpec(
        name="shortwave_radiation",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="W/m2",
        description="Incoming shortwave radiation at the surface",
    ),
    "net_radiation": VariableSpec(
        name="net_radiation",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="MJ/m2/dt",
        description="Net radiation available during the simulation time step",
    ),
    "antecedent_storage": VariableSpec(
        name="antecedent_storage",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="mm",
        description="Simplified antecedent-water storage",
    ),
    "antecedent_relative": VariableSpec(
        name="antecedent_relative",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="1",
        description="Antecedent-water storage normalized to [0, 1]",
    ),
    "antecedent_overflow": VariableSpec(
        name="antecedent_overflow",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="mm/dt",
        description="Excess antecedent-water input above storage capacity",
    ),
    # Hydrology
    "soil_moisture": VariableSpec(
        name="soil_moisture",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="mm",
        description="Soil water storage",
    ),
    "infiltration": VariableSpec(
        name="infiltration",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="mm/dt",
        description="Infiltrated water depth per simulation time step",
    ),
    "surface_runoff": VariableSpec(
        name="surface_runoff",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="mm/dt",
        description="Surface runoff depth per simulation time step",
    ),
    "subsurface_runoff": VariableSpec(
        name="subsurface_runoff",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="mm/dt",
        description="Subsurface runoff depth per simulation time step",
    ),
    "channel_flow": VariableSpec(
        name="channel_flow",
        dims=(TIME_DIM, Y_DIM, X_DIM),
        units="m3/s",
        description="Channel or routed flow",
    ),
    # Reservoirs
    "reservoir_inflow": VariableSpec(
        name="reservoir_inflow",
        dims=(TIME_DIM, RESERVOIR_DIM),
        units="m3/s",
        description="Reservoir inflow discharge",
    ),
    "reservoir_storage": VariableSpec(
        name="reservoir_storage",
        dims=(TIME_DIM, RESERVOIR_DIM),
        units="m3",
        description="Reservoir storage volume",
    ),
    "reservoir_release": VariableSpec(
        name="reservoir_release",
        dims=(TIME_DIM, RESERVOIR_DIM),
        units="m3/s",
        description="Reservoir release flow",
    ),
    "reservoir_spill": VariableSpec(
        name="reservoir_spill",
        dims=(TIME_DIM, RESERVOIR_DIM),
        units="m3/s",
        description="Reservoir spill flow",
    ),
    # Observation layer
    "obs_precipitation": VariableSpec(
        name="obs_precipitation",
        dims=(TIME_DIM, SENSOR_DIM),
        units="mm/dt",
        description="Observed precipitation at sensor locations",
    ),
    "obs_discharge": VariableSpec(
        name="obs_discharge",
        dims=(TIME_DIM, SENSOR_DIM),
        units="m3/s",
        description="Observed discharge at sensor locations",
    ),
    "obs_storage": VariableSpec(
        name="obs_storage",
        dims=(TIME_DIM, SENSOR_DIM),
        units="m3",
        description="Observed reservoir storage at sensor locations",
    ),
    "obs_mask": VariableSpec(
        name="obs_mask",
        dims=(TIME_DIM, SENSOR_DIM),
        units="1",
        description="Observation availability mask",
    ),
    "obs_quality_flag": VariableSpec(
        name="obs_quality_flag",
        dims=(TIME_DIM, SENSOR_DIM),
        units="1",
        description="Observation quality flag",
    ),
}


def _empty_spatial_time_array(n_steps: int, ny: int, nx: int) -> np.ndarray:
    """Create an empty 3D float array filled with NaN."""
    return np.full((n_steps, ny, nx), np.nan, dtype=float)


def _empty_time_reservoir_array(n_steps: int, n_reservoirs: int) -> np.ndarray:
    """Create an empty 2D float array filled with NaN."""
    return np.full((n_steps, n_reservoirs), np.nan, dtype=float)


def create_empty_dataset(domain: SimulationDomain) -> xr.Dataset:
    """Create the official empty simulation dataset for a domain.

    The resulting xarray.Dataset defines the standard coordinates,
    variables, dimensions and metadata for the simulator historical output.
    """
    # TODO añadir coordenada sensor y variables obs_* cuando se integre domain.sensors.
    ny, nx = domain.shape
    n_steps = domain.n_steps
    n_reservoirs = len(domain.reservoirs)

    ds = xr.Dataset(
        coords={
            TIME_DIM: domain.time.timestamps,
            Y_DIM: domain.spatial.y_coords,
            X_DIM: domain.spatial.x_coords,
            RESERVOIR_DIM: np.arange(n_reservoirs, dtype=int),
        },
        attrs={
            "title": "Synthetic Basin Simulator dataset",
            "spatial_shape": str(domain.shape),
            "n_steps": domain.n_steps,
            "dt_seconds": domain.time.dt_seconds,
        },
    )

    ds["basin_mask"] = xr.DataArray(
        domain.spatial.basin.mask.astype(bool),
        dims=VARIABLE_SPECS["basin_mask"].dims,
    )
    # Meteo
    ds["precipitation"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["precipitation"].dims,
    )
    ds["background_precipitation"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["background_precipitation"].dims,
    )
    ds["storm_mask"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["storm_mask"].dims,
    )
    ds["air_temperature"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["air_temperature"].dims,
    )
    # Energy
    ds["pet"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["pet"].dims,
    )
    ds["aet"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["aet"].dims,
    )
    ds["shortwave_radiation"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["shortwave_radiation"].dims,
    )
    ds["net_radiation"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["net_radiation"].dims,
    )
    ds["antecedent_storage"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["antecedent_storage"].dims,
    )
    ds["antecedent_relative"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["antecedent_relative"].dims,
    )
    ds["antecedent_overflow"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["antecedent_overflow"].dims,
    )
    # Hydrology
    ds["soil_moisture"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["soil_moisture"].dims,
    )
    ds["infiltration"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["infiltration"].dims,
    )
    ds["surface_runoff"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["surface_runoff"].dims,
    )
    ds["subsurface_runoff"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["subsurface_runoff"].dims,
    )
    ds["channel_flow"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=VARIABLE_SPECS["channel_flow"].dims,
    )
    ds["reservoir_inflow"] = xr.DataArray(
        _empty_time_reservoir_array(n_steps, n_reservoirs),
        dims=VARIABLE_SPECS["reservoir_inflow"].dims,
    )
    ds["reservoir_storage"] = xr.DataArray(
        _empty_time_reservoir_array(n_steps, n_reservoirs),
        dims=VARIABLE_SPECS["reservoir_storage"].dims,
    )
    ds["reservoir_release"] = xr.DataArray(
        _empty_time_reservoir_array(n_steps, n_reservoirs),
        dims=VARIABLE_SPECS["reservoir_release"].dims,
    )
    ds["reservoir_spill"] = xr.DataArray(
        _empty_time_reservoir_array(n_steps, n_reservoirs),
        dims=VARIABLE_SPECS["reservoir_spill"].dims,
    )

    for var_name, spec in VARIABLE_SPECS.items():
        if var_name in ds.data_vars:
            ds[var_name].attrs["units"] = spec.units
            ds[var_name].attrs["description"] = spec.description

    ds[Y_DIM].attrs["units"] = "m"
    ds[X_DIM].attrs["units"] = "m"
    ds[TIME_DIM].attrs["description"] = "simulation timestamps"
    ds[RESERVOIR_DIM].attrs["description"] = "reservoir index"

    return ds


def write_state_to_dataset(
    ds: xr.Dataset,
    state: SimulationState,
    step: int | None = None,
) -> xr.Dataset:
    """Write a SimulationState into the historical dataset at one time step."""
    target_step = state.step if step is None else step

    ds["precipitation"][target_step, :, :] = state.precipitation
    ds["air_temperature"][target_step, :, :] = state.air_temperature
    ds["pet"][target_step, :, :] = state.pet
    ds["soil_moisture"][target_step, :, :] = state.soil_moisture
    ds["surface_runoff"][target_step, :, :] = state.surface_runoff
    ds["channel_flow"][target_step, :, :] = state.channel_flow

    if state.background_precipitation is not None:
        ds["background_precipitation"][target_step, :, :] = state.background_precipitation

    if state.storm_mask is not None:
        ds["storm_mask"][target_step, :, :] = state.storm_mask

    if state.aet is not None:
        ds["aet"][target_step, :, :] = state.aet

    if state.shortwave_radiation is not None:
        ds["shortwave_radiation"][target_step, :, :] = state.shortwave_radiation

    if state.net_radiation is not None:
        ds["net_radiation"][target_step, :, :] = state.net_radiation

    if state.antecedent_storage is not None:
        ds["antecedent_storage"][target_step, :, :] = state.antecedent_storage

    if state.antecedent_relative is not None:
        ds["antecedent_relative"][target_step, :, :] = state.antecedent_relative

    if state.antecedent_overflow is not None:
        ds["antecedent_overflow"][target_step, :, :] = state.antecedent_overflow

    if state.infiltration is not None:
        ds["infiltration"][target_step, :, :] = state.infiltration

    if state.subsurface_runoff is not None:
        ds["subsurface_runoff"][target_step, :, :] = state.subsurface_runoff

    if state.reservoir_inflow is not None and ds.sizes[RESERVOIR_DIM] > 0:
        ds["reservoir_inflow"][target_step, :] = state.reservoir_inflow

    if state.reservoir_storage is not None and ds.sizes[RESERVOIR_DIM] > 0:
        ds["reservoir_storage"][target_step, :] = state.reservoir_storage

    if state.reservoir_release is not None and ds.sizes[RESERVOIR_DIM] > 0:
        ds["reservoir_release"][target_step, :] = state.reservoir_release

    if state.reservoir_spill is not None and ds.sizes[RESERVOIR_DIM] > 0:
        ds["reservoir_spill"][target_step, :] = state.reservoir_spill

    return ds
