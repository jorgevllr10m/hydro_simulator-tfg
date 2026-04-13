from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from simulator.core.contracts import ObservationOutput
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


TRUTH_VARIABLE_SPECS: dict[str, VariableSpec] = {
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
    "outlet_discharge": VariableSpec(
        name="outlet_discharge",
        dims=(TIME_DIM,),
        units="m3/s",
        description="Discharge at the basin outlet",
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
}

OBSERVATION_VARIABLE_SPECS: dict[str, VariableSpec] = {
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

ALL_VARIABLE_SPECS: dict[str, VariableSpec] = {
    **TRUTH_VARIABLE_SPECS,
    **OBSERVATION_VARIABLE_SPECS,
}


def _empty_spatial_time_array(n_steps: int, ny: int, nx: int) -> np.ndarray:
    """Create an empty 3D float array filled with NaN."""
    return np.full((n_steps, ny, nx), np.nan, dtype=float)


def _empty_spatial_time_bool_array(n_steps: int, ny: int, nx: int) -> np.ndarray:
    """Create an empty 3D boolean array filled with False."""
    return np.zeros((n_steps, ny, nx), dtype=bool)


def _empty_time_reservoir_array(n_steps: int, n_reservoirs: int) -> np.ndarray:
    """Create an empty 2D float array filled with NaN."""
    return np.full((n_steps, n_reservoirs), np.nan, dtype=float)


def _empty_time_sensor_float_array(n_steps: int, n_sensors: int) -> np.ndarray:
    """Create an empty 2D float array filled with NaN."""
    return np.full((n_steps, n_sensors), np.nan, dtype=float)


def _empty_time_sensor_bool_array(n_steps: int, n_sensors: int) -> np.ndarray:
    """Create an empty 2D boolean array filled with False."""
    return np.zeros((n_steps, n_sensors), dtype=bool)


def _empty_time_sensor_int_array(n_steps: int, n_sensors: int, *, fill_value: int = 0) -> np.ndarray:
    """Create an empty 2D integer array filled with a constant."""
    return np.full((n_steps, n_sensors), fill_value, dtype=int)


def _empty_time_array(n_steps: int) -> np.ndarray:
    """Create an empty 1D float array filled with NaN."""
    return np.full((n_steps,), np.nan, dtype=float)


def _build_base_coords(domain: SimulationDomain) -> dict[str, np.ndarray]:
    """Return the shared base coordinates used by all datasets."""
    return {
        TIME_DIM: domain.time.timestamps,
        Y_DIM: domain.spatial.y_coords,
        X_DIM: domain.spatial.x_coords,
        RESERVOIR_DIM: np.arange(len(domain.reservoirs), dtype=int),
        SENSOR_DIM: np.arange(len(domain.sensors), dtype=int),
    }


def _attach_variable_metadata(ds: xr.Dataset, specs: dict[str, VariableSpec]) -> xr.Dataset:
    """Attach units and descriptions to the variables present in a dataset."""
    for var_name, spec in specs.items():
        if var_name in ds.data_vars:
            ds[var_name].attrs["units"] = spec.units
            ds[var_name].attrs["description"] = spec.description
    return ds


def _attach_common_coord_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Attach common coordinate metadata."""
    if Y_DIM in ds.coords:
        ds[Y_DIM].attrs["units"] = "m"
    if X_DIM in ds.coords:
        ds[X_DIM].attrs["units"] = "m"
    if TIME_DIM in ds.coords:
        ds[TIME_DIM].attrs["description"] = "simulation timestamps"
    if RESERVOIR_DIM in ds.coords:
        ds[RESERVOIR_DIM].attrs["description"] = "reservoir index"
    if SENSOR_DIM in ds.coords:
        ds[SENSOR_DIM].attrs["description"] = "sensor index"
    return ds


def create_empty_truth_dataset(domain: SimulationDomain) -> xr.Dataset:
    """Create the official truth dataset for one simulation domain.

    This dataset stores the physical / simulated truth generated by the
    meteorology, energy, hydrology and routing modules.
    """
    ny, nx = domain.shape
    n_steps = domain.n_steps
    n_reservoirs = len(domain.reservoirs)

    ds = xr.Dataset(
        coords=_build_base_coords(domain),
        attrs={
            "title": "Synthetic Basin Simulator truth dataset",
            "dataset_role": "truth",
            "spatial_shape": str(domain.shape),
            "n_steps": domain.n_steps,
            "dt_seconds": domain.time.dt_seconds,
        },
    )

    ds["basin_mask"] = xr.DataArray(
        domain.spatial.basin.mask.astype(bool),
        dims=TRUTH_VARIABLE_SPECS["basin_mask"].dims,
    )

    # Meteorology
    ds["precipitation"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=TRUTH_VARIABLE_SPECS["precipitation"].dims,
    )
    ds["background_precipitation"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=TRUTH_VARIABLE_SPECS["background_precipitation"].dims,
    )
    ds["storm_mask"] = xr.DataArray(
        _empty_spatial_time_bool_array(n_steps, ny, nx),
        dims=TRUTH_VARIABLE_SPECS["storm_mask"].dims,
    )
    ds["air_temperature"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=TRUTH_VARIABLE_SPECS["air_temperature"].dims,
    )

    # Energy
    ds["pet"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=TRUTH_VARIABLE_SPECS["pet"].dims,
    )
    ds["aet"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=TRUTH_VARIABLE_SPECS["aet"].dims,
    )
    ds["shortwave_radiation"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=TRUTH_VARIABLE_SPECS["shortwave_radiation"].dims,
    )
    ds["net_radiation"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=TRUTH_VARIABLE_SPECS["net_radiation"].dims,
    )

    # Hydrology
    ds["soil_moisture"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=TRUTH_VARIABLE_SPECS["soil_moisture"].dims,
    )
    ds["infiltration"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=TRUTH_VARIABLE_SPECS["infiltration"].dims,
    )
    ds["surface_runoff"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=TRUTH_VARIABLE_SPECS["surface_runoff"].dims,
    )
    ds["subsurface_runoff"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=TRUTH_VARIABLE_SPECS["subsurface_runoff"].dims,
    )

    # Routing
    ds["channel_flow"] = xr.DataArray(
        _empty_spatial_time_array(n_steps, ny, nx),
        dims=TRUTH_VARIABLE_SPECS["channel_flow"].dims,
    )
    ds["outlet_discharge"] = xr.DataArray(
        _empty_time_array(n_steps),
        dims=TRUTH_VARIABLE_SPECS["outlet_discharge"].dims,
    )
    ds["reservoir_inflow"] = xr.DataArray(
        _empty_time_reservoir_array(n_steps, n_reservoirs),
        dims=TRUTH_VARIABLE_SPECS["reservoir_inflow"].dims,
    )
    ds["reservoir_storage"] = xr.DataArray(
        _empty_time_reservoir_array(n_steps, n_reservoirs),
        dims=TRUTH_VARIABLE_SPECS["reservoir_storage"].dims,
    )
    ds["reservoir_release"] = xr.DataArray(
        _empty_time_reservoir_array(n_steps, n_reservoirs),
        dims=TRUTH_VARIABLE_SPECS["reservoir_release"].dims,
    )
    ds["reservoir_spill"] = xr.DataArray(
        _empty_time_reservoir_array(n_steps, n_reservoirs),
        dims=TRUTH_VARIABLE_SPECS["reservoir_spill"].dims,
    )

    ds = _attach_variable_metadata(ds, TRUTH_VARIABLE_SPECS)
    ds = _attach_common_coord_metadata(ds)

    return ds


def create_empty_observation_dataset(domain: SimulationDomain) -> xr.Dataset:
    """Create the official observation dataset for one simulation domain.

    This dataset stores what the synthetic sensors observe, not the physical
    truth itself.
    """
    n_steps = domain.n_steps
    n_sensors = len(domain.sensors)

    ds = xr.Dataset(
        coords=_build_base_coords(domain),
        attrs={
            "title": "Synthetic Basin Simulator observation dataset",
            "dataset_role": "observation",
            "spatial_shape": str(domain.shape),
            "n_steps": domain.n_steps,
            "dt_seconds": domain.time.dt_seconds,
        },
    )

    ds["sensor_name"] = xr.DataArray(
        np.asarray([sensor.name for sensor in domain.sensors], dtype=object),
        dims=(SENSOR_DIM,),
    )
    ds["sensor_type"] = xr.DataArray(
        np.asarray([sensor.sensor_type for sensor in domain.sensors], dtype=object),
        dims=(SENSOR_DIM,),
    )
    ds["sensor_cell_y"] = xr.DataArray(
        np.asarray([sensor.cell_y for sensor in domain.sensors], dtype=int),
        dims=(SENSOR_DIM,),
    )
    ds["sensor_cell_x"] = xr.DataArray(
        np.asarray([sensor.cell_x for sensor in domain.sensors], dtype=int),
        dims=(SENSOR_DIM,),
    )

    ds["obs_precipitation"] = xr.DataArray(
        _empty_time_sensor_float_array(n_steps, n_sensors),
        dims=OBSERVATION_VARIABLE_SPECS["obs_precipitation"].dims,
    )
    ds["obs_discharge"] = xr.DataArray(
        _empty_time_sensor_float_array(n_steps, n_sensors),
        dims=OBSERVATION_VARIABLE_SPECS["obs_discharge"].dims,
    )
    ds["obs_storage"] = xr.DataArray(
        _empty_time_sensor_float_array(n_steps, n_sensors),
        dims=OBSERVATION_VARIABLE_SPECS["obs_storage"].dims,
    )
    ds["obs_mask"] = xr.DataArray(
        _empty_time_sensor_bool_array(n_steps, n_sensors),
        dims=OBSERVATION_VARIABLE_SPECS["obs_mask"].dims,
    )
    ds["obs_quality_flag"] = xr.DataArray(
        _empty_time_sensor_int_array(n_steps, n_sensors, fill_value=0),
        dims=OBSERVATION_VARIABLE_SPECS["obs_quality_flag"].dims,
    )

    ds["sensor_name"].attrs["description"] = "sensor name"
    ds["sensor_type"].attrs["description"] = "sensor type"
    ds["sensor_cell_y"].attrs["description"] = "sensor y-index in the grid"
    ds["sensor_cell_x"].attrs["description"] = "sensor x-index in the grid"

    ds = _attach_variable_metadata(ds, OBSERVATION_VARIABLE_SPECS)
    ds = _attach_common_coord_metadata(ds)

    return ds


def create_empty_dataset(domain: SimulationDomain) -> xr.Dataset:
    """Backward-compatible alias returning the truth dataset.

    Historically the project used a single dataset function. During phase 8,
    truth and observation are separated, so this alias is kept to avoid
    breaking older code while the runner is updated incrementally.
    """
    return create_empty_truth_dataset(domain)


def write_state_to_dataset(
    ds: xr.Dataset,
    state: SimulationState,
    step: int | None = None,
) -> xr.Dataset:
    """Write a SimulationState into a truth dataset at one time step."""
    target_step = state.step if step is None else step

    ds["precipitation"][target_step, :, :] = state.precipitation
    ds["air_temperature"][target_step, :, :] = state.air_temperature
    ds["pet"][target_step, :, :] = state.pet
    ds["soil_moisture"][target_step, :, :] = state.soil_moisture
    ds["surface_runoff"][target_step, :, :] = state.surface_runoff
    ds["channel_flow"][target_step, :, :] = state.channel_flow
    ds["outlet_discharge"][target_step] = state.outlet_discharge

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


def write_observation_to_dataset(
    ds: xr.Dataset,
    observation_output: ObservationOutput,
    *,
    step: int,
) -> xr.Dataset:
    """Write one ObservationOutput into an observation dataset at one time step."""
    if not isinstance(observation_output, ObservationOutput):
        raise TypeError(f"'observation_output' must be an ObservationOutput, got {type(observation_output).__name__}")

    if observation_output.obs_precipitation is not None:
        ds["obs_precipitation"][step, :] = observation_output.obs_precipitation

    if observation_output.obs_discharge is not None:
        ds["obs_discharge"][step, :] = observation_output.obs_discharge

    if observation_output.obs_storage is not None:
        ds["obs_storage"][step, :] = observation_output.obs_storage

    if observation_output.obs_mask is not None:
        ds["obs_mask"][step, :] = observation_output.obs_mask

    if observation_output.obs_quality_flag is not None:
        ds["obs_quality_flag"][step, :] = observation_output.obs_quality_flag

    return ds
