from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from simulator.common.validation import (
    validate_non_negative_scalar as _validate_non_negative_scalar,
)
from simulator.common.validation import (
    validate_positive_scalar as _validate_positive_scalar,
)
from simulator.common.validation import (
    validate_shape_2d as _validate_shape,
)
from simulator.common.validation import (
    validate_spatial_float_array as _validate_spatial_float_array,
)
from simulator.common.validation import (
    validate_vector_float_array as _validate_vector_float_array,
)
from simulator.core.contracts import RegulatedRoutingInput, RegulatedRoutingOutput
from simulator.core.types import FloatArray, SimulationDomain
from simulator.routing.network import DrainageNetwork
from simulator.routing.rules import (
    ReservoirOperationZone,
    ReservoirRulesConfig,
    apply_reservoir_operating_rules,
)
from simulator.routing.storage import (
    ReservoirStorageConfig,
    update_reservoir_storage,
)

VectorFloatArray = NDArray[np.float64]


def _build_initial_reservoir_storage_m3(
    domain: SimulationDomain,
) -> VectorFloatArray:
    """Return initial reservoir storage vector from the domain definitions."""
    return np.asarray(
        [float(reservoir.initial_storage) for reservoir in domain.reservoirs],
        dtype=float,
    )


def compute_cell_area_m2(
    domain: SimulationDomain,
) -> float:
    """Return the area of one grid cell in m²."""
    if not isinstance(domain, SimulationDomain):
        raise TypeError(f"'domain' must be a SimulationDomain, got {type(domain).__name__}")

    dx = _validate_positive_scalar("domain.spatial.grid.dx", domain.spatial.grid.dx)
    dy = _validate_positive_scalar("domain.spatial.grid.dy", domain.spatial.grid.dy)

    return dx * dy


def convert_runoff_mm_dt_to_discharge_m3s(
    runoff_mm_dt: FloatArray,
    *,
    cell_area_m2: int | float,
    dt_seconds: int | float,
) -> FloatArray:
    """Convert runoff depth [mm/dt] into discharge [m3/s] per cell.

    Formula
    -------
    volume_per_step_m3 = runoff_mm_dt / 1000 * cell_area_m2
    discharge_m3s = volume_per_step_m3 / dt_seconds
    """
    _validate_spatial_float_array("runoff_mm_dt", runoff_mm_dt)
    cell_area_m2 = _validate_positive_scalar("cell_area_m2", cell_area_m2)
    dt_seconds = _validate_positive_scalar("dt_seconds", dt_seconds)

    runoff_mm_dt = np.clip(runoff_mm_dt.astype(float, copy=False), 0.0, None)

    volume_per_step_m3 = runoff_mm_dt / 1000.0 * cell_area_m2
    discharge_m3s = volume_per_step_m3 / dt_seconds

    return np.clip(discharge_m3s, 0.0, None).astype(float, copy=False)


def build_lateral_inflow_m3s(
    surface_runoff_mm_dt: FloatArray,
    *,
    cell_area_m2: int | float,
    dt_seconds: int | float,
    subsurface_runoff_mm_dt: FloatArray | None = None,
    include_subsurface_runoff: bool = True,
) -> FloatArray:
    """Build local lateral inflow to the routing network in m3/s."""
    _validate_spatial_float_array("surface_runoff_mm_dt", surface_runoff_mm_dt)

    if subsurface_runoff_mm_dt is not None:
        _validate_spatial_float_array("subsurface_runoff_mm_dt", subsurface_runoff_mm_dt)

        if subsurface_runoff_mm_dt.shape != surface_runoff_mm_dt.shape:
            raise ValueError(
                "'subsurface_runoff_mm_dt' must have the same shape as "
                f"'surface_runoff_mm_dt', got {subsurface_runoff_mm_dt.shape} "
                f"and {surface_runoff_mm_dt.shape}"
            )

    surface_discharge_m3s = convert_runoff_mm_dt_to_discharge_m3s(
        surface_runoff_mm_dt,
        cell_area_m2=cell_area_m2,
        dt_seconds=dt_seconds,
    )

    if not include_subsurface_runoff or subsurface_runoff_mm_dt is None:
        return surface_discharge_m3s.astype(float, copy=False)

    subsurface_discharge_m3s = convert_runoff_mm_dt_to_discharge_m3s(
        subsurface_runoff_mm_dt,
        cell_area_m2=cell_area_m2,
        dt_seconds=dt_seconds,
    )

    return (surface_discharge_m3s + subsurface_discharge_m3s).astype(float, copy=False)


def compute_linear_reservoir_outflow_m3s(
    inflow_m3s: int | float,
    previous_outflow_m3s: int | float,
    *,
    dt_seconds: int | float,
    time_constant_hours: int | float,
) -> float:
    """Return routed outflow using a simple linear-reservoir response.

    Behavior
    --------
    - K = 0 -> instantaneous routing (outflow = inflow)
    - larger K -> stronger lag and attenuation

    Discrete update
    ---------------
    outflow_t = outflow_(t-1) + alpha * (inflow_t - outflow_(t-1))
    alpha = dt / (K + dt)
    """
    inflow_m3s = _validate_non_negative_scalar("inflow_m3s", inflow_m3s)
    previous_outflow_m3s = _validate_non_negative_scalar(
        "previous_outflow_m3s",
        previous_outflow_m3s,
    )
    dt_seconds = _validate_positive_scalar("dt_seconds", dt_seconds)
    time_constant_hours = _validate_non_negative_scalar(
        "time_constant_hours",
        time_constant_hours,
    )

    time_constant_seconds = time_constant_hours * 3600.0

    if time_constant_seconds == 0.0:
        return inflow_m3s

    alpha = dt_seconds / (time_constant_seconds + dt_seconds)
    outflow_m3s = previous_outflow_m3s + alpha * (inflow_m3s - previous_outflow_m3s)

    return max(0.0, float(outflow_m3s))


def extract_reservoir_pet_mm_dt(
    pet_mm_dt: FloatArray,
    *,
    domain: SimulationDomain,
) -> VectorFloatArray:
    """Extract one PET value per reservoir from the PET spatial field."""
    _validate_spatial_float_array("pet_mm_dt", pet_mm_dt)

    if pet_mm_dt.shape != domain.shape:
        raise ValueError(f"'pet_mm_dt' must have shape {domain.shape}, got {pet_mm_dt.shape}")

    values = np.asarray(
        [float(pet_mm_dt[reservoir.cell_y, reservoir.cell_x]) for reservoir in domain.reservoirs],
        dtype=float,
    )

    return np.clip(values, 0.0, None).astype(float, copy=False)


@dataclass(frozen=True)
class RegulatedRoutingConfig:
    """Configuration of the coupled routing + reservoir model.

    Notes
    -----
    - Non-reservoir cells use simple channel routing with lag + attenuation.
    - Reservoir cells do not apply the local channel routing operator.
      Instead, total cell inflow enters the reservoir directly, and the
      reservoir outflow (release + spill) is propagated downstream in the same
      simulation step.
    - Reservoir storage and operating-rule parameters are shared by all
      reservoirs in this MVP. Reservoir-specific capacity and initial storage
      still come from the domain definitions.
    """

    channel_time_constant_hours: float = 6.0
    include_subsurface_runoff: bool = True
    enable_reservoirs: bool = True

    reservoir_storage: ReservoirStorageConfig = field(default_factory=ReservoirStorageConfig)
    reservoir_rules: ReservoirRulesConfig = field(default_factory=ReservoirRulesConfig)

    def __post_init__(self) -> None:
        _validate_non_negative_scalar(
            "channel_time_constant_hours",
            self.channel_time_constant_hours,
        )

        if not isinstance(self.include_subsurface_runoff, bool):
            raise TypeError(f"'include_subsurface_runoff' must be bool, got {type(self.include_subsurface_runoff).__name__}")

        if not isinstance(self.enable_reservoirs, bool):
            raise TypeError(f"'enable_reservoirs' must be bool, got {type(self.enable_reservoirs).__name__}")

        if not isinstance(self.reservoir_storage, ReservoirStorageConfig):
            raise TypeError(f"'reservoir_storage' must be a ReservoirStorageConfig, got {type(self.reservoir_storage).__name__}")

        if not isinstance(self.reservoir_rules, ReservoirRulesConfig):
            raise TypeError(f"'reservoir_rules' must be a ReservoirRulesConfig, got {type(self.reservoir_rules).__name__}")


@dataclass(frozen=True)
class RegulatedRoutingState:
    """Persistent state of the coupled routing + reservoir model."""

    previous_cell_outflow_m3s_flat: VectorFloatArray
    reservoir_storage_m3: VectorFloatArray

    def __post_init__(self) -> None:
        _validate_vector_float_array(
            "previous_cell_outflow_m3s_flat",
            self.previous_cell_outflow_m3s_flat,
        )
        _validate_vector_float_array("reservoir_storage_m3", self.reservoir_storage_m3)

        if np.any(self.previous_cell_outflow_m3s_flat < 0.0):
            raise ValueError("'previous_cell_outflow_m3s_flat' must be >= 0 everywhere")

        if np.any(self.reservoir_storage_m3 < 0.0):
            raise ValueError("'reservoir_storage_m3' must be >= 0 everywhere")


class RegulatedRoutingModel:
    """Stateful coupled routing + reservoir model over a fixed drainage network.

    Step logic
    ----------
    For each active cell in upstream -> downstream order:

    Non-reservoir cell
    ------------------
    1. Sum lateral inflow + upstream propagated flow
    2. Apply channel lag + attenuation
    3. Propagate routed outflow downstream

    Reservoir cell
    --------------
    1. Sum lateral inflow + upstream propagated flow
    2. Treat that total as reservoir inflow
    3. Apply reservoir rules using storage at the beginning of the step
    4. Update reservoir storage with inflow, evaporation, controlled release and spill
    5. Propagate reservoir total outflow (release + spill) downstream
    """

    def __init__(
        self,
        config: RegulatedRoutingConfig,
        *,
        domain: SimulationDomain,
        network: DrainageNetwork,
    ) -> None:
        if not isinstance(config, RegulatedRoutingConfig):
            raise TypeError(f"'config' must be a RegulatedRoutingConfig, got {type(config).__name__}")

        if not isinstance(domain, SimulationDomain):
            raise TypeError(f"'domain' must be a SimulationDomain, got {type(domain).__name__}")

        if not isinstance(network, DrainageNetwork):
            raise TypeError(f"'network' must be a DrainageNetwork, got {type(network).__name__}")

        _validate_shape(domain.shape)

        if network.shape != domain.shape:
            raise ValueError(f"'network.shape' must match 'domain.shape', got {network.shape} and {domain.shape}")

        self.config = config
        self.domain = domain
        self.network = network

        self._shape = domain.shape
        self._cell_area_m2 = compute_cell_area_m2(domain)

        self._latest_state = RegulatedRoutingState(
            previous_cell_outflow_m3s_flat=np.zeros(self.network.n_cells, dtype=float),
            reservoir_storage_m3=_build_initial_reservoir_storage_m3(domain),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the fixed spatial shape of the model."""
        return self._shape

    @property
    def n_reservoirs(self) -> int:
        """Return number of reservoirs in the regulated system."""
        return len(self.domain.reservoirs)

    @property
    def latest_state(self) -> RegulatedRoutingState:
        """Return the latest persistent state of the regulated routing model."""
        return self._latest_state

    def reset(self) -> None:
        """Reset channel memory and reservoir storages to their initial state."""
        self._latest_state = RegulatedRoutingState(
            previous_cell_outflow_m3s_flat=np.zeros(self.network.n_cells, dtype=float),
            reservoir_storage_m3=_build_initial_reservoir_storage_m3(self.domain),
        )

    def step(
        self,
        *,
        routing_input: RegulatedRoutingInput,
    ) -> RegulatedRoutingOutput:
        """Advance the coupled routing + reservoir model by one simulation step."""
        if not isinstance(routing_input, RegulatedRoutingInput):
            raise TypeError(f"'routing_input' must be a RegulatedRoutingInput, got {type(routing_input).__name__}")

        if routing_input.domain.shape != self._shape:
            raise ValueError(f"'routing_input.domain.shape' must be {self._shape}, got {routing_input.domain.shape}")
        _validate_spatial_float_array("routing_input.surface_runoff", routing_input.surface_runoff)
        _validate_spatial_float_array("routing_input.pet", routing_input.pet)

        if routing_input.surface_runoff.shape != self._shape:
            raise ValueError(f"'routing_input.surface_runoff' must have shape {self._shape}, got {routing_input.surface_runoff.shape}")

        if routing_input.pet.shape != self._shape:
            raise ValueError(f"'routing_input.pet' must have shape {self._shape}, got {routing_input.pet.shape}")

        if routing_input.subsurface_runoff is not None:
            _validate_spatial_float_array(
                "routing_input.subsurface_runoff",
                routing_input.subsurface_runoff,
            )

            if routing_input.subsurface_runoff.shape != self._shape:
                raise ValueError(
                    f"'routing_input.subsurface_runoff' must have shape {self._shape}, got {routing_input.subsurface_runoff.shape}"
                )

        lateral_inflow_m3s = build_lateral_inflow_m3s(
            routing_input.surface_runoff,
            subsurface_runoff_mm_dt=routing_input.subsurface_runoff,
            include_subsurface_runoff=self.config.include_subsurface_runoff,
            cell_area_m2=self._cell_area_m2,
            dt_seconds=self.domain.time.dt_seconds,
        )

        lateral_inflow_flat = lateral_inflow_m3s.reshape(-1)
        pet_reservoir_mm_dt = extract_reservoir_pet_mm_dt(
            routing_input.pet,
            domain=self.domain,
        )

        cell_inflow_flat = np.zeros(self.network.n_cells, dtype=float)
        cell_outflow_flat = np.zeros(self.network.n_cells, dtype=float)

        reservoir_inflow_m3s = np.zeros(self.n_reservoirs, dtype=float)
        reservoir_requested_release_m3s = np.zeros(self.n_reservoirs, dtype=float)
        reservoir_storage_fraction = np.zeros(self.n_reservoirs, dtype=float)
        reservoir_surface_area_m2 = np.zeros(self.n_reservoirs, dtype=float)
        reservoir_evaporation_loss_m3 = np.zeros(self.n_reservoirs, dtype=float)
        reservoir_storage_next_m3 = self._latest_state.reservoir_storage_m3.astype(float, copy=True)
        reservoir_release_m3s = np.zeros(self.n_reservoirs, dtype=float)
        reservoir_spill_m3s = np.zeros(self.n_reservoirs, dtype=float)
        reservoir_total_outflow_m3s = np.zeros(self.n_reservoirs, dtype=float)
        reservoir_zones: list[ReservoirOperationZone | None] = [None] * self.n_reservoirs

        previous_cell_outflow_flat = self._latest_state.previous_cell_outflow_m3s_flat

        for linear_index in self.network.upstream_to_downstream_order:
            linear_index = int(linear_index)

            upstream_flow_sum_m3s = float(
                sum(cell_outflow_flat[upstream_index] for upstream_index in self.network.upstream_linear_indices[linear_index])
            )

            total_inflow_m3s = float(lateral_inflow_flat[linear_index] + upstream_flow_sum_m3s)
            cell_inflow_flat[linear_index] = total_inflow_m3s

            reservoir_id = int(self.network.reservoir_id_by_linear_index[linear_index])

            if reservoir_id >= 0 and self.config.enable_reservoirs:
                reservoir = self.domain.reservoirs[reservoir_id]
                storage_prev_m3 = float(reservoir_storage_next_m3[reservoir_id])

                decision = apply_reservoir_operating_rules(
                    storage_prev_m3,
                    capacity_m3=float(reservoir.capacity),
                    config=self.config.reservoir_rules,
                )

                update = update_reservoir_storage(
                    storage_prev_m3=storage_prev_m3,
                    inflow_m3s=total_inflow_m3s,
                    controlled_release_m3s=decision.requested_release_m3s,
                    pet_mm_dt=float(pet_reservoir_mm_dt[reservoir_id]),
                    capacity_m3=float(reservoir.capacity),
                    dt_seconds=self.domain.time.dt_seconds,
                    config=self.config.reservoir_storage,
                )

                reservoir_inflow_m3s[reservoir_id] = total_inflow_m3s
                reservoir_requested_release_m3s[reservoir_id] = decision.requested_release_m3s
                reservoir_storage_fraction[reservoir_id] = decision.storage_fraction
                reservoir_surface_area_m2[reservoir_id] = update.surface_area_m2
                reservoir_evaporation_loss_m3[reservoir_id] = update.evaporation_loss_m3
                reservoir_storage_next_m3[reservoir_id] = update.storage_m3
                reservoir_release_m3s[reservoir_id] = update.controlled_release_m3s
                reservoir_spill_m3s[reservoir_id] = update.spill_m3s
                reservoir_total_outflow_m3s[reservoir_id] = update.total_outflow_m3s
                reservoir_zones[reservoir_id] = decision.zone

                cell_outflow_flat[linear_index] = update.total_outflow_m3s
            else:
                routed_outflow_m3s = compute_linear_reservoir_outflow_m3s(
                    inflow_m3s=total_inflow_m3s,
                    previous_outflow_m3s=float(previous_cell_outflow_flat[linear_index]),
                    dt_seconds=self.domain.time.dt_seconds,
                    time_constant_hours=self.config.channel_time_constant_hours,
                )
                cell_outflow_flat[linear_index] = routed_outflow_m3s

        self._latest_state = RegulatedRoutingState(
            previous_cell_outflow_m3s_flat=cell_outflow_flat.astype(float, copy=True),
            reservoir_storage_m3=reservoir_storage_next_m3.astype(float, copy=True),
        )

        outlet_discharge_m3s = float(cell_outflow_flat[self.network.outlet_linear_index])

        if self.config.enable_reservoirs:
            if any(zone is None for zone in reservoir_zones):
                raise RuntimeError("Reservoir zones were not fully assigned during routing.")

        if not self.config.enable_reservoirs:
            reservoir_inflow_m3s = np.full(self.n_reservoirs, np.nan, dtype=float)
            reservoir_requested_release_m3s = np.full(self.n_reservoirs, np.nan, dtype=float)
            reservoir_storage_fraction = np.full(self.n_reservoirs, np.nan, dtype=float)
            reservoir_surface_area_m2 = np.full(self.n_reservoirs, np.nan, dtype=float)
            reservoir_evaporation_loss_m3 = np.full(self.n_reservoirs, np.nan, dtype=float)
            reservoir_storage_next_m3 = np.full(self.n_reservoirs, np.nan, dtype=float)
            reservoir_release_m3s = np.full(self.n_reservoirs, np.nan, dtype=float)
            reservoir_spill_m3s = np.full(self.n_reservoirs, np.nan, dtype=float)
            reservoir_total_outflow_m3s = np.full(self.n_reservoirs, np.nan, dtype=float)
            reservoir_zones = [None] * self.n_reservoirs

        final_reservoir_zones = tuple(reservoir_zones)

        return RegulatedRoutingOutput(
            lateral_inflow_m3s=lateral_inflow_m3s.astype(float, copy=False),
            cell_inflow_m3s=cell_inflow_flat.reshape(self._shape).astype(float, copy=False),
            channel_flow_m3s=cell_outflow_flat.reshape(self._shape).astype(float, copy=False),
            outlet_discharge_m3s=outlet_discharge_m3s,
            reservoir_inflow_m3s=reservoir_inflow_m3s.astype(float, copy=False),
            reservoir_requested_release_m3s=reservoir_requested_release_m3s.astype(float, copy=False),
            reservoir_storage_fraction=reservoir_storage_fraction.astype(float, copy=False),
            reservoir_surface_area_m2=reservoir_surface_area_m2.astype(float, copy=False),
            reservoir_evaporation_loss_m3=reservoir_evaporation_loss_m3.astype(float, copy=False),
            reservoir_storage_m3=reservoir_storage_next_m3.astype(float, copy=False),
            reservoir_release_m3s=reservoir_release_m3s.astype(float, copy=False),
            reservoir_spill_m3s=reservoir_spill_m3s.astype(float, copy=False),
            reservoir_total_outflow_m3s=reservoir_total_outflow_m3s.astype(float, copy=False),
            reservoir_zones=final_reservoir_zones,
        )
