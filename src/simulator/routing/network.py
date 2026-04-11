from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from simulator.core.types import BoolArray, SimulationDomain

IntArray = NDArray[np.int_]

OUTSIDE_BASIN_INDEX = -2  # The cell is outside the basin
NO_DOWNSTREAM_INDEX = -1  # That cell is indeed in the basin, but it doesn't have downstream because it's the outlet


def _validate_bool_mask(name: str, value: BoolArray) -> None:
    """Validate a 2D boolean NumPy mask."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"'{name}' must be a numpy.ndarray, got {type(value).__name__}")
    if value.ndim != 2:
        raise ValueError(f"'{name}' must be a 2D array with shape (ny, nx), got ndim={value.ndim}")
    if value.dtype != np.bool_:
        raise TypeError(f"'{name}' must have boolean dtype, got {value.dtype}")


def _validate_int_array(name: str, value: IntArray, *, ndim: int) -> None:
    """Validate a NumPy integer array with a fixed number of dimensions."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"'{name}' must be a numpy.ndarray, got {type(value).__name__}")
    if value.ndim != ndim:
        raise ValueError(f"'{name}' must be a {ndim}D array, got ndim={value.ndim}")
    if not np.issubdtype(value.dtype, np.integer):
        raise TypeError(f"'{name}' must have integer dtype, got {value.dtype}")


def _flatten_cell_index(
    cell_y: int,
    cell_x: int,
    *,
    nx: int,
) -> int:
    """Convert a 2D cell index (y, x) into a flat linear index."""
    return cell_y * nx + cell_x


def _unflatten_cell_index(
    linear_index: int,
    *,
    nx: int,
) -> tuple[int, int]:
    """Convert a flat linear index into a 2D cell index (y, x)."""
    return divmod(linear_index, nx)


def _iter_active_neighbors_4(
    active_mask: BoolArray,
    *,
    cell_y: int,
    cell_x: int,
) -> list[tuple[int, int]]:
    """Return 4-neighbour active cells around one cell."""
    ny, nx = active_mask.shape
    neighbors: list[tuple[int, int]] = []

    candidate_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dy, dx in candidate_offsets:
        neighbor_y = cell_y + dy
        neighbor_x = cell_x + dx

        if not (0 <= neighbor_y < ny and 0 <= neighbor_x < nx):
            continue

        if not active_mask[neighbor_y, neighbor_x]:
            continue

        neighbors.append((neighbor_y, neighbor_x))

    return neighbors


def _find_boundary_active_cells(
    active_mask: BoolArray,
) -> list[tuple[int, int]]:
    """Return all active cells that lie on the basin boundary."""
    ny, nx = active_mask.shape
    boundary_cells: list[tuple[int, int]] = []

    for cell_y in range(ny):
        for cell_x in range(nx):
            if not active_mask[cell_y, cell_x]:
                continue

            if cell_y in {0, ny - 1} or cell_x in {0, nx - 1}:
                boundary_cells.append((cell_y, cell_x))

    return boundary_cells


def _select_default_outlet_cell(
    active_mask: BoolArray,
) -> tuple[int, int]:
    """Select a deterministic outlet cell on the basin boundary.

    Current MVP choice:
    - choose the active boundary cell with the largest (y, x),
      i.e. the most south-eastern active boundary cell.

    For a fully active rectangular grid this becomes the bottom-right cell.
    """
    boundary_cells = _find_boundary_active_cells(active_mask)

    if not boundary_cells:
        raise ValueError("Cannot build a drainage network without at least one active boundary cell.")

    return max(boundary_cells, key=lambda cell: (cell[0], cell[1]))


def _compute_distance_to_outlet(
    active_mask: BoolArray,
    *,
    outlet_cell: tuple[int, int],
) -> IntArray:
    """Compute graph distance in number of 4-neighbour steps to the outlet."""
    ny, nx = active_mask.shape
    distance = np.full((ny, nx), -1, dtype=int)

    outlet_y, outlet_x = outlet_cell
    distance[outlet_y, outlet_x] = 0

    queue: deque[tuple[int, int]] = deque([outlet_cell])  # * FIFO; BFS

    while queue:
        current_y, current_x = queue.popleft()
        current_distance = int(distance[current_y, current_x])

        for neighbor_y, neighbor_x in _iter_active_neighbors_4(
            active_mask,
            cell_y=current_y,
            cell_x=current_x,
        ):
            if distance[neighbor_y, neighbor_x] != -1:
                continue

            distance[neighbor_y, neighbor_x] = current_distance + 1
            queue.append((neighbor_y, neighbor_x))

    unreachable_active = active_mask & (distance < 0)
    if np.any(unreachable_active):
        raise ValueError(
            "The basin mask is not fully connected to the selected outlet. "
            "This simplified drainage network requires one connected active component."
        )

    return distance.reshape(-1)


def _choose_downstream_cell(
    active_mask: BoolArray,
    distance_to_outlet: IntArray,
    *,
    cell_y: int,
    cell_x: int,
) -> tuple[int, int] | None:
    """Choose the unique downstream neighbour for one active cell.

    Rule:
    - the downstream cell must be an active 4-neighbour with strictly smaller
      distance to the outlet
    - if several candidates exist, choose the one with the largest (y, x)
      to keep the network deterministic and slightly biased south-eastward
    """
    ny, nx = active_mask.shape
    current_linear_index = _flatten_cell_index(cell_y, cell_x, nx=nx)
    current_distance = int(distance_to_outlet[current_linear_index])

    if current_distance == 0:
        return None

    candidates: list[tuple[int, int]] = []

    for neighbor_y, neighbor_x in _iter_active_neighbors_4(
        active_mask,
        cell_y=cell_y,
        cell_x=cell_x,
    ):
        neighbor_linear_index = _flatten_cell_index(neighbor_y, neighbor_x, nx=nx)
        neighbor_distance = int(distance_to_outlet[neighbor_linear_index])

        if neighbor_distance == current_distance - 1:
            candidates.append((neighbor_y, neighbor_x))

    if not candidates:
        raise RuntimeError(f"Active cell {(cell_y, cell_x)} with distance {current_distance} has no valid downstream neighbour.")

    return max(candidates, key=lambda cell: (cell[0], cell[1]))


def _build_downstream_linear_index(
    active_mask: BoolArray,
    distance_to_outlet: IntArray,
) -> IntArray:
    """Build flat downstream mapping for the whole grid."""
    ny, nx = active_mask.shape
    downstream = np.full(ny * nx, OUTSIDE_BASIN_INDEX, dtype=int)

    for cell_y in range(ny):
        for cell_x in range(nx):
            linear_index = _flatten_cell_index(cell_y, cell_x, nx=nx)

            if not active_mask[cell_y, cell_x]:
                continue

            downstream_cell = _choose_downstream_cell(
                active_mask,
                distance_to_outlet,
                cell_y=cell_y,
                cell_x=cell_x,
            )

            if downstream_cell is None:
                downstream[linear_index] = NO_DOWNSTREAM_INDEX
                continue

            downstream_y, downstream_x = downstream_cell
            downstream[linear_index] = _flatten_cell_index(
                downstream_y,
                downstream_x,
                nx=nx,
            )

    return downstream


def _build_upstream_linear_indices(
    downstream_linear_index: IntArray,
) -> tuple[tuple[int, ...], ...]:
    """Build upstream adjacency lists from the downstream mapping."""
    n_cells = int(downstream_linear_index.size)
    upstream_lists: list[list[int]] = [[] for _ in range(n_cells)]

    for linear_index, downstream_index in enumerate(downstream_linear_index):
        downstream_index = int(downstream_index)

        if downstream_index < 0:
            continue

        upstream_lists[downstream_index].append(linear_index)

    return tuple(tuple(items) for items in upstream_lists)


def _build_upstream_to_downstream_order(
    active_mask: BoolArray,
    distance_to_outlet: IntArray,
) -> IntArray:
    """Return active cells ordered from upstream to downstream.

    Since downstream cells always have smaller distance to the outlet,
    sorting by decreasing distance produces a valid topological order.
    """
    ny, nx = active_mask.shape
    active_linear_indices = np.flatnonzero(active_mask.reshape(-1))  # reshape(-1) flattens the mask

    def sort_key(linear_index: int) -> tuple[int, int, int]:
        cell_y, cell_x = _unflatten_cell_index(int(linear_index), nx=nx)
        distance = int(distance_to_outlet[int(linear_index)])
        return (-distance, cell_y, cell_x)

    ordered = sorted((int(index) for index in active_linear_indices), key=sort_key)
    return np.asarray(ordered, dtype=int)


def _build_reservoir_mappings(
    domain: SimulationDomain,
) -> tuple[IntArray, IntArray]:
    """Return reservoir cell indices and reverse cell->reservoir lookup."""
    ny, nx = domain.shape
    n_cells = ny * nx

    reservoir_linear_indices = np.empty(len(domain.reservoirs), dtype=int)
    reservoir_id_by_linear_index = np.full(n_cells, -1, dtype=int)

    for reservoir_id, reservoir in enumerate(domain.reservoirs):
        linear_index = _flatten_cell_index(
            reservoir.cell_y,
            reservoir.cell_x,
            nx=nx,
        )

        if reservoir_id_by_linear_index[linear_index] != -1:
            raise ValueError("Two reservoirs cannot occupy the same cell in this simplified routing network.")

        if not domain.spatial.basin.mask[reservoir.cell_y, reservoir.cell_x]:
            raise ValueError(f"Reservoir '{reservoir.name}' is placed outside the active basin mask.")

        reservoir_linear_indices[reservoir_id] = linear_index
        reservoir_id_by_linear_index[linear_index] = reservoir_id

    return reservoir_linear_indices, reservoir_id_by_linear_index


@dataclass(frozen=True)
class DrainageNetwork:
    """Simplified static drainage network over the simulation grid.

    Conventions
    -----------
    - indices are flat linear indices over the full grid
    - `downstream_linear_index[i]` gives the unique downstream cell of cell i
    - `NO_DOWNSTREAM_INDEX` marks the outlet
    - `OUTSIDE_BASIN_INDEX` marks cells outside the active basin
    """

    shape: tuple[int, int]
    active_mask: BoolArray

    outlet_linear_index: int
    distance_to_outlet: IntArray

    downstream_linear_index: IntArray
    upstream_linear_indices: tuple[tuple[int, ...], ...]

    upstream_to_downstream_order: IntArray

    reservoir_linear_indices: IntArray
    reservoir_id_by_linear_index: IntArray

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or len(self.shape) != 2:
            raise TypeError(f"'shape' must be a tuple[int, int], got {self.shape!r}")

        ny, nx = self.shape
        if not isinstance(ny, int) or not isinstance(nx, int) or ny <= 0 or nx <= 0:
            raise ValueError(f"'shape' must contain positive integers, got {self.shape!r}")

        _validate_bool_mask("active_mask", self.active_mask)

        if self.active_mask.shape != self.shape:
            raise ValueError(f"'active_mask' must have shape {self.shape}, got {self.active_mask.shape}")

        if not isinstance(self.outlet_linear_index, int):
            raise TypeError(f"'outlet_linear_index' must be an int, got {type(self.outlet_linear_index).__name__}")

        n_cells = ny * nx
        if not 0 <= self.outlet_linear_index < n_cells:
            raise ValueError(f"'outlet_linear_index' must be within [0, {n_cells - 1}], got {self.outlet_linear_index}")

        _validate_int_array("distance_to_outlet", self.distance_to_outlet, ndim=1)
        _validate_int_array("downstream_linear_index", self.downstream_linear_index, ndim=1)
        _validate_int_array("upstream_to_downstream_order", self.upstream_to_downstream_order, ndim=1)
        _validate_int_array("reservoir_linear_indices", self.reservoir_linear_indices, ndim=1)
        _validate_int_array("reservoir_id_by_linear_index", self.reservoir_id_by_linear_index, ndim=1)

        if self.distance_to_outlet.size != n_cells:
            raise ValueError(f"'distance_to_outlet' must have size {n_cells}, got {self.distance_to_outlet.size}")

        if self.downstream_linear_index.size != n_cells:
            raise ValueError(f"'downstream_linear_index' must have size {n_cells}, got {self.downstream_linear_index.size}")

        if self.reservoir_id_by_linear_index.size != n_cells:
            raise ValueError(f"'reservoir_id_by_linear_index' must have size {n_cells}, got {self.reservoir_id_by_linear_index.size}")

        if len(self.upstream_linear_indices) != n_cells:
            raise ValueError(
                f"'upstream_linear_indices' must contain one entry per grid cell, got {len(self.upstream_linear_indices)}"
            )

        if not self.active_mask.reshape(-1)[self.outlet_linear_index]:
            raise ValueError("The outlet cell must belong to the active basin.")

        if int(self.downstream_linear_index[self.outlet_linear_index]) != NO_DOWNSTREAM_INDEX:
            raise ValueError("The outlet cell must have NO_DOWNSTREAM_INDEX as downstream target.")

    @property
    def n_cells(self) -> int:
        """Return total number of grid cells."""
        ny, nx = self.shape
        return ny * nx

    @property
    def n_active_cells(self) -> int:
        """Return number of active basin cells."""
        return int(np.count_nonzero(self.active_mask))

    @property
    def outlet_cell(self) -> tuple[int, int]:
        """Return outlet cell as (y, x)."""
        return self.linear_to_cell(self.outlet_linear_index)

    def cell_to_linear(self, cell_y: int, cell_x: int) -> int:
        """Convert a cell (y, x) to a flat linear index."""
        ny, nx = self.shape

        if not isinstance(cell_y, int) or not isinstance(cell_x, int):
            raise TypeError(f"'cell_y' and 'cell_x' must be integers, got {cell_y!r}, {cell_x!r}")

        if not (0 <= cell_y < ny and 0 <= cell_x < nx):
            raise ValueError(f"Cell {(cell_y, cell_x)} is outside the grid. Valid ranges: y=[0, {ny - 1}], x=[0, {nx - 1}]")

        return _flatten_cell_index(cell_y, cell_x, nx=nx)

    def linear_to_cell(self, linear_index: int) -> tuple[int, int]:
        """Convert a flat linear index to a cell (y, x)."""
        if not isinstance(linear_index, int):
            raise TypeError(f"'linear_index' must be an int, got {type(linear_index).__name__}")

        if not 0 <= linear_index < self.n_cells:
            raise ValueError(f"'linear_index' must be within [0, {self.n_cells - 1}], got {linear_index}")

        _, nx = self.shape
        return _unflatten_cell_index(linear_index, nx=nx)

    def downstream_of(self, linear_index: int) -> int | None:
        """Return downstream cell index, or None for outlet/outside-basin."""
        if not isinstance(linear_index, int):
            raise TypeError(f"'linear_index' must be an int, got {type(linear_index).__name__}")

        if not 0 <= linear_index < self.n_cells:
            raise ValueError(f"'linear_index' must be within [0, {self.n_cells - 1}], got {linear_index}")

        downstream_index = int(self.downstream_linear_index[linear_index])

        if downstream_index < 0:
            return None

        return downstream_index

    def is_active_cell(self, linear_index: int) -> bool:
        """Return whether a linear cell index belongs to the active basin."""
        if not isinstance(linear_index, int):
            raise TypeError(f"'linear_index' must be an int, got {type(linear_index).__name__}")

        if not 0 <= linear_index < self.n_cells:
            raise ValueError(f"'linear_index' must be within [0, {self.n_cells - 1}], got {linear_index}")

        return bool(self.active_mask.reshape(-1)[linear_index])


def build_simplified_drainage_network(
    domain: SimulationDomain,
) -> DrainageNetwork:
    """Build the simplified static drainage network used by phase 7.

    Current MVP assumptions
    -----------------------
    - the basin mask is one connected component
    - each active cell drains to exactly one active 4-neighbour
    - the outlet is chosen automatically as the most south-eastern active
      boundary cell
    """
    if not isinstance(domain, SimulationDomain):
        raise TypeError(f"'domain' must be a SimulationDomain, got {type(domain).__name__}")

    active_mask = domain.spatial.basin.mask
    _validate_bool_mask("domain.spatial.basin.mask", active_mask)

    outlet_cell = _select_default_outlet_cell(active_mask)
    outlet_linear_index = _flatten_cell_index(
        outlet_cell[0],
        outlet_cell[1],
        nx=domain.shape[1],
    )

    distance_to_outlet = _compute_distance_to_outlet(
        active_mask,
        outlet_cell=outlet_cell,
    )

    downstream_linear_index = _build_downstream_linear_index(
        active_mask,
        distance_to_outlet,
    )

    upstream_linear_indices = _build_upstream_linear_indices(
        downstream_linear_index,
    )

    upstream_to_downstream_order = _build_upstream_to_downstream_order(
        active_mask,
        distance_to_outlet,
    )

    reservoir_linear_indices, reservoir_id_by_linear_index = _build_reservoir_mappings(
        domain,
    )

    return DrainageNetwork(
        shape=domain.shape,
        active_mask=active_mask.copy(),
        outlet_linear_index=outlet_linear_index,
        distance_to_outlet=distance_to_outlet,
        downstream_linear_index=downstream_linear_index,
        upstream_linear_indices=upstream_linear_indices,
        upstream_to_downstream_order=upstream_to_downstream_order,
        reservoir_linear_indices=reservoir_linear_indices,
        reservoir_id_by_linear_index=reservoir_id_by_linear_index,
    )
