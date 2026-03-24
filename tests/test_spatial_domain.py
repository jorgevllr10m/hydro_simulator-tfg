import numpy as np

from simulator.core.types import BasinDefinition, GridDefinition, SpatialDomain


def test_spatial_domain():
    # Define a grid 5x5 with 1000m cell size
    grid = GridDefinition(nx=5, ny=5, dx=1000, dy=1000)

    # Define a basin mask with the first 4 cells active
    mask = np.array(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )

    basin = BasinDefinition(mask=mask)

    # Create SpatialDomain
    domain = SpatialDomain(grid=grid, basin=basin)

    # Check properties
    assert domain.shape == (5, 5)
    assert domain.n_active_cells == 4
    assert domain.active_fraction == 0.16
    assert np.array_equal(domain.x_coords, np.array([500.0, 1500.0, 2500.0, 3500.0, 4500.0]))
    assert np.array_equal(domain.y_coords, np.array([500.0, 1500.0, 2500.0, 3500.0, 4500.0]))

    print("All tests passed!")
