from __future__ import annotations

import numpy as np
import pytest

from simulator.core.contracts import RegulatedRoutingInput
from simulator.routing.model import RegulatedRoutingConfig, RegulatedRoutingModel
from simulator.routing.network import build_simplified_drainage_network
from simulator.routing.rules import ReservoirRulesConfig, compute_requested_release_m3s


def test_reservoir_rules_reject_flood_fraction_equal_to_one() -> None:
    with pytest.raises(ValueError):
        ReservoirRulesConfig(
            conservation_fraction=0.5,
            flood_fraction=1.0,
        )


def test_requested_release_at_full_storage_returns_max_controlled_release() -> None:
    config = ReservoirRulesConfig(
        min_release_m3s=0.2,
        target_release_m3s=1.5,
        max_controlled_release_m3s=6.0,
        conservation_fraction=0.5,
        flood_fraction=0.9,
    )

    release = compute_requested_release_m3s(
        1000.0,
        capacity_m3=1000.0,
        config=config,
    )

    assert release == pytest.approx(6.0)


def test_routing_with_disabled_reservoirs_marks_reservoir_outputs_as_nan(
    small_domain_with_reservoir,
) -> None:
    network = build_simplified_drainage_network(small_domain_with_reservoir)
    model = RegulatedRoutingModel(
        RegulatedRoutingConfig(enable_reservoirs=False),
        domain=small_domain_with_reservoir,
        network=network,
    )

    zeros = np.zeros(small_domain_with_reservoir.shape, dtype=float)
    output = model.step(
        routing_input=RegulatedRoutingInput(
            domain=small_domain_with_reservoir,
            step=0,
            timestamp=small_domain_with_reservoir.time.timestamps[0],
            surface_runoff=zeros,
            subsurface_runoff=zeros,
            pet=zeros,
        )
    )

    assert output.channel_flow_m3s.shape == small_domain_with_reservoir.shape
    assert output.outlet_discharge_m3s >= 0.0
    assert np.isnan(output.reservoir_storage_m3).all()
    assert np.isnan(output.reservoir_release_m3s).all()
    assert output.reservoir_zones == (None,)
