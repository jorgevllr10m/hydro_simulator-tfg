from __future__ import annotations

import numpy as np

from simulator.core.contracts import ObservationInput
from simulator.obs.model import ObservationConfig, ObservationModel, ObservationQualityFlag


def test_reservoir_storage_sensor_returns_missing_when_storage_is_nan(
    small_domain_with_reservoir,
) -> None:
    model = ObservationModel(ObservationConfig(random_seed=1234))

    zeros = np.zeros(small_domain_with_reservoir.shape, dtype=float)
    output = model.step(
        ObservationInput(
            domain=small_domain_with_reservoir,
            step=0,
            timestamp=small_domain_with_reservoir.time.timestamps[0],
            precipitation=zeros,
            channel_flow=zeros,
            reservoir_storage=np.array([np.nan], dtype=float),
        )
    )

    assert output.obs_mask is not None
    assert output.obs_quality_flag is not None
    assert output.obs_storage is not None

    assert output.obs_mask[0] is np.False_ or output.obs_mask[0] == np.False_
    assert int(output.obs_quality_flag[0]) == int(ObservationQualityFlag.MISSING)
    assert np.isnan(output.obs_storage[0])
