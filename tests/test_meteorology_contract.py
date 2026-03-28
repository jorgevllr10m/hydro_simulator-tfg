import numpy as np

from simulator.core.contracts import MeteoOutput


def test_meteo():
    field = np.zeros((3, 4), dtype=float)

    meteo_out = MeteoOutput(
        precipitation=field,
        air_temperature=field + 15.0,
    )
    print("\nMETEO CONTRACT TEST")
    print(meteo_out.precipitation.shape)

    assert meteo_out.precipitation.shape == (3, 4)
    assert meteo_out.air_temperature.shape == (3, 4)
    assert np.allclose(meteo_out.air_temperature, 15.0)
