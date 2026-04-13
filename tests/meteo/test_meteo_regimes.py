from __future__ import annotations

import pytest

from simulator.meteo.regimes import (
    REGIME_PROFILES,
    MeteorologicalRegime,
    RegimeProfile,
    get_regime_profile,
)


def test_all_regimes_have_profiles() -> None:
    for regime in MeteorologicalRegime:
        profile = get_regime_profile(regime)
        assert isinstance(profile, RegimeProfile)


def test_regime_profile_indices_are_bounded() -> None:
    for profile in REGIME_PROFILES.values():
        assert 0.0 <= profile.cloudiness_index <= 1.0
        assert 0.0 <= profile.convective_potential_index <= 1.0
        assert 0.0 <= profile.wetness_equilibrium <= 1.0


def test_convective_profile_has_high_convective_potential() -> None:
    convective = get_regime_profile(MeteorologicalRegime.CONVECTIVE)
    stable_dry = get_regime_profile(MeteorologicalRegime.STABLE_DRY)

    assert convective.convective_potential_index > stable_dry.convective_potential_index


def test_frontal_profile_has_more_cloudiness_than_stable_dry() -> None:
    frontal = get_regime_profile(MeteorologicalRegime.FRONTAL_PERSISTENT)
    stable_dry = get_regime_profile(MeteorologicalRegime.STABLE_DRY)

    assert frontal.cloudiness_index > stable_dry.cloudiness_index


def test_regime_profile_rejects_out_of_bounds_values() -> None:
    with pytest.raises(ValueError):
        RegimeProfile(
            cloudiness_index=1.2,
            convective_potential_index=0.5,
            wetness_equilibrium=0.5,
            temperature_anomaly_c=0.0,
        )
