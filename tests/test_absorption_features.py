"""Absorption-feature pixel selection: windows, masks, and Spectrum modes."""
from __future__ import annotations

import numpy as np
import pytest
from sedpy_jax.observate import air2vac

from ceridwen.observation import Spectrum
from ceridwen.observation.absorption_features import (
    ABSORPTION_FEATURES,
    AbsorptionFeature,
    absorption_feature_mask,
    feature_windows,
    select_features,
)

C_KMS = 2.998e5


def _flat_spectrum(n=4000, lo=3600.0, hi=9000.0):
    wave = np.linspace(lo, hi, n)
    flux = np.full(n, 1e-27)
    unc = np.full(n, 1e-29)
    return Spectrum(wavelength=wave, flux=flux, uncertainty=unc, name="spectrum")


def test_catalogue_is_well_formed():
    names = [f.name for f in ABSORPTION_FEATURES]
    assert len(names) == len(set(names))
    for feature in ABSORPTION_FEATURES:
        assert feature.kind in {"band", "line"}
        if feature.kind == "band":
            assert feature.upper is not None and feature.upper > feature.lower
        else:
            assert feature.upper is None
    groups = {f.group for f in ABSORPTION_FEATURES}
    assert {"balmer", "ca", "g", "mg", "na", "fe", "tio"} <= groups


def test_band_window_is_vacuum_and_redshifted():
    (hbeta,) = select_features(["Hbeta"])
    window = feature_windows([hbeta], zred=0.6, window_kms=1000.0)
    expected = np.asarray(air2vac(np.array([hbeta.lower, hbeta.upper]))) * 1.6
    assert window.shape == (1, 2)
    np.testing.assert_allclose(window[0], expected, rtol=1e-12)


def test_line_window_uses_velocity_half_width():
    (cak,) = select_features(["CaK"])
    window = feature_windows([cak], zred=0.0, window_kms=500.0)
    centre = float(np.asarray(air2vac(np.array([cak.lower])))[0])
    half = centre * 500.0 / C_KMS
    np.testing.assert_allclose(window[0], [centre - half, centre + half], rtol=1e-12)


def test_mask_keeps_only_pixels_inside_windows():
    wave = np.linspace(3600.0, 9000.0, 20000)
    zred = 0.6
    windows = feature_windows(ABSORPTION_FEATURES, zred=zred)
    mask = absorption_feature_mask(wave, zred=zred)
    inside = np.zeros_like(wave, dtype=bool)
    for lo, hi in windows:
        inside |= (wave >= lo) & (wave <= hi)
    np.testing.assert_array_equal(mask, inside)
    assert 0 < mask.sum() < len(wave)


def test_select_by_name_group_and_object():
    by_group = select_features(["fe"])
    assert by_group and all(f.group == "fe" for f in by_group)
    custom = AbsorptionFeature("custom", "line", 5000.0, None, "custom")
    mixed = select_features(["Mgb", custom, "balmer"])
    assert mixed[0].name == "Mgb" and mixed[1] is custom
    assert any(f.group == "balmer" for f in mixed[2:])
    with pytest.raises(ValueError, match="unknown absorption feature"):
        select_features(["NotALine"])


def test_drop_mode_masks_continuum_and_keeps_uncertainty():
    spec = _flat_spectrum()
    spec.mask_wavelength_range(7000.0, 7100.0)
    before = np.asarray(spec.mask).copy()
    unc_before = np.asarray(spec.uncertainty).copy()
    in_feature = spec.select_absorption_features(zred=0.6, mode="drop")
    np.testing.assert_array_equal(np.asarray(spec.mask), before & in_feature)
    np.testing.assert_array_equal(np.asarray(spec.uncertainty), unc_before)
    assert 0 < spec.ndof < before.sum()
    assert spec.pixel_selection["mode"] == "drop"


def test_downweight_mode_inflates_only_continuum_sigma():
    spec = _flat_spectrum()
    before = np.asarray(spec.mask).copy()
    unc_before = np.asarray(spec.uncertainty).copy()
    in_feature = spec.select_absorption_features(
        zred=0.6, mode="downweight", downweight=5.0
    )
    unc_after = np.asarray(spec.uncertainty)
    np.testing.assert_array_equal(np.asarray(spec.mask), before)
    np.testing.assert_allclose(unc_after[in_feature], unc_before[in_feature])
    np.testing.assert_allclose(unc_after[~in_feature], 5.0 * unc_before[~in_feature])
    model = np.asarray(spec.flux) * 1.01
    chi = ((np.asarray(spec.flux) - model) / unc_before) ** 2
    expected = chi[in_feature].sum() + chi[~in_feature].sum() / 25.0
    assert spec.chi_sq(model) == pytest.approx(expected, rel=1e-10)


def test_invalid_mode_or_downweight_raise():
    spec = _flat_spectrum()
    with pytest.raises(ValueError, match="mode"):
        spec.select_absorption_features(mode="keep")
    with pytest.raises(ValueError, match="downweight"):
        spec.select_absorption_features(mode="downweight", downweight=0.5)


def test_drop_mode_with_no_feature_pixels_raises():
    wave = np.linspace(20000.0, 21000.0, 100)
    spec = Spectrum(wavelength=wave, flux=np.ones(100), uncertainty=np.ones(100))
    with pytest.raises(ValueError, match="no fitted pixels"):
        spec.select_absorption_features(zred=0.0, mode="drop")
