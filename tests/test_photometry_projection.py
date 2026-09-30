"""Photometry.setup_for_model: the sparse build returns the dense-matrix _T bit for bit."""

import numpy as np
import pytest

from ceridwen.observation import Photometry


def dense_projection(filterset, wave_model, zred):
    """The dense construction that setup_for_model used before 2026-09-30."""
    wm = (1.0 + float(zred)) * np.asarray(wave_model, dtype=np.float64)
    n_wave = len(wm)
    fnu_to_flam = 2.998e18 / wm**2
    lam_filt = np.asarray(filterset.lam, dtype=np.float64)
    n_lam = len(lam_filt)
    idx = np.clip(np.searchsorted(wm, lam_filt, side="right") - 1, 0, n_wave - 2)
    frac = np.clip((lam_filt - wm[idx]) / (wm[idx + 1] - wm[idx]), 0.0, 1.0)
    outside = (lam_filt < wm[0]) | (lam_filt > wm[-1])
    frac[outside] = 0.0
    H = np.zeros((n_lam, n_wave), dtype=np.float64)
    for j in range(n_lam):
        if outside[j]:
            continue
        H[j, idx[j]] = 1.0 - frac[j]
        H[j, idx[j] + 1] = frac[j]
    trans = np.asarray(filterset.trans, dtype=np.float64)
    return ((trans @ H) * fnu_to_flam[None, :]).astype(np.float32)


# Dense optical sampling like the C3K high-resolution grid: filter-grid and
# model-grid spacings are both about 3 A, so columns sum one to several terms.
WAVE = np.concatenate([np.geomspace(100.0, 3000.0, 1500),
                       np.linspace(3000.5, 25000.0, 8000),
                       np.geomspace(25010.0, 1e8, 1500)])


@pytest.mark.parametrize("filters, wave, zred", [
    (["hsc_g", "spitzer_irac_ch1"], WAVE, 0.6542),
    (["cfht_megacam_us_9301", "hsc_i", "vista_vircam_Ks"], WAVE, 0.0),
    # Model grid ends inside Ks: those filter-grid points get no weight.
    (["cfht_megacam_us_9301", "hsc_i", "vista_vircam_Ks"], WAVE[WAVE < 20000.0], 0.0),
])
def test_sparse_projection_matches_dense(filters, wave, zred):
    phot = Photometry(filters=filters, name="photometry")
    phot.setup_for_model(wave, zred=zred)
    expected = dense_projection(phot.filterset, wave, zred)
    actual = np.asarray(phot._T)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
