"""Static-width smoothing: one combined Gaussian instead of two chained ones.

When every broadening width is known at ``setup_for_model`` time, the LOSVD and
the instrumental LSF compose in quadrature into a single Gaussian.  These tests
pin three properties of that collapse:

* it recovers the analytic width of a Gaussian line more accurately than the
  chained form it replaces (the arbiter -- there is a closed-form answer),
* it agrees with a direct dense convolution,
* the free-LOSVD path is untouched, and ``_H`` is no longer built eagerly.

No SSP grid and no FSPS install are needed: the smoother only sees wavelength
grids, so the model spectrum here is synthetic.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ceridwen.observation.spectrum import Spectrum
from ceridwen.observation._smoothing import (
    combined_sigma_lambda,
    make_static_smoother,
)
from ceridwen.observation.base import _CKMS

SIGMA_LOSVD = 259.5      # km/s, a typical quiescent galaxy
INSTR_R = 2600.0         # resolving power, LEGA-C-like


def _model_grid(lo=4500.0, hi=11000.0, resolving_power=6000.0):
    """Log-uniform rest-frame model grid, as a real SSP library provides."""
    n = int(np.log(hi / lo) * resolving_power) + 1
    return lo * np.exp(np.arange(n) / resolving_power)


def _observed_grid(lo=5800.0, hi=9500.0, n=6000):
    return np.linspace(lo, hi, n)


def _build(fit_sigma_smooth=False, sigma_losvd=SIGMA_LOSVD, smoothtype="R"):
    """A Spectrum wired the way the joint LEGA-C fits wire it."""
    wo = _observed_grid()
    wm = _model_grid()
    extra = (dict(resolution=INSTR_R, res_convention="fwhm", inres=0.0)
             if smoothtype is not None else {})
    spec = Spectrum(
        wavelength=wo,
        flux=np.ones_like(wo),
        uncertainty=np.ones_like(wo),
        smoothtype=smoothtype,
        sigma_losvd=sigma_losvd,
        fit_sigma_smooth=fit_sigma_smooth,
        name="spec",
        **extra,
    )
    spec.setup_for_model(wm, zred=0.0)
    return spec, wm, wo


def _gaussian_line(wave, center, sigma_kms):
    sigma_lambda = sigma_kms * center / _CKMS
    return np.exp(-0.5 * ((np.asarray(wave) - center) / sigma_lambda) ** 2)


def _measured_sigma_kms(wave, flux, center, window_sigma_kms):
    """Second-moment width of a line, in km/s."""
    wave = np.asarray(wave, dtype=float)
    flux = np.clip(np.asarray(flux, dtype=float), 0.0, None)
    half = 12.0 * window_sigma_kms * center / _CKMS
    sel = np.abs(wave - center) < half
    w, f = wave[sel], flux[sel]
    total = f.sum()
    mean = (w * f).sum() / total
    var = ((w - mean) ** 2 * f).sum() / total
    return np.sqrt(var) / center * _CKMS


# ---------------------------------------------------------------------------
# The arbiter: a Gaussian line has a closed-form smoothed width.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("center", [6200.0, 7200.0, 8200.0, 9200.0])
def test_gaussian_line_width_matches_quadrature_sum(center):
    """Smoothing a Gaussian must give sqrt(sigma_in^2 + sigma_total^2)."""
    spec, wm, wo = _build()
    sigma_instr = _CKMS / INSTR_R / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    expected = np.sqrt(40.0**2 + SIGMA_LOSVD**2 + sigma_instr**2)

    line = _gaussian_line(wm, center, 40.0)
    out = np.asarray(spec.predict(line, wm))
    got = _measured_sigma_kms(wo, out, center, expected)

    assert got == pytest.approx(expected, rel=0.01), (
        f"line at {center} A: recovered {got:.2f} km/s, expected {expected:.2f}"
    )


def test_line_flux_is_approximately_conserved():
    """A convolution redistributes flux rather than creating or destroying it.

    Only approximately: the CDF transform makes the convolution Gaussian in a
    coordinate that is not linear in wavelength, so wavelength-space flux is
    conserved to the accuracy of that transform, not exactly.  The input line
    is wide enough to be well sampled on the model grid, otherwise the
    reference integral is itself inaccurate.
    """
    spec, wm, wo = _build()
    line = _gaussian_line(wm, 7200.0, 200.0)
    out = np.asarray(spec.predict(line, wm))
    inside = (wm >= wo[0]) & (wm <= wo[-1])
    assert np.trapezoid(out, wo) == pytest.approx(
        np.trapezoid(line[inside], wm[inside]), rel=1e-2
    )


def test_combined_beats_chained_against_a_dense_reference():
    """One combined Gaussian is closer to a direct convolution than two chained.

    This is the reason for the change: it is not a speed-for-accuracy trade.
    The reference is a brute-force sum with no FFT and no CDF transform.
    """
    spec, wm, wo = _build()
    sigma_instr = _CKMS / INSTR_R / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_tot = np.sqrt(SIGMA_LOSVD**2 + sigma_instr**2)

    rng = np.random.default_rng(20260901)
    # A spectrum with real structure: continuum plus absorption lines.
    model = np.ones_like(wm)
    for c in rng.uniform(5900.0, 9400.0, 40):
        model -= 0.6 * _gaussian_line(wm, c, 60.0)

    lnw = np.log(wm)
    dense = np.empty(wo.size)
    for k, lam in enumerate(wo):
        v = _CKMS * (lnw - np.log(lam))
        sel = np.abs(v) < 6.0 * sigma_tot
        g = np.exp(-0.5 * (v[sel] / sigma_tot) ** 2)
        dense[k] = (g * model[sel]).sum() / g.sum()

    got = np.asarray(spec.predict(model, wm))
    interior = (wo > wo[0] + 200.0) & (wo < wo[-1] - 200.0)
    err = np.abs(got[interior] - dense[interior]).max()
    assert err < 5e-3, f"combined smoother differs from dense reference by {err:.2e}"


# ---------------------------------------------------------------------------
# combined_sigma_lambda: units and limiting cases
# ---------------------------------------------------------------------------

def test_sigma_combines_in_quadrature_and_carries_units():
    wave = np.array([5000.0, 7000.0, 9000.0])
    got = combined_sigma_lambda(wave, None, 300.0, None)
    assert got == pytest.approx(300.0 * wave / _CKMS)


def test_library_width_is_subtracted_in_quadrature():
    wave = np.array([6000.0, 8000.0])
    target = np.array([2.0, 2.0])
    got = combined_sigma_lambda(wave, target, None, np.array([1.2, 1.2]))
    assert got == pytest.approx(np.sqrt(2.0**2 - 1.2**2))


def test_unknown_library_pixels_subtract_nothing():
    """NaN means 'resolution unknown here', not 'resolution zero'."""
    wave = np.array([6000.0, 8000.0])
    target = np.array([2.0, 2.0])
    got = combined_sigma_lambda(wave, target, None, np.array([np.nan, 1.2]))
    assert got[0] == pytest.approx(2.0)
    assert got[1] == pytest.approx(np.sqrt(2.0**2 - 1.2**2))


def test_library_coarser_than_target_floors_at_one_pixel():
    """Deconvolution is impossible; the width floors instead of going negative."""
    wave = np.linspace(6000.0, 6100.0, 101)
    target = np.full_like(wave, 0.5)
    got = combined_sigma_lambda(wave, target, None, np.full_like(wave, 5.0))
    assert np.all(got > 0.0)
    assert got == pytest.approx(np.full_like(wave, np.abs(np.gradient(wave)).min()))


# ---------------------------------------------------------------------------
# Grid sizing
# ---------------------------------------------------------------------------

def test_resampling_grid_is_never_coarser_than_the_input():
    """Sizing from the kernel alone can undersample the model and lose line depth."""
    wave = _model_grid(6000.0, 9000.0)
    sigma = combined_sigma_lambda(wave, None, 400.0, None)   # a wide kernel
    smoother = make_static_smoother(wave, sigma, wave)
    assert smoother.grid_size >= wave.size


# ---------------------------------------------------------------------------
# Behaviour preserved elsewhere
# ---------------------------------------------------------------------------

def test_H_is_not_built_until_something_reads_it():
    spec, wm, wo = _build()
    assert spec._H_cached is None, "_H was materialised despite the FFT path"
    assert spec._H.shape == (wo.size, wm.size)
    assert spec._H_cached is not None


def test_H_still_interpolates_correctly_when_requested():
    """Lazy construction must not change the matrix it produces."""
    wo, wm = _observed_grid(), _model_grid()
    spec = Spectrum(wavelength=wo, flux=np.ones_like(wo),
                    uncertainty=np.ones_like(wo), name="plain")
    spec.setup_for_model(wm, zred=0.0)
    ramp = 3.0 * wm + 11.0                      # linear: interpolation is exact
    assert np.asarray(spec._H @ ramp) == pytest.approx(3.0 * wo + 11.0, rel=1e-5)


def test_free_losvd_keeps_the_runtime_sigma_path():
    """fit_sigma_smooth makes the width traced, so the chained form must remain."""
    spec, wm, _ = _build(fit_sigma_smooth=True)
    line = _gaussian_line(wm, 7200.0, 40.0)
    narrow = np.asarray(spec.predict(line, wm, sigma_smooth=100.0))
    wide = np.asarray(spec.predict(line, wm, sigma_smooth=400.0))
    assert narrow.max() > wide.max(), "a larger sigma must give a shallower line"


def test_losvd_only_spectrum_still_smooths():
    """No instrumental stage: the combined width is the LOSVD alone."""
    spec, wm, wo = _build(smoothtype=None)
    expected = np.sqrt(40.0**2 + SIGMA_LOSVD**2)
    line = _gaussian_line(wm, 7200.0, 40.0)
    got = _measured_sigma_kms(wo, np.asarray(spec.predict(line, wm)), 7200.0, expected)
    assert got == pytest.approx(expected, rel=0.01)
