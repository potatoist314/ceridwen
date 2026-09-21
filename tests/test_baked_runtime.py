"""Baked runtime path for a sampled ``sigma_smooth`` and ``zred``.

``Spectrum(baked_runtime=True)`` precomputes the static indices, weights and
tapers of the two-stage smoothing chain and replaces the binary search of the
redshift stretch with a lookup table.  The operators are unchanged, so the
prediction and the log-likelihood must equal the ``baked_runtime=False`` path
(sedpy_jax smoothers and ``jnp.interp``) to rounding error.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ceridwen.likelihood import (
    DiagonalGaussianLikelihood,
    DiagonalNoiseModel,
    PolynomialCalibration,
)
from ceridwen.observation._smoothing import make_static_grid_interp
from ceridwen.observation.spectrum import Spectrum

Z0 = 0.70
N_POINTS = 25


def _model_grid(lo=2500.0, hi=6000.0, resolving_power=6000.0):
    n = int(np.log(hi / lo) * resolving_power) + 1
    return lo * np.exp(np.arange(n) / resolving_power)


def _model_spectrum(wm):
    flux = 1.0 + 0.2 * (wm - wm[0]) / (wm[-1] - wm[0])
    for centre, depth in ((3934.0, 0.5), (3969.0, 0.4), (4102.0, 0.3),
                          (4341.0, 0.3), (4861.0, 0.35), (5175.0, 0.25)):
        flux = flux * (1.0 - depth * np.exp(-0.5 * ((wm - centre) / 1.5) ** 2))
    return flux


def _observed_grid():
    # Slightly non-uniform, as an air-to-vacuum converted linear grid is.
    wo = np.arange(6300.0, 8800.0, 0.6)
    return wo + 2e-7 * (wo - wo[0]) ** 2 / 0.6


# smoothtype, resolution, inres, library curve: the scalar-FFT, the
# constant-wavelength and the wavelength-dependent LSF instrument stages.
ROUTES = {
    "vel_scalar": dict(smoothtype="R", resolution=2600.0, inres=0.0, lib=False),
    "lambda_scalar": dict(smoothtype="lambda", resolution=2.5, inres=0.0, lib=False),
    "vel_library_curve": dict(smoothtype="R", resolution=2600.0, inres="auto", lib=True),
}


def _build(route, baked):
    cfg = ROUTES[route]
    wo = _observed_grid()
    wm = _model_grid()
    rng = np.random.default_rng(3)
    data = np.interp(wo, wm * (1.0 + Z0), _model_spectrum(wm))
    spec = Spectrum(
        wavelength=wo, flux=data * (1.0 + 0.02 * rng.standard_normal(wo.size)),
        uncertainty=0.02 * data, smoothtype=cfg["smoothtype"],
        resolution=cfg["resolution"], res_convention="fwhm", inres=cfg["inres"],
        sigma_losvd=200.0, fit_sigma_smooth=True, free_z=True,
        baked_runtime=baked, name="spec",
    )
    lib = (wm, np.linspace(20.0, 45.0, wm.size)) if cfg["lib"] else None
    spec.setup_for_model(wm, zred=Z0, lib_resolution=lib)
    return spec, wm


def _sampled_points():
    rng = np.random.default_rng(11)
    wm = _model_grid()
    spectra = _model_spectrum(wm) * rng.uniform(0.5, 1.5, (N_POINTS, 1)) * (
        1.0 + 0.05 * rng.standard_normal((N_POINTS, wm.size)))
    return (jnp.asarray(spectra, dtype=jnp.float32),
            jnp.asarray(rng.uniform(80.0, 350.0, N_POINTS)),
            jnp.asarray(rng.uniform(Z0 - 0.1, Z0 + 0.1, N_POINTS)),
            jnp.asarray(rng.uniform(np.log(0.01), np.log(0.10), N_POINTS)))


@pytest.mark.parametrize("route", ROUTES)
def test_prediction_and_log_likelihood_match_unbaked_path(route):
    spectra, sigma, zred, log_f = _sampled_points()
    results = {}
    for baked in (False, True):
        spec, wm = _build(route, baked)
        likelihood = DiagonalGaussianLikelihood(
            noise_model=DiagonalNoiseModel(use_fractional=True),
            calibration=PolynomialCalibration.from_spectrum(
                spec, order=10, fit_constant=False, prior_sigma=0.1,
                marginalize=True),
        )

        def point(flux, s, z, lf, spec=spec, wm=wm, likelihood=likelihood):
            mu = spec.predict(flux, wm, sigma_smooth=s, zred=z)
            lnl, _ = likelihood(spec.flux, mu, spec.uncertainty, spec.mask,
                                params={"log_f_calib": lf[None]})
            return mu, lnl

        results[baked] = jax.jit(jax.vmap(point))(spectra, sigma, zred, log_f)

    mu_old, lnl_old = (np.asarray(v) for v in results[False])
    mu_new, lnl_new = (np.asarray(v) for v in results[True])
    assert np.all(np.isfinite(lnl_old))
    # The shorter zero pad moves the wrap of the taper's Nyquist cut-off, a
    # 1e-10 term of the FFT method itself for a kernel two pixels wide.
    np.testing.assert_allclose(mu_new, mu_old, rtol=1e-9, atol=0.0)
    # chi-square sums of order 1e6 carry the 1e-12 prediction rounding.
    np.testing.assert_allclose(lnl_new, lnl_old, rtol=1e-9, atol=1e-5)


def test_static_grid_interp_equals_jnp_interp():
    wo = _observed_grid()
    rng = np.random.default_rng(5)
    fp = jnp.asarray(rng.uniform(0.5, 1.5, wo.size))
    # Nodes, points one ulp either side of nodes, both out-of-range sides.
    x = np.concatenate([
        wo, np.nextafter(wo, -np.inf), np.nextafter(wo, np.inf),
        rng.uniform(wo[0] - 300.0, wo[-1] + 300.0, 20000),
    ])
    got = make_static_grid_interp(wo)(jnp.asarray(x), fp)
    want = jnp.interp(jnp.asarray(x), jnp.asarray(wo), fp)
    # A wrong interval changes the value at the 1e-3 level; 1e-13 allows only
    # the last-bit difference between the eager and the compiled arithmetic.
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-13, atol=0.0)
