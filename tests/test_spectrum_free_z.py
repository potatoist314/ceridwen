"""Free-redshift path for ``Spectrum``.

A ``Spectrum`` bakes its redshift into the projection at
``setup_for_model``.  With ``free_z=True`` the prediction can be re-evaluated
at a runtime ``zred`` by stretching the observed grid: the model at redshift
``z`` seen at observed pixel ``wo`` is the model at the setup redshift ``z0``
seen at ``wo * (1 + z0) / (1 + z)``.  Every smoothing kernel here is
velocity-shift-invariant, so the stretch commutes with the smoothing and the
only error is the linear interpolation between observed pixels.

The first tests need no SSP grid; the last one runs the CSP dispatch on the
alpha-enhanced schema-2.1 grid and skips when it is absent.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ceridwen.observation.spectrum import Spectrum
from ceridwen.observation.base import _CKMS

Z0 = 0.70
DV_KMS = 60.0                         # a plausible catalogue-redshift error
DZ = DV_KMS / _CKMS * (1.0 + Z0)
SIGMA_LOSVD = 200.0
INSTR_R = 2600.0


def _model_grid(lo=2500.0, hi=6000.0, resolving_power=6000.0):
    n = int(np.log(hi / lo) * resolving_power) + 1
    return lo * np.exp(np.arange(n) / resolving_power)


def _model_spectrum(wm):
    """Continuum with a handful of absorption lines (rest frame)."""
    flux = 1.0 + 0.2 * (wm - wm[0]) / (wm[-1] - wm[0])
    for centre, depth in ((3934.0, 0.5), (3969.0, 0.4), (4102.0, 0.3),
                          (4341.0, 0.3), (4861.0, 0.35), (5175.0, 0.25)):
        flux = flux * (1.0 - depth * np.exp(-0.5 * ((wm - centre) / 1.5) ** 2))
    return jnp.asarray(flux)


def _observed_grid(lo=6300.0, hi=8800.0, step=0.6):
    return np.arange(lo, hi, step)


def _build(zred, free_z, fit_sigma_smooth=False):
    wo = _observed_grid()
    spec = Spectrum(
        wavelength=wo, flux=np.ones_like(wo), uncertainty=np.ones_like(wo),
        smoothtype="R", resolution=INSTR_R, res_convention="fwhm", inres=0.0,
        sigma_losvd=SIGMA_LOSVD, fit_sigma_smooth=fit_sigma_smooth,
        free_z=free_z, name="spec",
    )
    spec.setup_for_model(_model_grid(), zred=zred)
    return spec


def _interior(wo, z_hi):
    """Pixels whose stretched position stays inside the observed grid."""
    lo, hi = wo[0] * (1.0 + z_hi) / (1.0 + Z0), wo[-1] * (1.0 + Z0) / (1.0 + z_hi)
    return (wo > lo + 5.0) & (wo < hi - 5.0)


def test_predict_at_setup_redshift_matches_baked_prediction():
    wm = _model_grid()
    flux = _model_spectrum(wm)
    spec = _build(Z0, free_z=True)
    baked = np.asarray(spec.predict(flux, wm))
    moved = np.asarray(spec.predict(flux, wm, zred=Z0))
    np.testing.assert_allclose(moved, baked, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("fit_sigma_smooth", [False, True])
def test_predict_at_shifted_redshift_matches_fresh_setup(fit_sigma_smooth):
    wm = _model_grid()
    flux = _model_spectrum(wm)
    z1 = Z0 + DZ
    free = _build(Z0, free_z=True, fit_sigma_smooth=fit_sigma_smooth)
    fresh = _build(z1, free_z=False, fit_sigma_smooth=fit_sigma_smooth)
    kw = dict(sigma_smooth=SIGMA_LOSVD) if fit_sigma_smooth else {}
    moved = np.asarray(free.predict(flux, wm, zred=z1, **kw))
    target = np.asarray(fresh.predict(flux, wm, **kw))
    wo = np.asarray(free.wavelength)
    inside = _interior(wo, z1)
    # The shift moves the lines by ~1.4 A on a 0.6 A grid, so the baked
    # prediction itself must be far from the target before the stretch.
    baked = np.asarray(free.predict(flux, wm, **kw))
    assert np.max(np.abs(baked - target)[inside]) > 1e-2
    # Residual floor: linear interpolation between 0.6 A pixels of a line
    # smoothed to ~5 A, h^2 f''/8 ~ 5e-4 of the continuum at the line
    # cores.  LEGA-C pixel noise is ~5e-2, so this is 100x below it.
    np.testing.assert_allclose(moved[inside], target[inside],
                               rtol=0.0, atol=1e-3)


def test_free_z_off_ignores_zred_kwarg_gracefully():
    """Without ``free_z`` the Spectrum has no stretch to apply and must say so."""
    wm = _model_grid()
    flux = _model_spectrum(wm)
    spec = _build(Z0, free_z=False)
    with pytest.raises(TypeError):
        spec.predict(flux, wm, zred=Z0 + DZ)


def test_csp_dispatch_threads_sampled_zred_into_spectrum():
    from _gridfixture import find_test_grid
    grid = find_test_grid("amist_c3k_hr_krou_afe.h5")
    if grid is None:
        pytest.skip("alpha-enhanced schema-2.1 grid not found")
    from ceridwen.csp import CSPBasis_afe
    from ceridwen.ssps import SSPDataAfe

    ssp = SSPDataAfe.load(str(grid))
    csp = CSPBasis_afe(
        ssp, lookback_time=jnp.linspace(0.0, 6.0, 5),
        zh_const=True, sfh_interp="step",
        add_dust=False, add_diffuse_dust=True,
        sigma_losvd_kms=0.0, verbose=False,
    )
    wo = _observed_grid()
    spec = Spectrum(
        wavelength=wo, flux=np.ones_like(wo), uncertainty=np.ones_like(wo),
        smoothtype="R", resolution=INSTR_R, res_convention="fwhm",
        sigma_losvd=SIGMA_LOSVD, fit_sigma_smooth=True, free_z=True,
        name="spec",
    )
    spec.setup_for_model(csp.wave, zred=Z0,
                         lib_resolution=getattr(csp, "lib_resolution", None))

    theta = dict(csp.all_params)
    theta["logmass"] = jnp.array([10.5])
    theta["afe"] = jnp.array([0.2])
    theta["sigma_smooth"] = jnp.array([SIGMA_LOSVD])
    theta["zred"] = jnp.array([Z0])
    at_z0 = np.asarray(csp.predict(theta, [spec])["spec"])
    theta["zred"] = jnp.array([Z0 + DZ])
    at_z1 = np.asarray(csp.predict(theta, [spec])["spec"])

    inside = _interior(wo, Z0 + DZ)
    # The flux factor alone changes the level by ~1e-3; moved lines
    # change the cores by ~1e-2.
    assert np.max(np.abs(at_z1 - at_z0)[inside] / at_z0[inside]) > 5e-3
    # The dispatch must do exactly what a direct predict(..., zred=) does.
    phot_spec, slit_spec, line_slit = csp._assemble_observer_spectra(theta)
    _, slit_spec, _ = csp._apply_mass_redshift_igm(phot_spec, slit_spec,
                                                   line_slit, theta)
    direct = np.asarray(spec.predict(slit_spec, csp.wave,
                                     sigma_smooth=SIGMA_LOSVD, zred=Z0 + DZ))
    np.testing.assert_allclose(at_z1, direct, rtol=1e-8)
