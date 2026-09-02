"""Analytic (profile) Chebyshev spectrophotometric calibration.

``PolynomialCalibration`` solves, at every likelihood call, the weighted
linear least-squares Chebyshev polynomial ``P(x)`` such that
``data ~= P(x) * model`` on the unmasked pixels, and hands the calibrated
model to the diagonal Gaussian kernel.  This is Prospector's ``polyopt``
(``PolySedModel.spec_calibration``) done in JAX, inside the jitted
log-posterior.

Pure-kernel tests need no SSP grid.  The one forward-model test loads the
cached DR2 alpha-enhanced grid and skips when it is absent.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebvander

from ceridwen.likelihood import (
    DiagonalGaussianLikelihood,
    DiagonalNoiseModel,
    MultiObservationLikelihood,
    PolynomialCalibration,
)
from ceridwen.observation import Spectrum

WAVE = np.linspace(6000.0, 9000.0, 2000)          # observed-frame A
X = (WAVE - 7500.0) / 1500.0                       # Chebyshev coordinate
TRUE_TILT = np.array([0.02, -0.01, 0.005])         # a_1..a_3 of P = 1 + sum a_m T_m


def _model_flux():
    """Smooth continuum with two absorption features (positive everywhere)."""
    lines = (0.3 * np.exp(-((WAVE - 7000.0) / 8.0) ** 2)
             + 0.2 * np.exp(-((WAVE - 8200.0) / 6.0) ** 2))
    return 1e-17 * (1.0 + 0.3 * X) * (1.0 - lines)


def _p_true(coeffs=TRUE_TILT, constant=1.0):
    return constant * (1.0 + chebvander(X, len(coeffs))[:, 1:] @ coeffs)


def _mock(seed=0, snr=20.0, tilt=TRUE_TILT, constant=1.0):
    mu = _model_flux()
    sigma = mu / snr
    noise = np.random.default_rng(seed).normal(0.0, sigma)
    y = _p_true(tilt, constant) * mu + noise
    return mu, y, sigma, np.ones(WAVE.shape, dtype=bool)


# ---------------------------------------------------------------------------
# Pure kernel
# ---------------------------------------------------------------------------
def test_recovers_noise_free_polynomial_exactly():
    mu = _model_flux()
    y = _p_true() * mu
    sigma = 0.05 * mu
    mask = np.ones(WAVE.shape, dtype=bool)
    cal = PolynomialCalibration.from_wavelength(WAVE, order=3)
    mu_cal, coeffs, ln_prior = cal.calibrate(y, mu, sigma, mask)
    np.testing.assert_allclose(np.asarray(coeffs), TRUE_TILT, atol=1e-9)
    np.testing.assert_allclose(np.asarray(mu_cal), y, rtol=1e-9)
    assert float(ln_prior) == 0.0


def test_masked_pixels_do_not_enter_the_fit():
    mu = _model_flux()
    y = _p_true() * mu
    sigma = 0.05 * mu
    mask = np.ones(WAVE.shape, dtype=bool)
    mask[400:450] = False
    mask[1500:1520] = False
    y = np.where(mask, y, 5.0 * y)          # garbage where masked
    cal = PolynomialCalibration.from_wavelength(WAVE, order=3, mask=mask)
    coeffs = cal.solve(y, mu, sigma, mask)
    np.testing.assert_allclose(np.asarray(coeffs), TRUE_TILT, atol=1e-9)


def test_recovers_injected_tilt_to_better_than_one_percent_with_noise():
    mu, y, sigma, mask = _mock(seed=1, snr=20.0)
    cal = PolynomialCalibration.from_wavelength(WAVE, order=3)
    mu_cal, coeffs, _ = cal.calibrate(y, mu, sigma, mask)
    p_hat = np.asarray(cal.polynomial(coeffs))
    assert np.max(np.abs(p_hat - _p_true())) < 0.01
    truth = _p_true() * mu
    assert np.max(np.abs(np.asarray(mu_cal) / truth - 1.0)) < 0.01


def test_constant_term_is_absorbed_only_when_fit_constant():
    mu = _model_flux()
    y = _p_true(np.array([0.03]), constant=0.8) * mu
    sigma = 0.05 * mu
    mask = np.ones(WAVE.shape, dtype=bool)

    with_c0 = PolynomialCalibration.from_wavelength(WAVE, order=1,
                                                    fit_constant=True)
    mu_cal, coeffs, _ = with_c0.calibrate(y, mu, sigma, mask)
    assert coeffs.shape == (2,)                     # a_0, a_1
    np.testing.assert_allclose(np.asarray(mu_cal), y, rtol=1e-9)

    without_c0 = PolynomialCalibration.from_wavelength(WAVE, order=1,
                                                       fit_constant=False)
    mu_cal, coeffs, _ = without_c0.calibrate(y, mu, sigma, mask)
    assert coeffs.shape == (1,)                     # a_1 only
    assert np.max(np.abs(np.asarray(mu_cal) / y - 1.0)) > 0.1


def test_gaussian_prior_shrinks_coefficients_and_penalises():
    mu, y, sigma, mask = _mock(seed=2)
    tight = PolynomialCalibration.from_wavelength(WAVE, order=3,
                                                  prior_sigma=1e-6)
    mu_cal, coeffs, ln_prior = tight.calibrate(y, mu, sigma, mask)
    assert np.max(np.abs(np.asarray(coeffs))) < 1e-4
    np.testing.assert_allclose(np.asarray(mu_cal), mu, rtol=1e-4)
    assert np.isfinite(float(ln_prior)) and float(ln_prior) <= 0.0

    loose = PolynomialCalibration.from_wavelength(WAVE, order=3,
                                                  prior_sigma=1.0)
    _, coeffs_loose, ln_prior_loose = loose.calibrate(y, mu, sigma, mask)
    np.testing.assert_allclose(np.asarray(coeffs_loose), TRUE_TILT, atol=5e-3)
    expected = -0.5 * float(jnp.sum(coeffs_loose ** 2))
    np.testing.assert_allclose(float(ln_prior_loose), expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# Inside the diagonal Gaussian likelihood
# ---------------------------------------------------------------------------
def test_profile_likelihood_recovers_the_untilted_log_likelihood():
    mu = _model_flux()
    sigma = mu / 20.0
    mask = np.ones(WAVE.shape, dtype=bool)
    noise = np.random.default_rng(3).normal(0.0, sigma)
    y_plain = mu + noise
    y_tilted = _p_true() * mu + noise

    plain = DiagonalGaussianLikelihood()
    cal = PolynomialCalibration.from_wavelength(WAVE, order=3)
    calibrated = DiagonalGaussianLikelihood(calibration=cal)

    lnl_ref = float(plain(y_plain, mu, sigma, mask)[0])
    lnl_tilt_plain = float(plain(y_tilted, mu, sigma, mask)[0])
    lnl_tilt_cal = float(calibrated(y_tilted, mu, sigma, mask)[0])

    assert lnl_tilt_plain < lnl_ref - 30.0         # the tilt costs ~60 nats
    assert abs(lnl_tilt_cal - lnl_ref) < 6.0       # ~chi^2_3 / 2 of overfit


def test_calibration_is_identity_when_absent():
    mu, y, sigma, mask = _mock(seed=4)
    a = DiagonalGaussianLikelihood()
    b = DiagonalGaussianLikelihood(calibration=None)
    assert float(a(y, mu, sigma, mask)[0]) == float(b(y, mu, sigma, mask)[0])


def test_make_lnprobfn_jit_matches_direct_call_and_is_differentiable():
    mu, y, sigma, mask = _mock(seed=5)
    spec = Spectrum(wavelength=WAVE, flux=y, uncertainty=sigma, mask=mask,
                    name="spec")
    cal = PolynomialCalibration.from_spectrum(spec, order=3)
    lhood = DiagonalGaussianLikelihood(
        noise_model=DiagonalNoiseModel(use_fractional=True), calibration=cal)

    class _Model:
        def predict(self, theta):
            return {"spec": jnp.asarray(mu) * theta["amp"][0]}

    class _Prior:
        def log_prob(self, theta):
            return jnp.zeros(())

    multi = MultiObservationLikelihood(keys=("spec",), likelihoods=(lhood,))
    lnprobfn = multi.make_lnprobfn({"spec": spec}, _Model(), _Prior())
    theta = {"amp": jnp.array([1.1]), "log_f_calib": jnp.array([np.log(0.02)])}
    direct = lhood(jnp.asarray(y), jnp.asarray(mu) * 1.1, jnp.asarray(sigma),
                   jnp.asarray(mask), params=theta)[0]
    np.testing.assert_allclose(float(lnprobfn(theta)), float(direct), rtol=1e-12)
    grad = jax.grad(lnprobfn)(theta)
    assert np.isfinite(float(grad["amp"][0]))


def test_from_spectrum_normalises_the_unmasked_wavelength_range():
    mu, y, sigma, mask = _mock(seed=6)
    mask[:100] = False
    mask[-50:] = False
    spec = Spectrum(wavelength=WAVE, flux=y, uncertainty=sigma, mask=mask,
                    name="spec")
    cal = PolynomialCalibration.from_spectrum(spec, order=2)
    x = np.asarray(cal.x)
    assert x[100] == pytest.approx(-1.0)
    assert x[-51] == pytest.approx(1.0)
    assert x[0] < -1.0 and x[-1] > 1.0
    assert cal.basis.shape == (WAVE.size, 2)      # T_1, T_2
    assert cal.order == 2 and cal.n_coeff == 2


def test_fit_polynomial_calibration_returns_absolute_chebyshev_coefficients():
    """Post-hoc helper contract: coefficients of P = sum c_n T_n (c_0 included)."""
    mu = _model_flux()
    y = _p_true() * mu
    sigma = 0.05 * mu
    mask = np.ones(WAVE.shape, dtype=bool)
    spec = Spectrum(wavelength=WAVE, flux=y, uncertainty=sigma, mask=mask,
                    name="spec")
    coeffs, calibrated = spec.fit_polynomial_calibration(mu, order=3)
    np.testing.assert_allclose(np.asarray(coeffs),
                               np.concatenate([[1.0], TRUE_TILT]), atol=1e-9)
    np.testing.assert_allclose(np.asarray(calibrated), y, rtol=1e-9)


# ---------------------------------------------------------------------------
# Through the alpha-enhanced forward model (the DR2 configuration)
# ---------------------------------------------------------------------------
def _afe_grid():
    """The DR2 production grid, if ``fetch_grid`` has cached it (never downloads)."""
    from ceridwen.ssps.grid_fetch import grid_cache_dir
    path = grid_cache_dir() / "amist_c3k_hr_krou_afe.h5"
    if not path.is_file():
        pytest.skip("cached amist_c3k_hr_krou_afe grid not found")
    return str(path)


def test_recovers_tilt_through_csp_afe_forward_model():
    from ceridwen.csp import CSPBasis_afe
    from ceridwen.ssps import SSPDataAfe

    zred = 0.7
    ssp = SSPDataAfe.load(_afe_grid())
    csp = CSPBasis_afe(
        ssp, lookback_time=jnp.linspace(0.0, 7.0, 5),
        zh_const=True, sfh_interp="step",
        add_dust=False, add_diffuse_dust=True,
        sigma_losvd_kms=0.0, verbose=False,
    )
    spec = Spectrum(wavelength=WAVE, resolution=200.0, smoothtype="vel",
                    name="spec")
    spec.setup_for_model(csp.wave, zred=zred,
                         lib_resolution=getattr(csp, "lib_resolution", None))
    theta = dict(csp.all_params)
    theta["logmass"] = jnp.array([11.0])
    theta["zred"] = jnp.array([zred])
    theta["afe"] = jnp.array([0.2])
    mu = np.asarray(csp.predict(theta, [spec])["spec"])
    assert np.all(mu > 0)

    sigma = mu / 30.0
    noise = np.random.default_rng(7).normal(0.0, sigma)
    y = _p_true() * mu + noise
    mask = np.ones(WAVE.shape, dtype=bool)
    data = Spectrum(wavelength=WAVE, flux=y, uncertainty=sigma, mask=mask,
                    name="spec")
    cal = PolynomialCalibration.from_spectrum(data, order=3)
    mu_cal, coeffs, _ = cal.calibrate(y, mu, sigma, mask)
    p_hat = np.asarray(cal.polynomial(coeffs))
    assert np.max(np.abs(p_hat - _p_true())) < 0.01
    assert np.max(np.abs(np.asarray(mu_cal) / (_p_true() * mu) - 1.0)) < 0.01
