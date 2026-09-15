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
    cal = PolynomialCalibration.from_wavelength(WAVE, order=3, marginalize=False)
    mu_cal, coeffs, ln_prior = cal.calibrate(y, mu, sigma, mask)
    np.testing.assert_allclose(np.asarray(coeffs), TRUE_TILT, atol=1e-9)
    np.testing.assert_allclose(np.asarray(mu_cal), y, rtol=1e-9)
    assert float(ln_prior) == 0.0                   # flat prior, profile: no extra term


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
                                                  prior_sigma=1e-6, marginalize=False)
    mu_cal, coeffs, ln_prior = tight.calibrate(y, mu, sigma, mask)
    assert np.max(np.abs(np.asarray(coeffs))) < 1e-4
    np.testing.assert_allclose(np.asarray(mu_cal), mu, rtol=1e-4)
    assert np.isfinite(float(ln_prior)) and float(ln_prior) <= 0.0

    loose = PolynomialCalibration.from_wavelength(WAVE, order=3,
                                                  prior_sigma=1.0, marginalize=False)
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
    cal = PolynomialCalibration.from_wavelength(WAVE, order=3, marginalize=False)
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


# ---------------------------------------------------------------------------
# Analytic marginalisation, per-coefficient priors, coefficient posterior
# ---------------------------------------------------------------------------
def _gaussian_lnl(y, model, sigma):
    """Diagonal Gaussian log-likelihood with its full normalisation."""
    r = (y - model) / sigma
    return float(-0.5 * np.sum(r ** 2) - np.sum(np.log(sigma))
                 - 0.5 * y.size * np.log(2.0 * np.pi))


def _brute_force_log_marginal(y, mu, sigma, basis, prior_sigma, half_width=0.25, n=801):
    """log integral over the coefficients of likelihood x Gaussian prior (2 coefficients)."""
    from scipy.special import logsumexp
    grid = np.linspace(-half_width, half_width, n)
    step = grid[1] - grid[0]
    a0, a1 = np.meshgrid(grid, grid, indexing="ij")
    coeffs = np.stack([a0.ravel(), a1.ravel()], axis=1)            # (n*n, 2)
    model = mu[None, :] * (1.0 + coeffs @ basis.T)                   # (n*n, n_pix)
    r = (y[None, :] - model) / sigma[None, :]
    lnl = (-0.5 * np.sum(r ** 2, axis=1) - np.sum(np.log(sigma))
           - 0.5 * y.size * np.log(2.0 * np.pi))
    prior = np.asarray(prior_sigma, dtype=float) * np.ones(2)
    ln_prior = (-0.5 * np.sum((coeffs / prior) ** 2, axis=1)
                - np.sum(np.log(prior)) - np.log(2.0 * np.pi))
    return float(logsumexp(lnl + ln_prior) + 2.0 * np.log(step))


def test_marginalised_likelihood_matches_brute_force_integration():
    """ln L returned with marginalize=True is the exact Gaussian integral over
    the coefficients, prior included (order 2 without the constant: a_1, a_2)."""
    wave = WAVE[::10]                                            # 200 pixels
    x = (wave - 7500.0) / 1500.0
    mu = 1e-17 * (1.0 + 0.3 * x)
    sigma = mu / 8.0                                             # low S/N: wide posterior
    rng = np.random.default_rng(11)
    y = mu * (1.0 + 0.03 * x - 0.02 * (2 * x ** 2 - 1)) + rng.normal(0.0, sigma)
    mask = np.ones(wave.shape, dtype=bool)
    cal = PolynomialCalibration.from_wavelength(wave, order=2, prior_sigma=0.05,
                                                marginalize=True)
    lhood = DiagonalGaussianLikelihood(calibration=cal)
    lnl_marg = float(lhood(y, mu, sigma, mask)[0])
    expected = _brute_force_log_marginal(y, mu, sigma, np.asarray(cal.basis), 0.05)
    assert abs(lnl_marg - expected) < 1e-3


def test_order_ten_marginal_matches_closed_form_gaussian_integral():
    """Order 10 (the 2026-09-15 calibration-order arms): the marginal ln L equals
    ln L(a=0) + b^T N^{-1} b / 2 - ln|N| / 2 + ln|S^{-1}| / 2 with b = D^T t and
    N = D^T D + S^{-1}, evaluated independently in numpy."""
    mu, y, sigma, mask = _mock(seed=13)
    prior_sigma = 0.1
    cal = PolynomialCalibration.from_wavelength(WAVE, order=10, prior_sigma=prior_sigma,
                                                marginalize=True)
    assert cal.n_coeff == 10 and np.asarray(cal.basis).shape == (WAVE.size, 10)
    lhood = DiagonalGaussianLikelihood(calibration=cal)
    lnl_marg = float(lhood(y, mu, sigma, mask)[0])
    design = np.asarray(cal.basis) * (mu / sigma)[:, None]
    t = (y - mu) / sigma
    precision = np.eye(10) / prior_sigma ** 2
    normal = design.T @ design + precision
    b = design.T @ t
    expected = (_gaussian_lnl(y, mu, sigma) + 0.5 * b @ np.linalg.solve(normal, b)
                - 0.5 * np.linalg.slogdet(normal)[1] + 0.5 * np.linalg.slogdet(precision)[1])
    assert np.isfinite(lnl_marg)
    assert abs(lnl_marg - expected) < 1e-6 * abs(expected)


def test_marginalised_flat_prior_is_the_lebesgue_integral():
    """Without a prior the marginal is exp(lnL(a_hat)) (2 pi)^{k/2} |D^T D|^{-1/2}."""
    mu, y, sigma, mask = _mock(seed=12)
    cal = PolynomialCalibration.from_wavelength(WAVE, order=3, marginalize=True)
    lhood = DiagonalGaussianLikelihood(calibration=cal)
    lnl_marg = float(lhood(y, mu, sigma, mask)[0])
    coeffs = np.asarray(cal.solve(y, mu, sigma, mask))
    design = np.asarray(cal.basis) * (mu / sigma)[:, None]
    normal = design.T @ design
    lnl_hat = _gaussian_lnl(y, mu * (1.0 + np.asarray(cal.basis) @ coeffs), sigma)
    expected = (lnl_hat + 0.5 * cal.n_coeff * np.log(2.0 * np.pi)
                - 0.5 * np.linalg.slogdet(normal)[1])
    assert abs(lnl_marg - expected) < 1e-6


def test_marginalize_false_reproduces_the_profile_likelihood():
    mu, y, sigma, mask = _mock(seed=13)
    cal = PolynomialCalibration.from_wavelength(WAVE, order=3, prior_sigma=0.1,
                                                marginalize=False)
    lhood = DiagonalGaussianLikelihood(calibration=cal)
    lnl = float(lhood(y, mu, sigma, mask)[0])
    coeffs = np.asarray(cal.solve(y, mu, sigma, mask))
    expected = (_gaussian_lnl(y, mu * (1.0 + np.asarray(cal.basis) @ coeffs), sigma)
                - 0.5 * np.sum(coeffs ** 2) / 0.1 ** 2)
    assert abs(lnl - expected) < 1e-6


def test_marginalisation_is_the_default_and_survives_jit():
    mu, y, sigma, mask = _mock(seed=14)
    cal = PolynomialCalibration.from_wavelength(WAVE, order=3, prior_sigma=0.1)
    assert cal.marginalize is True
    lhood = DiagonalGaussianLikelihood(calibration=cal)
    direct = float(lhood(y, mu, sigma, mask)[0])
    jitted = float(jax.jit(lambda m: lhood(y, m, sigma, mask)[0])(jnp.asarray(mu)))
    assert abs(direct - jitted) < 1e-8
    profile = PolynomialCalibration.from_wavelength(WAVE, order=3, prior_sigma=0.1,
                                                    marginalize=False)
    lnl_profile = float(DiagonalGaussianLikelihood(calibration=profile)(y, mu, sigma, mask)[0])
    # The Occam term is negative: a marginal is never above its profile.
    assert direct < lnl_profile


def test_vector_prior_widths_apply_per_coefficient():
    mu, y, sigma, mask = _mock(seed=15)
    cal = PolynomialCalibration.from_wavelength(
        WAVE, order=3, prior_sigma=[1.0, 1e-7, 1.0], marginalize=False)
    _, coeffs, ln_prior = cal.calibrate(y, mu, sigma, mask)
    coeffs = np.asarray(coeffs)
    assert abs(coeffs[1]) < 1e-6                      # the tight one is pinned
    assert abs(coeffs[0] - TRUE_TILT[0]) < 5e-3       # the loose ones still fit
    expected = -0.5 * np.sum((coeffs / np.array([1.0, 1e-7, 1.0])) ** 2)
    np.testing.assert_allclose(float(ln_prior), expected, rtol=1e-6, atol=1e-9)
    with pytest.raises(ValueError):
        PolynomialCalibration.from_wavelength(WAVE, order=3, prior_sigma=[0.1, 0.1])


def test_solve_uses_the_noise_model_sigma_not_the_raw_uncertainty():
    """Inside the likelihood the polynomial is solved with the effective sigma
    of the noise model (observational plus fractional term), so the profile
    maximises the likelihood that is actually evaluated."""
    mu = _model_flux()
    x = X
    sigma_obs = mu / 200.0 * (1.0 + 4.0 * (x > 0))     # S/N 200 on the blue half, 40 red
    rng = np.random.default_rng(16)
    y = _p_true() * mu + rng.normal(0.0, sigma_obs)
    mask = np.ones(WAVE.shape, dtype=bool)
    params = {"log_f_calib": jnp.array([np.log(0.05)])}
    noise = DiagonalNoiseModel(use_fractional=True)
    cal = PolynomialCalibration.from_wavelength(WAVE, order=3, marginalize=False)
    lhood = DiagonalGaussianLikelihood(noise_model=noise, calibration=cal)
    lnl, aux = lhood(y, mu, sigma_obs, mask, params=params)

    sigma_eff = 1.0 / np.sqrt(np.asarray(noise.compute(sigma_obs, mu, mask, params).inv_var))
    with_eff = np.asarray(cal.solve(y, mu, sigma_eff, mask))
    with_obs = np.asarray(cal.solve(y, mu, sigma_obs, mask))
    assert np.max(np.abs(with_eff - with_obs)) > 1e-4          # the two weightings differ here
    model_eff = mu * (1.0 + np.asarray(cal.basis) @ with_eff)
    np.testing.assert_allclose(float(lnl), _gaussian_lnl(y, model_eff, sigma_eff), rtol=0, atol=1e-6)
    # a_hat maximises that likelihood: every perturbation lowers it
    for delta in rng.normal(0.0, 1e-3, size=(5, 3)):
        model = mu * (1.0 + np.asarray(cal.basis) @ (with_eff + delta))
        assert _gaussian_lnl(y, model, sigma_eff) < float(lnl)


def test_covariance_is_the_inverse_normal_matrix():
    mu, y, sigma, mask = _mock(seed=17)
    mask[300:600] = False
    cal = PolynomialCalibration.from_wavelength(WAVE, order=3, prior_sigma=0.1, mask=mask)
    cov = np.asarray(cal.covariance(mu, sigma, mask))
    design = (np.asarray(cal.basis) * (mu / sigma)[:, None])[mask]
    expected = np.linalg.inv(design.T @ design + np.eye(3) / 0.1 ** 2)
    np.testing.assert_allclose(cov, expected, rtol=1e-8, atol=0)


def test_posterior_draws_have_the_conditional_mean_and_covariance():
    mu, y, sigma, mask = _mock(seed=18, snr=5.0)
    cal = PolynomialCalibration.from_wavelength(WAVE, order=2, prior_sigma=0.1)
    n_theta, n_per = 3, 4000
    mu_draws = np.stack([mu * s for s in (0.98, 1.0, 1.02)])
    sigma_draws = np.stack([sigma] * n_theta)
    coeffs, poly = cal.posterior_draws(y, mu_draws, sigma_draws, mask,
                                       jax.random.PRNGKey(0), draws_per_sample=n_per)
    assert coeffs.shape == (n_theta * n_per, 2) and poly.shape == (n_theta * n_per, WAVE.size)
    block = np.asarray(coeffs)[:n_per]
    a_hat = np.asarray(cal.solve(y, mu_draws[0], sigma, mask))
    cov = np.asarray(cal.covariance(mu_draws[0], sigma, mask))
    np.testing.assert_allclose(block.mean(axis=0), a_hat, atol=4.0 * np.sqrt(np.diag(cov) / n_per).max())
    # Chebyshev columns are nearly orthogonal here, so the off-diagonal terms
    # are ~0 and only an absolute tolerance (5% of the variances) is meaningful.
    np.testing.assert_allclose(np.cov(block.T), cov, rtol=0.15,
                               atol=0.05 * np.max(np.diag(cov)))
    np.testing.assert_allclose(np.asarray(poly)[0], 1.0 + np.asarray(cal.basis) @ block[0], rtol=1e-12)


def test_repr_states_marginalisation_and_prior():
    cal = PolynomialCalibration.from_wavelength(WAVE, order=3, prior_sigma=[0.1, 0.1, 0.1])
    text = repr(cal)
    assert "order=3" in text and "marginalize=True" in text and "0.1" in text


@pytest.mark.parametrize('order', [1, 3, 7])
def test_calibration_reduction_preserves_gram_matrix_and_gradient(order):
    from ceridwen.likelihood import PolynomialCalibration
    wave = np.linspace(6000., 9000., 321)
    mask = np.arange(len(wave)) % 5 != 0
    calibration = PolynomialCalibration.from_wavelength(
        wave, order=order, mask=mask, prior_sigma=.1)
    sigma = jnp.geomspace(.001, 1., len(wave))
    def reference(flux):
        design = calibration.design(flux, sigma, mask)
        return design.T @ design + calibration._precision()
    def candidate(flux):
        return calibration.normal_matrix(flux, sigma, mask)
    flux = jnp.stack([jnp.linspace(.1, scale, len(wave)) for scale in (.5, 1., 2.)])
    for function in (lambda f: f, lambda f: jax.grad(lambda x:jnp.linalg.slogdet(f(x))[1])):
        a = jax.jit(jax.vmap(function(reference)))(flux)
        b = jax.jit(jax.vmap(function(candidate)))(flux)
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
