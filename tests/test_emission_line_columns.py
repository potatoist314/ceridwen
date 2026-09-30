"""Emission lines as free-flux columns of the calibration solve.

``PolynomialCalibration.calibrate_with_lines`` integrates out the Chebyshev
coefficients and the line fluxes (flat prior on f >= 0) in one integral.
Pure-kernel tests; the line-list test needs ``$SPS_HOME`` and skips without it.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebvander
from scipy.integrate import quad
from scipy.stats import multivariate_normal

from dataclasses import replace

from ceridwen.likelihood import (
    DiagonalGaussianLikelihood,
    DiagonalNoiseModel,
    EmissionLineColumns,
    MultiObservationLikelihood,
    PolynomialCalibration,
)
from ceridwen.likelihood.calibration import _bvn_upper, log_positive_probability
from ceridwen.observation import Spectrum

jax.config.update("jax_enable_x64", True)

WAVE = np.linspace(6000.0, 9000.0, 2000)
X = (WAVE - 7500.0) / 1500.0
ZRED = 0.7
REST = np.array([3727.118, 4102.9514, 4862.7629])  # [O II], H-delta, H-beta (vacuum)
SIGMA_INST = 40.0                                  # km/s
SIGMA_GAS = 200.0                                  # km/s


def _lines(rest=REST, free=True):
    return EmissionLineColumns(
        wave_obs=WAVE, sigma_inst_kms=np.full(WAVE.shape, SIGMA_INST), wave_rest=rest,
        names=tuple(f"{w:.0f}" for w in rest), zred=ZRED, sigma_gas_kms=SIGMA_GAS,
        zred_key="zred" if free else None, sigma_key="sigma_smooth" if free else None)


def _continuum():
    return 1e-29 * (1.0 + 0.3 * X) * (1.0 - 0.3 * np.exp(-((WAVE - 7000.0) / 8.0) ** 2))


def _mock(fluxes, seed=0, snr=30.0):
    mu = _continuum()
    sigma = mu / snr
    L = np.asarray(_lines().columns({"zred": ZRED, "sigma_smooth": SIGMA_GAS}))
    p_true = 1.0 + chebvander(X, 2)[:, 1:] @ np.array([0.02, -0.01])
    y = p_true * mu + L @ fluxes + np.random.default_rng(seed).normal(0.0, sigma)
    return mu, y, sigma, np.ones(WAVE.shape, dtype=bool), L


def test_line_columns_have_unit_flux():
    L = np.asarray(_lines().columns({"zred": ZRED, "sigma_smooth": SIGMA_GAS}))
    f_lambda = L * 2.99792458e18 / WAVE[:, None] ** 2
    np.testing.assert_allclose(np.trapezoid(f_lambda, WAVE, axis=0), 1.0, rtol=1e-6)
    centre = WAVE[np.argmax(L, axis=0)]
    np.testing.assert_allclose(centre, REST * (1 + ZRED), atol=WAVE[1] - WAVE[0])


@pytest.mark.parametrize("rest", [REST, np.append(REST, 3540.0)])  # 3540 A: window clipped at the blue edge
def test_windowed_columns_equal_the_full_evaluation_bitwise(rest):
    full = replace(_lines(rest), zred_key=None)
    windowed = replace(full, sigma_max_kms=350.0)
    fixed = replace(windowed, sigma_key=None)
    assert windowed._window().shape[0] < WAVE.size
    sigmas = {"sigma_smooth": jnp.array([[5.0], [SIGMA_GAS], [350.0]])}
    np.testing.assert_array_equal(np.asarray(jax.jit(jax.vmap(windowed.columns))(sigmas)),
                                  np.asarray(jax.jit(jax.vmap(full.columns))(sigmas)))
    np.testing.assert_array_equal(np.asarray(jax.jit(fixed.columns)({})),
                                  np.asarray(jax.jit(full.columns)({"sigma_smooth": jnp.array([SIGMA_GAS])})))


@pytest.mark.parametrize("sigma_max_kms", [None, 350.0])
def test_tied_columns_equal_the_tie_product_bitwise(sigma_max_kms):
    tie = np.array([[1.0, 0.0], [0.0, 1.0 / 3.010], [0.0, 1.0]])   # [O III] 4960 tied to 5008
    raw = replace(_lines(np.array([3727.118, 4960.295, 5008.240])), zred_key=None, sigma_max_kms=sigma_max_kms)
    tied = replace(raw, tie=tie)
    sigmas = {"sigma_smooth": jnp.linspace(5.0, 350.0, 64)[:, None]}
    np.testing.assert_array_equal(np.asarray(jax.jit(jax.vmap(tied.columns))(sigmas)),
                                  np.asarray(jax.jit(jax.vmap(lambda t: raw.columns(t) @ jnp.asarray(tie)))(sigmas)))


@pytest.mark.parametrize("prior_sigma", [None, [0.3, 0.1, 0.1, 0.1]])
def test_zero_line_columns_equal_polynomial_marginalisation(prior_sigma):
    mu, y, sigma, mask, _ = _mock(np.zeros(3))
    cal = PolynomialCalibration.from_wavelength(WAVE, 3, fit_constant=True,
                                                prior_sigma=prior_sigma)
    mu_cal, coeffs, ln_extra = cal.calibrate(y, mu, sigma, mask)
    mu_j, coeffs_j, fluxes, ln_extra_j = cal.calibrate_with_lines(
        y, mu, sigma, mask, jnp.zeros((WAVE.size, 0)))
    assert fluxes.shape == (0,)
    np.testing.assert_allclose(coeffs_j, coeffs, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(mu_j, mu_cal, rtol=1e-12)
    np.testing.assert_allclose(ln_extra_j, ln_extra, rtol=0, atol=1e-8)


def test_line_posterior_equals_the_full_triangular_solve_bitwise():
    """The line block of N^{-1} from the line block of the factor alone."""
    cal = PolynomialCalibration.from_wavelength(WAVE, 3, fit_constant=True,
                                                prior_sigma=[0.3, 0.1, 0.1, 0.1])
    k = cal.n_coeff

    def full(factor, scale, solution):
        g = jax.scipy.linalg.solve_triangular(factor, jnp.eye(solution.shape[0])[:, k:], lower=True)
        return solution[k:], scale[k:, None] * (g.T @ g) * scale[None, k:]

    def both(seed):
        mu, y, sigma, mask, L = _mock(np.array([4e-17, 1e-17, 2e-17]), seed=seed)
        return (y, mu, sigma, mask, L)

    batch = [jnp.stack(v) for v in zip(*(both(seed) for seed in range(8)))]
    factors = jax.jit(jax.vmap(lambda *a: cal._joint_factor(*a)))(*batch)
    got = jax.jit(jax.vmap(cal._line_posterior))(*factors)
    want = jax.jit(jax.vmap(full))(*factors)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


@pytest.mark.parametrize("flux", [3e-17, 2e-18, -1e-17])
def test_joint_marginal_matches_quadrature_over_one_line_flux(flux):
    """ln of the integral over f >= 0 of the polynomial marginal of y - f L."""
    mu, y, sigma, mask, L = _mock(np.array([0.0, flux, 0.0]), snr=15.0)
    line = L[:, 1:2]
    cal = PolynomialCalibration.from_wavelength(WAVE, 2, fit_constant=True,
                                                prior_sigma=[0.3, 0.1, 0.1])
    lhood = DiagonalGaussianLikelihood(calibration=cal)
    joint = DiagonalGaussianLikelihood(calibration=cal, emission_lines=_lines(REST[1:2], False))

    def polynomial_marginal(f):
        return float(lhood(y - f * line[:, 0], mu, sigma, mask)[0])

    _, _, f_hat, _ = cal.calibrate_with_lines(y, mu, sigma, mask, jnp.asarray(line))
    f_hat = float(f_hat[0])
    peak = polynomial_marginal(max(f_hat, 0.0))
    width = 1.0 / np.sqrt(float(line[:, 0] @ (line[:, 0] / sigma ** 2)))
    lo, hi = max(0.0, f_hat - 12 * width), max(0.0, f_hat) + 12 * width
    integral, _ = quad(lambda f: np.exp(polynomial_marginal(f) - peak),
                       lo, hi, epsabs=0, epsrel=1e-11, limit=200)
    np.testing.assert_allclose(float(joint(y, mu, sigma, mask)[0]), peak + np.log(integral),
                               rtol=0, atol=1e-6)


def test_injected_lines_are_recovered():
    truth = np.array([4e-17, 1e-17, 2e-17])
    mu, y, sigma, mask, L = _mock(truth, seed=3)
    cal = PolynomialCalibration.from_wavelength(WAVE, 3, fit_constant=True,
                                                prior_sigma=[0.3, 0.1, 0.1, 0.1])
    _, coeffs, fluxes, _ = cal.calibrate_with_lines(y, mu, sigma, mask, jnp.asarray(L))
    normal, _ = cal._joint_system(y, mu, sigma, mask, jnp.asarray(L))
    sd = np.sqrt(np.diag(np.linalg.inv(np.asarray(normal))))[cal.n_coeff:]
    assert np.all(np.abs(np.asarray(fluxes) - truth) < 3 * sd)
    assert np.all(truth[[0, 2]] / sd[[0, 2]] > 10)
    # draws of lines far from zero are the untruncated Gaussian
    draws = np.asarray(cal.posterior_draws_with_lines(
        y, np.tile(mu, (2000, 1)), np.tile(sigma, (2000, 1)), jnp.tile(jnp.asarray(L), (2000, 1, 1)),
        mask, jax.random.PRNGKey(0), sweeps=20)[1])
    assert np.all(draws >= 0)
    np.testing.assert_allclose(np.std(draws, axis=0)[[0, 2]], sd[[0, 2]], rtol=0.06)
    np.testing.assert_allclose(np.mean(draws, axis=0)[[0, 2]], np.asarray(fluxes)[[0, 2]], atol=0.1 * sd.max())


@pytest.mark.parametrize("extra_depth", [0.0, 1.0])
def test_pure_stellar_absorption_gives_line_fluxes_near_zero(extra_depth):
    """Absorption at the line centres, in the model (0) or deeper in the data than
    in the model (twice as deep, template mismatch): the fluxes stay >= 0 and near zero."""
    mu = _continuum()
    L = np.asarray(_lines().columns({"zred": ZRED, "sigma_smooth": SIGMA_GAS}))
    absorption = 1.0 - 3.0 * (L / L.max(axis=0)).sum(axis=1) * 0.1
    sigma = mu / 30.0
    y = mu * absorption * (1.0 - extra_depth * (1.0 - absorption)) \
        + np.random.default_rng(5).normal(0.0, sigma)
    mu = mu * absorption
    mask = np.ones(WAVE.shape, dtype=bool)
    cal = PolynomialCalibration.from_wavelength(WAVE, 3, fit_constant=True,
                                                prior_sigma=[0.3, 0.1, 0.1, 0.1])
    _, _, f_hat, ln_extra = cal.calibrate_with_lines(y, mu, sigma, mask, jnp.asarray(L))
    factor, scale, solution = cal._joint_factor(y, mu, sigma, mask, jnp.asarray(L))
    sd = np.sqrt(np.diag(np.asarray(cal._line_posterior(factor, scale, solution)[1])))
    draws = np.asarray(cal.posterior_draws_with_lines(
        y, np.tile(mu, (500, 1)), np.tile(sigma, (500, 1)), jnp.tile(jnp.asarray(L), (500, 1, 1)),
        mask, jax.random.PRNGKey(1))[1])
    assert np.all(draws >= 0)
    assert np.all(np.mean(draws, axis=0) < 1.5 * sd)
    if extra_depth:
        assert np.all(np.asarray(f_hat) / sd < -5)       # unconstrained fluxes would be negative


def test_bivariate_orthant_matches_scipy():
    rng = np.random.default_rng(1)
    for r in [-0.996, -0.93, -0.5, 0.0, 0.3, 0.95]:
        for h, k in rng.normal(0.0, 3.0, (20, 2)):
            exact = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-h, -k])
            assert abs(float(_bvn_upper(h, k, r)) - exact) < 1e-12


def test_positive_probability_of_a_blended_pair_is_exact():
    mean = np.array([0.0, 0.0, 1e-17])
    sd = np.array([3e-17, 3e-17, 1e-17])
    corr = np.array([[1.0, -0.99, 0.0], [-0.99, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cov = corr * np.outer(sd, sd)
    exact = np.log(multivariate_normal(mean=-mean, cov=cov, abseps=1e-14, releps=1e-10,
                                       maxpts=10**7).cdf(np.zeros(3)))
    got = float(log_positive_probability(jnp.asarray(mean), jnp.asarray(cov), ((0, 1),)))
    assert abs(got - exact) < 1e-5
    product = float(log_positive_probability(jnp.asarray(mean), jnp.asarray(cov)))
    assert abs(product - exact) > 1.0


def test_likelihood_prefers_the_redshift_of_the_injected_lines():
    mu, y, sigma, mask, _ = _mock(np.array([4e-17, 0.0, 2e-17]), seed=4)
    cal = PolynomialCalibration.from_wavelength(WAVE, 3, fit_constant=True,
                                                prior_sigma=[0.3, 0.1, 0.1, 0.1])
    lhood = DiagonalGaussianLikelihood(noise_model=DiagonalNoiseModel(use_fractional=True),
                                       calibration=cal, emission_lines=_lines())
    zs = ZRED + np.linspace(-2e-3, 2e-3, 41)
    lnl = [float(lhood(y, mu, sigma, mask, {"zred": jnp.array([z]),
                                            "sigma_smooth": jnp.array([SIGMA_GAS]),
                                            "log_f_calib": jnp.array([np.log(0.01)])})[0])
           for z in zs]
    assert abs(zs[int(np.argmax(lnl))] - ZRED) <= 1e-4


def test_likelihood_without_lines_is_the_polynomial_likelihood():
    mu, y, sigma, mask, _ = _mock(np.zeros(3))
    cal = PolynomialCalibration.from_wavelength(WAVE, 3, fit_constant=True, prior_sigma=0.1)
    plain = DiagonalGaussianLikelihood(calibration=cal)
    assert DiagonalGaussianLikelihood(calibration=cal, emission_lines=None) == plain
    assert repr(plain) == f"DiagonalGaussianLikelihood(noise_model={plain.noise_model!r}, calibration={cal!r})"


@pytest.mark.skipif(not os.environ.get("SPS_HOME"), reason="needs $SPS_HOME for emlines_info.dat")
def test_from_spectrum_selects_covered_lines_with_enough_unmasked_pixels():
    spec = Spectrum(wavelength=WAVE, flux=_continuum(), uncertainty=_continuum() / 30,
                    resolution=3500.0, smoothtype="R", res_convention="fwhm",
                    sigma_losvd=SIGMA_GAS, fit_sigma_smooth=True, free_z=True, name="spectrum")
    spec.mask_wavelength_range(4862.7629 * (1 + ZRED) - 30, 4862.7629 * (1 + ZRED) + 30)
    lines = EmissionLineColumns.from_spectrum(spec, ZRED)
    assert "[O II] 3726" in lines.names and "[O III] 5007" in lines.names
    assert "Ba-beta 4861" not in lines.names          # masked: fewer than 3 pixels
    assert lines.zred_key == "zred" and lines.sigma_key == "sigma_smooth"
    assert np.all(lines.wave_rest * (1 + ZRED) > WAVE[0])
    assert np.all(lines.wave_rest * (1 + ZRED) < WAVE[-1])
    assert lines.covers(5006.8) and not lines.covers(4861.3)
    blended = {frozenset((lines.names[i], lines.names[j])) for i, j in lines.pairs}
    assert frozenset(("He I 3888.63A", "Ba-6 3889")) in blended
    assert len({i for pair in lines.pairs for i in pair}) == 2 * len(lines.pairs)


def test_photometry_shares_the_line_flux_exactly():
    """One line seen by the spectrum and by two bands: ln of the integral over
    f >= 0 of (polynomial marginal of y - f L) x (band Gaussian of y_p - mu_p - f B)."""
    mu, y, sigma, mask, L = _mock(np.array([0.0, 2e-17, 0.0]), snr=15.0)
    line = L[:, 1]
    band = np.array([[4e10], [1e10]])                      # maggies per unit line flux
    mu_p = np.array([2e-7, 3e-7])
    sigma_p = 0.05 * mu_p
    y_p = mu_p + band[:, 0] * 2e-17 + np.array([1e-9, -2e-9])
    mask_p = np.ones(2, dtype=bool)
    cal = PolynomialCalibration.from_wavelength(WAVE, 2, fit_constant=True, prior_sigma=[0.3, 0.1, 0.1])
    lines = replace(_lines(REST[1:2], False), band_matrix=band, photometry_key="photometry")
    joint = MultiObservationLikelihood(
        keys=("photometry", "spectrum"),
        likelihoods=(DiagonalGaussianLikelihood(),
                     DiagonalGaussianLikelihood(calibration=cal, emission_lines=lines)))
    data = {"spectrum": (y, sigma, mask), "photometry": (y_p, sigma_p, mask_p)}
    value = float(joint.loglike(data, {"spectrum": mu, "photometry": mu_p}, {}))

    plain = DiagonalGaussianLikelihood(calibration=cal)

    def integrand_log(f):
        spec = float(plain(y - f * line, mu, sigma, mask)[0])
        r = (y_p - mu_p - f * band[:, 0]) / sigma_p
        return spec - 0.5 * np.sum(r * r) - np.sum(np.log(np.sqrt(2 * np.pi) * sigma_p))

    _, _, f_hat, _ = cal.calibrate_with_lines(
        y, mu, sigma, mask, jnp.asarray(line[:, None]), photometry=(band, y_p, mu_p, sigma_p, mask_p))
    f_hat = float(f_hat[0])
    width = 1.0 / np.sqrt(float(line @ (line / sigma ** 2)) + float(band[:, 0] @ (band[:, 0] / sigma_p ** 2)))
    peak = integrand_log(max(f_hat, 0.0))
    integral, _ = quad(lambda f: np.exp(integrand_log(f) - peak),
                       max(0.0, f_hat - 12 * width), max(0.0, f_hat) + 12 * width,
                       epsabs=0, epsrel=1e-11, limit=200)
    np.testing.assert_allclose(value, peak + np.log(integral), rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="MultiObservationLikelihood.loglike"):
        joint.likelihoods[1](y, mu, sigma, mask, {})


def test_loglike_without_shared_lines_is_the_per_observation_sum():
    mu, y, sigma, mask, _ = _mock(np.zeros(3))
    cal = PolynomialCalibration.from_wavelength(WAVE, 3, fit_constant=True, prior_sigma=0.1)
    multi = MultiObservationLikelihood(
        keys=("a", "b"),
        likelihoods=(DiagonalGaussianLikelihood(), DiagonalGaussianLikelihood(calibration=cal)))
    data = {"a": (y, sigma, mask), "b": (y, sigma, mask)}
    expected = (jnp.zeros(()) + multi.likelihoods[0](y, mu, sigma, mask, params={})[0]
                + multi.likelihoods[1](y, mu, sigma, mask, params={})[0])
    assert float(multi.loglike(data, {"a": mu, "b": mu}, {})) == float(expected)


@pytest.mark.skipif(not os.environ.get("SPS_HOME"), reason="needs $SPS_HOME for emlines_info.dat")
def test_atomic_doublets_share_one_flux_at_the_fsps_ratio():
    spec = Spectrum(wavelength=WAVE, flux=_continuum(), uncertainty=_continuum() / 30,
                    resolution=3500.0, smoothtype="R", res_convention="fwhm",
                    sigma_losvd=SIGMA_GAS, name="spectrum")
    lines = EmissionLineColumns.from_spectrum(spec, ZRED)
    names = list(lines.names)
    assert "[O III] 5007 (+[O III] 4959)" in lines.free_names
    assert "[O III] 4959" not in lines.free_names
    assert lines.n_line == len(names) - 2                  # [O III] and [Ne III] doublets tied
    raw = np.asarray(replace(lines, tie=None).columns())
    tied = np.asarray(lines.columns())
    column = lines.free_names.index("[O III] 5007 (+[O III] 4959)")
    np.testing.assert_allclose(tied[:, column],
                               raw[:, names.index("[O III] 5007")] + raw[:, names.index("[O III] 4959")] / 3.010,
                               rtol=1e-12)
    assert lines.zred_key is None
    with pytest.raises(ValueError, match="fixed redshift"):
        replace(lines, zred_key="zred").with_photometry(None, "photometry")
