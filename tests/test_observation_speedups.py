"""Numerical contracts for linear spectral interpolation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ceridwen.observation import Spectrum


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_unsmoothed_gather_matches_dense_at_boundaries_and_in_batch(dtype):
    wave = np.array([1000., 1500., 2200., 3400., 5500.])
    out = np.array([900., 1200., 1500., 2150., 2640., 6600., 7000.])
    obs = Spectrum(wavelength=out, name="test")
    obs.setup_for_model(wave, zred=.2)
    flux = jnp.asarray([[.2, .8, 1.2, .7, 1.3], [1, 1, 1, 1, 1]], dtype=dtype)
    actual = jax.jit(jax.vmap(lambda s: obs.predict(s, wave)))(flux)
    assert obs._H_cached is None
    expected = jax.vmap(lambda s: obs._H @ s)(flux)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=0)
    np.testing.assert_allclose(actual[1], 1, rtol=1e-7)
    np.testing.assert_allclose(actual[:, [0, -1]], flux[:, [0, -1]], rtol=1e-7)
    a = jax.grad(lambda s: jnp.sum(obs.predict(s, wave)))(flux[0])
    b = jax.grad(lambda s: jnp.sum(obs._H @ s))(flux[0])
    np.testing.assert_allclose(a, b, rtol=1e-7)


def test_large_interpolation_retains_float32_accuracy():
    """CUDA's default TF32 matrix products must not lower interpolation accuracy."""
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(20260905)
    wave = np.geomspace(1000, 25000, 10992)
    obs = Spectrum(wavelength=np.linspace(6000, 8600, 6166))
    obs.setup_for_model(wave, zred=.6542)
    flux = rng.uniform(1e-6, 1e-5, wave.size).astype(np.float32)
    lo, hi, alpha, n_pix, n_wave = obs._H_factors
    weights = np.column_stack([(1 - alpha).astype(np.float32), alpha.astype(np.float32)])
    operator = csr_matrix((weights.astype(np.float64).ravel(),
                           np.column_stack([lo, hi]).ravel(),
                           np.arange(0, 2 * n_pix + 1, 2)), shape=(n_pix, n_wave))
    expected = operator @ flux.astype(np.float64)
    actual = jax.jit(lambda s: obs.predict(s, wave))(jnp.asarray(flux))
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=0)
    assert obs._H_cached is None
