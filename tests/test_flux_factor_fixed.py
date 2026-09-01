"""The fixed-redshift flux factor is hoisted out of the compiled hot path.

For a fixed ``zred`` the luminosity-distance quadrature inside
``flux_factor_maggies`` is a per-call constant computation (~10 kernels).
``SedModel`` computes it once and injects it as ``theta["flux_factor"]``;
the CSP uses the injected constant and must produce predictions identical
to the per-call quadrature. A sampled (traced) ``zred`` never sees the
constant and keeps the differentiable quadrature.
"""

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from ceridwen.cosmology import DEFAULT_COSMO, flux_factor_maggies
from ceridwen.csp.csp_afe import CSPBasis_afe
from ceridwen.model.model import SedModel

LOOKBACK_GYR = jnp.array([0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 5.0, 8.4])
ZRED = 0.6542


@pytest.fixture(scope="module")
def csp():
    rng = np.random.default_rng(20260901)
    ssp = SimpleNamespace(
        ssp_flux=rng.uniform(0.1, 2.0, size=(5, 13, 107, 11)).astype(np.float32),
        ssp_afe=np.linspace(-0.2, 0.6, 5, dtype=np.float32),
        ssp_wave=np.linspace(1000.0, 10000.0, 11, dtype=np.float32),
        ssp_lg_age_gyr=np.linspace(-3.0, 1.14, 107, dtype=np.float32),
        ssp_lgmet=np.linspace(-2.0, 0.4, 13, dtype=np.float32),
        ssp_resolution=None,
        isoc_type=None,
        spec_library=None,
    )
    return CSPBasis_afe(
        ssp,
        theta={
            "lookback_time": LOOKBACK_GYR,
            "sfh": jnp.ones(8),
            "Z": jnp.array([-0.8]),
            "afe": jnp.array([0.2]),
        },
        zh_const=True,
        sfh_interp="step",
        add_dust=False,
        add_diffuse_dust=False,
        add_dust_emission=False,
        sigma_losvd_kms=0.0,
        verbose=False,
    )


def _scale(csp, theta):
    ones = jnp.ones(csp.wave.shape[0], dtype=jnp.float32)
    phot, slit, line = csp._apply_mass_redshift_igm(ones, ones, ones, theta)
    return phot, slit, line


def test_injected_constant_matches_quadrature_exactly(csp):
    theta = {"zred": jnp.array([ZRED]), "logmass": jnp.array([10.0])}
    expected = _scale(csp, theta)
    injected = dict(theta)
    injected["flux_factor"] = jnp.float32(flux_factor_maggies(ZRED, csp.cosmo))
    actual = _scale(csp, injected)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_traced_zred_keeps_the_differentiable_quadrature(csp):
    def factor(z):
        theta = {"zred": jnp.array([z])}
        _, slit, _ = _scale(csp, theta)
        return slit[0].astype(jnp.float64)

    grad = jax.grad(factor)(jnp.float64(ZRED))
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0


class _StubCSP:
    def __init__(self):
        self.cosmo = DEFAULT_COSMO
        self.theta_init = {"logmass": jnp.array([10.0])}
        self.param_names = ["logmass"]
        self.wave = jnp.linspace(1000.0, 10000.0, 16)


def test_sedmodel_stores_the_native_fixed_flux_factor():
    model = SedModel(_StubCSP(), [], priors={}, zred=ZRED)
    assert model._flux_factor_fixed is not None
    expected = jnp.float32(flux_factor_maggies(ZRED, model.cosmo))
    assert float(model._flux_factor_fixed) == float(expected)
    assert model._flux_factor_fixed.dtype == jnp.float32


def test_sedmodel_free_zred_has_no_constant():
    model = SedModel(_StubCSP(), [], priors={}, zred=0.0)
    assert model._flux_factor_fixed is None
