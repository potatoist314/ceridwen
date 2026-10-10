"""The fixed-distance flux factor is hoisted out of the compiled hot path.

For a fixed ``zred`` the luminosity-distance quadrature inside
``flux_factor_cgs`` is a per-call constant computation.
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

from ceridwen import Cosmology
from ceridwen.cosmology import flux_factor_cgs
from ceridwen.csp.csp_afe import CSPBasis_afe
from ceridwen.model.model import SedModel

pytestmark = pytest.mark.filterwarnings("ignore:no prior for sampled parameter")

LOOKBACK_GYR = jnp.array([0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 5.0, 6.9])
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
        log10_zsun=0.0,
        ssp_resolution=None,
        isoc_type=None,
        spec_library=None,
    )
    return CSPBasis_afe(
        ssp,
        theta={
            "lookback_time": LOOKBACK_GYR,
            "sfh": jnp.ones(8),
            "logzsol": jnp.array([-0.8]),
            "afe": jnp.array([0.2]),
        },
        zh_const=True,
        sfh_interp="step",
        add_dust=False,
        add_diffuse_dust=False,
        add_dust_emission=False,
        cosmo=Cosmology.planck18(),
        verbose=False,
    )


def _scale(csp, theta):
    ones = jnp.ones(csp.wave.shape[0], dtype=jnp.float32)
    return csp._apply_mass_redshift_igm(ones, ones, theta)


def test_injected_constant_matches_quadrature_exactly(csp):
    theta = {"zred": jnp.array([ZRED]), "logmass": jnp.array([10.0])}
    model = SedModel(csp, [], zred=ZRED)
    injected = {**theta, "flux_factor": model._flux_factor_fixed}
    expected = jax.jit(lambda t: _scale(csp, t))(theta)
    actual = jax.jit(lambda t: _scale(csp, t))(injected)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_traced_zred_keeps_the_differentiable_quadrature(csp):
    def factor(z):
        _, slit = _scale(csp, {"zred": jnp.array([z])})
        return slit[0].astype(jnp.float64)

    grad = jax.grad(factor)(jnp.float64(ZRED))
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0


def test_sedmodel_stores_the_native_fixed_flux_factor(csp):
    model = SedModel(csp, [], zred=ZRED)
    expected = jax.jit(lambda z: flux_factor_cgs(z, csp.cosmo))(ZRED)
    assert float(model._flux_factor_fixed[0]) == float(expected)


def test_sedmodel_injects_the_constant_only_for_the_fixed_distance(csp):
    model = SedModel(csp, [], zred=ZRED)
    fixed = model._with_fixed_distance({"logmass": jnp.array([10.0])})
    assert fixed["flux_factor"] is model._flux_factor_fixed
    sampled = model._with_fixed_distance({"zred": jnp.array([0.7])})
    assert "flux_factor" not in sampled


def test_sedmodel_zero_redshift_has_no_constant(csp):
    model = SedModel(csp, [], zred=0.0)
    assert model._flux_factor_fixed is None
