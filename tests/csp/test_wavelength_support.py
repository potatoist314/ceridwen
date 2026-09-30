"""The CSP synthesises only the model wavelengths its observations read.

``predict`` computes the spectrum on the union of the observations'
``model_support`` and zero-fills the rest; the predictions must equal those
of the whole grid bit for bit.  The SFH basis table keeps each nonzero
``(group, bin)`` spectrum once and one shared zero row.
"""

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from ceridwen.csp.csp_afe import CSPBasis_afe
from ceridwen.observation import Photometry, Spectrum

LOOKBACK_GYR = jnp.array([0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 5.0, 8.4])
WAVE = np.geomspace(900.0, 60000.0, 600).astype(np.float32)
ZRED = 0.6


@pytest.fixture(scope="module")
def ssp():
    rng = np.random.default_rng(20261001)
    return SimpleNamespace(
        ssp_flux=rng.uniform(0.1, 2.0, size=(5, 13, 107, WAVE.size)).astype(np.float32),
        ssp_afe=np.linspace(-0.2, 0.6, 5, dtype=np.float32),
        ssp_wave=WAVE,
        ssp_lg_age_gyr=np.linspace(-3.0, 1.14, 107, dtype=np.float32),
        ssp_lgmet=np.linspace(-2.0, 0.4, 13, dtype=np.float32),
        ssp_resolution=None,
        isoc_type=None,
        spec_library=None,
    )


def build(ssp, zh_const):
    metallicity = {"Z": jnp.array([-0.8])} if zh_const else {"zh": jnp.full(7, -0.8)}
    csp = CSPBasis_afe(
        ssp,
        theta={"lookback_time": LOOKBACK_GYR, "sfh": jnp.ones(8),
               "afe": jnp.array([0.2]), **metallicity},
        zh_const=zh_const,
        sfh_interp="step",
        add_dust=True,
        add_diffuse_dust=True,
        add_dust_emission=False,
        sigma_losvd_kms=0.0,
        verbose=False,
    )
    phot = Photometry(filters=["hsc_g", "hsc_i"], name="photometry")
    phot.setup_for_model(csp.wave, zred=ZRED)
    wo = np.arange(6300.0, 8800.0, 2.0)
    spec = Spectrum(wavelength=wo, flux=np.ones_like(wo), uncertainty=np.ones_like(wo), name="spectrum")
    spec.setup_for_model(csp.wave, zred=ZRED)
    return csp, [phot, spec]


def thetas(csp, zh_const, n):
    rng = np.random.default_rng(7)
    base = {k: jnp.asarray(v) for k, v in csp.all_params.items()}
    base.update({k: jnp.full_like(jnp.asarray(v, dtype=float), 0.7)
                 for k, v in csp.dust_attn.get_default_fit_params().items()})
    base.update(logmass=jnp.array([10.5]), zred=jnp.array([ZRED]))
    draws = []
    for _ in range(n):
        theta = dict(base, sfh=jnp.asarray(rng.uniform(0.1, 2.0, 8)),
                     afe=jnp.asarray(rng.uniform(-0.2, 0.6, 1)))
        if zh_const:
            theta["Z"] = jnp.asarray(rng.uniform(-1.9, 0.3, 1))
        else:
            theta["zh"] = jnp.asarray(rng.uniform(-1.9, 0.3, 7))
        draws.append(theta)
    return draws


@pytest.mark.parametrize("zh_const", [True, False], ids=["constant-Z", "per-bin-Z"])
def test_support_prediction_is_bitwise_the_full_grid(ssp, zh_const):
    csp, observations = build(ssp, zh_const)
    draws = thetas(csp, zh_const, 3)
    support = csp._observation_support(observations, draws[0])
    assert 0 < support.start < support.stop < WAVE.size
    trimmed_one = jax.jit(lambda t: csp.predict(t, observations))
    trimmed_many = jax.jit(jax.vmap(lambda t: csp.predict(t, observations)))
    batch = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *draws)
    got_one = [trimmed_one(t) for t in draws]
    got_many = trimmed_many(batch)

    csp._observation_support = lambda observations, theta: slice(None)
    full_one = jax.jit(lambda t: csp.predict(t, observations))
    full_many = jax.jit(jax.vmap(lambda t: csp.predict(t, observations)))
    want_many = full_many(batch)
    for got, t in zip(got_one, draws):
        want = full_one(t)
        for name in want:
            assert np.asarray(got[name]).tobytes() == np.asarray(want[name]).tobytes()
            assert np.all(np.asarray(want[name]) != 0)
    for name in want_many:
        assert np.asarray(got_many[name]).tobytes() == np.asarray(want_many[name]).tobytes()


def test_basis_table_rows_rebuild_the_basis(ssp):
    csp, _ = build(ssp, zh_const=False)
    basis = np.asarray(csp._sfh_basis)
    empty = ~np.any(basis != 0, axis=(0, 1, 4))
    assert empty.any()
    for wave in (slice(None), slice(40, 500)):
        table, rows = csp._sfh_basis_table(wave)
        assert table.shape[2] == (~empty).sum() + 1
        assert np.asarray(table)[:, :, rows].tobytes() == basis[..., wave].tobytes()
