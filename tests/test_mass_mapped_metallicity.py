"""Contracts for the mass-mapped metallicity history (Gallazzi et al. 2026)."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ceridwen.csp.csp_afe import CSPBasis_afe
from ceridwen.model import mass_mapped_beta, mass_mapped_zh

jax.config.update("jax_enable_x64", True)

# amist_c3k_hr_krou_afe metallicity axis: [Fe/H] = -2.5 .. +0.5 minus 1.7328283.
LOGZ_SUN = -1.7328283
LOGZ_GRID = LOGZ_SUN + np.linspace(-2.5, 0.5, 13)
LOGZ_0_CAP = LOGZ_SUN - np.log10(50.0)
# Production layout: 0-30 Myr, 30-100 Myr, log-spaced to 0.85 t_univ, t_univ.
LOOKBACK_GYR = np.array(
    [0.0, 0.03] + (10.0 ** np.linspace(8.0, np.log10(0.85 * 6.9e9), 12) / 1e9).tolist() + [6.9]
)
TIMES_YR = LOOKBACK_GYR * 1e9


def sfh_draws(count, seed):
    """Per-node SFHs from the production StudentT(0, 0.3, 2) log-ratio prior."""
    rng = np.random.default_rng(seed)
    ratios = 0.3 * rng.standard_t(2.0, size=(count, LOOKBACK_GYR.size - 1))
    return 10.0 ** np.concatenate([np.zeros((count, 1)), -np.cumsum(ratios, axis=1)], axis=1)


def bin_mass_fractions(sfh):
    mass = 0.5 * (sfh[:-1] + sfh[1:]) * np.diff(TIMES_YR)
    return mass / mass.sum()


def history(sfh, logz_mean, beta_unit):
    logz_0 = min(LOGZ_0_CAP, logz_mean)
    beta = mass_mapped_beta(beta_unit, logz_mean, logz_0, LOGZ_GRID[-1])
    return np.asarray(mass_mapped_zh(sfh, TIMES_YR, logz_mean, beta, logz_0)), float(beta)


@pytest.mark.parametrize("logz_mean", [-3.9, -2.6, -1.5])
def test_small_beta_limit_is_constant_metallicity(logz_mean):
    sfh = sfh_draws(1, 1)[0]
    logz_0 = min(LOGZ_0_CAP, logz_mean)
    zh = mass_mapped_zh(sfh, TIMES_YR, logz_mean, 1e-6, logz_0)
    np.testing.assert_allclose(zh, logz_mean, atol=1e-5)


@pytest.mark.parametrize("beta", [0.05, 0.4, 0.8])
def test_mean_below_initial_cap_is_constant_metallicity(beta):
    logz_mean = LOGZ_0_CAP - 0.3
    zh = mass_mapped_zh(sfh_draws(1, 2)[0], TIMES_YR, logz_mean, beta, logz_mean)
    np.testing.assert_allclose(zh, logz_mean, rtol=0, atol=1e-12)


def test_formed_mass_weighted_metallicity_equals_mean():
    rng = np.random.default_rng(3)
    for sfh in sfh_draws(200, 4):
        logz_mean = rng.uniform(LOGZ_GRID[0], LOGZ_GRID[-1])
        zh, _ = history(sfh, logz_mean, rng.uniform())
        mean = np.sum(bin_mass_fractions(sfh) * 10.0 ** zh)
        np.testing.assert_allclose(np.log10(mean), logz_mean, rtol=0, atol=1e-9)


def test_metallicity_stays_on_grid_and_rises_with_formed_mass():
    edge = 1e-4  # production prior on Z stops this far inside the grid
    for sfh in sfh_draws(50, 5):
        for logz_mean in np.linspace(LOGZ_GRID[0] + edge, LOGZ_GRID[-1] - edge, 61):
            for beta_unit in (0.0, 0.5, 1.0):
                zh, beta = history(sfh, logz_mean, beta_unit)
                assert beta <= 0.8
                assert np.all(zh >= LOGZ_GRID[0] - 1e-12)
                assert np.all(zh <= LOGZ_GRID[-1] + 1e-12)
                assert np.all(np.diff(zh) <= 1e-12)  # index 0 = youngest bin


def test_beta_range_follows_gallazzi_prior_and_grid_limit():
    logz_max = LOGZ_GRID[-1]
    solar = mass_mapped_beta(np.array([0.0, 1.0]), LOGZ_SUN, LOGZ_0_CAP, logz_max)
    z_0, z_max = 10.0**LOGZ_0_CAP, 10.0**logz_max
    np.testing.assert_allclose(
        solar, [0.05, min(0.8, 1 - (10.0**LOGZ_SUN - z_0) / (z_max - z_0))]
    )
    low = mass_mapped_beta(np.array([0.0, 1.0]), LOGZ_GRID[0], LOGZ_GRID[0], logz_max)
    np.testing.assert_allclose(low, [0.05, 0.8])
    # At the upper limit the youngest stars sit on the grid top.
    zh, _ = history(np.ones(LOOKBACK_GYR.size), LOGZ_SUN + 0.3, 1.0)
    z_final = 10.0**zh[0]
    assert z_final <= z_max and z_final > 0.9 * z_max


@pytest.fixture(scope="module")
def grid_ssp():
    rng = np.random.default_rng(20260929)
    return SimpleNamespace(
        ssp_flux=rng.uniform(0.1, 2.0, size=(5, 13, 107, 11)).astype(np.float32),
        ssp_afe=np.linspace(-0.2, 0.6, 5, dtype=np.float32),
        ssp_wave=np.linspace(1000.0, 10000.0, 11, dtype=np.float32),
        ssp_lg_age_gyr=np.linspace(-3.0, 1.14, 107, dtype=np.float32),
        ssp_lgmet=LOGZ_GRID.astype(np.float32),
        ssp_resolution=None,
        isoc_type=None,
        spec_library=None,
    )


def build_csp(ssp, zh, *, reference=False, dust=False):
    csp = CSPBasis_afe(
        ssp,
        theta={
            "lookback_time": jnp.asarray(LOOKBACK_GYR),
            "sfh": jnp.ones(LOOKBACK_GYR.size),
            "zh": zh,
            "afe": jnp.array([0.2]),
        },
        zh_const=False,
        sfh_interp="step",
        add_dust=dust,
        add_diffuse_dust=dust,
        add_dust_emission=False,
        sigma_losvd_kms=0.0,
        verbose=False,
    )
    if reference:
        csp.sfh_basis_fastpath = False
        csp._sfh_basis = None
        csp._sfh_bin_to_age = None
    return csp


@pytest.mark.parametrize("dust", [False, True], ids=["no-dust", "birth-cloud"])
def test_fastpath_matches_general_path(grid_ssp, dust):
    init = jnp.full(LOOKBACK_GYR.size - 1, LOGZ_SUN)
    reference = build_csp(grid_ssp, init, reference=True, dust=dust)
    automatic = build_csp(grid_ssp, init, dust=dust)
    assert automatic.sfh_basis_fastpath and automatic.zh_per_bin
    inputs = {k: v for k, v in reference.theta_init.items()}
    if dust:
        defaults = automatic.dust_attn.get_default_fit_params()
        inputs.update({k: jnp.full_like(jnp.asarray(v, dtype=float), 0.7) for k, v in defaults.items()})
    for sfh, logz_mean, beta_unit in zip(sfh_draws(4, 6), (-3.8, -2.4, -1.7, -1.3), (0.2, 1.0, 0.6, 0.9)):
        zh, _ = history(sfh, logz_mean, beta_unit)
        inputs.update(afe=jnp.array([0.13]), sfh=jnp.asarray(sfh), zh=jnp.asarray(zh))
        np.testing.assert_allclose(
            automatic.get_spectrum(inputs), reference.get_spectrum(inputs), rtol=5e-5, atol=2e-2
        )


def test_per_bin_zh_matches_per_node_zh_with_same_bin_values(grid_ssp):
    node = jnp.linspace(-3.9, -1.3, LOOKBACK_GYR.size)
    per_node = build_csp(grid_ssp, node, reference=True)
    per_bin = build_csp(grid_ssp, 0.5 * (node[:-1] + node[1:]), reference=True)
    sfh = jnp.asarray(sfh_draws(1, 7)[0])
    np.testing.assert_allclose(
        per_bin.get_spectrum({"afe": jnp.array([0.13]), "sfh": sfh, "zh": 0.5 * (node[:-1] + node[1:])}),
        per_node.get_spectrum({"afe": jnp.array([0.13]), "sfh": sfh, "zh": node}),
        rtol=1e-6,
    )
