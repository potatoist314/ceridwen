"""Contracts for the automatic eight-node SFH-basis fast path."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ceridwen.csp.csp_afe import CSPBasis_afe


LOOKBACK_GYR = jnp.array([0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 5.0, 8.4])


@pytest.fixture(scope="module")
def fixed_ssp():
    rng = np.random.default_rng(20260827)
    return SimpleNamespace(
        ssp_flux=rng.uniform(0.1, 2.0, size=(5, 13, 107, 11)).astype(np.float32),
        ssp_afe=np.linspace(-0.2, 0.6, 5, dtype=np.float32),
        ssp_wave=np.linspace(1000.0, 10000.0, 11, dtype=np.float32),
        ssp_lg_age_gyr=np.linspace(-3.0, 1.14, 107, dtype=np.float32),
        ssp_lgmet=np.linspace(-2.0, 0.4, 13, dtype=np.float32),
        ssp_resolution=None,
        isoc_type=None,
        spec_library=None,
    )


def build_csp(ssp, *, reference=False, **overrides):
    options = {
        "theta": {
            "lookback_time": LOOKBACK_GYR,
            "sfh": jnp.ones(8),
            "Z": jnp.array([-0.8]),
            "afe": jnp.array([0.2]),
        },
        "zh_const": True,
        "sfh_interp": "step",
        "add_dust": False,
        "add_diffuse_dust": False,
        "add_dust_emission": False,
        "sigma_losvd_kms": 0.0,
        "verbose": False,
    }
    options.update(overrides)
    csp = CSPBasis_afe(ssp, **options)
    if reference:
        csp.sfh_basis_fastpath = False
        csp._sfh_basis = None
        csp._sfh_node_to_age = None
    return csp


@pytest.fixture(scope="module")
def models(fixed_ssp):
    return {
        "reference": build_csp(fixed_ssp, reference=True),
        "automatic": build_csp(fixed_ssp),
    }


def theta(afe, metallicity, sfh=None):
    if sfh is None:
        sfh = jnp.array([0.2, 1.1, 0.4, 2.0, 0.7, 1.6, 0.3, 0.9])
    return {
        "afe": jnp.array([afe]),
        "Z": jnp.array([metallicity]),
        "sfh": jnp.asarray(sfh),
    }


def test_supported_model_uses_fastpath_by_default(models):
    automatic = models["automatic"]
    assert automatic.sfh_basis_fastpath
    assert automatic._sfh_node_to_age.shape == (8, 107)
    assert automatic._sfh_basis.shape == (5, 13, 8, 11)


@pytest.mark.parametrize(
    ("afe", "metallicity"),
    [
        (-0.2, -2.0),
        (-0.2, 0.4),
        (0.6, -2.0),
        (0.6, 0.4),
        (0.13, -0.73),
    ],
)
def test_fastpath_matches_reference_at_edges_and_between_knots(
    models, afe, metallicity
):
    inputs = theta(afe, metallicity)
    expected = models["reference"].get_spectrum(inputs)
    actual = models["automatic"].get_spectrum(inputs)
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-2)


def test_alpha_free_fastpath_matches_reference_solar_plane(models):
    inputs = theta(0.0, -0.73)
    inputs.pop("afe")
    expected = models["reference"].get_spectrum(inputs)
    actual = models["automatic"].get_spectrum(inputs)
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-2)


def test_diffuse_dust_fastpath_matches_reference(fixed_ssp):
    reference = build_csp(fixed_ssp, reference=True, add_diffuse_dust=True)
    automatic = build_csp(fixed_ssp, add_diffuse_dust=True)
    inputs = dict(reference.theta_init)
    inputs.update(theta(0.13, -0.73))
    expected = reference.get_spectrum(inputs)
    actual = automatic.get_spectrum(inputs)
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-2)


def test_static_operator_matches_reference_age_weights(models):
    sfh = jnp.array([0.3, 1.7, 0.2, 0.9, 2.1, 0.5, 1.2, 0.4])
    reference_age_weights = models["reference"].calculate_ssp_weights(
        theta(0.13, -0.73, sfh)
    ).sum(axis=0)
    fast_age_weights = sfh @ models["automatic"]._sfh_node_to_age
    np.testing.assert_allclose(
        fast_age_weights,
        reference_age_weights,
        rtol=2e-6,
        atol=16.0,
    )


def test_operator_rows_are_trapezoid_time_weights_in_years(models):
    times_yr = LOOKBACK_GYR * 1e9
    widths_yr = jnp.diff(times_yr)
    expected = jnp.concatenate(
        (
            0.5 * widths_yr[:1],
            0.5 * (widths_yr[:-1] + widths_yr[1:]),
            0.5 * widths_yr[-1:],
        )
    )
    actual = models["automatic"]._sfh_node_to_age.sum(axis=1)
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=64.0)


def test_constant_ssp_limit_is_sfh_time_integral_in_years(fixed_ssp):
    data = vars(fixed_ssp).copy()
    data["ssp_flux"] = np.ones_like(fixed_ssp.ssp_flux)
    unit_ssp = SimpleNamespace(**data)
    sfh = jnp.array([0.2, 1.1, 0.4, 2.0, 0.7, 1.6, 0.3, 0.9])
    expected = jnp.trapezoid(sfh, LOOKBACK_GYR * 1e9)

    for reference in (True, False):
        actual = build_csp(unit_ssp, reference=reference).get_spectrum(
            theta(0.13, -0.73, sfh)
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=128.0)


def test_gradients_match_reference(models):
    sfh = jnp.array([0.2, 1.1, 0.4, 2.0, 0.7, 1.6, 0.3, 0.9])

    def objective(csp, afe, metallicity, sfh_values):
        return jnp.sum(csp.get_spectrum(theta(afe, metallicity, sfh_values)))

    expected = jax.grad(objective, argnums=(1, 2, 3))(
        models["reference"], 0.13, -0.73, sfh
    )
    actual = jax.grad(objective, argnums=(1, 2, 3))(
        models["automatic"], 0.13, -0.73, sfh
    )
    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_allclose(
            actual_part,
            expected_part,
            rtol=8e-5,
            atol=3e-2,
        )


def test_runtime_lookback_grid_uses_general_path(models):
    inputs = theta(0.13, -0.73)
    inputs["lookback_time"] = jnp.array(
        [0.0, 0.02, 0.08, 0.25, 0.8, 2.6, 4.8, 7.9]
    )
    expected = models["reference"].get_spectrum(inputs)
    actual = models["automatic"].get_spectrum(inputs)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-3)


def test_fastpath_hlo_excludes_age_cube_shapes(models):
    lowered = jax.jit(models["automatic"].get_spectrum).lower(theta(0.13, -0.73))
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    for excluded in (
        "tensor<5x13x107x11xf32>",
        "tensor<13x107x11xf32>",
        "tensor<107x11xf32>",
        "tensor<13x107xf32>",
    ):
        assert excluded not in stablehlo
    assert "tensor<5x13x8x11xf32>" in stablehlo
    # compare_all searchsorted and the corner dynamic_slice must not lower
    # to sequential loops.
    assert "stablehlo.while" not in stablehlo


@pytest.mark.parametrize(
    "overrides",
    [
        {"lookback_time": jnp.linspace(0.0, 8.4, 7), "theta": None},
        {"sfh_interp": "linear"},
        {"track_zred_age": True},
        {"add_dust": True},
    ],
)
def test_unsupported_model_uses_general_path(fixed_ssp, overrides):
    csp = build_csp(fixed_ssp, **overrides)
    assert not csp.sfh_basis_fastpath
    assert csp._sfh_basis is None
    assert csp._sfh_node_to_age is None
