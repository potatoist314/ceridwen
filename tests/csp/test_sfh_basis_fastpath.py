"""Fixed-grid tests for the experimental eight-node SFH-basis kernels."""

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


def build_csp(ssp, selector=None, **overrides):
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
        "sfh_basis_fastpath": selector,
    }
    options.update(overrides)
    return CSPBasis_afe(ssp, **options)


@pytest.fixture(scope="module")
def models(fixed_ssp):
    return {
        selector: build_csp(fixed_ssp, selector)
        for selector in (None, "A", "B")
    }


def theta(afe, metallicity, sfh=None):
    if sfh is None:
        sfh = jnp.array([0.2, 1.1, 0.4, 2.0, 0.7, 1.6, 0.3, 0.9])
    return {
        "afe": jnp.array([afe]),
        "Z": jnp.array([metallicity]),
        "sfh": jnp.asarray(sfh),
    }


def test_baseline_is_default_and_operator_shapes(models):
    assert models[None].sfh_basis_fastpath is None
    assert models[None]._sfh_node_to_age is None
    assert models["A"]._sfh_node_to_age.shape == (8, 107)
    assert models["A"]._sfh_basis.shape == (5, 13, 8, 11)
    assert models["B"]._sfh_basis_flat.shape == (5 * 13 * 8, 11)


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
def test_fp32_variants_match_baseline_at_edges_and_between_knots(
    models, afe, metallicity
):
    inputs = theta(afe, metallicity)
    expected = models[None].get_spectrum(inputs)
    assert expected.dtype == jnp.float32
    for selector in ("A", "B"):
        actual = models[selector].get_spectrum(inputs)
        assert actual.dtype == jnp.float32
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-2)


def test_static_operator_matches_baseline_age_weights(models):
    sfh = jnp.array([0.3, 1.7, 0.2, 0.9, 2.1, 0.5, 1.2, 0.4])
    baseline_age_weights = models[None].calculate_ssp_weights(
        theta(0.13, -0.73, sfh)
    ).sum(axis=0)
    fast_age_weights = sfh @ models["A"]._sfh_node_to_age
    np.testing.assert_allclose(
        fast_age_weights,
        baseline_age_weights,
        rtol=2e-6,
        atol=16.0,
    )


def test_gradients_match_away_from_alpha_and_metallicity_knots(models):
    sfh = jnp.array([0.2, 1.1, 0.4, 2.0, 0.7, 1.6, 0.3, 0.9])

    def objective(csp, afe, metallicity, sfh_values):
        return jnp.sum(csp.get_spectrum(theta(afe, metallicity, sfh_values)))

    expected = jax.grad(objective, argnums=(1, 2, 3))(
        models[None], 0.13, -0.73, sfh
    )
    for selector in ("A", "B"):
        actual = jax.grad(objective, argnums=(1, 2, 3))(
            models[selector], 0.13, -0.73, sfh
        )
        for actual_part, expected_part in zip(actual, expected):
            np.testing.assert_allclose(
                actual_part,
                expected_part,
                rtol=8e-5,
                atol=3e-2,
            )


def test_runtime_lookback_grid_is_outside_static_contract(models):
    inputs = theta(0.13, -0.73)
    inputs["lookback_time"] = LOOKBACK_GYR
    with pytest.raises(ValueError, match="construction-time lookback grid"):
        models["A"].get_spectrum(inputs)


@pytest.mark.parametrize("selector", ["A", "B"])
def test_hlo_excludes_age_cube_shapes(models, selector):
    lowered = jax.jit(models[selector].get_spectrum).lower(theta(0.13, -0.73))
    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    for excluded in (
        "f32[5,13,107,11]",
        "f32[13,107,11]",
        "f32[107,11]",
        "f32[13,107]",
    ):
        assert excluded not in hlo
    expected_basis_shape = (
        "f32[5,13,8,11]" if selector == "A" else "f32[520,11]"
    )
    assert expected_basis_shape in hlo


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"lookback_time": jnp.linspace(0.0, 8.4, 7), "theta": None}, "eight"),
        ({"sfh_interp": "linear"}, "sfh_interp='step'"),
        ({"track_zred_age": True}, "static lookback"),
        ({"add_dust": True}, "age-dependent dust"),
    ],
)
def test_selector_enforces_only_fixed_grid_contract(
    fixed_ssp, overrides, message
):
    with pytest.raises(ValueError, match=message):
        build_csp(fixed_ssp, "A", **overrides)
