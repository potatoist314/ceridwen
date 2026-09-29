"""Contracts for the automatic SFH-basis fast path."""

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
        csp._sfh_bin_to_age = None
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
    assert automatic._sfh_bin_to_age.shape == (7, 107)
    assert automatic._sfh_basis.shape == (5, 13, 1, 7, 11)


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
    fast_age_weights = 0.5 * (sfh[:-1] + sfh[1:]) @ models["automatic"]._sfh_bin_to_age
    np.testing.assert_allclose(
        fast_age_weights,
        reference_age_weights,
        rtol=2e-6,
        atol=16.0,
    )


def test_operator_rows_are_bin_widths_in_years(models):
    expected = jnp.diff(LOOKBACK_GYR * 1e9)
    actual = models["automatic"]._sfh_bin_to_age.sum(axis=1)
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
    with pytest.warns(UserWarning, match="fast path is bypassed"):
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
    assert "tensor<5x13x1x7x11xf32>" in stablehlo
    # compare_all searchsorted and the corner dynamic_slice must not lower
    # to sequential loops.
    assert "stablehlo.while" not in stablehlo


@pytest.mark.parametrize(
    "overrides",
    [{"sfh_interp": "linear"}, {"track_zred_age": True}],
)
def test_unsupported_model_warns_and_uses_general_path(fixed_ssp, overrides):
    with pytest.warns(UserWarning, match="fast path is off"):
        csp = build_csp(fixed_ssp, **overrides)
    assert not csp.sfh_basis_fastpath
    assert csp._sfh_basis is None
    assert csp._sfh_bin_to_age is None


EIGHT_BINS_GYR = jnp.array([0.0, 0.03, 0.1, 0.3, 0.8, 1.6, 3.0, 5.0, 8.4])
FOURTEEN_BINS_GYR = jnp.array(
    [0.0, 0.03, 0.1, 0.16, 0.25, 0.4, 0.63, 1.0, 1.6, 2.5, 3.2, 4.0, 5.0, 6.2, 8.4]
)


def model_pair(ssp, lookback, *, per_bin=False, zh_const=True, dust=False):
    n_sfh = lookback.size - 1 if per_bin else lookback.size
    initial = {"lookback_time": lookback, "sfh": jnp.ones(n_sfh), "afe": jnp.array([0.2])}
    if zh_const:
        initial["Z"] = jnp.array([-0.8])
    else:
        initial["zh"] = jnp.full(lookback.size, -0.8)
    options = dict(theta=initial, zh_const=zh_const, add_dust=dust, add_diffuse_dust=dust)
    reference = build_csp(ssp, reference=True, **options)
    automatic = build_csp(ssp, **options)
    assert automatic.sfh_basis_fastpath

    inputs = {k: v for k, v in reference.theta_init.items() if k != "lookback_time"}
    # Young-weighted SFH, so stars under 10 Myr carry a visible share of the light.
    inputs.update(afe=jnp.array([0.13]), sfh=jnp.geomspace(300.0, 0.3, n_sfh))
    if zh_const:
        inputs["Z"] = jnp.array([-0.73])
    else:
        # Off-grid values, both grid edges and values beyond them.
        inputs["zh"] = jnp.linspace(-2.1, 0.5, lookback.size) + 0.07 * jnp.sin(
            jnp.arange(lookback.size)
        )
    if dust:
        defaults = automatic.dust_attn.get_default_fit_params()
        inputs.update({k: jnp.full_like(jnp.asarray(v, dtype=float), 0.7) for k, v in defaults.items()})
    return reference, automatic, inputs


def assert_spectra_match(reference, automatic, inputs):
    np.testing.assert_allclose(
        automatic.get_spectrum(inputs), reference.get_spectrum(inputs), rtol=5e-5, atol=2e-2
    )


@pytest.mark.parametrize("dust", [False, True], ids=["no-dust", "birth-cloud"])
@pytest.mark.parametrize("zh_const", [True, False], ids=["const-Z", "evolving-Z"])
@pytest.mark.parametrize(
    "lookback", [EIGHT_BINS_GYR, FOURTEEN_BINS_GYR], ids=["8-bin", "14-bin"]
)
def test_bins_metallicity_and_birth_cloud_dust_match_reference(fixed_ssp, lookback, zh_const, dust):
    reference, automatic, inputs = model_pair(fixed_ssp, lookback, zh_const=zh_const, dust=dust)
    assert_spectra_match(reference, automatic, inputs)
    if dust:
        # Birth-cloud dust must change the spectrum for the check to test it.
        dust_free = {k: v for k, v in inputs.items() if k not in automatic.dust_param_names}
        dust_free.update({k: jnp.zeros_like(v) for k, v in inputs.items() if k in automatic.dust_param_names})
        change = automatic.get_spectrum(inputs) / automatic.get_spectrum(dust_free) - 1
        assert np.max(np.abs(change)) > 1e-2
        assert_spectra_match(reference, automatic, {**inputs, "frac_obrun": jnp.array([0.3])})


@pytest.mark.parametrize("zh_const", [True, False], ids=["const-Z", "evolving-Z"])
def test_per_bin_sfh_matches_reference(fixed_ssp, zh_const):
    assert_spectra_match(*model_pair(fixed_ssp, FOURTEEN_BINS_GYR, per_bin=True,
                                     zh_const=zh_const, dust=True))


def test_evolving_metallicity_gradients_match_reference(fixed_ssp):
    reference, automatic, inputs = model_pair(fixed_ssp, FOURTEEN_BINS_GYR, zh_const=False, dust=True)

    def objective(csp, zh, sfh):
        return jnp.sum(csp.get_spectrum({**inputs, "zh": zh, "sfh": sfh}))

    expected = jax.grad(objective, argnums=(1, 2))(reference, inputs["zh"], inputs["sfh"])
    actual = jax.grad(objective, argnums=(1, 2))(automatic, inputs["zh"], inputs["sfh"])
    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_allclose(actual_part, expected_part, rtol=8e-5, atol=3e-2)


def test_single_alpha_plane_grid_matches_reference(fixed_ssp):
    data = vars(fixed_ssp).copy()
    data.update(ssp_flux=fixed_ssp.ssp_flux[2:3], ssp_afe=fixed_ssp.ssp_afe[2:3])
    single = SimpleNamespace(**data)
    reference = build_csp(single, reference=True)
    automatic = build_csp(single)
    assert automatic.sfh_basis_fastpath
    for inputs in (theta(0.2, -0.73), {k: v for k, v in theta(0.2, -0.73).items() if k != "afe"}):
        assert_spectra_match(reference, automatic, inputs)
