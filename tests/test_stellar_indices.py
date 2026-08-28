"""Equation and serialization tests for integrated stellar indices."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from ceridwen.fit import read_result_h5, write_result_h5
from ceridwen.observation import StellarIndexDefinition, StellarIndices
from ceridwen.sampler.runner import SamplingResult


DEFINITIONS = (
    StellarIndexDefinition(
        "TestEW", "equivalent_width", (4000.0, 4010.0),
        (4020.0, 4030.0), (4040.0, 4050.0),
    ),
    StellarIndexDefinition(
        "TestMag", "magnitude", (4060.0, 4070.0),
        (4080.0, 4090.0), (4100.0, 4110.0),
    ),
    StellarIndexDefinition(
        "TestRatio", "flux_ratio", (4120.0, 4130.0), None,
        (4140.0, 4150.0),
    ),
)


def _observation():
    wave = np.arange(3900.0, 4200.5, 0.5)
    observation = StellarIndices(
        definitions=DEFINITIONS,
        values=np.zeros(3),
        uncertainty=np.ones(3),
        name="stellar_indices",
    )
    observation.setup_for_model(wave)
    return observation, wave


def test_index_equations_and_units():
    observation, wave = _observation()
    vacuum_bands = observation.bandpasses_vacuum
    flam = np.ones_like(wave)
    for feature in vacuum_bands[:2, 1]:
        flam[(wave >= feature[0] - 0.5) & (wave <= feature[1] + 0.5)] = 0.8
    fnu = flam * wave**2
    ratio_bands = vacuum_bands[2, (0, 2), :]
    fnu[
        (wave >= ratio_bands.min() - 0.5)
        & (wave <= ratio_bands.max() + 0.5)
    ] = 1.0

    predicted = np.asarray(observation.predict(jnp.asarray(fnu), wave))
    np.testing.assert_allclose(predicted[0], 2.0, rtol=0.0, atol=1e-3)
    np.testing.assert_allclose(
        predicted[1], -2.5 * np.log10(0.8), rtol=0.0, atol=1e-4
    )
    np.testing.assert_allclose(predicted[2], 1.0, rtol=0.0, atol=1e-6)
    assert observation.index_units == ["angstrom", "mag", "dimensionless"]


def test_predictions_are_scale_invariant_and_jittable():
    observation, wave = _observation()
    spectrum = jnp.asarray(wave**2 * (1.0 + 0.05 * np.sin(wave / 20.0)))

    predict = jax.jit(lambda flux: observation.predict(flux, wave))
    base = np.asarray(predict(spectrum))
    scaled = np.asarray(predict(17.0 * spectrum))
    np.testing.assert_allclose(scaled, base, rtol=2e-6, atol=2e-6)
    assert np.all(np.isfinite(base))


def test_mask_and_selection_follow_index_names():
    observation, _ = _observation()
    observation.mask_by_name(["TestMag"])
    np.testing.assert_array_equal(observation.mask, [True, False, True])

    selected = observation.select_by_name(["TestRatio", "TestEW"])
    assert selected.index_names == ["TestRatio", "TestEW"]
    np.testing.assert_array_equal(selected.mask, [True, True])


def test_hdf5_preserves_index_metadata(tmp_path):
    observation, wave = _observation()
    model = SimpleNamespace(
        observations=[observation],
        csp=SimpleNamespace(wave=wave),
        theta_init={"logmass": jnp.array([10.0])},
        priors={},
        transforms={},
        param_names=["logmass"],
        zred=0.0,
    )
    result = SamplingResult(
        samples={"logmass": jnp.array([10.0, 10.1])},
        log_evidence=0.0,
        log_evidence_err=0.1,
        log_weights=jnp.zeros(2),
        log_likelihoods=jnp.zeros(2),
        param_names=["logmass"],
        n_likelihood_calls=2,
        wall_time_s=1.0,
        sampler_name="test",
    )
    path = tmp_path / "indices.h5"
    write_result_h5(path, model, result)
    restored = read_result_h5(path)["obs"]["stellar_indices"]

    assert restored["type"] == "StellarIndices"
    assert "bandpasses_vacuum" in restored
    assert restored["bandpasses_vacuum"].shape == (3, 3, 2)
    assert "TestRatio" in restored["index_names"]
