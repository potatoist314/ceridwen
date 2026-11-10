"""C/N response units, resolution, SFH integration and zero-offset limits."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ceridwen import Cosmology
from ceridwen.broadening import Instrument, Kinematics
from ceridwen.csp import CSPBasis_afe
from ceridwen.model import SedModel
from ceridwen.observation import Spectrum
from ceridwen.ssps.abundance_response import AlfCNResponse


@pytest.fixture(scope="module")
def grid():
    wave = np.geomspace(3501., 10000., 1000)
    return SimpleNamespace(
        ssp_flux=np.ones((2, 3, 10, len(wave)), dtype=np.float32),
        ssp_afe=np.array([0., .4]), ssp_wave=wave,
        ssp_lg_age_gyr=np.linspace(-3., 1.1, 10),
        ssp_lgmet=np.array([-1., -.5, 0.]), log10_zsun=0.,
        ssp_resolution=np.full(len(wave), 42.), isoc_type=None, spec_library=None)


def response(grid):
    templates = np.ones((9, 4, 5, len(grid.ssp_wave)))
    templates[1:3] = .1
    templates[3:5] = .2
    templates[5:] = .02
    return AlfCNResponse(grid.ssp_wave, templates, sigma_catalog=200.)


def model(grid, cn=None, evolving=False):
    theta = {"lookback_time": jnp.array([0., 1., 3., 8.]),
             "sfh": jnp.array([1., 2., 3., 1.]), "afe": jnp.array([.1])}
    theta.update({"logzsol_hist": jnp.array([-.9, -.5, -.1, 0.])} if evolving
                 else {"logzsol": jnp.array([-.5])})
    csp = CSPBasis_afe(grid, theta=theta, zh_const=not evolving,
                     add_dust=False, add_diffuse_dust=False,
                     cosmo=Cosmology.planck18(), abundance_response=cn, verbose=False)
    obs = Spectrum(wavelength=np.linspace(4000., 5000., 100),
                   instrument=Instrument.sigma_kms(60.), name="spectrum")
    return SedModel(csp, observations=[obs],
                    kinematics=Kinematics(sigma_gal=200.), zred=0.)


@pytest.mark.parametrize("evolving", [False, True])
def test_zero_offsets_are_bitwise_current_model(grid, evolving):
    plain, enabled = model(grid, evolving=evolving), model(grid, response(grid), evolving)
    assert enabled.csp.sfh_basis_fastpath
    assert {"cfe", "nfe"}.isdisjoint(plain.theta_init)
    for predict in (lambda m, t: m.predict(t),
                    lambda m, t: jax.jit(m.predict)(t)):
        expected = predict(plain, plain.theta_init)["spectrum"]
        actual = predict(enabled, enabled.theta_init)["spectrum"]
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("evolving", [False, True])
def test_abundance_steps_and_cross_term(grid, evolving):
    cn = response(grid)
    fitted = model(grid, cn, evolving)
    theta = fitted.apply_transforms(fitted.theta_init)
    theta.update(cfe=jnp.array([.15]), nfe=jnp.array([.3]))
    # One native abundance step gives (1 + 0.1) * (1 + 0.2).
    np.testing.assert_allclose(cn.factor(fitted.csp, theta, "spectrum"), 1.32, rtol=2e-6)
    theta["nfe"] = jnp.array([0.])
    np.testing.assert_allclose(cn.factor(fitted.csp, theta, "spectrum"), 1.1, rtol=2e-6)


def test_source_columns_wavelength_and_fnu_units(tmp_path):
    wave = np.array([4000., 5000., 6000.])
    table = np.ones((3, 35))
    table[:, 0] = wave
    table[:, 1] = [2., 3., 4.]
    table[:, 8] = table[:, 1] * 1.1
    table[:, 9] = table[:, 1] * .9
    table[:, 11] = table[:, 1] * 1.2
    table[:, 12] = table[:, 1] * .8
    for z in ("m1.5", "m1.0", "m0.5", "p0.0"):
        for age in ("01", "03", "05", "09", "13"):
            np.savetxt(tmp_path / f"atlas_ssp_t{age}_Z{z}.abund.krpa.s100", table)
    cn = AlfCNResponse.load(tmp_path, sigma_catalog=200.)
    np.testing.assert_array_equal(cn.wave, wave)
    np.testing.assert_allclose(cn.templates[0, 0, 0], table[:, 1] * wave ** 2)
    np.testing.assert_allclose(cn.templates[1, 0, 0] / cn.templates[0, 0, 0], .1)
    np.testing.assert_allclose(cn.templates[2, 0, 0] / cn.templates[0, 0, 0], -.1)
    np.testing.assert_allclose(cn.templates[3, 0, 0] / cn.templates[0, 0, 0], .2)
    np.testing.assert_allclose(cn.templates[4, 0, 0] / cn.templates[0, 0, 0], -.2)


def test_total_resolution_below_source_is_refused(grid):
    cn = response(grid)
    obs = Spectrum(wavelength=np.linspace(4000., 5000., 100),
                   instrument=Instrument.sigma_kms(20.), name="spectrum")
    with pytest.raises(ValueError, match="total sigma >=100"):
        cn.projected_templates(obs, 0., 50.)


def test_broadening_subtracts_intrinsic_width_from_combined_resolution():
    wave = np.geomspace(4700., 5300., 12000)
    velocity = 2.998e5 * np.log(wave / 5000.)
    templates = np.zeros((9, 4, 5, len(wave)))
    templates[0] = 1.
    templates[1] = -.1 * np.exp(-.5 * (velocity / 100.) ** 2)
    cn = AlfCNResponse(wave, templates, sigma_catalog=200.)
    pixels = np.geomspace(4850., 5150., 3000)
    obs = Spectrum(wavelength=pixels, instrument=Instrument.sigma_kms(60.), name="spectrum")
    projected, minimum = cn.projected_templates(obs, 0., 200.)
    deficit = -projected[1, 0, 0]
    v = 2.998e5 * np.log(pixels / 5000.)
    centre = np.sum(v * deficit) / deficit.sum()
    width = np.sqrt(np.sum((v - centre) ** 2 * deficit) / deficit.sum())
    np.testing.assert_allclose(width, np.hypot(200., 60.), rtol=.005)
    np.testing.assert_allclose(minimum, np.hypot(200., 60.))
