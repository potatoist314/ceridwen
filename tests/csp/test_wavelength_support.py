"""``CSPBasis(wavelength_support=True)`` synthesises only the model pixels the observations
read and zero-fills the rest after the mass / flux / IGM scaling.

Reference: the same model on the whole grid (the default).  The two are not byte-identical:
XLA's CPU reductions (the age contraction of the CSP einsum) change their summation order with
the array length, so a pixel can differ in the last float32 bit.  Tolerance: 2 float32 ulps
relative per prediction (4.8e-7), and the log-likelihood to 1e-6 absolute.  Also: every
fall-back to the whole grid listed in ``CSPBasis._support_for``, the observations' declared
supports, the default being off, and finite gradients through the support path.
"""
import os
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _gridfixture import require_test_grid, find_named_grid  # noqa: E402

import ceridwen  # noqa: F401  (sets the precision: float64, or float32 on an Apple GPU)

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology   # noqa: E402
from ceridwen.observation import Photometry, Spectrum, Lines  # noqa: E402
from ceridwen.broadening import Instrument, Kinematics        # noqa: E402
from _f32bounds import B8, lnl_bound, fft_atol   # noqa: E402,F401  (float32 bounds)

GRID = require_test_grid()
NEB = pytest.mark.skipif(not os.environ.get("SPS_HOME"), reason="nebular model needs $SPS_HOME")
ZRED = 3.0
FILTERS = ["acs_wfc_f606w", "jwst_f115w", "jwst_f200w", "jwst_f356w", "jwst_f444w"]
RTOL = 4.8e-7                              # 2 float32 ulps


def _ssp():
    return SSPData.load(str(GRID))


def _obs(spec=True, lines=False):
    out = [Photometry(filters=FILTERS, flux=[1e-9] * len(FILTERS),
                      uncertainty=[1e-10] * len(FILTERS), name="phot")]
    if spec:
        lam = np.exp(np.arange(np.log(28700.0), np.log(51000.0), 1.0 / 1500.0))
        out.append(Spectrum(wavelength=lam, flux=np.full(lam.size, 1e-30),
                            uncertainty=np.full(lam.size, 1e-31),
                            instrument=Instrument.R_fwhm(1000.0), name="spec"))
    if lines:
        out.append(Lines(line_ind=np.arange(2), line_names=["Hbeta", "Halpha"],
                         wavelength=np.array([4862.71, 6564.61]), flux=np.full(2, 1e-17),
                         uncertainty=np.full(2, 1e-18), name="lines"))
    return out


def _model(support, obs=None, zred=ZRED, neb=True, kin=None, fpi=None, **kw):
    cosmo = Cosmology.planck18()
    csp_kw = dict(lookback_time=jnp.linspace(0.0, float(cosmo.age(zred)), 5), zh_const=True,
                  add_neb=neb, add_igm=True, verbose=False, cosmo=cosmo,
                  wavelength_support=support)
    if neb or kw.get("add_dust_emission"):
        csp_kw["sps_home"] = os.environ["SPS_HOME"]
    csp_kw.update(kw)
    csp = CSPBasis(_ssp(), **csp_kw)
    return SedModel(csp, _obs() if obs is None else obs, priors={}, zred=zred,
                    kinematics=kin if kin is not None else Kinematics(sigma_gal=200.0),
                    free_param_init=fpi)


def _theta(model, k=0):
    rng = np.random.default_rng(k)
    return {n: jnp.asarray(np.asarray(model.theta_init[n]) * (1.0 + 0.02 * rng.standard_normal(
        np.shape(model.theta_init[n])))) for n in model.param_names}


def _close(a, b, n_pad=2 ** 16):
    """rtol RTOL (8 float32 roundings) plus the float32 FFT smoothing's absolute error,
    fft_atol (the two paths smooth different windows; GOTCHAS section 24). n_pad bounds the FFT
    length from above (2**14 .. 2**15 in these models)."""
    for k in a:
        x, y = np.asarray(a[k], dtype=np.float64), np.asarray(b[k], dtype=np.float64)
        assert x.shape == y.shape and np.all(np.isfinite(y)), k
        np.testing.assert_allclose(y, x, rtol=RTOL, atol=fft_atol(x, n_pad), err_msg=k)


def _pair(build):
    """(whole-grid model, support model) built the same way."""
    return build(False), build(True)


def test_default_is_off():
    csp = CSPBasis(_ssp(), lookback_time=jnp.linspace(0.0, 5.0, 4), add_neb=False,
                   verbose=False, cosmo=Cosmology.planck18())
    assert csp._wave_support is False
    m = SedModel(csp, _obs(), priors={}, zred=0.5)
    assert csp._support_for(m.observations, {"zred": jnp.array([0.5])}) is None


@NEB
def test_observation_supports_cover_what_they_read():
    m = _model(True)
    phot, spec = m.observations
    T = np.asarray(phot._T)
    cols = np.flatnonzero(np.any(T != 0.0, axis=0))
    lo, hi = phot.model_support
    assert lo <= cols[0] and hi >= cols[-1] + 1
    idx = np.asarray(phot._broadener.window.grid.idx)
    assert lo <= idx[0] and hi >= idx[-1] + 1
    idx = np.asarray(spec._proj.window.grid.idx)
    assert spec.model_support == (int(idx[0]), int(idx[-1]) + 1)
    assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1))      # contiguous window
    s = m.csp._support_for(m.observations, {"zred": jnp.array([ZRED])})
    assert s is not None and s.stop - s.start < m.csp.wave.shape[0]
    assert s.start == min(phot.model_support[0], spec.model_support[0])
    assert s.stop == max(phot.model_support[1], spec.model_support[1])


@NEB
@pytest.mark.parametrize("variant", ["neb", "neb_frac_obrun", "picket", "noneb", "nodust",
                                     "phot_only", "lines", "sampled_sigma_gal"])
def test_support_equals_whole_grid(variant):
    kw, obs, kin, extra, neb = {}, None, None, {}, True
    if variant == "neb_frac_obrun":
        extra["frac_obrun"] = jnp.array([0.2])
    elif variant == "picket":
        kw["fesc_geometry"] = "picket"
        extra["frac_obrun"] = jnp.array([0.2])
    elif variant == "noneb":
        neb = False
    elif variant == "nodust":
        kw.update(add_dust=False, add_diffuse_dust=False)
    elif variant == "phot_only":
        obs = "phot"
    elif variant == "lines":
        obs = "lines"
    elif variant == "sampled_sigma_gal":
        kin = Kinematics(sigma_gal="sigma_gal", sigma_gas=150.0)
        extra["sigma_gal"] = jnp.array([180.0])

    def build(flag):
        o = (_obs(spec=False) if obs == "phot" else _obs(lines=True) if obs == "lines"
             else None)
        return _model(flag, obs=o, kin=kin, neb=neb,
                      fpi={"sigma_gal": jnp.array([180.0])} if kin is not None else None, **kw)
    full, sup = _pair(build)
    th = {**_theta(full), **extra}
    s = sup.csp._support_for(sup.observations, {**th, "zred": jnp.array([ZRED])})
    assert s is not None
    _close(jax.jit(full.predict)(th), jax.jit(sup.predict)(th))
    tb = {k: jnp.stack([v, v * 1.01]) for k, v in th.items()}
    _close(jax.jit(jax.vmap(full.predict))(tb), jax.jit(jax.vmap(sup.predict))(tb))


@NEB
def test_free_z_spectrum_uses_its_window():
    """A Spectrum at a sampled redshift reads its projector's window, which covers the whole
    zred_range; without Photometry the support path is used."""
    def build(flag):
        cosmo = Cosmology.planck18()
        csp = CSPBasis(_ssp(), lookback_time=jnp.linspace(0.0, float(cosmo.age(3.2)), 5),
                       zh_const=True, add_neb=True, add_igm=True, verbose=False, cosmo=cosmo,
                       sps_home=os.environ["SPS_HOME"], wavelength_support=flag)
        from ceridwen.priors import Uniform
        return SedModel(csp, _obs()[1:], priors={"zred": Uniform(low=2.9, high=3.2)},
                        free_param_init={"zred": jnp.array([3.0])},
                        kinematics=Kinematics(sigma_gal=200.0))
    full, sup = _pair(build)
    th = _theta(full)
    assert sup.csp._support_for(sup.observations, th) is not None
    for z in (2.9, 3.05, 3.2):
        th["zred"] = jnp.array([z])
        _close(jax.jit(full.predict)(th), jax.jit(sup.predict)(th))


@NEB
def test_fall_backs_to_the_whole_grid():
    z = {"zred": jnp.array([ZRED])}
    m = _model(True, add_dust_emission=True)               # energy balance: every pixel
    assert m.csp._support_for(m.observations, z) is None
    m = _model(True)
    assert m.csp._support_for(m.observations, z, paint_lines=True) is None
    m.observations[0].free_z = True                         # Photometry at a sampled z
    assert m.csp._support_for(m.observations, z) is None
    m.observations[0].free_z = False

    class Custom(Photometry):                               # no declared support: whole grid
        model_support = None
    c = Custom(filters=FILTERS, flux=[1e-9] * 5, uncertainty=[1e-10] * 5, name="c")
    assert m.csp._support_for([c], z) is None
    lines = _obs(spec=False, lines=True)[1:]                # Lines read no pixel
    assert m.csp._support_for(lines, z) is None
    m.csp.get_spectrum = lambda theta, include_lines=False: None   # a replaced variant
    assert m.csp._support_for(m.observations, z) is None


def test_lnprob_and_gradient():
    """fitSED's log-posterior with the support path (no nebular model, so no $SPS_HOME):
    the value within the first-order bound of a 2-ulp change of every prediction,
    sum |y - mu| RTOL |mu| / sigma^2, and the gradient finite and close."""
    from ceridwen.fit import _likelihood_for
    from ceridwen.likelihood import MultiObservationLikelihood
    full, sup = _pair(lambda f: _model(f, neb=False))
    th = _theta(full)
    pred = full.predict(th)
    for m in (full, sup):
        for o in m.observations:
            y = np.asarray(pred[o.name])
            o.flux = jnp.asarray(y * 1.05)
            o.uncertainty = jnp.asarray(0.1 * np.abs(y) + 1e-35)
    vals = []
    for m in (full, sup):
        obs = m.obs_dict
        keys = tuple(obs)
        lnp = MultiObservationLikelihood(
            keys=keys, likelihoods=tuple(_likelihood_for(obs[k], m.param_names) for k in keys)
        ).make_lnprobfn(obs, m, m)
        vals.append(jax.jit(jax.value_and_grad(lnp))(th))
    (v0, g0), (v1, g1) = vals
    bound = sum(float(np.sum(np.abs(np.asarray(o.flux) - np.asarray(pred[o.name]))
                             * RTOL * np.abs(np.asarray(pred[o.name]))
                             / np.asarray(o.uncertainty) ** 2)) for o in full.observations)
    assert np.isfinite(float(v1)) and abs(float(v1) - float(v0)) <= bound, (v1 - v0, bound)
    for k in g0:
        a, b = np.asarray(g0[k]), np.asarray(g1[k])
        assert np.all(np.isfinite(b)), k
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-6 * np.max(np.abs(a)), err_msg=k)


def test_afe_support_equals_whole_grid():
    path = find_named_grid("amist_c3k_hr_krou_afe.h5")
    if path is None:
        pytest.skip("amist_c3k_hr_krou_afe.h5 not present")
    from ceridwen.csp import CSPBasis_afe
    from ceridwen.ssps import SSPDataAfe
    ssp = SSPDataAfe.load(str(path))
    cosmo = Cosmology.planck18()
    ms = []
    for flag in (False, True):
        csp = CSPBasis_afe(ssp, lookback_time=jnp.linspace(0.0, float(cosmo.age(1.0)), 5),
                           zh_const=True, add_igm=True, verbose=False, cosmo=cosmo,
                           wavelength_support=flag)
        lam = np.linspace(8000.0, 9500.0, 400)
        ms.append(SedModel(csp, [Photometry(filters=FILTERS[:3], flux=[1e-9] * 3,
                                            uncertainty=[1e-10] * 3, name="phot"),
                                 Spectrum(wavelength=lam, flux=np.full(lam.size, 1e-30),
                                          uncertainty=np.full(lam.size, 1e-31),
                                          instrument=Instrument.R_fwhm(3000.0), name="spec")],
                           priors={}, zred=1.0, kinematics=Kinematics(sigma_gal=200.0),
                           free_param_init={"afe": jnp.array([0.0])}))
    full, sup = ms
    assert sup.csp._support_for(sup.observations, {"zred": jnp.array([1.0])}) is not None
    for afe in (None, 0.13):
        th = _theta(full)
        if afe is not None:
            th["afe"] = jnp.array([afe])
        _close(jax.jit(full.predict)(th), jax.jit(sup.predict)(th))


def test_result_file_records_wavelength_support_only_when_on():
    """csp_config names the opt-in, so a rebuilt model with the other setting is a difference;
    with it off the record is the one written before the option existed."""
    import json
    from ceridwen.resultfile import _model_record
    on, off = (_model(s, neb=False) for s in (True, False))
    cfg_on, cfg_off = (json.loads(_model_record(m)["/model@csp_config"]) for m in (on, off))
    assert "wavelength_support" not in cfg_off
    assert cfg_on.pop("wavelength_support") is True and cfg_on == cfg_off


# CSPBasis_afe's SFH-basis fast path (fork 578b310, 37f04b0) on the support path: a synthetic
# alpha grid, two Photometry observations, no IGM.  The fast path sums over SFH bins, not over
# wavelength, so the support predictions equal the whole grid bit for bit.  The SFH basis table
# keeps each nonzero ``(group, bin)`` spectrum once and one shared zero row.
from types import SimpleNamespace  # noqa: E402

from ceridwen.csp.csp_afe import CSPBasis_afe  # noqa: E402

AFE_LOOKBACK_GYR = jnp.array([0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 5.0, 8.4])
AFE_WAVE = np.geomspace(900.0, 60000.0, 600).astype(np.float32)
AFE_ZRED = 0.6


@pytest.fixture(scope="module")
def afe_ssp():
    rng = np.random.default_rng(20261001)
    return SimpleNamespace(
        ssp_flux=rng.uniform(0.1, 2.0, size=(5, 13, 107, AFE_WAVE.size)).astype(np.float32),
        ssp_afe=np.linspace(-0.2, 0.6, 5, dtype=np.float32),
        ssp_wave=AFE_WAVE,
        ssp_lg_age_gyr=np.linspace(-3.0, 1.14, 107, dtype=np.float32),
        ssp_lgmet=np.linspace(-2.0, 0.4, 13, dtype=np.float32),
        log10_zsun=0.0,
        ssp_resolution=None,
        isoc_type=None,
        spec_library=None,
    )


def _afe_build(ssp, zh_const, support):
    metallicity = ({"logzsol": jnp.array([-0.8])} if zh_const
                   else {"logzsol_hist": jnp.full(7, -0.8)})
    csp = CSPBasis_afe(
        ssp,
        theta={"lookback_time": AFE_LOOKBACK_GYR, "sfh": jnp.ones(8),
               "afe": jnp.array([0.2]), **metallicity},
        zh_const=zh_const,
        sfh_interp="step",
        add_dust=True,
        add_diffuse_dust=True,
        add_dust_emission=False,
        cosmo=Cosmology.planck18(),
        verbose=False,
        wavelength_support=support,
    )
    assert csp.sfh_basis_fastpath
    blue = Photometry(filters=["hsc_g"], name="blue")
    red = Photometry(filters=["hsc_i"], name="red")
    for phot in (blue, red):
        phot.setup_for_model(csp.wave, zred=AFE_ZRED)
    return csp, [blue, red]


def _afe_thetas(csp, zh_const, n):
    rng = np.random.default_rng(7)
    base = {k: jnp.asarray(v) for k, v in csp.all_params.items()}
    base.update({k: jnp.full_like(jnp.asarray(v, dtype=float), 0.7)
                 for k, v in csp.dust_attn.get_default_fit_params().items()})
    base.update(logmass=jnp.array([10.5]), zred=jnp.array([AFE_ZRED]))
    draws = []
    for _ in range(n):
        theta = dict(base, sfh=jnp.asarray(rng.uniform(0.1, 2.0, 8)),
                     afe=jnp.asarray(rng.uniform(-0.2, 0.6, 1)))
        if zh_const:
            theta["logzsol"] = jnp.asarray(rng.uniform(-1.9, 0.3, 1))
        else:
            theta["logzsol_hist"] = jnp.asarray(rng.uniform(-1.9, 0.3, 7))
        draws.append(theta)
    return draws


@pytest.mark.parametrize("zh_const", [True, False], ids=["constant-Z", "per-bin-Z"])
def test_sfh_basis_support_is_bitwise_the_whole_grid(afe_ssp, zh_const):
    (full, full_obs), (sup, sup_obs) = (_afe_build(afe_ssp, zh_const, s) for s in (False, True))
    draws = _afe_thetas(sup, zh_const, 3)
    support = sup._support_for(sup_obs, draws[0])
    assert support is not None and 0 < support.start < support.stop < AFE_WAVE.size
    assert full._support_for(full_obs, draws[0]) is None
    batch = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *draws)
    got_one = [jax.jit(lambda t: sup.predict(t, sup_obs))(t) for t in draws]
    got_many = jax.jit(jax.vmap(lambda t: sup.predict(t, sup_obs)))(batch)
    want_many = jax.jit(jax.vmap(lambda t: full.predict(t, full_obs)))(batch)
    for got, t in zip(got_one, draws):
        want = jax.jit(lambda t: full.predict(t, full_obs))(t)
        for name in want:
            assert np.asarray(got[name]).tobytes() == np.asarray(want[name]).tobytes()
            assert np.all(np.asarray(want[name]) != 0)
    for name in want_many:
        assert np.asarray(got_many[name]).tobytes() == np.asarray(want_many[name]).tobytes()


def test_basis_table_rows_rebuild_the_basis(afe_ssp):
    csp, _ = _afe_build(afe_ssp, zh_const=False, support=False)
    basis = np.asarray(csp._sfh_basis)
    empty = ~np.any(basis != 0, axis=(0, 1, 4))
    assert empty.any()
    for support in (None, slice(40, 500)):
        table, rows = csp._sfh_basis_table(support)
        assert table.shape[2] == (~empty).sum() + 1
        want = basis if support is None else basis[..., support]
        assert np.asarray(table)[:, :, rows].tobytes() == want.tobytes()
