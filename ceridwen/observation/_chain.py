"""
The LOSVD + LSF projection chain of ``Spectrum(projector="chain")``.

This is the projection of the speedups branch (2d5edc9,
``observation/spectrum.py``), kept as the production path of the LEGA-C fits:
the model is trimmed to the observed window, broadened by the galaxy LOSVD
(FFT on sedpy's log grid), then by the instrument minus the SSP library width
(CDF-transform smoother) onto the observed pixels.  With every width fixed at
setup the two stages collapse into one Gaussian (:mod:`._smoothing`).

``Spectrum(projector="fft")`` (the default) is ``broadening.SpectralProjector``.
The chain reads the same ``Instrument`` and ``Kinematics``; it does not paint
nebular lines, apply an IGM transmission after the LOSVD, sample the redshift
or the instrument scale, and ``Spectrum`` refuses those settings with it.

Needs ``sedpy_jax`` (its log/linear/LSF grid helpers, via :mod:`._smoothing`).
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from . import _smoothing
from ._smoothing import _CKMS, combined_sigma_lambda, make_static_smoother

# FWHM = 2 sqrt(2 ln 2) sigma for a Gaussian kernel.
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # 1/2.3548

__all__ = ["ChainProjector", "chain_resolution", "resolution_as_sigma"]


def chain_resolution(instrument, wavelength):
    """``(smoothtype, resolution, res_convention)`` of the chain for ``instrument``.

    R_fwhm / R_sigma -> "R" (scalar R); sigma_kms / fwhm_kms -> "vel" (scalar);
    sigma_aa / fwhm_aa -> "lambda" (scalar) or "lsf" (tabulated, read at the
    observed pixels ``wavelength``).  ``None`` -> no instrument.
    """
    if instrument is None:
        return None, None, None
    if isinstance(instrument.scale, str) or instrument.scale != 1.0:
        raise ValueError("Spectrum(projector='chain') has no instrument scale; "
                         "use projector='fft' for Instrument(scale=...)")
    kind = instrument.kind
    convention = "fwhm" if "fwhm" in kind else "sigma"
    value = instrument.value
    if value.ndim == 1:
        if kind not in ("sigma_aa", "fwhm_aa"):
            raise ValueError("Spectrum(projector='chain') takes a tabulated instrument width "
                             "in Angstrom only (Instrument.sigma_aa / fwhm_aa); use "
                             "projector='fft'")
        return "lsf", np.interp(np.asarray(wavelength, dtype=np.float64),
                                instrument.wave, value), convention
    smoothtype = {"R_fwhm": "R", "R_sigma": "R", "sigma_kms": "vel", "fwhm_kms": "vel",
                  "sigma_aa": "lambda", "fwhm_aa": "lambda"}[kind]
    return smoothtype, float(value), convention


def resolution_as_sigma(smoothtype, resolution, res_convention):
    """``resolution`` as a Gaussian sigma: sigma_v [km/s] for "vel" / "R",
    sigma_lambda [A] for "lambda" (scalar) and "lsf" (array)."""
    f = (_FWHM_TO_SIGMA if res_convention == "fwhm" else 1.0)
    if smoothtype == "vel":
        return float(resolution) * f
    if smoothtype == "R":
        return float(_CKMS / resolution) * f
    if smoothtype == "lambda":
        return float(resolution) * f
    return np.asarray(resolution, dtype=np.float64) * f  # "lsf"


def _warn_floor(name, target, lib, units, wave=None):
    """Setup-time diagnostics for the library-resolution subtraction.

    Warns (once, Python-level) when the quadrature floor engages -- the
    library is coarser than the requested resolution, so the model CANNOT be
    sharpened to the target there -- and when the library curve is unknown
    (NaN) over part of the fitted window, so nothing is subtracted there.
    """
    target = np.asarray(target, dtype=np.float64)
    lib    = np.asarray(lib, dtype=np.float64)
    fin    = np.isfinite(lib)
    if fin.any():
        bad = fin & (lib >= target)
        if bad.any():
            frac = 100.0 * bad.sum() / bad.size
            rng = ("" if wave is None else
                   f" over [{np.asarray(wave)[bad].min():.0f}, "
                   f"{np.asarray(wave)[bad].max():.0f}] AA")
            warnings.warn(
                f"Spectrum {name!r}: the SSP library resolution "
                f"meets or exceeds the target resolution on "
                f"{frac:.0f}% of pixels{rng} ({units}); those pixels "
                f"stay at LIBRARY resolution (no deconvolution is "
                f"attempted).  The model cannot be as sharp as the "
                f"data there -- use a higher-resolution SSP grid.",
                stacklevel=4)
    if (~fin).any():
        frac = 100.0 * (~fin).sum() / fin.size
        rng = ("" if wave is None else
               f" (e.g. [{np.asarray(wave)[~fin].min():.0f}, "
               f"{np.asarray(wave)[~fin].max():.0f}] AA)")
        warnings.warn(
            f"Spectrum {name!r}: the library resolution is "
            f"unknown (NaN) on {frac:.0f}% of the fitted pixels{rng}; "
            f"no library subtraction is applied there and the model "
            f"is over-broadened by the (unknown) library width.",
            stacklevel=4)


class ChainProjector:
    """Built once by ``Spectrum.setup_for_model`` (projector="chain").

    ``predict(spectrum, theta)`` returns the model on the observed pixels;
    ``model_support`` is the half-open range of model-grid indices it reads.
    """

    def __init__(self, name, wavelength, instrument, subtract_library, kinematics,
                 wave_model, zred=0.0, lib_resolution=None):
        smoothtype, resolution, res_convention = chain_resolution(instrument, wavelength)
        inres = "auto" if subtract_library else 0.0
        sigma_gal = kinematics.sigma_gal
        # A sampled sigma_gal is read from theta[sigma_key] per call and the
        # LOSVD FFT is padded for kinematics.sigma_max; a fixed one is baked in.
        self.sigma_key = sigma_gal if isinstance(sigma_gal, str) else None
        fit_lo = self.sigma_key is not None
        sigma_losvd = None if fit_lo or float(sigma_gal) == 0.0 else float(sigma_gal)
        sigma_smooth_max = float(kinematics.sigma_max) if fit_lo else None
        self.fit_lo = fit_lo

        wm_rest = np.asarray(wave_model, dtype=np.float64)
        opz = 1.0 + float(zred)
        wm = opz * wm_rest
        wo = np.asarray(wavelength, dtype=np.float64)
        n_wave = len(wm)
        n_pix  = len(wo)

        # ── Record the interpolation matrix _H, but do not build it ───────
        j_hi = np.searchsorted(wm, wo, side='right')
        j_hi = np.clip(j_hi, 1, n_wave - 1)
        j_lo = j_hi - 1

        dw    = wm[j_hi] - wm[j_lo]
        alpha = np.where(dw > 0, (wo - wm[j_lo]) / dw, 0.0)
        alpha = np.clip(alpha, 0.0, 1.0)

        self._H_factors = (j_lo, j_hi, alpha, n_pix, n_wave)
        self._H_cached = None

        make_vel_smoother  = _smoothing.make_vel_smoother
        make_wave_smoother = _smoothing.make_wave_smoother
        make_lsf_smoother  = _smoothing.make_lsf_smoother

        # ── Build _predict_fn ──────────────────────────────────────────────
        st         = smoothtype
        has_instr  = st is not None
        has_losvd  = fit_lo or sigma_losvd is not None

        if not has_instr and not has_losvd:
            # Preserve the dense matrix's independently rounded coefficients.
            lo, hi = jnp.asarray(j_lo), jnp.asarray(j_hi)
            w_lo = jnp.asarray((1.0 - alpha).astype(np.float32))
            w_hi = jnp.asarray(alpha.astype(np.float32))
            self._predict_fn = lambda spec: w_lo * spec[lo] + w_hi * spec[hi]
            self.model_support = (int(j_lo.min()), int(j_hi.max()) + 1)
            return

        # ── Trim model grid to observed wavelength range ────────────────
        # The FFT grids span the model grid they are given; restricting it to
        # the observed window (plus a buffer for the kernel wings) keeps the
        # FFT pixel small enough to resolve line profiles.
        _buf     = max(500.0, 0.15 * float(wo.max() - wo.min()))
        _trim    = ((wm >= float(wo.min()) - _buf) &
                    (wm <= float(wo.max()) + _buf))
        _wm_trim = wm[_trim]
        _idx = jnp.array(np.where(_trim)[0])
        self.model_support = (int(_idx.min()), int(_idx.max()) + 1)

        # ── Optional LOSVD pre-smoothing stage (sampled sigma_gal) ──────
        if fit_lo:
            _pad = {"sigma_max": sigma_smooth_max}
            # Output stays on the trimmed model grid when an instrumental
            # stage follows, else goes straight to the observed pixels.
            _losvd_sm = make_vel_smoother(
                _wm_trim, _wm_trim if has_instr else wo, inres=0.0, **_pad)

            def _apply_losvd_rt(spec_trim, sigma_v, _s=_losvd_sm):
                return _s(spec_trim, sigma_v)

        if has_instr:
            # ── Library (input) resolution on the trimmed model grid ──
            # sigma_v is frame-invariant, so the rest-frame curve is
            # evaluated at lambda_rest = lambda_obs/(1+z) and used unchanged.
            # NaN = "unknown here" -> nothing subtracted at that pixel.
            auto = (inres == "auto")
            if auto and lib_resolution is None:
                raise ValueError(
                    f"Spectrum {name!r}: the chain projector needs the SSP library "
                    "resolution to subtract in quadrature, but no lib_resolution "
                    "curve was provided.  Load a schema-2.0 SSP grid (SedModel "
                    "threads its curve in), or pass subtract_library=False.")
            if auto:
                _lw = np.asarray(lib_resolution[0], dtype=np.float64)
                _ls = np.asarray(lib_resolution[1], dtype=np.float64)
                if _lw.shape != _ls.shape:
                    raise ValueError(
                        "lib_resolution wave/sigma shape mismatch: "
                        f"{_lw.shape} vs {_ls.shape}")
                # NaN-aware interpolation onto the trimmed grid: NaN
                # segments stay NaN (np.interp would smear them).
                _rest_trim = _wm_trim / opz
                fin = np.isfinite(_ls)
                sig_v_lib = np.full(_wm_trim.shape, np.nan)
                if fin.any():
                    sig_v_lib = np.where(
                        (_rest_trim >= _lw[fin].min())
                        & (_rest_trim <= _lw[fin].max()),
                        np.interp(_rest_trim, _lw[fin], _ls[fin]),
                        np.nan)
                    # pixels inside coverage but adjacent to NaN gaps:
                    # mark as NaN when the nearest native pixel is NaN
                    near = np.clip(np.searchsorted(_lw, _rest_trim),
                                   0, _lw.size - 1)
                    sig_v_lib = np.where(np.isfinite(_ls[near]),
                                         sig_v_lib, np.nan)
            else:
                sig_v_lib = None      # no library subtraction

            # ── Target width and the library curve, per smoothtype ────
            # Everything reduces to ONE of two smoother kinds: scalar-sigma
            # FFT (shift-invariant target, constant or no library curve) or
            # the CDF-transform LSF smoother (any wavelength dependence).
            def _flat(a):
                f = a[np.isfinite(a)]
                return f.size and (np.ptp(f) < 1e-6) and \
                    np.isfinite(a).all()

            _sig_lam = _wm_trim / _CKMS   # dlambda/dv at each trimmed pixel

            # An all-NaN curve (resolution unknown everywhere) subtracts
            # nothing: take the scalar inres=0 path and warn.
            all_nan = auto and not np.isfinite(sig_v_lib).any()
            if all_nan:
                _warn_floor(name, np.full_like(_wm_trim, np.inf),
                            sig_v_lib, "km/s", _wm_trim)

            if st in ("vel", "R"):
                sigma_v_t = resolution_as_sigma(st, resolution, res_convention)
                if not auto or all_nan:
                    _in = 0.0
                    _sig_tgt_lam = sigma_v_t * _sig_lam
                    _sig_lib_lam = _in * _sig_lam
                    _instr_sm = make_vel_smoother(
                        _wm_trim, wo, inres=_in)
                    _apply_instr = (lambda spec_trim, _sm=_instr_sm,
                                    _sv=sigma_v_t: _sm(spec_trim, _sv))
                    _warn_floor(name, np.full(1, sigma_v_t),
                                np.full(1, _in), "km/s")
                elif _flat(sig_v_lib):
                    _in = float(sig_v_lib[np.isfinite(sig_v_lib)][0])
                    _sig_tgt_lam = sigma_v_t * _sig_lam
                    _sig_lib_lam = _in * _sig_lam
                    _instr_sm = make_vel_smoother(_wm_trim, wo, inres=_in)
                    _apply_instr = (lambda spec_trim, _sm=_instr_sm,
                                    _sv=sigma_v_t: _sm(spec_trim, _sv))
                    _warn_floor(name, np.full(1, sigma_v_t),
                                np.full(1, _in), "km/s")
                else:
                    # wavelength-dependent library curve -> LSF route in
                    # wavelength space: sigma_lambda = lambda sigma_v / c
                    # for both target and library.
                    tgt = sigma_v_t * _sig_lam
                    lib = sig_v_lib * _sig_lam
                    _sig_tgt_lam, _sig_lib_lam = tgt, lib
                    _instr_sm = make_lsf_smoother(
                        _wm_trim, tgt, wo, inres=lib)
                    _apply_instr = (lambda spec_trim, _sm=_instr_sm:
                                    _sm(spec_trim))
                    _warn_floor(name, np.full_like(_wm_trim, sigma_v_t),
                                sig_v_lib, "km/s", _wm_trim)

            elif st == "lambda":
                sigma_l_t = resolution_as_sigma(st, resolution, res_convention)
                if not auto or all_nan:
                    _in = 0.0
                    _sig_tgt_lam = np.full_like(_wm_trim, sigma_l_t)
                    _sig_lib_lam = np.full_like(_wm_trim, _in)
                    _instr_sm = make_wave_smoother(
                        _wm_trim, wo, inres=_in)
                    _apply_instr = (lambda spec_trim, _sm=_instr_sm,
                                    _sl=sigma_l_t: _sm(spec_trim, _sl))
                    _warn_floor(name, np.full(1, sigma_l_t),
                                np.full(1, _in), "AA")
                else:
                    # constant-lambda target, velocity-defined library curve
                    # -> always wavelength-dependent in lambda: LSF route.
                    tgt = np.full_like(_wm_trim, sigma_l_t)
                    lib = sig_v_lib * _sig_lam
                    _sig_tgt_lam, _sig_lib_lam = tgt, lib
                    _instr_sm = make_lsf_smoother(
                        _wm_trim, tgt, wo, inres=lib)
                    _apply_instr = (lambda spec_trim, _sm=_instr_sm:
                                    _sm(spec_trim))
                    _warn_floor(name, tgt, lib, "AA", _wm_trim)

            else:   # st == "lsf"
                # Wavelength-dependent target LSF: sigma(lambda) [A] at the
                # observed pixels -> trimmed model grid.
                res_obs        = resolution_as_sigma(st, resolution, res_convention)
                sigma_lsf_trim = np.interp(_wm_trim, wo, res_obs)
                if not auto or all_nan:
                    lib = 0.0
                    _sig_tgt_lam = sigma_lsf_trim
                    _sig_lib_lam = np.full_like(_wm_trim, lib)
                    _instr_sm = make_lsf_smoother(
                        _wm_trim, sigma_lsf_trim, wo, inres=lib)
                    _warn_floor(name, sigma_lsf_trim,
                                np.full_like(sigma_lsf_trim, lib),
                                "AA", _wm_trim)
                else:
                    lib = sig_v_lib * _sig_lam
                    _sig_tgt_lam, _sig_lib_lam = sigma_lsf_trim, lib
                    _instr_sm = make_lsf_smoother(
                        _wm_trim, sigma_lsf_trim, wo, inres=lib)
                    _warn_floor(name, sigma_lsf_trim, lib, "AA", _wm_trim)
                _apply_instr = (lambda spec_trim, _sm=_instr_sm:
                                _sm(spec_trim))

            # With a sampled LOSVD the total width is traced, so the two
            # stages stay chained.  Otherwise every width is known now and
            # the whole chain collapses into one Gaussian (see _smoothing).
            if fit_lo:
                self._predict_fn = (
                    lambda spec, sigma_lo, _A=_apply_instr, _L=_apply_losvd_rt:
                        _A(_L(spec[_idx], sigma_lo))
                )
            else:
                _S = self._build_static_smoother(
                    _wm_trim, wo, _sig_tgt_lam, _sig_lib_lam, sigma_losvd)
                self._predict_fn = lambda spec, _S=_S: _S(spec[_idx])

        else:
            # LOSVD only (no instrumental smoothing).
            if fit_lo:
                self._predict_fn = (
                    lambda spec, sigma_lo, _L=_apply_losvd_rt:
                        _L(spec[_idx], sigma_lo)
                )
            else:
                _S = self._build_static_smoother(
                    _wm_trim, wo, None, None, sigma_losvd)
                self._predict_fn = lambda spec, _S=_S: _S(spec[_idx])

    def _build_static_smoother(self, wave_trim, wave_out, sigma_target,
                               sigma_library, sigma_losvd):
        """Collapse the whole static smoothing chain into one Gaussian.

        Gaussians compose in quadrature, so a LOSVD followed by an
        instrumental LSF is one Gaussian of the combined width at each
        wavelength.  Records ``smoother_grid_size`` for diagnostics.
        """
        sigma_total = combined_sigma_lambda(
            wave_trim, sigma_target, sigma_losvd, sigma_library)
        smoother = make_static_smoother(wave_trim, sigma_total, wave_out)
        self.smoother_grid_size = smoother.grid_size
        self.smoother_sigma_lambda = sigma_total
        return smoother

    @property
    def H(self):
        """Dense (n_pix, n_wave) linear-interpolation matrix, built on first
        access (diagnostics; prediction uses its two nonzero coefficients per row)."""
        if self._H_cached is None:
            j_lo, j_hi, alpha, n_pix, n_wave = self._H_factors
            H = np.zeros((n_pix, n_wave), dtype=np.float32)
            rows = np.arange(n_pix)
            H[rows, j_lo] += (1.0 - alpha).astype(np.float32)
            H[rows, j_hi] += alpha.astype(np.float32)
            self._H_cached = jnp.array(H)
        return self._H_cached

    def predict(self, spectrum, theta):
        """The model ``spectrum`` (n_wave,) on the observed pixels; a sampled
        sigma_gal is ``theta[sigma_key]``."""
        if self.fit_lo:
            sv = jnp.asarray(theta[self.sigma_key]).reshape(())
            return self._predict_fn(spectrum, sv)
        return self._predict_fn(spectrum)
