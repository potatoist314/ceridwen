"""
ceridwen.likelihood.emission_lines
==================================

Emission lines as free-amplitude columns of the calibration solve.

The line list, the line profile, the coverage rules and the flat prior on
the line fluxes are those of upstream Ceridwen's emission-line
marginalisation (Espe13/ceridwen ``be852282``,
``ceridwen/likelihood/eline_marginal.py``):

* lines and vacuum rest wavelengths from FSPS
  ``$SPS_HOME/data/emlines_info.dat`` (no nebular grid is needed, so the
  alpha-enhanced basis works);
* each line is a unit-flux Gaussian in ``ln(lambda)`` [erg s^-1 cm^-2,
  observed frame, drawn in F_nu] centred at ``lambda_rest (1 + z)``, with
  width ``sqrt(sigma_gas^2 + sigma_inst^2)``; ``sigma_gas`` is tied to the
  stellar dispersion (upstream's default ``Kinematics(sigma_gas=TIED)``)
  and ``sigma_inst`` is the spectrum's instrumental width at the line;
* every line whose centre stays inside the spectrum over the redshift
  range, 3 sigma from the edges, with at least 3 unmasked pixels within
  2 sigma, is fitted; its flux has a flat prior.

Here the lines join the Chebyshev polynomial in one linear model of the
spectrum,

.. math::

    y = P(a)\\,\\mu + \\sum_k f_k L_k + \\epsilon ,

so :meth:`PolynomialCalibration.calibrate_with_lines` integrates out the
coefficients ``a`` and the line fluxes ``f`` in the same Gaussian integral.
Three approximations are made, each small for the weak lines of quiescent
galaxies:

* The lines are added after the polynomial.  With a free flux this is the
  same as ``P(lambda_k) f_k L_k``, except for the change of ``P`` across one
  line width.
* The noise model's effective sigma (for example the fractional term
  ``f_calib mu``) is evaluated at the stellar model ``mu`` only, as for the
  polynomial alone.  The line flux does not enter the variance.
* The lines are not added to the photometry.  A line of 1 A equivalent
  width in a filter wider than 1000 A changes the band flux by less than
  0.1 %, well under a 5 % photometric error floor.

With the flat prior, ln Z is defined only up to a constant per line: do
not compare the evidence of a fit with lines against one without.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
_CKMS = 2.99792458e5
_C_AA_S = 2.99792458e18
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
COND_MAX = 1e10

__all__ = ["EmissionLineColumns", "read_fsps_line_list"]


def read_fsps_line_list(sps_home: Optional[str] = None) -> tuple[np.ndarray, list[str]]:
    """Vacuum rest wavelengths [A] and names of FSPS's emission lines."""
    sps_home = sps_home or os.environ.get("SPS_HOME")
    path = None if not sps_home else Path(sps_home) / "data" / "emlines_info.dat"
    if path is None or not path.is_file():
        raise ValueError(
            "emission lines need FSPS's $SPS_HOME/data/emlines_info.dat, which "
            + (f"was not found at {path}" if path else "needs $SPS_HOME"))
    wave, names = [], []
    for row in path.read_text().splitlines():
        parts = row.split(",")
        if len(parts) >= 2:
            wave.append(float(parts[0]))
            names.append(parts[1].strip())
    return np.asarray(wave, dtype=np.float64), names


def _sigma_inst_kms(spectrum, wave) -> np.ndarray:
    """The spectrum's instrumental Gaussian sigma [km/s] on its pixels."""
    if spectrum.smoothtype is None:
        return np.zeros_like(wave)
    sigma = spectrum._resolution_as_sigma()
    if spectrum.smoothtype in ("vel", "R"):
        return np.full_like(wave, float(sigma))
    return _CKMS * np.broadcast_to(np.asarray(sigma, dtype=np.float64), wave.shape) / wave


@dataclass(frozen=True, eq=False)
class EmissionLineColumns:
    """
    Unit-flux line profiles on the spectrum pixels, one column per line.

    Build with :meth:`from_spectrum` and pass as
    ``DiagonalGaussianLikelihood(calibration=..., emission_lines=...)``.
    :meth:`columns` evaluates the profiles at the sampled redshift and
    dispersion of ``theta``.

    Attributes
    ----------
    wave_obs : ndarray, shape (n_pix,)
        Observed vacuum pixel wavelengths [A].
    sigma_inst_kms : ndarray, shape (n_pix,)
        Instrumental sigma on the pixels [km/s].
    wave_rest : ndarray, shape (n_line,)
        Vacuum rest wavelengths of the fitted lines [A].
    names : tuple of str
        FSPS names of the fitted lines.
    zred, sigma_gas_kms : float
        Values used when ``zred_key`` / ``sigma_key`` is None.
    zred_key, sigma_key : str or None
        ``theta`` keys of a sampled redshift / line dispersion.
    """

    wave_obs: np.ndarray
    sigma_inst_kms: np.ndarray
    wave_rest: np.ndarray
    names: tuple
    zred: float
    sigma_gas_kms: float
    zred_key: Optional[str] = None
    sigma_key: Optional[str] = None

    @classmethod
    def from_spectrum(
        cls,
        spectrum,
        zred: float,
        *,
        zred_range: Optional[tuple[float, float]] = None,
        names: Optional[Sequence[str]] = None,
        sps_home: Optional[str] = None,
    ) -> "EmissionLineColumns":
        """
        Select the lines the spectrum constrains, with upstream's rules.

        Parameters
        ----------
        spectrum : Spectrum
            Its wavelength grid, mask and instrumental resolution are used.
            A sampled redshift (``free_z``) and dispersion
            (``fit_sigma_smooth``) are read from ``theta["zred"]`` and
            ``theta["sigma_smooth"]``; otherwise ``zred`` and
            ``spectrum.sigma_losvd`` are fixed.
        zred : float
            Redshift of the model (the start value when it is sampled).
        zred_range : (float, float), optional
            Redshift prior range; a line is fitted only if it is covered at
            both ends and at ``zred``.
        names : sequence of str, optional
            FSPS names of the candidate lines.  Default: every FSPS line.
        """
        wave = np.asarray(spectrum.wavelength, dtype=np.float64)
        used = np.asarray(spectrum.mask, dtype=bool)
        s_inst = _sigma_inst_kms(spectrum, wave)
        s_gas = float(spectrum.sigma_losvd or 0.0)
        table_wave, table_names = read_fsps_line_list(sps_home)
        rows = (range(table_wave.size) if names is None
                else [table_names.index(n) for n in names])
        zs = [zred] if zred_range is None else [zred_range[0], zred, zred_range[1]]
        keep = []
        for r in rows:
            fits = True
            for z in zs:
                lo = table_wave[r] * (1.0 + z)
                s = np.hypot(s_gas, np.interp(lo, wave, s_inst)) / _CKMS
                if not (wave[0] * np.exp(3 * s) < lo < wave[-1] * np.exp(-3 * s)
                        and np.sum(used & (np.abs(np.log(wave / lo)) < 2 * s)) >= 3):
                    fits = False
                    break
            if fits:
                keep.append(r)
        lines = cls(
            wave_obs=wave, sigma_inst_kms=s_inst,
            wave_rest=table_wave[keep], names=tuple(table_names[r] for r in keep),
            zred=float(zred), sigma_gas_kms=s_gas,
            zred_key="zred" if spectrum.free_z else None,
            sigma_key="sigma_smooth" if spectrum.fit_sigma_smooth else None,
        )
        lines._check_conditioning(spectrum)
        return lines

    @property
    def n_line(self) -> int:
        return int(self.wave_rest.size)

    def columns(self, params: Optional[dict] = None) -> Array:
        """(n_pix, n_line) F_nu profiles of unit-flux lines [erg s^-1 cm^-2]."""
        def value(key, fixed):
            if key is None:
                return jnp.asarray(fixed)
            return jnp.ravel(jnp.asarray(params[key]))[0]
        opz = 1.0 + value(self.zred_key, self.zred)
        s_gas = value(self.sigma_key, self.sigma_gas_kms)
        centre = jnp.asarray(self.wave_rest) * opz
        s_inst = jnp.interp(centre, jnp.asarray(self.wave_obs),
                            jnp.asarray(self.sigma_inst_kms))
        s = jnp.sqrt(s_gas ** 2 + s_inst ** 2) / _CKMS
        x = (jnp.asarray(np.log(self.wave_obs))[:, None] - jnp.log(centre)[None, :]) / s[None, :]
        phi = jnp.exp(-0.5 * x * x) / (jnp.sqrt(2.0 * jnp.pi) * s[None, :])
        return phi * jnp.asarray(self.wave_obs / _C_AA_S)[:, None]

    def covers(self, rest_wave: float, tol: float = 2.0) -> bool:
        """Whether a fitted line lies within ``tol`` A of ``rest_wave``."""
        return bool(np.any(np.abs(self.wave_rest - float(rest_wave)) < tol))

    def _check_conditioning(self, spectrum) -> None:
        """Refuse near-degenerate line sets (upstream's rule, cond > 1e10)."""
        unc = np.asarray(spectrum.uncertainty, dtype=np.float64)
        used = np.asarray(spectrum.mask, dtype=bool) & np.isfinite(unc) & (unc > 0)
        w = np.where(used, 1.0 / np.where(unc > 0, unc, 1.0) ** 2, 0.0)
        A = np.asarray(self.columns({"zred": self.zred, "sigma_smooth": self.sigma_gas_kms}))
        M = A.T @ (w[:, None] * A)
        d = 1.0 / np.sqrt(np.maximum(np.diag(M), np.finfo(float).tiny))
        cond = float(np.linalg.cond(d[:, None] * M * d[None, :]))
        if not np.isfinite(cond) or cond > COND_MAX:
            raise ValueError(
                f"the emission lines are nearly degenerate (condition number {cond:.2e}); "
                "select fewer lines with names=...")

    def __repr__(self) -> str:
        return (f"EmissionLineColumns({self.n_line} lines, flat prior, "
                f"zred={self.zred_key or self.zred}, "
                f"sigma_gas={self.sigma_key or self.sigma_gas_kms}: {', '.join(self.names)})")
