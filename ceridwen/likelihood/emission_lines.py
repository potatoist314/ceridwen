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
* every line 3 sigma inside the spectrum, with at least 3 unmasked
  pixels within 2 sigma, is fitted.  Upstream tests this at the ends of
  the redshift prior; here it is tested at the catalogue redshift, because
  the fit's +/- 0.1 prior would drop every line within ~500 A of an edge
  ([O III] 4959, 5007 for M1_210210).

The line fluxes have a flat prior on f >= 0: stellar absorption is in the
stellar model, so a line column may only add light.

Doublets from one upper level have a flux ratio fixed by atomic physics and
share one flux (:data:`TIED_RATIOS`, the ratios of FSPS's Cloudy line table
``$SPS_HOME/nebular/ZAU_ND_mist.lines``, constant to 1e-4 over its 770 grid
points).  Density-sensitive pairs such as [O II] 3726/3729 and
[S II] 6716/6731 stay free.

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

With :meth:`EmissionLineColumns.with_photometry` (fixed redshift only, as
upstream) the same fluxes also enter a Photometry observation, through the
band flux of each line (:attr:`band_matrix`); ``MultiObservationLikelihood``
then solves the spectrum and the photometry together.  A line of 1 A
equivalent width changes the flux of a band 1000 A wide by about 0.1 %.

With the flat prior, ln Z is defined only up to a constant per line: do
not compare the evidence of a fit with lines against one without.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
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
RIDGE = 1e-12
_CGS_PER_MAGGIE = 3631e-23
# (weaker, stronger) FSPS line names -> flux ratio stronger / weaker; each
# pair shares one upper level (1D2), so the ratio is set by the Einstein A
# values.  Values: FSPS ZAU_ND_mist.lines (Cloudy), median over the grid.
# FSPS names that are not real lines.  "[O II] 3867" (3868.16 A) has zero flux
# in every Cloudy model, and NIST ASD lists no forbidden [O II] line between
# 3860 and 3875 A; the forbidden line there is [Ne III] 3869, which stays.
# https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=O+II&low_w=3860&upp_w=3875
EXCLUDED_LINES = ("[O II] 3867",)
TIED_RATIOS = {
    ("[O III] 4959", "[O III] 5007"): 3.010,
    ("[Ne III] 3968", "[Ne III] 3869"): 3.318,
    ("[N II] 6548", "[N II] 6584"): 2.951,
    ("[O I] 6363", "[O I] 6300"): 3.136,
    ("[S III] 9069", "[S III] 9532"): 2.480,
}

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
    pairs : tuple of (int, int)
        Blended pairs: lines whose flux posterior correlation exceeds
        ``blend_correlation``; their ``P(f >= 0)`` is computed jointly.
    tie : ndarray, shape (n_raw, n_line), optional
        Profile of each free flux as a combination of the raw lines: an
        identity column for a free line; for a tied doublet, 1 on the
        stronger line and ``1 / ratio`` on the weaker.  ``None``: identity.
    band_matrix : ndarray, shape (n_band, n_line), optional
        Maggies per unit line flux in each band of ``photometry_key``.
    photometry_key : str, optional
        Key of the Photometry observation that shares the line fluxes.
    ridge : ndarray, shape (n_line,), optional
        Precision added to each line flux: ``1e-12`` times its information
        from the spectrum at ``zred`` (a prior 10^6 times wider than the
        flux error).  It keeps the solve finite where a sampled redshift
        moves a line off the unmasked pixels (zero column).
    """

    wave_obs: np.ndarray
    sigma_inst_kms: np.ndarray
    wave_rest: np.ndarray
    names: tuple
    zred: float
    sigma_gas_kms: float
    zred_key: Optional[str] = None
    sigma_key: Optional[str] = None
    pairs: tuple = ()
    ridge: Optional[np.ndarray] = None
    tie: Optional[np.ndarray] = None
    band_matrix: Optional[np.ndarray] = None
    photometry_key: Optional[str] = None

    @classmethod
    def from_spectrum(
        cls,
        spectrum,
        zred: float,
        *,
        names: Optional[Sequence[str]] = None,
        sps_home: Optional[str] = None,
        blend_correlation: float = 0.5,
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
            Catalogue redshift; line coverage is tested here.
        names : sequence of str, optional
            FSPS names of the candidate lines.  Default: every FSPS line.
        blend_correlation : float
            Lines whose flux posterior correlation (spectrum weights, at
            ``zred``) exceeds this in absolute value form a pair, strongest
            first; a line joins at most one pair.
        """
        wave = np.asarray(spectrum.wavelength, dtype=np.float64)
        used = np.asarray(spectrum.mask, dtype=bool)
        s_inst = _sigma_inst_kms(spectrum, wave)
        s_gas = float(spectrum.sigma_losvd or 0.0)
        table_wave, table_names = read_fsps_line_list(sps_home)
        rows = (range(table_wave.size) if names is None
                else [table_names.index(n) for n in names])
        keep = []
        for r in rows:
            if table_names[r] in EXCLUDED_LINES:
                continue
            lo = table_wave[r] * (1.0 + zred)
            s = np.hypot(s_gas, np.interp(lo, wave, s_inst)) / _CKMS
            if (wave[0] * np.exp(3 * s) < lo < wave[-1] * np.exp(-3 * s)
                    and np.sum(used & (np.abs(np.log(wave / lo)) < 2 * s)) >= 3):
                keep.append(r)
        kept = [table_names[r] for r in keep]
        tie, weaker = np.eye(len(kept)), []
        for (weak, strong), ratio in TIED_RATIOS.items():
            if weak in kept and strong in kept:
                tie[kept.index(weak), kept.index(strong)] = 1.0 / ratio
                weaker.append(kept.index(weak))
        tie = np.delete(tie, weaker, axis=1)
        lines = cls(
            wave_obs=wave, sigma_inst_kms=s_inst,
            wave_rest=table_wave[keep], names=tuple(kept),
            zred=float(zred), sigma_gas_kms=s_gas,
            zred_key="zred" if spectrum.free_z else None,
            sigma_key="sigma_smooth" if spectrum.fit_sigma_smooth else None,
            tie=tie,
        )
        information, diag = lines._check_conditioning(spectrum)
        correlation = np.linalg.inv(information)
        d = np.sqrt(np.diag(correlation))
        correlation = correlation / np.outer(d, d)
        candidates = sorted(
            ((i, j) for i in range(lines.n_line) for j in range(i + 1, lines.n_line)
             if abs(correlation[i, j]) > blend_correlation),
            key=lambda p: -abs(correlation[p]))
        pairs, used = [], set()
        for i, j in candidates:          # strongest first; a line joins one pair only
            if i not in used and j not in used:
                pairs.append((i, j))
                used |= {i, j}
        return replace(lines, pairs=tuple(sorted(pairs)), ridge=RIDGE * diag)

    @property
    def n_line(self) -> int:
        """Number of free fluxes."""
        return int(self.wave_rest.size if self.tie is None else self.tie.shape[1])

    @property
    def free_names(self) -> tuple:
        """Name of each free flux; a tied doublet is named after its stronger
        line, with the weaker in brackets."""
        if self.tie is None:
            return self.names
        out = []
        for c in range(self.tie.shape[1]):
            rows = np.flatnonzero(self.tie[:, c])
            main = rows[np.argmax(self.tie[rows, c])]
            out.append(self.names[main] + "".join(f" (+{self.names[r]})" for r in rows if r != main))
        return tuple(out)

    def with_photometry(self, photometry, key: str) -> "EmissionLineColumns":
        """The same lines, also added to the Photometry ``photometry`` (its
        likelihood key ``key``).  Needs a fixed redshift."""
        if self.zred_key is not None:
            raise ValueError("emission lines in the photometry need a fixed redshift")
        opz = 1.0 + self.zred
        centre = self.wave_rest * opz
        s = np.hypot(self.sigma_gas_kms, np.interp(centre, self.wave_obs, self.sigma_inst_kms)) / _CKMS
        # AB maggies = int f_lambda T lambda dlambda / ab_zero_counts (sedpy's
        # normalisation), integrated over each line profile on its own fine
        # grid: FilterSet.get_sed_maggies samples f_lambda at the filter-grid
        # points, which are coarser than a line.
        x = np.linspace(-8.0, 8.0, 801)
        band = np.zeros((len(photometry.filters), self.wave_rest.size))
        for k, (lam, width) in enumerate(zip(centre, s)):
            wave = lam * np.exp(width * x)
            flam = np.exp(-0.5 * x * x) / (np.sqrt(2.0 * np.pi) * width * wave)
            for b, curve in enumerate(photometry.filterset.filters):
                trans = np.interp(wave, np.asarray(curve.wavelength, dtype=np.float64),
                                  np.asarray(curve.transmission, dtype=np.float64), left=0.0, right=0.0)
                band[b, k] = np.trapezoid(flam * trans * wave, wave) / float(curve.ab_zero_counts)
        tie = np.eye(self.wave_rest.size) if self.tie is None else self.tie
        return replace(self, band_matrix=band @ tie, photometry_key=key)

    def columns(self, params: Optional[dict] = None) -> Array:
        """(n_pix, n_line) F_nu profiles of unit-flux lines [erg s^-1 cm^-2],
        one per free flux."""
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
        profiles = phi * jnp.asarray(self.wave_obs / _C_AA_S)[:, None]
        return profiles if self.tie is None else profiles @ jnp.asarray(self.tie)

    def covers(self, rest_wave: float, tol: float = 2.0) -> bool:
        """Whether a fitted line lies within ``tol`` A of ``rest_wave``."""
        return bool(np.any(np.abs(self.wave_rest - float(rest_wave)) < tol))

    def _check_conditioning(self, spectrum) -> tuple[np.ndarray, np.ndarray]:
        """Refuse near-degenerate line sets (upstream's rule, cond > 1e10);
        return the unit-diagonal information matrix of the line fluxes and
        its diagonal."""
        unc = np.asarray(spectrum.uncertainty, dtype=np.float64)
        used = np.asarray(spectrum.mask, dtype=bool) & np.isfinite(unc) & (unc > 0)
        w = np.where(used, 1.0 / np.where(unc > 0, unc, 1.0) ** 2, 0.0)
        A = np.asarray(self.columns({"zred": self.zred, "sigma_smooth": self.sigma_gas_kms}))
        M = A.T @ (w[:, None] * A)
        d = 1.0 / np.sqrt(np.maximum(np.diag(M), np.finfo(float).tiny))
        C = d[:, None] * M * d[None, :]
        cond = float(np.linalg.cond(C))
        if not np.isfinite(cond) or cond > COND_MAX:
            raise ValueError(
                f"the emission lines are nearly degenerate (condition number {cond:.2e}); "
                "select fewer lines with names=...")
        return C, np.diag(M)

    def __repr__(self) -> str:
        phot = "" if self.photometry_key is None else f", shared with {self.photometry_key!r}"
        return (f"EmissionLineColumns({self.n_line} fluxes, flat prior f >= 0, "
                f"{len(self.pairs)} blended pairs{phot}, "
                f"zred={self.zred_key or self.zred}, "
                f"sigma_gas={self.sigma_key or self.sigma_gas_kms}: {', '.join(self.free_names)})")
