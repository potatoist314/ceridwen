"""C/N differential spectra from alf, transferred at the observed resolution.

alf revision cda32111c721efe586945d7dfcb9f605a9f6a105:
doc/alf.pdf §§1.4.1, 1.8; src/setup.f90; src/add_response.f90.
Conroy & van Dokkum (2012), Conroy et al. (2018).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d

from ceridwen.broadening import Instrument, Kinematics, SpectralProjector


ALF_REVISION = "cda32111c721efe586945d7dfcb9f605a9f6a105"


class AlfCNResponse:
    """Multiplicative C/N offsets in dex, with isochrones unchanged.

    Each alf SSP has five ages (1, 3, 5, 9, 13 Gyr) and four usable solar-scaled
    metallicities (-1.5, -1, -0.5, 0 dex). Interpolate fluxes in log age and [Fe/H],
    clamping to the endpoints. As alf does, do not use its broken +0.2 files.
    Integrate these theoretical SSPs over the Park model's SFH and metallicity
    history. Their enhanced/base ratio modifies the Park continuum.

    Spectrum response templates are broadened once at the catalogue velocity
    dispersion plus the observed instrument LSF. The Park spectrum retains its
    sampled dispersion. Birth-cloud attenuation weights the same age groups as
    the CSP, after template broadening. Diffuse attenuation cancels in the ratio.
    Photometry uses the unsmoothed template ratio on the Park wavelength grid.
    Outside alf's wavelength coverage the response is unity.
    """

    def __init__(self, wave, templates, *, sigma_catalog, source=None):
        self.wave = np.asarray(wave, dtype=float)
        # (9 polynomial terms, 4 metallicities, 5 ages, wavelength), in F_nu.
        self.templates = np.asarray(templates, dtype=np.float64)
        self.sigma_catalog = float(sigma_catalog)
        self.source = source
        self.logages = np.log10([1., 3., 5., 9., 13.])
        self.metallicities = np.array([-1.5, -1., -.5, 0.])
        self.tables = {}
        self.minimum_total_sigma = {}

    @classmethod
    def load(cls, directory, *, sigma_catalog):
        """Read unchanged Kroupa tables. Wavelengths are vacuum Angstroms."""
        directory = Path(directory)
        rows = []
        wave = None
        for metallicity in ("m1.5", "m1.0", "m0.5", "p0.0"):
            ages = []
            for age in ("01", "03", "05", "09", "13"):
                raw = np.loadtxt(directory / f"atlas_ssp_t{age}_Z{metallicity}.abund.krpa.s100")
                if wave is None:
                    wave = raw[:, 0]
                # Published files differ by up to 0.051 A in their rounded NIR
                # wavelength columns. Align fluxes, leaving every raw file unchanged.
                flux = interp1d(raw[:, 0], raw[:, 1:], axis=0, bounds_error=False,
                                fill_value=(raw[0, 1:], raw[-1, 1:]))(wave)
                raw = np.column_stack((wave, flux))
                solar = raw[:, 1]
                # Zero-based columns 8/9 are C +/-0.15 dex; 11/12 are N +/-0.3.
                dc = raw[:, [8, 9]].T / solar - 1.
                dn = raw[:, [11, 12]].T / solar - 1.
                terms = np.stack([solar, solar * dc[0], solar * dc[1],
                                  solar * dn[0], solar * dn[1],
                                  solar * dc[0] * dn[0], solar * dc[0] * dn[1],
                                  solar * dc[1] * dn[0], solar * dc[1] * dn[1]])
                # alf L_lambda [L_sun/um] -> proportional L_nu. The common
                # dimensional constant cancels, but lambda^2 must precede smoothing.
                ages.append(terms * wave[None, :] ** 2)
            rows.append(np.stack(ages, axis=1))
        templates = np.stack(rows, axis=1)
        return cls(wave, templates, sigma_catalog=sigma_catalog,
                   source=str(directory))

    def configure(self, csp):
        if not csp.sfh_basis_fastpath:
            raise ValueError("C/N responses require the fixed-grid step-SFH basis")
        # Map Park SSP age weights onto alf's log-age interpolation knots.
        age_weights = interp1d(self.logages, np.eye(5), axis=0,
                               bounds_error=False,
                               fill_value=(np.eye(5)[0], np.eye(5)[-1]))(
                                   np.asarray(csp.ages))
        if csp._has_age_dependent_dust:
            groups = np.asarray(csp._age_bin_mix)
            _, labels = np.unique(groups, axis=0, return_inverse=True)
        else:
            labels = np.zeros(len(csp.ages), dtype=int)
        membership = labels[None, :] == np.arange(len(csp._dust_group_mix))[:, None]
        self.bin_to_response_age = np.einsum(
            "ga,na,at->gnt", membership,
            np.asarray(csp._sfh_bin_to_age, dtype=float), age_weights)
        return self

    def _table(self, templates):
        # The contraction is static. The sampler still sums SFH bins, not SSP ages.
        return jnp.asarray(np.einsum("gnt,rztw->rzgnw",
                                    self.bin_to_response_age, templates), dtype=jnp.float32)

    def projected_templates(self, observation, zred, sigma):
        """Broaden base and differential spectra with the same linear projector."""
        wave_obs = np.asarray(observation.wavelength)
        inst = (np.zeros_like(wave_obs) if observation.instrument is None else
                observation.instrument.sigma_kms_at(wave_obs))
        total = np.hypot(sigma, inst)
        if np.min(total) < 100.:
            raise ValueError(f"alf response needs total sigma >=100 km/s; got {total.min():.3f}")
        # Put the combined *additional* width in the instrument. In particular,
        # do not let a projector clamp (sigma_inst^2 - 100^2) before adding sigma_gal.
        extra = np.sqrt(total ** 2 - 100. ** 2)
        # LEGA-C includes UV pixels below the response library. Extend its base
        # continuum and set differential templates to zero outside the source.
        # The final ratio is also explicitly unity outside that coverage.
        margin = np.exp(32. * np.max(extra) / 2.998e5)
        low = min(wave_obs.min() / (1. + zred) / margin, self.wave[0] * .99)
        high = max(wave_obs.max() / (1. + zred) * margin, self.wave[-1] * 1.01)
        wave_model = np.r_[low, self.wave[0] - np.diff(self.wave)[0], self.wave,
                           self.wave[-1] + np.diff(self.wave)[-1], high]
        templates = np.pad(self.templates, ((0, 0), (0, 0), (0, 0), (2, 2)))
        templates[0, :, :, :2] = self.templates[0, :, :, :1]
        templates[0, :, :, -2:] = self.templates[0, :, :, -1:]
        projector = SpectralProjector.build(
            Kinematics(sigma_gal=0., sigma_max=1.), Instrument.sigma_kms(extra, wave=wave_obs),
            wave_model, wave_obs, zred, subtract_library=False)
        shape = templates.shape
        project = jax.jit(jax.vmap(lambda spectrum: projector.predict(spectrum, None, {})))
        flat = templates.reshape(-1, shape[-1])
        # Bound the setup-time FFT batch memory independently of the source grid.
        projected = np.concatenate([np.asarray(project(jnp.asarray(batch)))
                                    for batch in np.array_split(flat, 12)])
        return projected.reshape(*shape[:-1], wave_obs.size), float(total.min())

    def setup_observations(self, model):
        """Prepare fixed-resolution ratios without changing the Park projectors."""
        if model.zred_is_free:
            raise ValueError("C/N response transfer currently requires a fixed catalogue redshift")
        csp = model.csp
        grid = interp1d(self.wave, self.templates, axis=-1, bounds_error=False,
                        fill_value=0.)(np.asarray(csp.wave))
        self.tables = {"grid": (self._table(grid), jnp.asarray(csp.wave))}
        self.minimum_total_sigma = {}
        for obs in model.observations:
            if getattr(obs, "_kind", None) != "spectrum":
                continue
            projected, minimum = self.projected_templates(obs, model.zred, self.sigma_catalog)
            self.tables[obs.name] = (self._table(projected),
                                    jnp.asarray(obs.wavelength) / (1. + model.zred))
            self.minimum_total_sigma[obs.name] = minimum

    def factor(self, csp, theta, key):
        """Enhanced/base ratio for the current SFH and metallicity history."""
        table, wave = self.tables[key]
        sfh = jnp.clip(theta["sfh"], 1e-30, None).astype(jnp.float32)
        sfh_bin = sfh if csp.sfh_per_bin else .5 * (sfh[:-1] + sfh[1:])
        if csp.zh_const:
            metallicity = jnp.broadcast_to(jnp.ravel(theta["logzsol"])[0], sfh_bin.shape)
        else:
            history = theta["logzsol_hist"]
            metallicity = history if csp.zh_per_bin else .5 * (history[:-1] + history[1:])
        knots = jnp.asarray(self.metallicities)
        hi = jnp.clip(jnp.searchsorted(knots, metallicity), 1, len(knots) - 1)
        lo = hi - 1
        weight = jnp.clip((metallicity - knots[lo]) / (knots[hi] - knots[lo]), 0., 1.)
        bins = jnp.arange(sfh_bin.size)
        lower = table[:, lo, :, bins, :].transpose(1, 2, 0, 3)
        upper = table[:, hi, :, bins, :].transpose(1, 2, 0, 3)
        basis = lower * (1. - weight[None, None, :, None]) + upper * weight[None, None, :, None]
        spectra = jnp.einsum("n,rgnw->rgw", sfh_bin, basis)
        if csp._has_age_dependent_dust:
            attn, _ = csp.attenuate_dust(wave.astype(jnp.float32), theta)
            spectra *= csp._dust_group_attenuation(attn, theta)[None, :, :]
        spectra = spectra.sum(axis=1)
        c = jnp.ravel(theta["cfe"])[0]
        n = jnp.ravel(theta["nfe"])[0]
        ci = jnp.where(c >= 0., 1, 2)
        ni = jnp.where(n >= 0., 3, 4)
        cross = 5 + (ci - 1) * 2 + ni - 3
        ca, na = jnp.abs(c) / .15, jnp.abs(n) / .3
        delta = ca * spectra[ci] + na * spectra[ni] + ca * na * spectra[cross]
        base = spectra[0]
        inside = (wave >= self.wave[0]) & (wave <= self.wave[-1])
        return jnp.where((base > 0.) & inside,
                         1. + delta / jnp.where(base > 0., base, 1.), 1.)
