"""Parameter transforms for SedModel: callables ``fn(free_theta) -> derived value``
substituted into the CSP theta before ``csp.predict``."""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "logsfr_ratios_to_sfh",
    "mass_mapped_beta",
    "mass_mapped_zh",
    "sfh_to_logsfr_ratios",
]


def logsfr_ratios_to_sfh(
    logsfr_ratios,
    sfh_times_yr=None,
):
    """Unit-mass SFH weight vector (n,) from ``logsfr_ratios[i] = log10(SFR[i]/SFR[i+1])``
    (n-1,), index 0 = today. With ``sfh_times_yr`` (lookback yr, (n,)) the trapezoidal
    integral is normalised to 1 Msun; otherwise sum(sfh) = 1."""
    ratios  = jnp.asarray(logsfr_ratios, dtype=float)
    log_sfr = jnp.concatenate([jnp.zeros(1),
                                -jnp.cumsum(ratios)])
    sfr     = 10.0 ** log_sfr

    if sfh_times_yr is not None:
        times = jnp.asarray(sfh_times_yr, dtype=float)
        dt    = jnp.abs(jnp.diff(times))
        w_lo  = jnp.concatenate([jnp.zeros(1), dt])
        w_hi  = jnp.concatenate([dt, jnp.zeros(1)])
        w     = 0.5 * (w_lo + w_hi)
        # normalise to total mass (1 Msun), NOT to mean SFR
        total_mass = jnp.sum(sfr * w)
        sfh   = sfr / total_mass
    else:
        sfh = sfr / jnp.sum(sfr)

    return sfh


def sfh_to_logsfr_ratios(sfh):
    """Invert :func:`logsfr_ratios_to_sfh`: (n-1,) log10 ratios of consecutive bins."""
    sfh     = jnp.asarray(sfh, dtype=float)
    sfh     = jnp.clip(sfh, 1e-30)
    log_sfr = jnp.log10(sfh)
    return log_sfr[:-1] - log_sfr[1:]


def mass_mapped_beta(
    beta_unit,
    logz_mean,
    logz_0,
    logz_max,
    beta_min=0.05,
    beta_max=0.80,
):
    """
    Map a unit-interval coordinate to the enrichment shape ``beta`` of
    :func:`mass_mapped_zh`.

    ``beta = 1 / (1 + alpha)`` runs linearly from ``beta_min`` at
    ``beta_unit = 0`` to an upper limit at ``beta_unit = 1``.  The upper
    limit keeps the final metallicity on the grid::

        beta_hi = min(beta_max, 1 - (<Z> - Z_0) / (Z_max - Z_0)),

    so a uniform prior on ``beta_unit`` is a uniform prior on ``beta`` at
    fixed ``<Z>``.  Gallazzi et al. (2026, A&A, arXiv:2512.07952, Table 1)
    sample ``beta`` uniformly in [0.05, 0.80].  If ``beta_hi < beta_min``
    (``<Z>`` just below ``Z_max``), ``beta = beta_hi`` and ``Z_f = Z_max``.

    Parameters
    ----------
    beta_unit : array_like, shape (1,)
        Coordinate in [0, 1].
    logz_mean, logz_0, logz_max : array_like or float
        log10 of the formed-mass-weighted metallicity ``<Z>``, the initial
        metallicity ``Z_0 <= <Z>`` and the grid top, in ``ssp_lgmet`` units.
    beta_min, beta_max : float
        Prior range of ``beta`` before the grid limit.

    Returns
    -------
    beta : jnp.ndarray, shape (1,)
    """
    z_mean, z_0, z_max = 10.0 ** logz_mean, 10.0 ** logz_0, 10.0 ** logz_max
    upper = jnp.minimum(beta_max, 1.0 - (z_mean - z_0) / (z_max - z_0))
    lower = jnp.minimum(beta_min, upper)
    return lower + beta_unit * (upper - lower)


def mass_mapped_zh(sfh, sfh_times_yr, logz_mean, beta, logz_0, sfh_per_bin=False):
    """
    Per-bin metallicity of a history that rises with the formed stellar mass.

    Gallazzi et al. (2026, A&A, arXiv:2512.07952, Table 1)::

        Z(m) = Z_f - (Z_f - Z_0) (1 - m)^alpha,

    with ``m`` the formed-mass fraction (0 at the first star, 1 today) and
    ``Z`` linear.  With ``beta = 1 / (1 + alpha)`` the formed-mass-weighted
    metallicity is ``<Z> = Z_f - beta (Z_f - Z_0)``, so
    ``Z_f = Z_0 + (<Z> - Z_0) / (1 - beta)``.  ``beta -> 0`` gives
    ``Z = <Z>`` in every bin.

    Each step-SFH bin gets the formed-mass average of ``Z(m)`` over its
    mass interval, so the bin masses reproduce ``<Z>`` exactly::

        Z_i = Z_f - (Z_f - Z_0) beta (r_old^(1/beta) - r_young^(1/beta)) / w_i,

    with ``w_i`` the bin's formed-mass fraction and ``r = 1 - m`` at its
    older and younger edges.  Bin masses are ``SFR_bin * dt``, as in the
    CSP step weights.

    Parameters
    ----------
    sfh : array_like, shape (n,) or (n - 1,)
        Linear SFR per node or per bin (``sfh_per_bin``), index 0 = today.
    sfh_times_yr : array_like, shape (n,)
        Lookback-time nodes in years (``CSPBasis.sfh_times``).
    logz_mean : array_like, shape (1,)
        log10 ``<Z>`` in ``ssp_lgmet`` units.
    beta : array_like, shape (1,)
        Enrichment shape in (0, 1), e.g. from :func:`mass_mapped_beta`.
    logz_0 : array_like, shape (1,)
        log10 ``Z_0 <= <Z>`` in ``ssp_lgmet`` units.
    sfh_per_bin : bool
        ``sfh`` holds one SFR per bin.

    Returns
    -------
    zh : jnp.ndarray, shape (n - 1,)
        log10 metallicity per bin, index 0 = youngest bin; the per-bin
        ``theta["zh"]`` of ``CSPBasis_afe``.
    """
    sfh = jnp.clip(jnp.asarray(sfh, dtype=float), 1e-30, None)
    sfh_bin = sfh if sfh_per_bin else 0.5 * (sfh[:-1] + sfh[1:])
    mass = sfh_bin * jnp.diff(jnp.asarray(sfh_times_yr, dtype=float))
    w = mass / jnp.sum(mass)
    r_old = jnp.cumsum(w)                                  # 1 - m at older edge
    r_young = jnp.concatenate([jnp.zeros(1), r_old[:-1]])  # 1 - m at younger edge

    z_mean, z_0 = 10.0 ** logz_mean, 10.0 ** logz_0
    z_f = z_0 + (z_mean - z_0) / (1.0 - beta)
    power = 1.0 / beta
    # Near-empty bins lose the difference quotient to rounding; they take
    # (1 - m)^alpha at their mass midpoint instead.
    wide = w > 1e-9
    w_safe = jnp.where(wide, w, 1.0)
    shape = jnp.where(
        wide,
        beta * (r_old ** power - r_young ** power) / w_safe,
        (r_young + 0.5 * w) ** (power - 1.0),
    )
    return jnp.log10(z_f - (z_f - z_0) * shape)
