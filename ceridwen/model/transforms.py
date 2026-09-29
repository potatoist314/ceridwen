"""
ceridwen/model/transforms.py
============================
Parameter transform functions for SedModel.

These functions implement the ``transforms`` mechanism that allows fitting
reparameterised quantities instead of the raw CSP parameters.  The
canonical example is fitting ``logsfr_ratios`` — log-ratios of consecutive
SFR bins — instead of the raw SFH weight vector ``sfh``.

Design
------
A *transform* is a callable with signature::

    derived_value = fn(free_theta: dict[str, Array]) -> Array

It receives the *full* free-parameter dict and returns the derived
parameter value that is substituted into the CSP theta before calling
``csp.predict``.

Example usage with SedModel
----------------------------
::

    from ceridwen.model.transforms import logsfr_ratios_to_sfh

    model = SedModel(
        csp,
        observations=[spec],
        priors={"logsfr_ratios": Normal(mean=0.0, sigma=0.3)},
        transforms={
            "sfh": lambda t: logsfr_ratios_to_sfh(
                t["logsfr_ratios"],
                sfh_times_yr=csp.sfh_times,
            )
        },
        free_param_init={
            "logsfr_ratios": jnp.zeros(csp.n_time - 1),
        },
    )

The ``SedModel`` will:

- Remove ``"sfh"`` from its free-parameter list (it becomes derived).
- Add ``"logsfr_ratios"`` to the free-parameter list with the supplied
  initial values.
- At every ``predict(free_theta)`` call, evaluate the transform to obtain
  the CSP-compatible model_theta and pass it to ``csp.predict``.
- Evaluate ``ln_prior`` against ``free_theta`` (i.e., the prior on
  ``logsfr_ratios``, not on ``sfh``).

Parametrisation details
-----------------------
For ``n`` SFH time bins the free parameters are ``n − 1`` log-ratios::

    logsfr_ratios[i] = log10( SFR[i] / SFR[i+1] )

The first SFR bin is anchored (``log10 SFR[0] = 0``), and each subsequent
bin is derived via::

    log10 SFR[i+1] = log10 SFR[i] - logsfr_ratios[i]

The resulting linear SFR values are normalised so that the
mass-weighted mean equals unity using bin-width trapezoidal weights.
This matches the Prospector ``logsfr_ratios`` parametrisation described in
Leja et al. (2019, ApJ, 876, 3).
"""

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
    """
    Convert log-ratios of consecutive SFR bins to a unit-mass SFH weight
    vector.

    Parameters
    ----------
    logsfr_ratios : array_like, shape (n - 1,)
        Log10 ratios of consecutive SFR bins.
        ``logsfr_ratios[i] = log10( SFR[i] / SFR[i+1] )``.

        Under the lookback-time convention (index 0 =
        today, last index = oldest), ``SFR[0]`` is the most-recent SFR
        and ``SFR[i+1]`` is at a slightly older lookback time.  Positive
        ``logsfr_ratios[i]`` therefore mean ``SFR[i] > SFR[i+1]``, i.e.
        the SFR is *higher today than in the past* — a late-assembly
        history; negative values mean an earlier burst with the SFR
        declining toward the present.
    sfh_times_yr : array_like, shape (n,), optional
        Lookback-time grid in years (same as ``CSPBasis.sfh_times``).
        When provided, trapezoidal integration weights are used for
        normalisation so that the integral of the SFH equals 1 Msun.
        If *None*, the discrete sum is used instead.

    Returns
    -------
    sfh : jnp.ndarray, shape (n,)
        Unit-mass SFH weight vector suitable for ``theta["sfh"]``.

    Notes
    -----
    The normalisation enforces ``sum(sfh * w) = 1`` where ``w`` are
    the standard trapezoidal quadrature weights (half-width at
    boundaries), so that the trapezoidal integral of the SFH over the
    lookback-time grid equals **1 Msun**:

    .. math::
        \\int_0^{t_{\\rm univ}} \\mathrm{SFR}(t)\\,dt = 1\\;\\mathrm{M_\\odot}.

    The total stellar mass of the model is then set by
    ``theta["logmass"]`` (applied as a multiplicative ``10**logmass``
    inside ``CSPBasis.predict``), matching the Prospector / FSPS
    convention.

    This is consistent with the per-bin mass integral
    ``m2 = sfh_mid * dt`` computed inside
    ``CSPBasis.calculate_ssp_weights_const_zh_step`` and the
    piecewise-linear integral used by
    ``calculate_ssp_weights_const_zh``: on a shared node grid the
    trapezoid sum here and the midpoint sum there are algebraically
    identical, so no compensating rescale is needed inside the CSP.

    JAX compatibility
    -----------------
    The function is fully JIT-compatible: every operation is a
    ``jnp`` primitive on traced arrays, the ``sfh_times_yr is not
    None`` branch resolves at trace time (it is a Python-level check
    on the closure argument, not a runtime decision on a traced
    value), and there are no data-dependent shapes.
    """
    ratios  = jnp.asarray(logsfr_ratios, dtype=float)            # (n-1,)
    # Anchor log10(SFR[0]) = 0, then cumulate the negative ratios
    log_sfr = jnp.concatenate([jnp.zeros(1),
                                -jnp.cumsum(ratios)])             # (n,)
    sfr     = 10.0 ** log_sfr                                     # (n,)

    if sfh_times_yr is not None:
        times = jnp.asarray(sfh_times_yr, dtype=float)            # (n,)
        dt    = jnp.abs(jnp.diff(times))                          # (n-1,)
        # Standard trapezoidal quadrature weights (yr):
        #   w[0]   = 0.5 * dt[0]
        #   w[i]   = 0.5 * (dt[i-1] + dt[i])   for 0 < i < n-1
        #   w[n-1] = 0.5 * dt[n-2]
        w_lo  = jnp.concatenate([jnp.zeros(1), dt])               # (n,)
        w_hi  = jnp.concatenate([dt, jnp.zeros(1)])               # (n,)
        w     = 0.5 * (w_lo + w_hi)                               # (n,)
        # Unit-mass normalisation: ∫SFR dt = sum(sfr * w) = 1 Msun.
        # GOTCHA: normalise to total mass, NOT mean SFR — dividing by
        # sum(w) instead would leave an implicit factor of t_universe[yr]
        # (~1.4e10) in the spectrum and bias every logmass estimate by
        # ~10 dex at z=0 (less at higher z).
        total_mass = jnp.sum(sfr * w)
        sfh   = sfr / total_mass
    else:
        # Discrete fallback: sum(sfh) = 1.
        sfh = sfr / jnp.sum(sfr)

    return sfh


def sfh_to_logsfr_ratios(sfh):
    """
    Invert :func:`logsfr_ratios_to_sfh` — useful for initialising free
    parameters from a known SFH.

    Parameters
    ----------
    sfh : array_like, shape (n,)
        SFH weight vector (need not be normalised).

    Returns
    -------
    logsfr_ratios : jnp.ndarray, shape (n − 1,)
        Log10 ratios of consecutive SFR bins satisfying
        ``logsfr_ratios[i] = log10( sfh[i] / sfh[i+1] )``.
    """
    sfh     = jnp.asarray(sfh, dtype=float)
    sfh     = jnp.clip(sfh, 1e-30)
    log_sfr = jnp.log10(sfh)
    return log_sfr[:-1] - log_sfr[1:]                             # (n-1,)


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

