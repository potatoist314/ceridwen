"""
ceridwen.likelihood.calibration
===============================

Analytic spectrophotometric calibration polynomial.

The flux calibration of a slit or fibre spectrum is uncertain at the few
percent level and varies smoothly with wavelength (flux-standard errors,
slit losses, atmospheric differential refraction, sky-subtraction and
extraction systematics).  Broadband photometry does not share these
errors.  A joint fit therefore lets the photometry set the absolute SED
shape and treats the spectrum's smooth calibration as a nuisance:

.. math::

    d_i \\approx P(x_i)\\, \\mu_i, \\qquad
    P(x) = 1 + \\sum_{n} a_n T_n(x)

where :math:`T_n` are Chebyshev polynomials of a normalised wavelength
coordinate :math:`x \\in [-1, 1]` over the unmasked pixels, and
:math:`\\mu` is the model spectrum on the data pixels.  Because
:math:`P` is linear in the coefficients, the maximum-likelihood
coefficients conditional on every other parameter are a weighted linear
least-squares solution, obtained analytically at each likelihood call.
The coefficients are therefore *profiled out* rather than sampled --
Prospector's ``polyopt`` (``PolySedModel.spec_calibration``) -- at the
cost of one tiny ``(n_coeff x n_coeff)`` solve per call.

Two static choices:

* ``fit_constant`` -- whether the basis includes :math:`T_0`.  With
  ``False`` (default, Prospector convention) the polynomial only bends the
  spectrum; its overall normalisation stays with the sampled scalar
  ``spectrum_scaling`` (Prospector ``spec_norm``).  With ``True`` the
  polynomial also absorbs the normalisation and ``spectrum_scaling`` is
  redundant.
* ``prior_sigma`` -- optional Gaussian prior width on every coefficient
  (a ridge term ``I / prior_sigma**2`` on the normal equations, and
  ``-0.5 * sum(a**2) / prior_sigma**2`` added to the log-likelihood).  It
  keeps the polynomial close to unity when the data cannot pin it down.

The weights are the static observational uncertainties ``sigma_obs`` (as
in Prospector), not the noise-model-inflated variance, so the solution is
a deterministic function of ``theta`` and never feeds back through a
model-anchored noise term.  The log-determinant term of a full Gaussian
marginalisation is not included (a profile likelihood, as in Prospector).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.chebyshev import chebvander

Array = jax.Array

__all__ = ["PolynomialCalibration"]


@dataclass(frozen=True, eq=False)
class PolynomialCalibration:
    """
    Chebyshev calibration polynomial profiled out at every likelihood call.

    Build one with :meth:`from_spectrum` (or :meth:`from_wavelength`) and
    pass it as ``DiagonalGaussianLikelihood(calibration=...)``.  The
    likelihood then replaces the model spectrum ``mu`` by
    ``polynomial(solve(...)) * mu`` before the noise model and the
    Gaussian kernel.

    Attributes
    ----------
    x : Array, shape (n_pix,)
        Chebyshev coordinate; the unmasked wavelength range maps onto
        ``[-1, 1]`` (masked edge pixels may lie outside).
    basis : Array, shape (n_pix, n_coeff)
        Columns ``T_n(x)`` for ``n = 1..order`` (``fit_constant=False``) or
        ``n = 0..order`` (``fit_constant=True``).
    order : int
        Polynomial order.
    fit_constant : bool
        Whether ``T_0`` is part of the basis (see the module docstring).
    prior_sigma : float or None
        Gaussian prior width on every coefficient; ``None`` = flat.

    Examples
    --------
    >>> cal = PolynomialCalibration.from_spectrum(spec, order=3)
    >>> lhood = DiagonalGaussianLikelihood(
    ...     noise_model=DiagonalNoiseModel(use_fractional=True),
    ...     calibration=cal)
    >>> mu_cal, coeffs, ln_prior = cal.calibrate(spec.flux, mu,
    ...                                          spec.uncertainty, spec.mask)
    >>> P = cal.polynomial(coeffs)          # the calibration vector on the pixels
    """

    x: Array
    basis: Array
    order: int
    fit_constant: bool = False
    prior_sigma: Optional[float] = None

    # ------------------------------------------------------------------
    @classmethod
    def from_wavelength(
        cls,
        wavelength,
        order: int = 3,
        *,
        mask=None,
        fit_constant: bool = False,
        prior_sigma: Optional[float] = None,
    ) -> "PolynomialCalibration":
        """
        Build the static basis for a pixel grid.

        Parameters
        ----------
        wavelength : array-like, shape (n_pix,)
            Observed-frame pixel wavelengths.
        order : int
            Polynomial order (``0`` is allowed only with ``fit_constant``).
        mask : array-like of bool, optional
            Pixels that define the ``[-1, 1]`` range.  Default: all.
        fit_constant, prior_sigma
            See the class docstring.
        """
        order = int(order)
        if order < 0:
            raise ValueError("PolynomialCalibration: order must be >= 0")
        if order == 0 and not fit_constant:
            raise ValueError(
                "PolynomialCalibration: order=0 without fit_constant has no "
                "coefficient to solve for; use order>=1 or fit_constant=True"
            )
        wave = np.asarray(wavelength, dtype=float)
        if wave.ndim != 1:
            raise ValueError("PolynomialCalibration: wavelength must be 1-D")
        keep = (np.ones(wave.shape, dtype=bool) if mask is None
                else np.asarray(mask, dtype=bool))
        if not keep.any():
            raise ValueError("PolynomialCalibration: mask leaves no pixels")
        lo, hi = float(wave[keep].min()), float(wave[keep].max())
        mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
        x = (wave - mid) / (half if half > 0.0 else 1.0)
        basis = chebvander(x, order)                 # (n_pix, order + 1)
        if not fit_constant:
            basis = basis[:, 1:]
        return cls(
            x=jnp.asarray(x),
            basis=jnp.asarray(basis),
            order=order,
            fit_constant=bool(fit_constant),
            prior_sigma=None if prior_sigma is None else float(prior_sigma),
        )

    @classmethod
    def from_spectrum(cls, spectrum, order: int = 3, **kwargs
                      ) -> "PolynomialCalibration":
        """Build from a ``Spectrum`` (its ``wavelength`` and, by default, its
        ``mask`` define the ``[-1, 1]`` range)."""
        if spectrum.wavelength is None:
            raise ValueError("PolynomialCalibration: the Spectrum has no "
                             "wavelength grid")
        if "mask" not in kwargs:
            mask = np.asarray(spectrum.mask, dtype=bool)
            kwargs["mask"] = mask if mask.size else None
        return cls.from_wavelength(spectrum.wavelength, order, **kwargs)

    # ------------------------------------------------------------------
    @property
    def n_coeff(self) -> int:
        return int(self.basis.shape[1])

    def solve(self, y, mu, sigma, mask) -> Array:
        """
        Weighted least-squares coefficients ``a`` of ``y ~= mu (1 + basis @ a)``.

        Pure JAX (jit/grad safe).  Masked pixels carry zero weight.  With a
        ``prior_sigma`` the normal equations get the ridge term
        ``I / prior_sigma**2``.
        """
        y = jnp.asarray(y)
        mu = jnp.asarray(mu)
        mask = jnp.asarray(mask, dtype=bool)
        safe_sigma = jnp.where(mask, jnp.asarray(sigma), 1.0)
        # Whitened, masked design matrix and residual: rows outside the mask
        # are exactly zero, so they drop out of the normal equations.
        weight = jnp.where(mask, mu / safe_sigma, 0.0)          # (n_pix,)
        design = self.basis * weight[:, None]                    # (n_pix, k)
        target = jnp.where(mask, (y - mu) / safe_sigma, 0.0)     # (n_pix,)
        normal = design.T @ design                               # (k, k)
        rhs = design.T @ target                                  # (k,)
        if self.prior_sigma is not None:
            normal = normal + jnp.eye(self.n_coeff) / self.prior_sigma ** 2
        return jnp.linalg.solve(normal, rhs)

    def polynomial(self, coeffs) -> Array:
        """``P(x) = 1 + basis @ coeffs`` on the full pixel grid."""
        return 1.0 + self.basis @ jnp.asarray(coeffs)

    def log_prior(self, coeffs) -> Array:
        """``-0.5 * sum(coeffs**2) / prior_sigma**2`` (zero for a flat prior)."""
        if self.prior_sigma is None:
            return jnp.zeros(())
        return -0.5 * jnp.sum(jnp.asarray(coeffs) ** 2) / self.prior_sigma ** 2

    def calibrate(self, y, mu, sigma, mask) -> tuple[Array, Array, Array]:
        """
        Solve and apply the calibration.

        Returns
        -------
        mu_cal : Array, shape (n_pix,)
            ``P(x) * mu`` -- the model to compare with ``y``.
        coeffs : Array, shape (n_coeff,)
            The profiled coefficients.
        ln_prior : Array, scalar
            The coefficient log-prior at the solution (zero when flat); the
            likelihood adds it so the profile respects ``prior_sigma``.
        """
        coeffs = self.solve(y, mu, sigma, mask)
        return self.polynomial(coeffs) * jnp.asarray(mu), coeffs, self.log_prior(coeffs)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"PolynomialCalibration(order={self.order}, "
                f"fit_constant={self.fit_constant}, "
                f"prior_sigma={self.prior_sigma}, n_pix={self.basis.shape[0]})")
