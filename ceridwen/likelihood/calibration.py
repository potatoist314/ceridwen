"""
ceridwen.likelihood.calibration
===============================

Spectrophotometric calibration polynomial, marginalised analytically.

The flux calibration of a slit or fibre spectrum is uncertain at the few
percent level and varies smoothly with wavelength (flux-standard and
response errors, slit losses that change with seeing and wavelength,
atmospheric differential refraction, sky-subtraction and extraction
systematics, and the aperture of the slit against the total light the
photometry measures).  Broadband photometry does not share these errors.
A joint fit therefore lets the photometry set the absolute SED shape and
treats the spectrum's smooth calibration as a nuisance:

.. math::

    d_i \\approx P(x_i)\\, \\mu_i, \\qquad
    P(x) = 1 + \\sum_{n} a_n T_n(x)

where :math:`T_n` are Chebyshev polynomials of a normalised wavelength
coordinate :math:`x \\in [-1, 1]` over the unmasked pixels, and
:math:`\\mu` is the model spectrum on the data pixels.  :math:`P` is
linear in the coefficients :math:`a`, so with a diagonal Gaussian
likelihood and a Gaussian prior :math:`a \\sim N(0, \\Sigma_p)` the
coefficients can be integrated out in closed form.  Writing
:math:`D_{in} = T_n(x_i)\\,\\mu_i/\\sigma_i` (whitened design matrix),
:math:`t_i = (d_i - \\mu_i)/\\sigma_i`, :math:`N = D^T D + \\Sigma_p^{-1}`
and :math:`\\hat a = N^{-1} D^T t`:

.. math::

    \\ln \\int \\mathcal{L}(\\theta, a)\\, p(a)\\, da
      = \\ln \\mathcal{L}(\\theta, \\hat a)
        - \\tfrac12 \\hat a^T \\Sigma_p^{-1} \\hat a
        + \\tfrac12 \\ln|\\Sigma_p^{-1}| - \\tfrac12 \\ln|N| .

The first two terms are the *profile* likelihood (Prospector's
``polyopt`` in ``PolySedModel.spec_calibration`` stops here); the last
two are the Occam factor of the marginalisation.  ``marginalize=True``
(default) returns the full expression; ``False`` returns the profile.
Without a prior the flat-prior integral is used,
:math:`\\ln \\mathcal{L}(\\hat a) + \\tfrac{k}{2}\\ln 2\\pi - \\tfrac12 \\ln|D^T D|`.
Either way the polynomial costs one ``(k x k)`` solve per likelihood
call and adds no sampled dimension.

Two static choices:

* ``fit_constant`` -- whether the basis includes :math:`T_0`.  With
  ``False`` (default, Prospector convention) the polynomial only bends the
  spectrum; its overall normalisation stays with the sampled scalar
  ``spectrum_scaling`` (Prospector ``spec_norm``).  With ``True`` the
  polynomial also absorbs the normalisation and ``spectrum_scaling`` is
  redundant.
* ``prior_sigma`` -- Gaussian prior width per coefficient, one float for
  all or one per coefficient (``BAGPIPES`` gives each ``calib:n`` its
  own prior).  ``None`` is a flat prior.

Inside :class:`~ceridwen.likelihood.DiagonalGaussianLikelihood` the
weights :math:`\\sigma_i` are the noise model's effective uncertainties
evaluated at the *uncalibrated* model (observational plus any fractional
or jitter term), so :math:`\\hat a` maximises exactly the Gaussian kernel
that is then evaluated, and the marginal above is exact for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.chebyshev import chebvander

Array = jax.Array
_LOG_2PI = float(np.log(2.0 * np.pi))

__all__ = ["PolynomialCalibration"]


@dataclass(frozen=True, eq=False)
class PolynomialCalibration:
    """
    Chebyshev calibration polynomial integrated out at every likelihood call.

    Build one with :meth:`from_spectrum` (or :meth:`from_wavelength`) and
    pass it as ``DiagonalGaussianLikelihood(calibration=...)``.  The
    likelihood then replaces the model spectrum ``mu`` by
    ``polynomial(solve(...)) * mu`` inside the Gaussian kernel and adds
    :meth:`log_marginal_terms` (the coefficient prior at the solution plus,
    with ``marginalize=True``, the Occam factor).

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
    prior_sigma : tuple of float or None
        Gaussian prior width of each coefficient; ``None`` = flat.
    marginalize : bool
        Add the Occam factor of the analytic marginalisation (default) or
        return the profile likelihood.

    Examples
    --------
    >>> cal = PolynomialCalibration.from_spectrum(spec, order=3, prior_sigma=0.1)
    >>> lhood = DiagonalGaussianLikelihood(
    ...     noise_model=DiagonalNoiseModel(use_fractional=True),
    ...     calibration=cal)
    >>> mu_cal, coeffs, ln_extra = cal.calibrate(spec.flux, mu,
    ...                                          spec.uncertainty, spec.mask)
    >>> P = cal.polynomial(coeffs)          # the calibration vector on the pixels
    >>> a, P_draws = cal.posterior_draws(spec.flux, mu_draws, sigma_draws,
    ...                                  spec.mask, jax.random.PRNGKey(0))
    """

    x: Array
    basis: Array
    order: int
    fit_constant: bool = False
    prior_sigma: Optional[tuple[float, ...]] = None
    marginalize: bool = True

    # ------------------------------------------------------------------
    @classmethod
    def from_wavelength(
        cls,
        wavelength,
        order: int = 3,
        *,
        mask=None,
        fit_constant: bool = False,
        prior_sigma: Union[float, Sequence[float], None] = None,
        marginalize: bool = True,
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
        fit_constant, prior_sigma, marginalize
            See the class docstring.  ``prior_sigma`` is one width for
            every coefficient or a sequence with one width per coefficient
            (``order`` entries, or ``order + 1`` with ``fit_constant``).
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
        n_coeff = basis.shape[1]
        widths: Optional[tuple[float, ...]] = None
        if prior_sigma is not None:
            values = np.atleast_1d(np.asarray(prior_sigma, dtype=float))
            if values.size == 1:
                values = np.repeat(values, n_coeff)
            if values.shape != (n_coeff,):
                raise ValueError(
                    f"PolynomialCalibration: prior_sigma needs {n_coeff} "
                    f"widths (one per coefficient), got {values.size}"
                )
            if not np.all(values > 0.0):
                raise ValueError("PolynomialCalibration: prior_sigma must be > 0")
            widths = tuple(float(v) for v in values)
        return cls(
            x=jnp.asarray(x),
            basis=jnp.asarray(basis),
            order=order,
            fit_constant=bool(fit_constant),
            prior_sigma=widths,
            marginalize=bool(marginalize),
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

    def _precision(self) -> Optional[Array]:
        """Prior precision matrix ``diag(1 / prior_sigma**2)`` or ``None``."""
        if self.prior_sigma is None:
            return None
        return jnp.diag(1.0 / jnp.asarray(self.prior_sigma) ** 2)

    def design(self, mu, sigma, mask) -> Array:
        """Whitened, masked design matrix ``D[i, n] = T_n(x_i) mu_i / sigma_i``
        (rows outside the mask are exactly zero)."""
        mask = jnp.asarray(mask, dtype=bool)
        safe_sigma = jnp.where(mask, jnp.asarray(sigma), 1.0)
        weight = jnp.where(mask, jnp.asarray(mu) / safe_sigma, 0.0)
        return self.basis * weight[:, None]

    def normal_matrix(self, mu, sigma, mask) -> Array:
        """``D^T D + Sigma_p^{-1}`` -- the posterior precision of the coefficients."""
        design = self.design(mu, sigma, mask)
        normal = design.T @ design
        precision = self._precision()
        return normal if precision is None else normal + precision

    def solve(self, y, mu, sigma, mask) -> Array:
        """
        Coefficients ``a_hat`` of ``y ~= mu (1 + basis @ a)`` that maximise
        the Gaussian log-likelihood times the coefficient prior.

        Pure JAX (jit/grad safe).  Masked pixels carry zero weight.
        """
        mask = jnp.asarray(mask, dtype=bool)
        mu = jnp.asarray(mu)
        safe_sigma = jnp.where(mask, jnp.asarray(sigma), 1.0)
        design = self.design(mu, safe_sigma, mask)
        target = jnp.where(mask, (jnp.asarray(y) - mu) / safe_sigma, 0.0)
        rhs = design.T @ target
        return jnp.linalg.solve(self.normal_matrix(mu, safe_sigma, mask), rhs)

    def covariance(self, mu, sigma, mask) -> Array:
        """Posterior covariance of the coefficients given ``theta``: ``N^{-1}``."""
        return jnp.linalg.inv(self.normal_matrix(mu, sigma, mask))

    def polynomial(self, coeffs) -> Array:
        """``P(x) = 1 + basis @ coeffs`` on the full pixel grid."""
        return 1.0 + self.basis @ jnp.asarray(coeffs)

    def log_prior(self, coeffs) -> Array:
        """``-0.5 * sum((a / prior_sigma)**2)`` (zero for a flat prior)."""
        if self.prior_sigma is None:
            return jnp.zeros(())
        return -0.5 * jnp.sum((jnp.asarray(coeffs) / jnp.asarray(self.prior_sigma)) ** 2)

    def log_marginal_terms(self, coeffs, normal) -> Array:
        """
        Terms added to the Gaussian log-likelihood at ``a_hat``.

        Profile (``marginalize=False``): the coefficient log-prior at the
        solution.  Marginal (``marginalize=True``): additionally
        ``+0.5 ln|Sigma_p^{-1}| - 0.5 ln|N|`` with a prior, or
        ``+(k/2) ln 2 pi - 0.5 ln|N|`` for the flat-prior integral.
        """
        total = self.log_prior(coeffs)
        if not self.marginalize:
            return total
        _, log_det_normal = jnp.linalg.slogdet(normal)
        total = total - 0.5 * log_det_normal
        if self.prior_sigma is None:
            return total + 0.5 * self.n_coeff * _LOG_2PI
        return total - jnp.sum(jnp.log(jnp.asarray(self.prior_sigma)))

    def calibrate(self, y, mu, sigma, mask) -> tuple[Array, Array, Array]:
        """
        Solve and apply the calibration.

        Returns
        -------
        mu_cal : Array, shape (n_pix,)
            ``P(x) * mu`` -- the model to compare with ``y``.
        coeffs : Array, shape (n_coeff,)
            The conditional maximum-likelihood coefficients ``a_hat``.
        ln_extra : Array, scalar
            :meth:`log_marginal_terms` at ``a_hat``; the likelihood adds it.
        """
        mask = jnp.asarray(mask, dtype=bool)
        mu = jnp.asarray(mu)
        safe_sigma = jnp.where(mask, jnp.asarray(sigma), 1.0)
        normal = self.normal_matrix(mu, safe_sigma, mask)
        design = self.design(mu, safe_sigma, mask)
        target = jnp.where(mask, (jnp.asarray(y) - mu) / safe_sigma, 0.0)
        coeffs = jnp.linalg.solve(normal, design.T @ target)
        return (self.polynomial(coeffs) * mu, coeffs,
                self.log_marginal_terms(coeffs, normal))

    def posterior_draws(self, y, mu_draws, sigma_draws, mask, key,
                        draws_per_sample: int = 1) -> tuple[Array, Array]:
        """
        Draw calibration coefficients from their posterior.

        For each posterior sample of ``theta`` (row of ``mu_draws`` and
        ``sigma_draws``) the coefficients are Gaussian,
        ``a ~ N(a_hat(theta), N(theta)^{-1})``; this draws
        ``draws_per_sample`` of them per row so the returned band carries
        both the spread of ``a_hat`` over ``theta`` and the conditional
        uncertainty given ``theta``.

        Returns
        -------
        coeffs : Array, shape (n_draws * draws_per_sample, n_coeff)
        polynomial : Array, shape (n_draws * draws_per_sample, n_pix)
            ``P(x)`` for every coefficient draw.
        """
        mask = jnp.asarray(mask, dtype=bool)
        y = jnp.asarray(y)
        mu_draws = jnp.asarray(mu_draws)
        sigma_draws = jnp.asarray(sigma_draws)
        n_draws = mu_draws.shape[0]
        noise = jax.random.normal(
            key, (n_draws, int(draws_per_sample), self.n_coeff))

        def one(mu, sigma, z):
            a_hat = self.solve(y, mu, sigma, mask)
            chol = jnp.linalg.cholesky(self.covariance(mu, sigma, mask))
            return a_hat[None, :] + z @ chol.T

        coeffs = jax.vmap(one)(mu_draws, sigma_draws, noise)
        coeffs = coeffs.reshape(-1, self.n_coeff)
        return coeffs, 1.0 + coeffs @ self.basis.T

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"PolynomialCalibration(order={self.order}, "
                f"fit_constant={self.fit_constant}, "
                f"prior_sigma={self.prior_sigma}, "
                f"marginalize={self.marginalize}, n_pix={self.basis.shape[0]})")
