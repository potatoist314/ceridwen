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
Either way the polynomial adds no sampled dimension.  Per likelihood
call it costs one weighted moment vector of the pixels and one ``(k x k)``
Cholesky factorisation: the Gram matrix :math:`D^T D` is formed from the
Chebyshev product identity :math:`T_m T_n = \\tfrac12 (T_{m+n} + T_{|m-n|})`,
so :math:`(D^T D)_{mn} = \\tfrac12 (M_{m+n} + M_{|m-n|})` with moments
:math:`M_j = \\sum_i w_i T_j(x_i)`, :math:`w_i = \\mu_i^2 / \\sigma_i^2`,
:math:`j = 0 \\dots 2k`.  That is linear in the order instead of quadratic.

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
import jax.scipy.linalg as jsl
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
    moment_basis : Array, shape (n_pix, 2 * order + 1)
        Columns ``T_j(x)`` for ``j = 0..2 order``; the Gram matrix is read
        off the weighted column sums (module docstring).
    pair_plus, pair_minus : ndarray of int, shape (n_coeff, n_coeff)
        Static degree indices ``m + n`` and ``|m - n|`` of each coefficient pair.
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
    moment_basis: Array
    pair_plus: np.ndarray
    pair_minus: np.ndarray
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
        moment_basis = chebvander(x, 2 * order)      # (n_pix, 2 order + 1)
        basis = moment_basis[:, :order + 1]          # (n_pix, order + 1)
        if not fit_constant:
            basis = basis[:, 1:]
        n_coeff = basis.shape[1]
        degrees = np.arange(order + 1 - n_coeff, order + 1)
        pair_plus = degrees[:, None] + degrees[None, :]
        pair_minus = np.abs(degrees[:, None] - degrees[None, :])
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
            moment_basis=jnp.asarray(moment_basis),
            pair_plus=pair_plus,
            pair_minus=pair_minus,
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

    def _weights(self, mu, sigma, mask) -> Array:
        """Masked pixel weights ``w_i = mu_i^2 / sigma_i^2`` (zero outside the mask)."""
        mask = jnp.asarray(mask, dtype=bool)
        safe_sigma = jnp.where(mask, jnp.asarray(sigma), 1.0)
        return jnp.where(mask, (jnp.asarray(mu) / safe_sigma) ** 2, 0.0)

    def _gram(self, weights) -> Array:
        """``D^T D`` from the Chebyshev moments ``M_j = sum_i w_i T_j(x_i)``.

        ``T_m T_n = (T_{m+n} + T_{|m-n|}) / 2`` turns the ``(n_pix, k, k)``
        reduction into one ``(n_pix, 2k + 1)`` matvec, linear in the order.
        """
        moments = weights @ self.moment_basis
        return 0.5 * (moments[self.pair_plus] + moments[self.pair_minus])

    def normal_matrix(self, mu, sigma, mask) -> Array:
        """``D^T D + Sigma_p^{-1}`` -- the posterior precision of the coefficients."""
        normal = self._gram(self._weights(mu, sigma, mask))
        precision = self._precision()
        return normal if precision is None else normal + precision

    def _rhs(self, y, mu, sigma, mask) -> Array:
        """``D^T t`` with ``t_i = (y_i - mu_i) / sigma_i`` (zero outside the mask)."""
        mask = jnp.asarray(mask, dtype=bool)
        mu = jnp.asarray(mu)
        safe_sigma = jnp.where(mask, jnp.asarray(sigma), 1.0)
        target = jnp.where(mask, mu * (jnp.asarray(y) - mu) / safe_sigma ** 2, 0.0)
        return target @ self.basis

    def _factor(self, mu, sigma, mask) -> Array:
        """Lower Cholesky factor of the (symmetric positive definite) normal matrix."""
        return jnp.linalg.cholesky(self.normal_matrix(mu, sigma, mask))

    def solve(self, y, mu, sigma, mask) -> Array:
        """
        Coefficients ``a_hat`` of ``y ~= mu (1 + basis @ a)`` that maximise
        the Gaussian log-likelihood times the coefficient prior.

        Pure JAX (jit/grad safe).  Masked pixels carry zero weight.
        """
        return jsl.cho_solve((self._factor(mu, sigma, mask), True),
                             self._rhs(y, mu, sigma, mask))

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
        _, log_det_normal = jnp.linalg.slogdet(normal)
        return self._marginal_terms(coeffs, log_det_normal)

    def _marginal_terms(self, coeffs, log_det_normal) -> Array:
        """:meth:`log_marginal_terms` given ``ln|N|`` directly."""
        total = self.log_prior(coeffs)
        if not self.marginalize:
            return total
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
        mu = jnp.asarray(mu)
        factor = self._factor(mu, sigma, mask)
        coeffs = jsl.cho_solve((factor, True), self._rhs(y, mu, sigma, mask))
        log_det_normal = 2.0 * jnp.sum(jnp.log(jnp.diagonal(factor)))
        return (self.polynomial(coeffs) * mu, coeffs,
                self._marginal_terms(coeffs, log_det_normal))

    def _joint_system(self, y, mu, sigma, mask, lines, ridge=None) -> tuple[Array, Array]:
        """Normal matrix and right-hand side of ``y ~= mu (1 + basis @ a) + lines @ f``:
        the polynomial block as in :meth:`normal_matrix`, the line block
        ``L^T W L`` (plus ``diag(ridge)``) and their cross terms."""
        mask = jnp.asarray(mask, dtype=bool)
        mu = jnp.asarray(mu)
        safe_sigma = jnp.where(mask, jnp.asarray(sigma), 1.0)
        weighted_lines = jnp.where(mask, 1.0 / safe_sigma ** 2, 0.0)[:, None] * lines
        cross = self.basis.T @ (mu[:, None] * weighted_lines)
        line_block = lines.T @ weighted_lines
        if ridge is not None:
            line_block = line_block + jnp.diag(jnp.asarray(ridge))
        normal = jnp.block([[self.normal_matrix(mu, sigma, mask), cross],
                            [cross.T, line_block]])
        residual = jnp.where(mask, jnp.asarray(y) - mu, 0.0)
        rhs = jnp.concatenate([self._rhs(y, mu, sigma, mask),
                               weighted_lines.T @ residual])
        return normal, rhs

    def _joint_factor(self, y, mu, sigma, mask, lines, ridge=None):
        """Jacobi-scaled Cholesky factor, scale and solution of the joint system."""
        normal, rhs = self._joint_system(y, mu, sigma, mask, lines, ridge)
        scale = 1.0 / jnp.sqrt(jnp.diagonal(normal))
        factor = jnp.linalg.cholesky(scale[:, None] * normal * scale[None, :])
        solution = scale * jsl.cho_solve((factor, True), scale * rhs)
        return factor, scale, solution

    def _line_posterior(self, factor, scale, solution):
        """Mean and covariance of the line fluxes with the coefficients
        integrated out: the line block of ``N^{-1}``."""
        n_line = solution.shape[0] - self.n_coeff
        unit = jnp.eye(solution.shape[0])[:, self.n_coeff:]
        g = jsl.solve_triangular(factor, unit, lower=True)
        s = scale[self.n_coeff:]
        return solution[self.n_coeff:], s[:, None] * (g.T @ g) * s[None, :]

    def calibrate_with_lines(self, y, mu, sigma, mask, lines, pairs=(), ridge=None
                             ) -> tuple[Array, Array, Array, Array]:
        """
        :meth:`calibrate` with emission lines added after the polynomial.

        The model is ``P(x) * mu + lines @ f``; the coefficients (Gaussian
        prior) and the line fluxes ``f`` (flat prior on ``f >= 0``) are
        integrated out together.  ``lines`` is the ``(n_pix, n_line)``
        matrix of :meth:`EmissionLineColumns.columns`; with zero columns the
        result equals :meth:`calibrate`.

        The integral over ``f >= 0`` is the unconstrained integral times
        ``P(f >= 0)`` under the Gaussian posterior of ``f``, ``N(f_hat,
        Sigma)``.  ``P`` is the product of ``Phi(f_hat_k / sigma_k)`` over
        the lines, with the exact bivariate probability for each blended
        pair in ``pairs`` (index pairs, disjoint).  ``ridge`` adds a small
        precision to each line flux (:class:`EmissionLineColumns`).

        Returns
        -------
        mu_cal : Array, shape (n_pix,)
            ``P(x) * mu + lines @ f_hat`` (unconstrained maximum; the
            Gaussian kernel at this point plus ``ln_extra`` is the marginal).
        coeffs : Array, shape (n_coeff,)
        fluxes : Array, shape (n_line,)
            Unconstrained line fluxes ``f_hat``.
        ln_extra : Array, scalar
            Coefficient prior at the solution plus, with ``marginalize``,
            ``-0.5 ln|N|``, the prior and flat-prior normalisations and
            ``ln P(f >= 0)``.
        """
        mu = jnp.asarray(mu)
        factor, scale, solution = self._joint_factor(y, mu, sigma, mask, lines, ridge)
        coeffs, fluxes = solution[:self.n_coeff], solution[self.n_coeff:]
        log_det_normal = (2.0 * jnp.sum(jnp.log(jnp.diagonal(factor)))
                          - 2.0 * jnp.sum(jnp.log(scale)))
        ln_extra = self._marginal_terms(coeffs, log_det_normal)
        if self.marginalize:
            mean, cov = self._line_posterior(factor, scale, solution)
            ln_extra = (ln_extra + 0.5 * lines.shape[1] * _LOG_2PI
                        + log_positive_probability(mean, cov, pairs))
        return (self.polynomial(coeffs) * mu + lines @ fluxes, coeffs, fluxes,
                ln_extra)

    def posterior_draws_with_lines(self, y, mu_draws, sigma_draws, lines_draws,
                                   mask, key, sweeps: int = 1000, ridge=None
                                   ) -> tuple[Array, Array, Array]:
        """
        One joint draw of coefficients and line fluxes per posterior sample
        from ``N((a_hat, f_hat), N^{-1})`` truncated to ``f >= 0``.

        The fluxes come from ``sweeps`` Gibbs sweeps of one-dimensional
        truncated normals (started at ``max(f_hat, 0)``), the coefficients
        from their Gaussian conditional given the fluxes.

        Returns
        -------
        coeffs : Array, shape (n_draws, n_coeff)
        fluxes : Array, shape (n_draws, n_line), all ``>= 0``
        model : Array, shape (n_draws, n_pix)
            ``P(x) * mu + lines @ f`` for every draw.
        """
        mask = jnp.asarray(mask, dtype=bool)
        y = jnp.asarray(y)
        n_draws = jnp.shape(lines_draws)[0]
        keys = jax.random.split(key, n_draws)
        k = self.n_coeff

        def one(mu, sigma, lines, key):
            normal, _ = self._joint_system(y, mu, sigma, mask, lines, ridge)
            factor, scale, solution = self._joint_factor(y, mu, sigma, mask, lines, ridge)
            mean, cov = self._line_posterior(factor, scale, solution)
            fluxes = truncated_gibbs(key, mean, jnp.linalg.inv(cov), sweeps)
            n_aa = normal[:k, :k]
            chol = jnp.linalg.cholesky(n_aa)
            shift = jsl.cho_solve((chol, True), normal[:k, k:] @ (fluxes - mean))
            z = jax.random.normal(jax.random.fold_in(key, 1), (k,))
            coeffs = (solution[:k] - shift
                      + jsl.solve_triangular(chol.T, z, lower=False))
            return coeffs, fluxes, self.polynomial(coeffs) * mu + lines @ fluxes

        return jax.vmap(one)(jnp.asarray(mu_draws), jnp.asarray(sigma_draws),
                             jnp.asarray(lines_draws), keys)

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


# ---------------------------------------------------------------------------
# Positivity of the line fluxes
# ---------------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(20)


def _bvn_upper(h, k, r):
    """``P(X > h, Y > k)`` for a standard bivariate normal with correlation
    ``r`` (Genz 2004, Stat. Comput. 14, 251; his ``bvnu`` with 20 Gauss-
    Legendre points, absolute error ~1e-15)."""
    ndtr = jax.scipy.special.ndtr
    x, w = jnp.asarray(_GL_X), jnp.asarray(_GL_W)
    hk = h * k
    # |r| < 0.925: Drezner-Wesolowsky integral over asin(r)
    asr = jnp.arcsin(r)
    sn = jnp.sin(asr * (1.0 + x) / 2.0)
    small = (jnp.sum(w * jnp.exp((sn * hk - (h * h + k * k) / 2.0) / (1.0 - sn * sn)))
             * asr / (4.0 * jnp.pi) + ndtr(-h) * ndtr(-k))
    # |r| >= 0.925: Genz's expansion about |r| = 1
    kk = jnp.where(r < 0, -k, k)
    hk = jnp.where(r < 0, -hk, hk)
    as_ = (1.0 - r) * (1.0 + r)
    a = jnp.sqrt(as_)
    bs = (h - kk) ** 2
    c = (4.0 - hk) / 8.0
    d = (12.0 - hk) / 16.0
    e = -(bs / as_ + hk) / 2.0
    big = jnp.where(e > -100.0, a * jnp.exp(e) * (1.0 - c * (bs - as_) * (1.0 - d * bs / 5.0) / 3.0
                                                  + c * d * as_ * as_ / 5.0), 0.0)
    b = jnp.sqrt(bs)
    big = big - jnp.where(hk > -100.0, jnp.exp(-hk / 2.0) * jnp.sqrt(2.0 * jnp.pi) * ndtr(-b / a)
                          * b * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0), 0.0)
    xs = (a / 2.0 * (1.0 + x)) ** 2
    rs = jnp.sqrt(1.0 - xs)
    e = -(bs / xs + hk) / 2.0
    term = jnp.exp(e) * (jnp.exp(-hk * xs / (2.0 * (1.0 + rs) ** 2)) / rs
                         - (1.0 + c * xs * (1.0 + d * xs)))
    big = -(big + a / 2.0 * jnp.sum(jnp.where(e > -100.0, w * term, 0.0))) / (2.0 * jnp.pi)
    lower = jnp.where(h < 0, ndtr(kk) - ndtr(h), ndtr(-h) - ndtr(-kk))
    big = jnp.where(r > 0, big + ndtr(-jnp.maximum(h, kk)),
                    jnp.where(h >= kk, -big, lower - big))
    return jnp.clip(jnp.where(jnp.abs(r) < 0.925, small, big), 0.0, 1.0)


def log_positive_probability(mean, cov, pairs=()) -> Array:
    """``ln P(f >= 0)`` for ``f ~ N(mean, cov)``: independent lines, times the
    exact bivariate probability of each blended pair in ``pairs``.  Where a
    pair's probability is below 1e-10 (beyond the absolute accuracy of the
    bivariate formula) the pair keeps the product of its two ``Phi``."""
    sd = jnp.sqrt(jnp.diagonal(cov))
    z = mean / sd
    single = jax.scipy.special.log_ndtr(z)
    total = jnp.sum(single)
    for i, j in pairs:
        p = _bvn_upper(-z[i], -z[j], cov[i, j] / (sd[i] * sd[j]))
        product = single[i] + single[j]
        total = total + jnp.where(p > 1e-10, jnp.log(jnp.maximum(p, 1e-300)), product) - product
    return total


def truncated_gibbs(key, mean, precision, sweeps: int) -> Array:
    """One draw of ``N(mean, precision^{-1})`` truncated to the positive
    orthant, by ``sweeps`` Gibbs sweeps of one-dimensional truncated normals
    started at ``max(mean, 0)``."""
    m = mean.shape[0]
    cond_sd = 1.0 / jnp.sqrt(jnp.diagonal(precision))

    def update(i, carry):
        f, key = carry
        key, sub = jax.random.split(key)
        k = i % m
        centre = mean[k] - (precision[k] @ (f - mean) - precision[k, k] * (f[k] - mean[k])) / precision[k, k]
        # inverse CDF of the upper tail; beyond 30 sigma, where ndtr underflows,
        # the tail inverse sqrt(a^2 - 2 ln U) (relative error < 1e-3)
        a = -centre / cond_sd[k]
        uniform = jax.random.uniform(sub, dtype=mean.dtype, minval=1e-300)
        u = jnp.where(a < 30.0,
                      -jax.scipy.special.ndtri(uniform * jax.scipy.special.ndtr(-jnp.minimum(a, 30.0))),
                      jnp.sqrt(a * a - 2.0 * jnp.log(uniform)))
        return f.at[k].set(centre + cond_sd[k] * u), key

    f, _ = jax.lax.fori_loop(0, sweeps * m, update, (jnp.maximum(mean, 0.0), key))
    return f
