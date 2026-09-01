"""Static-width Gaussian smoothing built once at setup time.

The observation-side smoothing chain is a fixed linear operator whenever every
width is known at ``setup_for_model`` time.  This module exploits that twice:

1. **One convolution, not two.**  Gaussians compose in quadrature, so a galaxy
   LOSVD followed by an instrumental LSF is a single Gaussian of width
   ``sqrt(sigma_losvd^2 + sigma_instr^2 - sigma_library^2)`` at each wavelength.
   Chaining two resample -> FFT -> resample round trips costs one extra
   interpolation pair and is measurably *less* accurate than doing one.

2. **No per-call setup.**  ``jax_interp`` runs ``searchsorted`` and
   ``smooth_fft_padded`` builds its Gaussian taper on every evaluation, even
   though both depend only on grids fixed at setup.  XLA does not constant-fold
   them (the arrays exceed its folding budget), so they are baked here instead.

The grid rule differs from ``sedpy_jax._lsf_grid``: the resampling grid is never
allowed to be coarser than the input grid.  Sizing it from the kernel width
alone lets a wide kernel pick a grid that undersamples the model spectrum, which
loses line depth before the convolution ever runs.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = ["combined_sigma_lambda", "make_static_smoother"]


def _bake_interp(x, xp, dtype=jnp.float64):
    """Return ``f(fp) -> interp(x; xp, fp)`` with indices and weights baked in.

    ``x`` and ``xp`` are static, so the gather indices and the linear weights
    are computed once here with NumPy.  Only ``fp`` stays traced.  Matches
    ``sedpy_jax.observate.jax_interp``, including its zero-fill outside
    ``[xp[0], xp[-1]]``.
    """
    x = np.asarray(x, dtype=np.float64)
    xp = np.asarray(xp, dtype=np.float64)
    idx = np.clip(np.searchsorted(xp, x, side="left") - 1, 0, len(xp) - 2)
    x0, x1 = xp[idx], xp[idx + 1]
    # A CDF-transform grid can repeat a wavelength where the transform is
    # locally flat, which would divide by zero here.  Such an interval has no
    # width to interpolate across, so take the left node.
    span = x1 - x0
    frac = np.where(span > 0.0, (x - x0) / np.where(span > 0.0, span, 1.0), 0.0)
    inside = (x >= xp[0]) & (x <= xp[-1])
    frac = np.where(inside, frac, 0.0)

    i0 = jnp.asarray(idx)
    i1 = jnp.asarray(idx + 1)
    w = jnp.asarray(frac.astype(np.float64)).astype(dtype)
    keep = jnp.asarray(inside.astype(np.float64)).astype(dtype)

    def interp(fp):
        y0 = fp[i0]
        return keep * (y0 + (fp[i1] - y0) * w)

    return interp


def _bake_gaussian_fft(dx, n, sigma, dtype=jnp.float64):
    """Return ``f(spec) -> Gaussian convolution`` with the taper baked in.

    Zero-pads to ``2n`` before the transform so the cyclic wrap lands entirely
    in the pad, exactly as ``sedpy_jax.smoothing.smooth_fft_padded`` does.
    """
    m = 2 * n
    nu = np.fft.rfftfreq(m, d=float(dx))
    taper = jnp.asarray(
        np.exp(-2.0 * np.pi**2 * float(sigma) ** 2 * nu**2).astype(np.float64)
    ).astype(dtype)
    pad = jnp.zeros((n,), dtype=dtype)

    def convolve(spec):
        padded = jnp.concatenate([spec.astype(dtype), pad])
        return jnp.fft.irfft(jnp.fft.rfft(padded) * taper, n=m)[:n]

    return convolve


def combined_sigma_lambda(wave, sigma_target, sigma_losvd_kms, sigma_library):
    """Quadrature-combine every static broadening term into one sigma_lambda.

    Parameters
    ----------
    wave : (n,) array
        Wavelength grid the smoother reads [AA], observed frame.
    sigma_target : float or (n,) array or None
        Instrumental target width [AA].  ``None`` means no instrumental stage.
    sigma_losvd_kms : float or None
        Galaxy velocity dispersion [km/s].  ``None`` means no LOSVD stage.
    sigma_library : float or (n,) array or None
        Width already present in the input spectrum [AA], subtracted in
        quadrature.  ``NaN`` entries mean "unknown here" and subtract nothing.

    Returns
    -------
    (n,) ndarray
        Effective Gaussian sigma [AA] at each wavelength, floored at one
        wavelength pixel where the subtraction would leave zero or less.
    """
    from .base import _CKMS

    wave = np.asarray(wave, dtype=np.float64)
    var = np.zeros_like(wave)

    if sigma_target is not None:
        var = var + np.broadcast_to(
            np.asarray(sigma_target, dtype=np.float64), wave.shape
        ) ** 2
    if sigma_losvd_kms is not None:
        var = var + (float(sigma_losvd_kms) * wave / _CKMS) ** 2
    if sigma_library is not None:
        lib = np.broadcast_to(
            np.asarray(sigma_library, dtype=np.float64), wave.shape
        )
        var = var - np.nan_to_num(lib, nan=0.0) ** 2

    sigma = np.sqrt(np.maximum(var, 0.0))
    # A zero width is not representable on the CDF grid; one pixel is a delta
    # within Nyquist, i.e. no smoothing.  Same convention as sedpy_jax.
    dw = np.abs(np.gradient(wave))
    floor = float(dw.min()) if dw.size else 0.0
    return np.where(sigma > 0.0, sigma, floor)


def _cdf_grid(wave, sigma_lambda, pix_per_sigma=2):
    """CDF-transform resampling grid, never coarser than the input grid.

    ``x(lambda) = integral dlambda' / sigma(lambda')`` maps the input onto a
    coordinate where the kernel width is constant, so one FFT does a
    wavelength-dependent convolution.

    Differs from ``sedpy_jax.smoothing._lsf_grid`` in the final size: that
    function sizes the grid from the kernel width alone, which for a wide kernel
    can undersample the input spectrum.  Here the size is also floored at the
    next power of two above the input length, so resampling never discards
    resolution the model already carries.
    """
    wave = np.asarray(wave, dtype=np.float64)
    sigma_lambda = np.asarray(sigma_lambda, dtype=np.float64)

    dw = np.gradient(wave)
    cdf = np.cumsum(dw / sigma_lambda)
    cdf /= cdf[-1]

    x_per_pixel = np.gradient(cdf)
    sigma_per_pixel = dw / sigma_lambda
    x_per_sigma = float(np.nanmedian(x_per_pixel / sigma_per_pixel))

    n_kernel = int(2 ** np.ceil(np.log2(pix_per_sigma / x_per_sigma)))
    n_input = int(2 ** np.ceil(np.log2(max(wave.size, 2))))
    nx = max(n_kernel, n_input)

    x = np.linspace(0.0, 1.0, nx)
    lam = np.interp(x, cdf, wave)
    return lam, 1.0 / nx, x_per_sigma


def make_static_smoother(
    wave_in, sigma_lambda, wave_out, pix_per_sigma=2, dtype=jnp.float64
):
    """Return ``smoother(spec) -> (len(wave_out),)`` for one static Gaussian.

    ``spec`` is the only traced argument.  Every grid, index, weight and taper
    is computed here with NumPy and enters the compiled graph as a constant.
    """
    lam, dx, x_per_sigma = _cdf_grid(wave_in, sigma_lambda, pix_per_sigma)
    to_x = _bake_interp(lam, wave_in, dtype)
    convolve = _bake_gaussian_fft(dx, lam.size, x_per_sigma, dtype)
    to_out = _bake_interp(wave_out, lam, dtype)

    def smoother(spec):
        return to_out(convolve(to_x(spec)))

    smoother.grid_size = int(lam.size)
    return smoother
