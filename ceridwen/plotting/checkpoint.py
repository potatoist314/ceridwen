"""Compact spectrum summaries for nested-sampling checkpoint animations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def spectrum_credible_interval(
    predictions,
    quantiles: tuple[float, float, float] = (0.16, 0.50, 0.84),
) -> np.ndarray:
    """Return per-pixel posterior model-spectrum quantiles.

    The input is an equal-weight posterior sample with shape
    ``(n_draws, n_pixels)``.  The interval describes parameter uncertainty in
    the noiseless model spectrum.  It does not include measurement noise.
    """
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2:
        raise ValueError("predictions must have shape (n_draws, n_pixels)")
    return np.quantile(values, quantiles, axis=0)


def spectrum_checkpoint_frame_fn(
    model,
    observation_name: str,
    *,
    source_run: str = "",
    source_data: str = "",
) -> Callable[[dict], dict]:
    """Build a callback for ``BlackJAXNestedSamplerAdapter`` checkpoints.

    The callback evaluates the model on the adapter's deterministic
    equal-weight sample.  It stores the median and 16--84 percent interval on
    the observation grid.  Excluded pixels stay on that grid as NaN values.
    """
    observation = model.obs_dict[observation_name]
    wavelength = np.asarray(observation.wavelength, dtype=float)
    observed = np.asarray(observation.flux, dtype=float)
    uncertainty = np.asarray(observation.uncertainty, dtype=float)
    mask = np.asarray(observation.mask, dtype=bool)

    def build(draws: dict) -> dict:
        import jax.numpy as jnp

        prediction = model.predict_vmap(
            {name: jnp.asarray(value) for name, value in draws.items()}
        )[observation_name]
        q16, q50, q84 = spectrum_credible_interval(np.asarray(prediction))
        calibration_fraction = (
            float(np.median(np.exp(np.asarray(draws["log_f_calib"], dtype=float))))
            if "log_f_calib" in draws
            else 0.0
        )
        residual_uncertainty = np.hypot(
            uncertainty, calibration_fraction * np.abs(q50)
        )
        keep = mask & np.isfinite(observed) & np.isfinite(uncertainty)

        def connected(values):
            result = np.asarray(values, dtype=float).copy()
            result[~keep] = np.nan
            return result

        return {
            "schema_version": 1,
            "observation_name": observation_name,
            "wavelength": wavelength,
            "observed": connected(observed),
            "uncertainty": connected(uncertainty),
            "residual_uncertainty": connected(residual_uncertainty),
            "model_q16": connected(q16),
            "model_q50": connected(q50),
            "model_q84": connected(q84),
            "calibration_fraction": calibration_fraction,
            "quantiles": (0.16, 0.50, 0.84),
            "interval_kind": "posterior model-spectrum credible interval",
            "includes_measurement_noise": False,
            "source_run": source_run,
            "source_data": source_data,
        }

    return build
