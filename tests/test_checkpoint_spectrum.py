from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ceridwen.plotting import spectrum_checkpoint_frame_fn, spectrum_credible_interval


def test_spectrum_credible_interval_is_per_pixel():
    predictions = np.array(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [3.0, 13.0],
            [4.0, 14.0],
        ]
    )

    interval = spectrum_credible_interval(predictions, (0.25, 0.5, 0.75))

    np.testing.assert_allclose(interval, [[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])


def test_checkpoint_frame_prediction_and_mask_contract():
    observation = SimpleNamespace(
        wavelength=np.array([4000.0, 4100.0, 4200.0]),
        flux=np.array([2.0, 4.0, 6.0]),
        uncertainty=np.array([0.5, 0.5, 1.0]),
        mask=np.array([True, False, True]),
    )

    class Model:
        obs_dict = {"spectrum": observation}

        @staticmethod
        def predict_vmap(draws):
            amplitude = np.asarray(draws["amplitude"])
            return {"spectrum": amplitude * np.array([1.0, 2.0, 3.0])}

    build = spectrum_checkpoint_frame_fn(Model(), "spectrum")
    frame = build(
        {
            "amplitude": np.array([[1.0], [2.0], [3.0]]),
            "log_f_calib": np.log(np.array([[0.1], [0.2], [0.3]])),
        }
    )

    np.testing.assert_allclose(frame["model_q50"][[0, 2]], [2.0, 6.0])
    assert np.isnan(frame["model_q50"][1])
    assert np.isnan(frame["observed"][1])
    np.testing.assert_allclose(
        frame["residual_uncertainty"][[0, 2]],
        np.hypot([0.5, 1.0], 0.2 * np.array([2.0, 6.0])),
    )
    assert frame["calibration_fraction"] == pytest.approx(0.2)
    assert frame["includes_measurement_noise"] is False
