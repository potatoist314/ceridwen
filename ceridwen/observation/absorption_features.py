"""Stellar absorption-feature windows for spectral pixel selection.

A ``Spectrum`` normally enters the likelihood with every valid pixel.  This
module defines a standard catalogue of stellar absorption features and turns
it into an observed-frame pixel mask, so a fit can use the feature pixels
only (or down-weight everything else).  See
``Spectrum.select_absorption_features``.

Wavelengths in ``ABSORPTION_FEATURES`` are REST-FRAME, IN-AIR angstroms, as
tabulated in the Lick/IDS system (Worthey et al. 1994; Trager et al. 1998)
and the usual line lists.  ``feature_windows`` converts them to vacuum with
``sedpy_jax.observate.air2vac`` -- the same conversion ``StellarIndices``
uses -- and redshifts them onto the observed pixel grid.

Two kinds of entry exist:

``band``
    A Lick-style feature bandpass ``(lower, upper)``.  The window is the
    bandpass itself; ``window_kms`` is ignored.
``line``
    A single line centre.  The window is ``centre * (1 +- window_kms / c)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from sedpy_jax.observate import air2vac

C_KMS = 2.998e5


@dataclass(frozen=True)
class AbsorptionFeature:
    """One absorption feature: a Lick bandpass or a line centre (air, rest)."""

    name: str
    kind: Literal["band", "line"]
    lower: float
    upper: float | None
    group: str

    def __post_init__(self):
        if self.kind not in {"band", "line"}:
            raise ValueError(f"{self.name}: kind must be 'band' or 'line'")
        if self.kind == "band" and (self.upper is None or self.upper <= self.lower):
            raise ValueError(f"{self.name}: a band needs upper > lower")
        if self.kind == "line" and self.upper is not None:
            raise ValueError(f"{self.name}: a line has no upper edge")


def _band(name, lower, upper, group):
    return AbsorptionFeature(name, "band", float(lower), float(upper), group)


def _line(name, centre, group):
    return AbsorptionFeature(name, "line", float(centre), None, group)


ABSORPTION_FEATURES: tuple[AbsorptionFeature, ...] = (
    # Balmer series.  H-delta, H-gamma and H-beta use the wide Lick "A"/Hbeta
    # feature bandpasses; the higher-order lines are centres.
    _line("H10", 3797.898, "balmer"),
    _line("H9", 3835.386, "balmer"),
    _line("H8", 3889.049, "balmer"),
    _band("HdA", 4083.500, 4122.250, "balmer"),
    _band("HgA", 4319.750, 4363.500, "balmer"),
    _band("Hbeta", 4847.875, 4876.625, "balmer"),
    _line("Halpha", 6562.801, "balmer"),
    # Calcium: H & K, the Lick Ca features and the near-infrared triplet.
    _line("CaK", 3933.663, "ca"),
    _line("CaH", 3968.469, "ca"),
    _band("Ca4227", 4222.250, 4234.750, "ca"),
    _band("Ca4455", 4452.125, 4474.625, "ca"),
    _line("CaT1", 8498.018, "ca"),
    _line("CaT2", 8542.089, "ca"),
    _line("CaT3", 8662.140, "ca"),
    # Molecular bands and the G band.
    _band("CN4150", 4142.125, 4177.125, "cn"),
    _band("G4300", 4281.375, 4316.375, "g"),
    _band("C4668", 4634.000, 4720.250, "c"),
    # Iron.
    _band("Fe4383", 4369.125, 4420.375, "fe"),
    _band("Fe4531", 4514.250, 4559.250, "fe"),
    _band("Fe5015", 4977.750, 5054.000, "fe"),
    _band("Fe5270", 5245.650, 5285.650, "fe"),
    _band("Fe5335", 5312.125, 5352.125, "fe"),
    _band("Fe5406", 5387.500, 5415.000, "fe"),
    _band("Fe5709", 5696.625, 5720.375, "fe"),
    _band("Fe5782", 5776.625, 5796.625, "fe"),
    # Magnesium and sodium.
    _band("Mgb", 5160.125, 5192.625, "mg"),
    _band("NaD", 5876.875, 5909.375, "na"),
    # TiO bands (cool giants and dwarfs).
    _band("TiO1", 5936.625, 5994.125, "tio"),
    _band("TiO2", 6189.625, 6272.125, "tio"),
)

_BY_NAME = {feature.name: feature for feature in ABSORPTION_FEATURES}
_GROUPS = {feature.group for feature in ABSORPTION_FEATURES}


def select_features(
    items: Iterable[str | AbsorptionFeature] | None = None,
) -> tuple[AbsorptionFeature, ...]:
    """Resolve names, group names, or feature objects into feature objects.

    ``None`` returns the whole catalogue.  A string that matches a feature
    name selects that feature; a string that matches a group (``"fe"``,
    ``"balmer"``, ...) selects every feature of that group, in catalogue
    order.  ``AbsorptionFeature`` instances pass through unchanged.
    """
    if items is None:
        return ABSORPTION_FEATURES
    selected: list[AbsorptionFeature] = []
    for item in items:
        if isinstance(item, AbsorptionFeature):
            selected.append(item)
        elif item in _BY_NAME:
            selected.append(_BY_NAME[item])
        elif item in _GROUPS:
            selected.extend(f for f in ABSORPTION_FEATURES if f.group == item)
        else:
            raise ValueError(
                f"unknown absorption feature {item!r}; "
                f"names: {sorted(_BY_NAME)}; groups: {sorted(_GROUPS)}"
            )
    return tuple(selected)


def feature_windows(
    features: Iterable[str | AbsorptionFeature] | None = None,
    zred: float = 0.0,
    window_kms: float = 1000.0,
) -> np.ndarray:
    """Observed-frame vacuum ``(lower, upper)`` edges, shape ``(n, 2)``."""
    resolved = select_features(features)
    if not resolved:
        return np.zeros((0, 2), dtype=float)
    opz = 1.0 + float(zred)
    half = float(window_kms) / C_KMS
    edges = []
    for feature in resolved:
        if feature.kind == "band":
            edges.append([feature.lower, feature.upper])
        else:
            edges.append([feature.lower, feature.lower])
    vacuum = np.asarray(air2vac(np.asarray(edges, dtype=float).ravel()), dtype=float)
    vacuum = vacuum.reshape(-1, 2) * opz
    for k, feature in enumerate(resolved):
        if feature.kind == "line":
            centre = vacuum[k, 0]
            vacuum[k] = [centre * (1.0 - half), centre * (1.0 + half)]
    return vacuum


def absorption_feature_mask(
    wave_obs: Sequence[float] | np.ndarray,
    zred: float = 0.0,
    features: Iterable[str | AbsorptionFeature] | None = None,
    window_kms: float = 1000.0,
) -> np.ndarray:
    """Boolean mask, True for observed-frame pixels inside any feature window."""
    wave = np.asarray(wave_obs, dtype=float)
    inside = np.zeros(wave.shape, dtype=bool)
    for lower, upper in feature_windows(features, zred=zred, window_kms=window_kms):
        inside |= (wave >= lower) & (wave <= upper)
    return inside


__all__ = [
    "ABSORPTION_FEATURES",
    "AbsorptionFeature",
    "absorption_feature_mask",
    "feature_windows",
    "select_features",
]
