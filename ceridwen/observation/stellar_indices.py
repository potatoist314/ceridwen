"""Integrated stellar absorption indices and continuum-break measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import jax.numpy as jnp
import numpy as np
from sedpy_jax.observate import air2vac

from .base import Observation
from .spectrum import Spectrum


IndexKind = Literal["equivalent_width", "magnitude", "flux_ratio"]


@dataclass(frozen=True)
class StellarIndexDefinition:
    """One rest-frame stellar index definition in air wavelengths."""

    name: str
    kind: IndexKind
    blue: tuple[float, float]
    feature: tuple[float, float] | None
    red: tuple[float, float]

    @property
    def unit(self) -> str:
        """Human-readable unit of the measured index."""
        return {
            "equivalent_width": "angstrom",
            "magnitude": "mag",
            "flux_ratio": "dimensionless",
        }[self.kind]


LEGAC_DR2_STELLAR_INDEX_DEFINITIONS = (
    StellarIndexDefinition(
        "CN1", "magnitude", (4080.125, 4117.625),
        (4142.125, 4177.125), (4244.125, 4284.125),
    ),
    StellarIndexDefinition(
        "CN2", "magnitude", (4083.875, 4096.375),
        (4142.125, 4177.125), (4244.125, 4284.125),
    ),
    StellarIndexDefinition(
        "Ca4227", "equivalent_width", (4211.0, 4219.75),
        (4222.25, 4234.75), (4241.0, 4251.0),
    ),
    StellarIndexDefinition(
        "G4300", "equivalent_width", (4266.375, 4282.625),
        (4281.375, 4316.375), (4318.875, 4335.125),
    ),
    StellarIndexDefinition(
        "Fe4383", "equivalent_width", (4359.125, 4370.375),
        (4369.125, 4420.375), (4442.875, 4455.375),
    ),
    StellarIndexDefinition(
        "Ca4455", "equivalent_width", (4445.875, 4454.625),
        (4452.125, 4474.625), (4477.125, 4492.125),
    ),
    StellarIndexDefinition(
        "Fe4531", "equivalent_width", (4504.25, 4514.25),
        (4514.25, 4559.25), (4560.5, 4579.25),
    ),
    StellarIndexDefinition(
        "C4668", "equivalent_width", (4611.5, 4630.25),
        (4634.0, 4720.25), (4742.75, 4756.5),
    ),
    StellarIndexDefinition(
        "Hbeta", "equivalent_width", (4827.875, 4847.875),
        (4847.875, 4876.625), (4876.625, 4891.625),
    ),
    StellarIndexDefinition(
        "HdA", "equivalent_width", (4041.6, 4079.75),
        (4083.5, 4122.25), (4128.5, 4161.0),
    ),
    StellarIndexDefinition(
        "HgA", "equivalent_width", (4283.5, 4319.75),
        (4319.75, 4363.5), (4367.25, 4419.75),
    ),
    StellarIndexDefinition(
        "HdF", "equivalent_width", (4057.25, 4088.5),
        (4091.0, 4112.25), (4114.75, 4137.25),
    ),
    StellarIndexDefinition(
        "HgF", "equivalent_width", (4283.5, 4319.75),
        (4331.25, 4352.25), (4354.75, 4384.75),
    ),
    StellarIndexDefinition(
        "Dn4000", "flux_ratio", (3850.0, 3950.0), None,
        (4000.0, 4100.0),
    ),
)


def _band_weights(grid: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Trapezoidal integration weights divided by the band width."""
    selected = np.flatnonzero((grid >= lower) & (grid <= upper))
    if selected.size < 2:
        raise ValueError(f"Band [{lower}, {upper}] has fewer than two samples")

    wave = grid[selected]
    weights = np.zeros_like(grid)
    local = np.empty_like(wave)
    local[0] = 0.5 * (wave[1] - wave[0])
    local[-1] = 0.5 * (wave[-1] - wave[-2])
    local[1:-1] = 0.5 * (wave[2:] - wave[:-2])
    weights[selected] = local / (upper - lower)
    return weights


class StellarIndices(Observation):
    """Observed integrated stellar indices predicted from a model spectrum.

    The supplied definitions use rest-frame air wavelengths. During model
    setup, the band edges are converted to vacuum wavelengths and a
    :class:`Spectrum` projector applies the SSP-library, instrumental, and
    stellar-velocity broadening. Lick indices are measured from ``F_lambda``;
    ``Dn4000``-style flux ratios are measured from ``F_nu``.
    """

    _kind = "stellar_indices"
    alias = {
        "values": "flux",
        "index_values": "flux",
        "index_uncertainty": "uncertainty",
    }
    _meta = ("kind", "name")
    _data = ("wavelength", "flux", "uncertainty", "mask")

    def __init__(
        self,
        definitions: Sequence[StellarIndexDefinition],
        values=None,
        uncertainty=None,
        mask=slice(None),
        resolution=None,
        smoothtype=None,
        res_convention=None,
        inres="auto",
        sigma_losvd=None,
        name=None,
        **kwargs,
    ):
        self.definitions = tuple(definitions)
        if not self.definitions:
            raise ValueError("StellarIndices requires at least one definition")

        self.index_names = [definition.name for definition in self.definitions]
        self.index_kinds = [definition.kind for definition in self.definitions]
        self.index_units = [definition.unit for definition in self.definitions]
        self.resolution = resolution
        self.smoothtype = smoothtype
        self.res_convention = res_convention
        self.inres = inres
        self.sigma_losvd = sigma_losvd

        air_centres = []
        for definition in self.definitions:
            if definition.feature is None:
                air_centres.append(
                    0.25 * sum((*definition.blue, *definition.red))
                )
            else:
                air_centres.append(0.5 * sum(definition.feature))
        self._wavelength = jnp.asarray(
            air2vac(jnp.asarray(air_centres, dtype=float)), dtype=float
        )

        super().__init__(
            flux=values,
            uncertainty=uncertainty,
            mask=mask,
            name=name,
            **kwargs,
        )

        if self.flux is not None and len(self.flux) != len(self.definitions):
            raise ValueError("Index values and definitions must have equal length")

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = None if value is None else jnp.asarray(value, dtype=float)

    @property
    def bandpasses_vacuum(self) -> np.ndarray:
        """Band edges as ``(n_indices, 3, 2)`` vacuum-wavelength array."""
        rows = []
        for definition in self.definitions:
            feature = definition.feature or (np.nan, np.nan)
            air_bands = np.asarray(
                [definition.blue, feature, definition.red], dtype=float
            )
            finite = np.isfinite(air_bands)
            vacuum = air_bands.copy()
            vacuum[finite] = np.asarray(
                air2vac(jnp.asarray(air_bands[finite])), dtype=float
            )
            rows.append(vacuum)
        return np.asarray(rows)

    def setup_for_model(self, wave_model, zred=0.0, lib_resolution=None):
        """Build the broadened spectral projector and band integrals."""
        wave_model = np.asarray(wave_model, dtype=float)
        bands = self.bandpasses_vacuum
        finite_edges = bands[np.isfinite(bands)]
        lower = float(finite_edges.min())
        upper = float(finite_edges.max())
        inside = wave_model[(wave_model >= lower) & (wave_model <= upper)]
        evaluation_wave = np.unique(np.concatenate((inside, finite_edges)))
        self._evaluation_wave = jnp.asarray(evaluation_wave)

        self._spectrum_projector = Spectrum(
            wavelength=evaluation_wave * (1.0 + float(zred)),
            resolution=self.resolution,
            smoothtype=self.smoothtype,
            res_convention=self.res_convention,
            inres=self.inres,
            sigma_losvd=self.sigma_losvd,
            name=f"{self.name}_projector",
        )
        self._spectrum_projector.setup_for_model(
            wave_model,
            zred=zred,
            lib_resolution=lib_resolution,
        )

        blue_weights = []
        feature_weights = []
        red_weights = []
        blue_midpoints = []
        red_midpoints = []
        feature_lower = []
        feature_upper = []
        kind_codes = []
        kind_code = {
            "equivalent_width": 0,
            "magnitude": 1,
            "flux_ratio": 2,
        }

        for definition, vacuum_bands in zip(
            self.definitions,
            bands,
            strict=True,
        ):
            blue, feature, red = vacuum_bands
            blue_weights.append(_band_weights(evaluation_wave, *blue))
            red_weights.append(_band_weights(evaluation_wave, *red))
            blue_midpoints.append(float(np.mean(blue)))
            red_midpoints.append(float(np.mean(red)))
            kind_codes.append(kind_code[definition.kind])

            if definition.feature is None:
                feature_weights.append(np.zeros_like(evaluation_wave))
                feature_lower.append(0.0)
                feature_upper.append(1.0)
            else:
                feature_weights.append(_band_weights(evaluation_wave, *feature))
                feature_lower.append(float(feature[0]))
                feature_upper.append(float(feature[1]))

        self._blue_weights = jnp.asarray(blue_weights)
        self._feature_weights = jnp.asarray(feature_weights)
        self._red_weights = jnp.asarray(red_weights)
        self._blue_midpoints = jnp.asarray(blue_midpoints)
        self._red_midpoints = jnp.asarray(red_midpoints)
        self._feature_lower = jnp.asarray(feature_lower)
        self._feature_upper = jnp.asarray(feature_upper)
        self._kind_codes = jnp.asarray(kind_codes)

    def predict(self, spectrum, wave_model):
        """Measure all configured indices from a broadened model spectrum."""
        projected_fnu = self._spectrum_projector.predict(spectrum, wave_model)
        projected_flam = projected_fnu / self._evaluation_wave**2

        blue_flam = self._blue_weights @ projected_flam
        red_flam = self._red_weights @ projected_flam
        feature_flam = self._feature_weights @ projected_flam
        feature_width = self._feature_upper - self._feature_lower

        continuum_slope = (
            (red_flam - blue_flam)
            / (self._red_midpoints - self._blue_midpoints)
        )
        continuum_lower = blue_flam + continuum_slope * (
            self._feature_lower - self._blue_midpoints
        )
        continuum_upper = blue_flam + continuum_slope * (
            self._feature_upper - self._blue_midpoints
        )
        continuum_mean = 0.5 * (continuum_lower + continuum_upper)
        is_ratio = self._kind_codes == 2
        feature_ratio = jnp.where(
            is_ratio,
            jnp.ones_like(feature_flam),
            feature_flam / continuum_mean,
        )

        equivalent_width = (1.0 - feature_ratio) * feature_width
        magnitude = -2.5 * jnp.log10(feature_ratio)
        blue_fnu = self._blue_weights @ projected_fnu
        red_fnu = self._red_weights @ projected_fnu
        flux_ratio = red_fnu / blue_fnu

        return jnp.where(
            self._kind_codes == 0,
            equivalent_width,
            jnp.where(self._kind_codes == 1, magnitude, flux_ratio),
        )

    def mask_by_name(self, names):
        """Exclude named indices from the likelihood."""
        excluded = set(names)
        remove = jnp.asarray(
            [name in excluded for name in self.index_names], dtype=bool
        )
        self.mask = self.mask & ~remove

    def select_by_name(self, names):
        """Return a new observation containing only the requested indices."""
        positions = []
        for name in names:
            if name not in self.index_names:
                raise KeyError(
                    f"Index {name!r} not found; available: {self.index_names}"
                )
            positions.append(self.index_names.index(name))
        positions = np.asarray(positions)

        def select(values):
            return None if values is None else np.asarray(values)[positions]

        return StellarIndices(
            definitions=[self.definitions[index] for index in positions],
            values=select(self.flux),
            uncertainty=select(self.uncertainty),
            mask=np.asarray(self.mask)[positions],
            resolution=self.resolution,
            smoothtype=self.smoothtype,
            res_convention=self.res_convention,
            inres=self.inres,
            sigma_losvd=self.sigma_losvd,
            name=f"{self.name}_sel",
        )

    def _display_str(self, max_rows=80):
        rows = ["name       value       uncertainty  unit          mask"]
        for index, (name, unit) in enumerate(
            zip(self.index_names, self.index_units, strict=True)
        ):
            value = np.nan if self.flux is None else float(self.flux[index])
            error = (
                np.nan
                if self.uncertainty is None
                else float(self.uncertainty[index])
            )
            mask = False if self.mask.size == 0 else bool(self.mask[index])
            rows.append(
                f"{name:<10} {value:>10.5g}  {error:>11.5g}  {unit:<13} {mask}"
            )
        return "\n".join(rows)


__all__ = [
    "LEGAC_DR2_STELLAR_INDEX_DEFINITIONS",
    "StellarIndexDefinition",
    "StellarIndices",
]
