"""FilterSet._build_super_trans: the host-side padding returns the per-filter jnp build bit for bit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import vmap

from ceridwen.observation.filters import FilterSet, pad_to

jax.config.update("jax_enable_x64", True)


def per_filter_build(filterset):
    """The construction _build_super_trans used before 2026-10-14."""
    n_lam = len(filterset.lam)
    ab_counts = jnp.array([f.ab_zero_counts for f in filterset.filters])
    inds_start = jnp.array([f.inds.start for f in filterset.filters])
    inds_stop = jnp.array([f.inds.stop for f in filterset.filters])
    trans = jnp.stack([pad_to(filterset.lam, f.transmission, f.inds) for f in filterset.filters])
    wave = jnp.stack([pad_to(filterset.lam, f.wavelength, f.inds) for f in filterset.filters])
    dwave = jnp.stack([pad_to(filterset.lam, jnp.gradient(f.wavelength), f.inds) for f in filterset.filters])

    def compute_row(trans_j, wave_j, dwave_j, ab_j, istart, istop):
        full = trans_j * wave_j * dwave_j / ab_j
        mask = (jnp.arange(n_lam) >= istart) & (jnp.arange(n_lam) < istop)
        return jnp.where(mask, full, 0.0)

    rows = vmap(compute_row)(trans, wave, dwave, ab_counts, inds_start, inds_stop)
    return (rows, [rows[i, inds_start[i]:inds_stop[i]] for i in range(len(filterset.filters))],
            jnp.stack([inds_start, inds_stop], axis=1))


@pytest.mark.parametrize("filters", [
    ["hsc_g", "spitzer_irac_ch1"],
    ["cfht_megacam_us_9301", "hsc_i", "vista_vircam_Ks", "galex_FUV"],
])
def test_super_trans_matches_per_filter_build(filters):
    filterset = FilterSet(filters)
    trans, trans_list, frange = per_filter_build(filterset)
    assert filterset.trans.dtype == trans.dtype
    np.testing.assert_array_equal(np.asarray(filterset.trans), np.asarray(trans), strict=True)
    assert len(filterset.trans_list) == len(trans_list)
    for new, old in zip(filterset.trans_list, trans_list):
        np.testing.assert_array_equal(np.asarray(new), np.asarray(old), strict=True)
    np.testing.assert_array_equal(np.asarray(filterset.frange), np.asarray(frange), strict=True)
