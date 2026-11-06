"""Photometry(filter_directory=): curves read from a directory outside the library."""

import numpy as np

from ceridwen.observation import Photometry


def test_filter_directory_reads_its_own_curves(tmp_path):
    """Photometry(filter_directory=) loads <name>.par from that directory, cached apart from the library."""
    from pathlib import Path
    import ceridwen
    src = Path(ceridwen.__file__).parent / "data" / "filters" / "hsc_g.par"
    rows = [line.split() for line in src.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    (tmp_path / "hsc_g.par").write_text(
        "\n".join(f"{float(w)} {float(t) ** 2}" for w, t in rows) + "\n")
    library = Photometry(filters=["hsc_g"], flux=np.ones(1), uncertainty=np.ones(1), name="a")
    own = Photometry(filters=["hsc_g"], flux=np.ones(1), uncertainty=np.ones(1), name="b",
                     filter_directory=str(tmp_path))
    assert own.filter_directory == str(tmp_path)
    assert own.filterset is not library.filterset
    assert own.filters[0].wave_effective != library.filters[0].wave_effective
