"""Tests for `coordinate_system.input = cartesian`.

The pipeline is Cartesian-native underneath (BAOReconstructor takes (N, 3)
positions), so this mode is a pair of short-circuits: skip the sky conversion on
the way in, and skip the inverse on the way out. The tests below pin both ends,
because skipping only the first would silently write RA/DEC/z invented from a
box through a cosmology that never entered the run.
"""

import numpy as np
import pytest
import yaml
from astropy.table import Table

from baorecon.io.config import resolve_coordinate_input
from baorecon.pipeline.bao_pipeline import ReconstructionPipeline


# ==========================================
# RESOLUTION / VALIDATION
# ==========================================
def test_default_is_ra_dec_z():
    assert resolve_coordinate_input({}) == "ra_dec_z"
    assert resolve_coordinate_input({"input": None}) == "ra_dec_z"


def test_value_is_normalised():
    assert resolve_coordinate_input({"input": "  CARTESIAN "}) == "cartesian"


def test_unknown_value_is_rejected():
    """Falling back to the default would read x/y/z as degrees and a redshift."""
    with pytest.raises(ValueError, match="galactic"):
        resolve_coordinate_input({"input": "galactic"})


# ==========================================
# END-TO-END
# ==========================================
@pytest.fixture
def cartesian_catalogs(tmp_path):
    """Clustered data + densely sampled randoms, in Cartesian columns.

    Both properties matter. The data must be clustered or delta is ~0 and the
    displacement field is trivially zero; the randoms must sample the mesh well
    (here ~15 per cell at nmesh=16) or most cells have no randoms at all, delta
    is undefined almost everywhere, and the reconstruction returns the input
    unchanged -- which would let these tests pass without reconstructing.
    """
    rng = np.random.default_rng(1)
    centres = rng.uniform(80, 420, (30, 3))
    data = np.clip(np.repeat(centres, 30, axis=0) + rng.normal(0, 12, (900, 3)), 1, 499)
    randoms = rng.uniform(0, 500, (60_000, 3))

    paths = []
    for name, xyz in (("data.fits", data), ("random.fits", randoms)):
        t = Table()
        t["X"], t["Y"], t["Z"] = xyz.T
        t["WEIGHT"] = np.ones(len(xyz))
        path = tmp_path / name
        t.write(path, overwrite=True)
        paths.append(str(path))
    return paths


def _config(tmp_path, catalogs, coord_input, columns):
    data_path, random_path = catalogs
    cfg = {
        "catalog": {"data_path": data_path, "random_path": random_path},
        "columns": {
            "coordinates": dict(zip(("ra", "dec", "redshift"), columns)),
            "weights": {"data": "WEIGHT", "random": "WEIGHT"},
        },
        "coordinate_system": {"input": coord_input},
        "cosmology": {"H0": 67.11, "Om0": 0.3175},
        "reconstruction": {
            "nmesh": 16, "R_sm": 15.0, "f": 0.8, "bias": 1.5,
            "pbc": False, "padding": 10.0, "solver_type": "ifft", "device": "cpu",
        },
        "output": {"folder": str(tmp_path / "out"), "naming_pattern": "run",
                   "save": ["catalogs"], "save_metadata": False},
        "catalog_name": "cart",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg))
    return str(path)


def test_cartesian_input_skips_both_conversions(tmp_path, cartesian_catalogs):
    """x/y/z in, x/y/z out, and the positions must be read through unchanged."""
    pipeline = ReconstructionPipeline(
        _config(tmp_path, cartesian_catalogs, "cartesian", ("X", "Y", "Z")))

    pipeline.load_catalogs()
    raw = np.column_stack((pipeline.data_pos_ra, pipeline.data_pos_dec, pipeline.data_pos_z))

    pipeline.convert_to_xyz()
    # No conversion happened: the stacked columns ARE the Cartesian positions.
    assert np.allclose(pipeline.data_pos_xyz, raw)

    pipeline.reconstruct()
    pipeline.convert_back()
    # The "sky" attributes carry the Cartesian components straight through.
    back = np.column_stack((pipeline.data_rec_ra, pipeline.data_rec_dec, pipeline.data_rec_z))
    assert np.allclose(back, pipeline.data_rec_xyz)


def test_cartesian_run_writes_cartesian_columns(tmp_path, cartesian_catalogs):
    """The output catalogue keeps the X/Y/Z columns, with shifted values."""
    saved = ReconstructionPipeline(
        _config(tmp_path, cartesian_catalogs, "cartesian", ("X", "Y", "Z"))).run()

    out = Table.read(saved["data_catalog"])
    assert {"X", "Y", "Z"}.issubset(set(out.colnames))

    original = Table.read(cartesian_catalogs[0])
    shift = np.abs(np.asarray(out["X"]) - np.asarray(original["X"]))
    # A MEANINGFUL displacement, not numerical noise: `not allclose` would also
    # be satisfied by 1e-9, which is what an empty density field produces.
    assert shift.mean() > 1.0, "tracers barely moved: did the reconstruction run?"
    # Still positions in Mpc/h, i.e. bounded by the box -- not degrees, not a
    # redshift, which is what a wrongly-taken sky branch would have written.
    assert shift.max() < 500.0
    assert np.asarray(out["X"]).min() > -500.0 and np.asarray(out["X"]).max() < 1000.0


def test_cosmology_is_irrelevant_on_the_cartesian_path(tmp_path, cartesian_catalogs):
    """Two very different cosmologies must give identical Cartesian results."""
    results = []
    for H0, Om0, sub in ((67.11, 0.3175, "planck"), (50.0, 0.9, "absurd")):
        (tmp_path / sub).mkdir()
        cfg_path = _config(tmp_path / sub, cartesian_catalogs, "cartesian", ("X", "Y", "Z"))
        cfg = yaml.safe_load(open(cfg_path))
        cfg["cosmology"] = {"H0": H0, "Om0": Om0}
        open(cfg_path, "w").write(yaml.dump(cfg))

        pipeline = ReconstructionPipeline(cfg_path)
        pipeline.load_catalogs()
        pipeline.convert_to_xyz()
        pipeline.reconstruct()
        results.append(pipeline.data_rec_xyz.copy())

    assert np.allclose(results[0], results[1])


def test_unknown_input_fails_the_run(tmp_path, cartesian_catalogs):
    pipeline = ReconstructionPipeline(
        _config(tmp_path, cartesian_catalogs, "spherical", ("X", "Y", "Z")))
    with pytest.raises(ValueError, match="spherical"):
        pipeline.convert_to_xyz()
