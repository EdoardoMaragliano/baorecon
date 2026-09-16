"""The two pipeline implementations must agree.

``ReconstructionPipeline`` and ``ReconstructionPipelineInteractive`` are parallel
implementations that differ only in *when* they release intermediate arrays: one
frees them mid-``run()`` to cap peak memory, the other keeps everything reachable
for inspection. Nothing about the result should depend on that choice.

They had drifted out of test coverage — 86% against 39% — which matters because
every wiring change has to be made twice. These tests pin them together, stage by
stage and on the written products.
"""

import numpy as np
import pytest
import yaml
from astropy.io import fits
from astropy.table import Table

from baorecon.pipeline.bao_pipeline import ReconstructionPipeline
from baorecon.pipeline.bao_pipeline_interactive import ReconstructionPipelineInteractive

PIPELINES = [ReconstructionPipeline, ReconstructionPipelineInteractive]


@pytest.fixture(scope="module")
def catalogs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("equiv_catalogs")
    rng = np.random.RandomState(17)

    def write(n, path):
        t = Table()
        t["RA"] = rng.uniform(150.0, 170.0, n)
        t["DEC"] = rng.uniform(5.0, 25.0, n)
        t["REDSHIFT"] = rng.uniform(0.8, 1.0, n)
        t["WEIGHT"] = np.ones(n)
        t.write(path, overwrite=True)
        return str(path)

    return write(800, tmp / "data.fits"), write(4000, tmp / "random.fits")


def _config(tmp_path, catalogs, name, align_cone=False, save=("catalogs",)):
    data_path, random_path = catalogs
    cfg = {
        "catalog": {"data_path": data_path, "random_path": random_path,
                    "data_hdu": 1, "random_hdu": 1},
        "columns": {"coordinates": {"ra": "RA", "dec": "DEC", "redshift": "REDSHIFT"},
                    "weights": {"data": "WEIGHT", "random": "WEIGHT"}},
        "coordinate_system": {"input": "ra_dec_z"},
        "cosmology": {"H0": 70.0, "Om0": 0.3},
        "reconstruction": {"redshift": 0.9, "nmesh": 32, "R_sm": 15.0, "f": 0.8,
                           "bias": 1.5, "pbc": False, "padding": 100.0,
                           "solver_type": "ifft", "device": "cpu",
                           "threshold_randoms": 0.01, "align_cone": align_cone},
        "output": {"folder": str(tmp_path / f"out_{name}"), "naming_pattern": name,
                   "format": "fits", "save_metadata": True, "save": list(save)},
        "catalog_name": "equiv",
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.dump(cfg))
    return str(path)


def _drive(Pipeline, config_path):
    """Run the chain step by step, so nothing is released before we read it."""
    p = Pipeline(config_path)
    p.load_catalogs()
    p.convert_to_xyz()
    p.reconstruct()
    p.convert_back()
    return p


@pytest.mark.parametrize("align_cone", [False, True])
def test_pipelines_agree_stage_by_stage(tmp_path, catalogs, align_cone):
    """Same config in, same arrays out, at every stage of the chain."""
    capped = _drive(ReconstructionPipeline,
                    _config(tmp_path, catalogs, "capped", align_cone))
    interactive = _drive(ReconstructionPipelineInteractive,
                         _config(tmp_path, catalogs, "interactive", align_cone))

    for attr in ("data_pos_xyz", "random_pos_xyz", "data_rec_xyz", "random_rec_xyz",
                 "data_rec_ra", "data_rec_dec", "data_rec_z",
                 "random_rec_ra", "random_rec_dec", "random_rec_z"):
        a, b = getattr(capped, attr), getattr(interactive, attr)
        assert (a is None) == (b is None), f"{attr}: one pipeline has it, the other does not"
        if a is not None:
            np.testing.assert_allclose(a, b, rtol=0, atol=0, err_msg=f"{attr} differs")

    # Not vacuous: the reconstruction moved something.
    assert np.abs(capped.data_rec_xyz - capped.data_pos_xyz).max() > 1e-3


def test_pipelines_agree_on_the_cone_frame(tmp_path, catalogs):
    capped = _drive(ReconstructionPipeline, _config(tmp_path, catalogs, "c_frame", True))
    interactive = _drive(ReconstructionPipelineInteractive,
                         _config(tmp_path, catalogs, "i_frame", True))

    assert capped.cone_frame is not None and interactive.cone_frame is not None
    np.testing.assert_array_equal(capped.cone_frame.matrix, interactive.cone_frame.matrix)
    np.testing.assert_array_equal(capped.cone_frame.mean_direction,
                                  interactive.cone_frame.mean_direction)


def test_pipelines_write_identical_catalogues(tmp_path, catalogs):
    """The full ``run()``, compared on the products rather than in memory.

    This is the path that differs most: the capped pipeline interleaves writing
    with releasing, the interactive one writes at the end.
    """
    saved_capped = ReconstructionPipeline(
        _config(tmp_path, catalogs, "run_capped")).run()
    saved_interactive = ReconstructionPipelineInteractive(
        _config(tmp_path, catalogs, "run_interactive")).run()

    assert set(saved_capped) == set(saved_interactive), "different products written"

    for key in ("data_catalog", "random_catalog"):
        a = Table.read(saved_capped[key]).to_pandas()
        b = Table.read(saved_interactive[key]).to_pandas()
        assert list(a.columns) == list(b.columns), f"{key}: column sets differ"
        assert len(a) == len(b), f"{key}: row counts differ"
        for col in a.columns:
            np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(),
                                       rtol=0, atol=0, err_msg=f"{key}.{col} differs")


@pytest.mark.parametrize("Pipeline", PIPELINES)
def test_run_releases_or_keeps_as_documented(tmp_path, catalogs, Pipeline):
    """The one difference that *is* expected, asserted so it stays intentional."""
    p = Pipeline(_config(tmp_path, catalogs, f"rel_{Pipeline.__name__}"))
    p.run()

    kept = p.data_rec_ra is not None
    if Pipeline is ReconstructionPipelineInteractive:
        assert kept, "the interactive pipeline must keep its intermediates reachable"
    else:
        assert not kept, ("the capped pipeline must release the catalogue arrays "
                          "as it writes them")
