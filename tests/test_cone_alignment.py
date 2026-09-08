"""Tests for cone alignment wired into the pipelines (``reconstruction.align_cone``)."""

import numpy as np
import pytest
import yaml
from astropy.table import Table

from baorecon.io.config import CatalogConfig
from baorecon.pipeline.bao_pipeline import ReconstructionPipeline
from baorecon.pipeline.bao_pipeline_interactive import ReconstructionPipelineInteractive
from baorecon.utils.frames import ConeFrame

PIPELINES = [ReconstructionPipeline, ReconstructionPipelineInteractive]


@pytest.fixture(scope="module")
def catalogs(tmp_path_factory):
    """Data and random catalogues on a patch far from the z axis.

    Centred at DEC ~ +15 so the cone starts well off-axis: an alignment that
    quietly did nothing would still leave the mean direction there.
    """
    tmp = tmp_path_factory.mktemp("cone_catalogs")
    rng = np.random.default_rng(4)

    def write(n, path):
        table = Table()
        table["RA"] = rng.uniform(150.0, 190.0, n)
        table["DEC"] = rng.uniform(5.0, 25.0, n)
        table["REDSHIFT"] = rng.uniform(0.4, 0.6, n)
        table["WEIGHT"] = np.ones(n)
        table.write(path, overwrite=True)
        return str(path)

    return write(400, tmp / "data.fits"), write(2000, tmp / "random.fits")


def make_config(tmp_path, catalogs, align_cone, extra_output=None, name="run"):
    data_path, random_path = catalogs
    config = {
        "catalog": {"data_path": data_path, "random_path": random_path,
                    "data_hdu": 1, "random_hdu": 1},
        "columns": {
            "coordinates": {"ra": "RA", "dec": "DEC", "redshift": "REDSHIFT"},
            "weights": {"data": "WEIGHT", "random": "WEIGHT"},
        },
        "coordinate_system": {"input": "ra_dec_z"},
        "cosmology": {"Om0": 0.3, "H0": 70},
        "reconstruction": {
            "redshift": 0.5, "nmesh": 16, "R_sm": 15.0, "f": 0.7, "bias": 1.5,
            "pbc": False, "padding": 100.0, "solver_type": "ifft", "device": "cpu",
            "align_cone": align_cone,
        },
        "output": {
            "folder": str(tmp_path / "out"), "naming_pattern": name,
            "format": "fits", "save_metadata": True,
            "save": extra_output or ["catalogs"],
        },
        "catalog_name": "cone_test",
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


def mean_direction(positions):
    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    total = unit.sum(axis=0)
    return total / np.linalg.norm(total)


# ---------------------------------------------------------------------------
# off by default; on when asked
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Pipeline", PIPELINES)
def test_off_by_default(tmp_path, catalogs, Pipeline):
    """Existing configs must behave exactly as before."""
    pipeline = Pipeline(make_config(tmp_path, catalogs, align_cone=False))
    pipeline.load_catalogs()
    pipeline.convert_to_xyz()

    assert pipeline.align_cone is False
    assert pipeline.cone_frame is None
    # The cone is left pointing where the survey points, nowhere near +z.
    assert mean_direction(pipeline.random_pos_xyz)[2] < 0.5


def test_absent_key_means_off(tmp_path, catalogs):
    config_path = make_config(tmp_path, catalogs, align_cone=False)
    config = yaml.safe_load(open(config_path))
    del config["reconstruction"]["align_cone"]
    open(config_path, "w").write(yaml.dump(config))
    assert ReconstructionPipeline(config_path).align_cone is False


@pytest.mark.parametrize("Pipeline", PIPELINES)
def test_alignment_puts_the_cone_on_z(tmp_path, catalogs, Pipeline):
    pipeline = Pipeline(make_config(tmp_path, catalogs, align_cone=True))
    pipeline.load_catalogs()
    pipeline.convert_to_xyz()

    assert isinstance(pipeline.cone_frame, ConeFrame)
    assert pipeline.cone_frame.dtype == pipeline.dtype

    # Randoms define the frame, so theirs lands on +z; the data share it, so
    # they land near it -- not exactly on it, which is the point of one frame.
    assert mean_direction(pipeline.random_pos_xyz)[2] == pytest.approx(1.0, abs=1e-6)
    assert mean_direction(pipeline.data_pos_xyz)[2] > 0.99


@pytest.mark.parametrize("Pipeline", PIPELINES)
def test_frame_comes_from_the_randoms(tmp_path, catalogs, Pipeline):
    """Not from the data: the geometry must not follow the data's variance."""
    pipeline = Pipeline(make_config(tmp_path, catalogs, align_cone=True))
    pipeline.load_catalogs()
    expected = ConeFrame.from_angles(
        pipeline.random_pos_ra, pipeline.random_pos_dec, dtype=pipeline.dtype
    )
    pipeline.convert_to_xyz()
    np.testing.assert_allclose(
        pipeline.cone_frame.mean_direction, expected.mean_direction, rtol=0, atol=0
    )


# ---------------------------------------------------------------------------
# the round trip is closed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Pipeline", PIPELINES)
def test_sky_coordinates_survive_the_round_trip(tmp_path, catalogs, Pipeline):
    """Rotate in, identity 'reconstruction', rotate out -> the original sky."""
    pipeline = Pipeline(make_config(tmp_path, catalogs, align_cone=True))
    pipeline.load_catalogs()
    ra_in = np.array(pipeline.data_pos_ra, dtype=np.float64)
    dec_in = np.array(pipeline.data_pos_dec, dtype=np.float64)
    pipeline.convert_to_xyz()

    # Stand in for the solver: no displacement at all.
    pipeline.data_rec_xyz = pipeline.data_pos_xyz.copy()
    pipeline.random_rec_xyz = pipeline.random_pos_xyz.copy()
    pipeline.convert_back()

    assert np.abs((pipeline.data_rec_ra - ra_in + 180.0) % 360.0 - 180.0).max() < 1e-3
    assert np.abs(pipeline.data_rec_dec - dec_in).max() < 1e-3


@pytest.mark.parametrize("Pipeline", PIPELINES)
def test_displacements_do_not_straddle_two_frames(tmp_path, catalogs, Pipeline):
    """convert_back must bring the *pre*-reconstruction arrays back as well.

    _save_catalogs builds the tracer displacements as ``pos_xyz - rec_xyz``. If
    only the reconstructed arrays were un-rotated, that subtraction would mix a
    survey-frame array with a cone-frame one and produce large bogus vectors
    instead of the zeros an identity reconstruction must give.
    """
    pipeline = Pipeline(make_config(tmp_path, catalogs, align_cone=True))
    pipeline.load_catalogs()
    pipeline.convert_to_xyz()
    pipeline.data_rec_xyz = pipeline.data_pos_xyz.copy()
    pipeline.random_rec_xyz = pipeline.random_pos_xyz.copy()
    pipeline.convert_back()

    displacements = pipeline.data_pos_xyz - pipeline.data_rec_xyz
    assert np.abs(displacements).max() < 1e-3


# ---------------------------------------------------------------------------
# Cartesian input path
# ---------------------------------------------------------------------------

def test_cartesian_input_uses_the_positions(tmp_path, catalogs):
    """That path has no angles at all, so the axis must come from the positions."""
    rng = np.random.default_rng(9)
    n = 1500
    ra, dec = rng.uniform(150.0, 190.0, n), rng.uniform(5.0, 25.0, n)
    r = rng.uniform(1000.0, 2000.0, n)
    cos_dec = np.cos(np.radians(dec))
    table = Table()
    table["X"] = cos_dec * np.cos(np.radians(ra)) * r
    table["Y"] = cos_dec * np.sin(np.radians(ra)) * r
    table["Z"] = np.sin(np.radians(dec)) * r
    table["WEIGHT"] = np.ones(n)
    path = tmp_path / "cart.fits"
    table.write(path, overwrite=True)

    config_path = make_config(tmp_path, catalogs, align_cone=True, name="cart")
    config = yaml.safe_load(open(config_path))
    config["catalog"]["data_path"] = config["catalog"]["random_path"] = str(path)
    config["columns"]["coordinates"] = {"ra": "X", "dec": "Y", "redshift": "Z"}
    config["coordinate_system"] = {"input": "cartesian"}
    open(config_path, "w").write(yaml.dump(config))

    pipeline = ReconstructionPipeline(config_path)
    pipeline.load_catalogs()
    pipeline.convert_to_xyz()

    assert isinstance(pipeline.cone_frame, ConeFrame)
    assert mean_direction(pipeline.random_pos_xyz)[2] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# end to end, and the metadata that makes the grids readable
# ---------------------------------------------------------------------------

def test_full_run_records_the_matrix(tmp_path, catalogs):
    """Saved grids live in the cone frame, so the matrix must travel with them."""
    pipeline = ReconstructionPipeline(
        make_config(tmp_path, catalogs, align_cone=True,
                    extra_output=["catalogs", "grid_density"], name="full")
    )
    saved = pipeline.run()

    metadata = open(saved["metadata"]).read()
    assert "cone_frame" in metadata
    assert "mean_direction" in metadata

    restored = ConeFrame.from_dict(pipeline.cone_frame.to_dict())
    np.testing.assert_array_equal(restored.matrix, pipeline.cone_frame.matrix)


def test_metadata_has_no_frame_when_disabled(tmp_path, catalogs):
    pipeline = ReconstructionPipeline(
        make_config(tmp_path, catalogs, align_cone=False, name="plain")
    )
    assert "cone_frame" not in open(pipeline.run()["metadata"]).read()


def test_full_run_writes_sky_coordinates(tmp_path, catalogs):
    """The saved catalogue must be on the sky, not left in the cone frame.

    Read back from disk rather than off the pipeline: ``run()`` releases the
    arrays as it writes them, which is the whole point of that implementation.
    """
    pipeline = ReconstructionPipeline(
        make_config(tmp_path, catalogs, align_cone=True, name="sky")
    )
    saved = pipeline.run()
    written = Table.read(saved["data_catalog"]).to_pandas()

    assert 140.0 < written["RA"].median() < 200.0
    assert 0.0 < written["DEC"].median() < 30.0


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_align_cone_reaches_the_config_from_yaml(tmp_path, catalogs):
    """The INI half lives in test_config_ini.py, next to the parfile machinery."""
    config = CatalogConfig.from_file(
        make_config(tmp_path, catalogs, align_cone=True, name="cfg")
    )
    assert config.reconstruction["align_cone"] is True


def test_alignment_is_logged(tmp_path, catalogs):
    """`align_cone` is a run-level fact: it must be visible in the log.

    baorecon's loggers set ``propagate = False``, so caplog (which listens on the
    root logger) never sees them; the handler goes on the module's own logger.
    """
    import logging

    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("baorecon.pipeline._cone")
    handler = Collect(level=logging.DEBUG)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        pipeline = ReconstructionPipeline(make_config(tmp_path, catalogs, align_cone=True))
        pipeline.load_catalogs()
        pipeline.convert_to_xyz()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)

    messages = " ".join(records)
    assert "Cone alignment ON" in messages
    assert "before rotation" in messages
    assert "after rotation" in messages
    # The centre lands on the pole, not on RA/DEC = (0, 0).
    after = messages.split("after rotation: DEC = ")[1]
    assert float(after.split(" deg")[0]) == pytest.approx(90.0, abs=1e-3)
