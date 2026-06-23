"""Tests for the end-to-end ReconstructionPipeline."""

import pytest
import yaml
from pathlib import Path
import numpy as np
from astropy.table import Table

from baorecon.pipeline.bao_pipeline import ReconstructionPipeline
from baorecon.utils.backend import CUPY_AVAILABLE

# Marker to skip GPU tests if CUDA is not available
gpu_test = pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available or CuPy not installed")
DEVICES = ["cpu", pytest.param("gpu", marks=gpu_test)]


@pytest.fixture(scope="module")
def dummy_catalogs(tmp_path_factory):
    """Create dummy FITS catalogs for data and randoms."""
    tmp_path = tmp_path_factory.mktemp("catalogs")
    data_path = tmp_path / "data.fits"
    random_path = tmp_path / "random.fits"

    def create_catalog(num_rows, path):
        table = Table()
        table["RA"] = np.random.uniform(10, 20, num_rows)
        table["DEC"] = np.random.uniform(10, 20, num_rows)
        table["REDSHIFT"] = np.random.uniform(0.4, 0.6, num_rows)
        table["WEIGHT"] = np.ones(num_rows)
        table.write(path, overwrite=True)

    create_catalog(100, data_path)
    create_catalog(500, random_path)

    return str(data_path), str(random_path)


@pytest.fixture(scope="module", params=DEVICES)
def test_config_all_outputs(request, tmp_path_factory, dummy_catalogs):
    """Create a YAML config file that enables all save options, for both
    the 'cpu' and 'gpu' compute backends."""
    device = request.param
    tmp_path = tmp_path_factory.mktemp("config")
    data_path, random_path = dummy_catalogs
    output_dir = tmp_path / "output"

    config_dict = {
        "catalog": {
            "data_path": data_path,
            "random_path": random_path,
            "data_hdu": 1,
            "random_hdu": 1,
        },
        "columns": {
            "coordinates": {"ra": "RA", "dec": "DEC", "redshift": "REDSHIFT"},
            "weights": {"data": "WEIGHT", "random": "WEIGHT"},
        },
        "coordinate_system": {"input": "ra_dec_z"},
        "cosmology": {"Om0": 0.3, "H0": 70},
        "reconstruction": {
            "redshift": 0.5,
            "nmesh": 16,  # Use a small mesh for speed
            "R_sm": 15.0,
            "f": 0.7,
            "bias": 1.5,
            "pbc": True,
            "solver_type": "ifft",
            "device": device,
        },
        "output": {
            "folder": str(output_dir),
            "naming_pattern": f"test_run_{device}",
            "save_metadata": True,
            "save": [
                "catalogs",
                "tracer_displacements",
                "grid_potential",
                "grid_displacement",
                "reconstructor_object",
            ],
        },
        "catalog_name": "test_survey",
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    return str(config_path)


def test_pipeline_saves_all_outputs(test_config_all_outputs):
    """
    Run the pipeline (on both 'cpu' and 'gpu' devices) with a config that
    saves all possible outputs and verify that all files are created correctly.
    """
    config_path = test_config_all_outputs
    pipeline = ReconstructionPipeline(config_path)
    saved_files = pipeline.run()

    # 1. Check that the returned dictionary contains all expected keys
    expected_keys = {
        "data_catalog",
        "random_catalog",
        "grid_potential",
        "grid_displacement",
        "reconstructor_object",
        "metadata",
    }
    assert set(saved_files.keys()) == expected_keys

    # 2. Check that all files physically exist at the returned paths
    for key, path_str in saved_files.items():
        path = Path(path_str)
        assert path.exists(), f"File for '{key}' not found at {path}"
        assert path.stat().st_size > 0, f"File for '{key}' is empty: {path}"

    # 3. Deeper check for 'tracer_displacements' option
    # The output FITS catalog should contain the displacement vector columns.
    data_table = Table.read(saved_files["data_catalog"])
    expected_displacement_cols = {"S_X", "S_Y", "S_Z"}
    assert expected_displacement_cols.issubset(
        data_table.colnames
    ), "Displacement columns S_X, S_Y, S_Z not found in output catalog."

    random_table = Table.read(saved_files["random_catalog"])
    assert expected_displacement_cols.issubset(random_table.colnames)