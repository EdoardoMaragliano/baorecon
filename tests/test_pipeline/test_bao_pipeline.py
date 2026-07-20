"""Tests for the end-to-end ReconstructionPipeline."""

import pytest
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table

from baorecon.pipeline.bao_pipeline import ReconstructionPipeline
from baorecon.utils.backend import CUPY_AVAILABLE

# Marker to skip GPU tests if CUDA is not available
gpu_test = pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available or CuPy not installed")
DEVICES = ["cpu", pytest.param("gpu", marks=gpu_test)]


def _catalog_columns(path_str):
    """Return the column names of a saved catalog (FITS or Parquet)."""
    path = Path(path_str)
    if path.suffix == ".parquet":
        return set(pd.read_parquet(path).columns)
    return set(Table.read(path).colnames)


def _read_catalog(path_str):
    """Read a saved catalog (FITS or Parquet) into a DataFrame (native endianness)."""
    path = Path(path_str)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return Table.read(path).to_pandas()


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


@pytest.fixture(scope="module", params=["fits", "parquet"])
def output_format(request):
    """Exercise both on-disk catalog output formats."""
    return request.param


@pytest.fixture(scope="module", params=DEVICES)
def test_config_all_outputs(request, tmp_path_factory, dummy_catalogs, output_format):
    """Create a YAML config file that enables all save options, for both
    the 'cpu' and 'gpu' compute backends and both output formats."""
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
            "naming_pattern": f"test_run_{device}_{output_format}",
            "format": output_format,
            "save_metadata": True,
            "save": [
                "catalogs",
                "tracer_displacements",
                "grid_density",
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
        "grid_density",
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
    # The output catalog (FITS or Parquet) should contain the displacement columns.
    expected_displacement_cols = {"S_X", "S_Y", "S_Z"}
    data_cols = _catalog_columns(saved_files["data_catalog"])
    assert expected_displacement_cols.issubset(
        data_cols
    ), "Displacement columns S_X, S_Y, S_Z not found in output catalog."

    random_cols = _catalog_columns(saved_files["random_catalog"])
    assert expected_displacement_cols.issubset(random_cols)


def test_pipeline_float64_dtype_preserved(tmp_path, dummy_catalogs):
    """reconstruction.dtype=float64 must survive end-to-end to the saved catalog.

    The default working precision is float32; when float64 is requested the
    reconstructed coordinate and displacement columns written to disk must be
    float64, not silently downcast.
    """
    data_path, random_path = dummy_catalogs
    output_dir = tmp_path / "output"

    config_dict = {
        "catalog": {"data_path": data_path, "random_path": random_path},
        "columns": {
            "coordinates": {"ra": "RA", "dec": "DEC", "redshift": "REDSHIFT"},
            "weights": {"data": "WEIGHT", "random": "WEIGHT"},
        },
        "coordinate_system": {"input": "ra_dec_z"},
        "cosmology": {"Om0": 0.3, "H0": 70},
        "reconstruction": {
            "redshift": 0.5,
            "nmesh": 16,
            "R_sm": 15.0,
            "f": 0.7,
            "bias": 1.5,
            "pbc": True,
            "solver_type": "ifft",
            "device": "cpu",
            "dtype": "float64",
        },
        "output": {
            "folder": str(output_dir),
            "naming_pattern": "test_float64",
            "save": ["catalogs", "tracer_displacements"],
        },
        "catalog_name": "test_survey",
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    pipeline = ReconstructionPipeline(str(config_path))
    saved_files = pipeline.run()

    data = _read_catalog(saved_files["data_catalog"])
    for col in ("RA", "DEC", "REDSHIFT", "S_X", "S_Y", "S_Z"):
        assert data[col].dtype == np.float64, f"{col} is {data[col].dtype}, expected float64"