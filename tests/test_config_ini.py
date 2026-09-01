"""Tests for the INI (2PCF/Euclid-style) configuration front-end.

The central guarantee is the round-trip in
:func:`test_ini_and_yaml_produce_equivalent_config`: the shipped parfile and the
shipped YAML example describe the same run, so the INI front-end is a pure
re-spelling of the YAML one and nothing downstream of ``CatalogConfig`` can tell
them apart.
"""

import logging
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from astropy.table import Table
import numpy as np

from baorecon.io.config import CatalogConfig

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
PARFILE = EXAMPLES / "bao_pipeline_parfile.ini"
YAML_EXAMPLE = EXAMPLES / "bao_pipeline_example.yaml"


@pytest.fixture
def catalogs(tmp_path):
    """Two dummy FITS catalogues; CatalogConfig.__post_init__ requires them."""
    paths = []
    for name, rows in (("data.fits", 20), ("random.fits", 50)):
        table = Table()
        table["RA"] = np.random.uniform(10, 20, rows)
        table["DEC"] = np.random.uniform(10, 20, rows)
        table["REDSHIFT"] = np.random.uniform(0.4, 0.6, rows)
        table["WEIGHT"] = np.ones(rows)
        path = tmp_path / name
        table.write(path, overwrite=True)
        paths.append(str(path))
    return paths


def _write_parfile(tmp_path, catalogs, source=PARFILE, **replacements):
    """Materialise the shipped parfile with real catalogue paths.

    Each replacement must match exactly one line of the file. Substrings such as
    ``pbc = true`` also occur inside the explanatory comments, so an unanchored
    replace can silently edit prose instead of the setting and leave the test
    passing for the wrong reason; the uniqueness check below rules that out.
    """
    data_path, random_path = catalogs
    text = source.read_text()
    text = text.replace("/path/to/data_catalog.fits", data_path)
    text = text.replace("/path/to/random_catalog.fits", random_path)
    for old, new in replacements.items():
        anchored = "\n{0}\n".format(old)
        assert text.count(anchored) == 1, (
            "{0!r} must match exactly one whole line (found {1})".format(
                old, text.count(anchored))
        )
        text = text.replace(anchored, "\n{0}\n".format(new))
    path = tmp_path / "parfile.ini"
    path.write_text(text)
    return str(path)


# ==========================================
# THE SHIPPED PARFILE
# ==========================================
def test_shipped_parfile_loads(tmp_path, catalogs):
    """examples/bao_pipeline_parfile.ini must stay loadable as it is shipped."""
    config = CatalogConfig.from_ini(_write_parfile(tmp_path, catalogs))

    assert config.columns.ra == "RA"
    assert config.columns.dec == "DEC"
    assert config.columns.redshift == "REDSHIFT"
    assert config.columns.weight_data == "WEIGHT"
    assert config.columns.id_data is None          # present but empty -> None
    assert config.catalog_format is None           # empty -> infer from extension
    assert config.catalog_name == "sample_survey"
    assert config.data_hdu == 1


def test_cosmology_is_translated_from_2pcf_names(tmp_path, catalogs):
    """H0 is a product, not a rename, and the inert 2PCF keys are dropped."""
    config = CatalogConfig.from_ini(_write_parfile(tmp_path, catalogs))

    # Hubble * hubble = 100.0 * 0.6711
    assert config.cosmology["H0"] == pytest.approx(67.11)
    assert config.cosmology["Om0"] == pytest.approx(0.3175)
    assert config.cosmology["Ob0"] == pytest.approx(0.049)
    assert config.cosmology["Tcmb0"] == pytest.approx(2.7255)
    assert config.cosmology["name"] == "LCDM"      # from cosmology_ID

    # create_cosmology is called with **, so a stray key would be a TypeError.
    assert set(config.cosmology) <= {"H0", "Om0", "Ob0", "Tcmb0", "Mnu", "name"}
    for dropped in ("om_vac", "om_k", "sigma8", "w_eos", "N_eff", "spectral_index"):
        assert dropped not in config.cosmology


def test_cosmology_kwargs_build_a_cosmology(tmp_path, catalogs):
    """The translated block must be directly splattable into create_cosmology."""
    from baorecon.utils.coordinates import create_cosmology

    config = CatalogConfig.from_ini(_write_parfile(tmp_path, catalogs))
    cosmo = create_cosmology(**config.cosmology)
    assert cosmo.H0.value == pytest.approx(67.11)


def test_types_are_coerced(tmp_path, catalogs):
    """configparser yields strings; everything must come back typed."""
    config = CatalogConfig.from_ini(_write_parfile(tmp_path, catalogs))
    recon = config.reconstruction

    assert recon["pbc"] is False                   # not the string "false"
    assert isinstance(recon["nmesh"], int) and recon["nmesh"] == 256
    assert isinstance(recon["n_iterations"], int)
    assert recon["R_sm"] == pytest.approx(15.0)
    # Key case survived optionxform: these would be "mas"/"rsdspace" by default.
    assert recon["MAS"] == "CIC"
    assert "RSDspace" in recon
    assert recon["RSDspace"] in ("RealSpace", "RedshiftSpace")
    assert config.output["save_metadata"] is True
    assert config.output["save"] == ["catalogs"]
    assert config.columns.keep_cols == ["RA", "DEC", "REDSHIFT", "WEIGHT"]


def test_booleans_are_coerced_both_ways(tmp_path, catalogs):
    """The classic INI trap: every non-empty string is truthy, "false" included."""
    # The shipped parfile sets pbc = false; check the loader reads it as False
    # (not as the truthy string) and still parses the flipped value correctly.
    assert CatalogConfig.from_ini(
        _write_parfile(tmp_path, catalogs)).reconstruction["pbc"] is False

    flipped = _write_parfile(tmp_path, catalogs, **{"pbc = false": "pbc = true"})
    assert CatalogConfig.from_ini(flipped).reconstruction["pbc"] is True

    bad = _write_parfile(tmp_path, catalogs, **{"pbc = false": "pbc = maybe"})
    with pytest.raises(ValueError, match="boolean"):
        CatalogConfig.from_ini(bad)


def test_empty_means_null_absent_means_default(tmp_path, catalogs):
    """`key =` is null; a missing key is left out so the pipeline default wins."""
    config = CatalogConfig.from_ini(_write_parfile(tmp_path, catalogs))

    # Present but empty in the shipped parfile -> explicit None.
    assert config.reconstruction["los"] is None        # radial LOS
    assert config.reconstruction["boxsize"] is None
    assert config.reconstruction["cellsize"] is None

    # Absent from [Recon] entirely -> not in the dict at all.
    dropped = _write_parfile(tmp_path, catalogs, **{"padding = 100.0": ""})
    assert "padding" not in CatalogConfig.from_ini(dropped).reconstruction


def test_list_valued_keys(tmp_path, catalogs):
    """nmesh/boxsize take a scalar or a comma-separated triple; save is a list."""
    path = _write_parfile(
        tmp_path, catalogs,
        **{"nmesh = 256": "nmesh = 64, 128, 256",
           "save = catalogs": "save = catalogs, grid_density, grid_potential"},
    )
    config = CatalogConfig.from_ini(path)
    assert config.reconstruction["nmesh"] == [64, 128, 256]
    assert config.output["save"] == ["catalogs", "grid_density", "grid_potential"]


# ==========================================
# VALIDATION
# ==========================================
def test_non_flat_cosmology_is_rejected(tmp_path, catalogs):
    """om_k != 0 is a different cosmology, not extra detail to ignore."""
    path = _write_parfile(tmp_path, catalogs, **{"om_k = 0.0": "om_k = 0.05"})
    with pytest.raises(ValueError, match="om_k"):
        CatalogConfig.from_ini(path)


def test_non_lambda_dark_energy_is_rejected(tmp_path, catalogs):
    path = _write_parfile(tmp_path, catalogs, **{"w_eos = -1.0": "w_eos = -0.9"})
    with pytest.raises(ValueError, match="w_eos"):
        CatalogConfig.from_ini(path)


def test_inconsistent_om_vac_warns_but_loads(tmp_path, catalogs, caplog):
    """om_vac is derived, so a stale value is a warning rather than an error."""
    path = _write_parfile(tmp_path, catalogs, **{"om_vac = 0.6825": "om_vac = 0.70"})

    # setup_logger sets propagate=False, so caplog's root handler never sees
    # these records; attach it to the module logger directly.
    module_logger = logging.getLogger("baorecon.io.config")
    module_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="baorecon.io.config"):
            config = CatalogConfig.from_ini(path)
    finally:
        module_logger.removeHandler(caplog.handler)

    assert config.cosmology["Om0"] == pytest.approx(0.3175)
    assert "om_vac" in caplog.text


def test_mismatched_coordinate_columns_are_rejected(tmp_path, catalogs):
    """The columns are per-catalogue in the parfile but global in the config."""
    text = PARFILE.read_text().replace("/path/to/data_catalog.fits", catalogs[0])
    text = text.replace("/path/to/random_catalog.fits", catalogs[1])
    # Only the [Catalog.Random] occurrence, i.e. the last one.
    head, _, tail = text.rpartition("coord1 = RA")
    path = tmp_path / "mismatch.ini"
    path.write_text(head + "coord1 = RA_RANDOM" + tail)
    with pytest.raises(ValueError, match="coord1"):
        CatalogConfig.from_ini(str(path))


def test_missing_required_key_is_rejected(tmp_path, catalogs):
    """An empty coord column is missing, not null: the read would have no column."""
    text = PARFILE.read_text().replace("/path/to/data_catalog.fits", catalogs[0])
    text = text.replace("/path/to/random_catalog.fits", catalogs[1])
    # `coord2 = DEC` appears in both catalogue sections; blank the [Catalog.Galaxy]
    # one, i.e. the first.
    path = tmp_path / "missing.ini"
    path.write_text(text.replace("\ncoord2 = DEC\n", "\ncoord2 =\n", 1))
    with pytest.raises(ValueError, match="coord2"):
        CatalogConfig.from_ini(str(path))


def test_missing_section_is_rejected(tmp_path, catalogs):
    text = Path(_write_parfile(tmp_path, catalogs)).read_text()
    path = tmp_path / "nogalaxy.ini"
    path.write_text(text.replace("[Catalog.Galaxy]", "[Catalog.Ignored]"))
    with pytest.raises(ValueError, match=r"Catalog\.Galaxy"):
        CatalogConfig.from_ini(str(path))


# ==========================================
# TOLERANCE OF A REAL 2PCF PARFILE
# ==========================================
def test_unknown_2pcf_sections_and_keys_are_ignored(tmp_path, catalogs):
    """A file carrying the 2PCF's own blocks must still load.

    This is what lets one parfile serve both codes: baorecon reads the blocks it
    knows and steps over [2PCF], [Catalog.Reconstructed], density, mask, ...
    """
    text = Path(_write_parfile(tmp_path, catalogs)).read_text()
    # Anchor on the whole line: "[Catalog.Random]" also appears inside comments,
    # and injecting there would land the keys in the previous section instead.
    assert text.count("\n[Catalog.Random]\n") == 1
    text = text.replace(
        "\n[Catalog.Random]\n",
        "\n[Catalog.Random]\ndensity = 999\nmask = 999\n",
    )
    text += (
        "\n[Path]\n"
        "\n[Catalog.Reconstructed]\n"
        "filename = /nonexistent/rec_random.fits\n"
        "\n[2PCF]\n"
        "statistics = AUTO_REC_2DPOL\n"
        "method = LINKED_LIST\n"
        "compute_DD = true\n"
    )
    path = tmp_path / "full_2pcf.ini"
    path.write_text(text)

    config = CatalogConfig.from_ini(str(path))
    assert config.columns.ra == "RA"
    assert config.cosmology["H0"] == pytest.approx(67.11)


# ==========================================
# DISPATCH + EQUIVALENCE WITH THE YAML FRONT-END
# ==========================================
def test_from_file_dispatches_on_extension(tmp_path, catalogs):
    ini_path = _write_parfile(tmp_path, catalogs)
    assert CatalogConfig.from_file(ini_path).columns.ra == "RA"

    with pytest.raises(ValueError, match="Unsupported config extension"):
        CatalogConfig.from_file(str(tmp_path / "config.toml"))


def _yaml_twin(tmp_path, catalogs):
    """The shipped YAML example, pointed at the dummy catalogues."""
    config = yaml.safe_load(YAML_EXAMPLE.read_text())
    config["catalog"]["data_path"], config["catalog"]["random_path"] = catalogs
    path = tmp_path / "twin.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


def test_ini_and_yaml_produce_equivalent_config(tmp_path, catalogs):
    """The two shipped examples must describe the same run.

    Compared field by field rather than with a blanket asdict() equality, so the
    handful of deliberate differences are named explicitly instead of hidden.
    """
    from_ini = CatalogConfig.from_ini(_write_parfile(tmp_path, catalogs))
    from_yaml = CatalogConfig.from_yaml(_yaml_twin(tmp_path, catalogs))

    assert from_ini.data_path == from_yaml.data_path
    assert from_ini.random_path == from_yaml.random_path
    assert from_ini.data_hdu == from_yaml.data_hdu
    assert from_ini.random_hdu == from_yaml.random_hdu
    assert from_ini.catalog_name == from_yaml.catalog_name

    # Column mapping: identical, keep_cols included.
    assert asdict(from_ini.columns) == asdict(from_yaml.columns)

    # Cosmology: the INI carries `name` from cosmology_ID, the YAML spells its
    # own label, so compare the numbers and check both name the same universe.
    for key in ("H0", "Om0", "Ob0", "Tcmb0"):
        assert from_ini.cosmology[key] == pytest.approx(from_yaml.cosmology[key]), key

    # Coordinate system. `input` and `frame` are dead settings (nothing reads
    # them), so neither example carries them and only the live pair is compared.
    for key in ("ra_dec_unit", "distance_unit"):
        assert from_ini.coordinate_system[key] == from_yaml.coordinate_system[key], key
    assert "frame" not in from_ini.coordinate_system

    # Reconstruction. The YAML example leaves a few keys commented out and lets
    # the pipeline default apply, while the parfile spells them; those are the
    # only permitted differences, and each must equal that same default so the
    # two files still describe the same run.
    ini_only_defaults = {"n_iterations": 3, "dtype": "float32", "cellsize": None}
    for key, value in from_ini.reconstruction.items():
        if key in from_yaml.reconstruction:
            assert value == from_yaml.reconstruction[key], key
        else:
            assert key in ini_only_defaults, "{0} is set only in the INI".format(key)
            assert value == ini_only_defaults[key], key
    for key in from_yaml.reconstruction:
        assert key in from_ini.reconstruction, "{0} missing from the parfile".format(key)

    for key in ("folder", "naming_pattern", "format", "save_metadata", "save"):
        assert from_ini.output[key] == from_yaml.output[key], key
