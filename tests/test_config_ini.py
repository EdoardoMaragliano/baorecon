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


def test_align_cone_is_read_from_the_parfile(tmp_path, catalogs):
    """The whitelist in from_ini must carry align_cone through, as a real bool."""
    assert CatalogConfig.from_ini(
        _write_parfile(tmp_path, catalogs)).reconstruction["align_cone"] is False

    flipped = _write_parfile(
        tmp_path, catalogs, **{"align_cone = false": "align_cone = true"})
    assert CatalogConfig.from_ini(flipped).reconstruction["align_cone"] is True


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


def test_randoms_may_name_their_coordinates_differently(tmp_path, catalogs):
    """Observed data against randoms written to another convention.

    The columns are per-catalogue in the parfile and per-catalogue in the
    config, so the two need not agree.
    """
    text = PARFILE.read_text().replace("/path/to/data_catalog.fits", catalogs[0])
    text = text.replace("/path/to/random_catalog.fits", catalogs[1])
    # Only the [Catalog.Random] occurrence, i.e. the last one.
    head, _, tail = text.rpartition("coord1 = RA")
    path = tmp_path / "mismatch.ini"
    path.write_text(head + "coord1 = RIGHT_ASCENSION" + tail)

    config = CatalogConfig.from_ini(str(path))
    assert config.columns.coordinates(is_data=True) == ("RA", "DEC", "REDSHIFT")
    assert config.columns.coordinates(is_data=False) == (
        "RIGHT_ASCENSION", "DEC", "REDSHIFT")


def test_random_coordinates_default_to_the_data_ones(tmp_path, catalogs):
    """Dropping them from [Catalog.Random] inherits, which is the common case."""
    text = Path(_write_parfile(tmp_path, catalogs)).read_text()
    head, _, tail = text.rpartition("coord1 = RA")
    path = tmp_path / "inherit.ini"
    path.write_text(head + tail)

    config = CatalogConfig.from_ini(str(path))
    assert config.columns.coordinates(is_data=False) == ("RA", "DEC", "REDSHIFT")
    assert config.columns.ra_random is None


# ==========================================
# [Recon] IS THE ONE STRICT SECTION
# ==========================================
def test_typo_in_recon_is_rejected(tmp_path, catalogs):
    """A dropped key is a run that silently used the default instead."""
    text = Path(_write_parfile(tmp_path, catalogs)).read_text()
    path = tmp_path / "typo.ini"
    path.write_text(text.replace("\nR_sm = 15.0\n", "\nR_smm = 15.0\n"))
    with pytest.raises(ValueError, match="R_smm"):
        CatalogConfig.from_ini(str(path))


def test_foreign_keys_in_recon_are_rejected(tmp_path, catalogs):
    """A processing element's own settings do not belong in baorecon's section."""
    text = Path(_write_parfile(tmp_path, catalogs)).read_text()
    path = tmp_path / "foreign.ini"
    path.write_text(text.replace("\n[Recon]\n",
                                 "\n[Recon]\nnside_mask = 1024\nopt_nmesh = false\n"))
    with pytest.raises(ValueError, match="nside_mask"):
        CatalogConfig.from_ini(str(path))


def test_foreign_keys_in_their_own_section_are_ignored(tmp_path, catalogs):
    """...and that is where they go: unknown sections are the extension point."""
    text = Path(_write_parfile(tmp_path, catalogs)).read_text()
    path = tmp_path / "scoped.ini"
    path.write_text(text + "\n[Recon.Euclid]\nnside_mask = 1024\nopt_nmesh = false\n")

    config = CatalogConfig.from_ini(str(path))
    assert "nside_mask" not in config.reconstruction
    assert config.reconstruction["R_sm"] == pytest.approx(15.0)


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
# COORDINATE SYSTEM (Euclid spelling)
# ==========================================
def test_coordinates_uses_the_euclid_vocabulary(tmp_path, catalogs):
    """The parfile speaks PSEUDO_EQUATORIAL/CARTESIAN; the internal names differ."""
    config = CatalogConfig.from_ini(_write_parfile(tmp_path, catalogs))
    assert config.coordinate_system["input"] == "ra_dec_z"

    cart = _write_parfile(
        tmp_path, catalogs,
        **{"coordinates = PSEUDO_EQUATORIAL": "coordinates = CARTESIAN"})
    assert CatalogConfig.from_ini(cart).coordinate_system["input"] == "cartesian"


def test_coordinates_value_is_case_insensitive(tmp_path, catalogs):
    path = _write_parfile(
        tmp_path, catalogs,
        **{"coordinates = PSEUDO_EQUATORIAL": "coordinates = cartesian"})
    assert CatalogConfig.from_ini(path).coordinate_system["input"] == "cartesian"


def test_internal_names_are_not_accepted_in_the_parfile(tmp_path, catalogs):
    """baorecon's own spelling must not leak into the Euclid-facing file."""
    path = _write_parfile(
        tmp_path, catalogs,
        **{"coordinates = PSEUDO_EQUATORIAL": "coordinates = ra_dec_z"})
    with pytest.raises(ValueError, match="PSEUDO_EQUATORIAL"):
        CatalogConfig.from_ini(path)


def test_unknown_coordinates_fails_at_load_time(tmp_path, catalogs):
    """Not at conversion time, i.e. before both catalogues are read from disk."""
    path = _write_parfile(
        tmp_path, catalogs,
        **{"coordinates = PSEUDO_EQUATORIAL": "coordinates = GALACTIC"})
    with pytest.raises(ValueError, match="GALACTIC"):
        CatalogConfig.from_ini(path)


def test_coordinates_may_be_omitted(tmp_path, catalogs):
    """Absent -> the key is not set, and the pipeline default (sky) applies."""
    path = _write_parfile(
        tmp_path, catalogs, **{"coordinates = PSEUDO_EQUATORIAL": ""})
    config = CatalogConfig.from_ini(path)
    assert "input" not in config.coordinate_system
    from baorecon.io.config import resolve_coordinate_input
    assert resolve_coordinate_input(config.coordinate_system) == "ra_dec_z"


# ==========================================
# TOLERANCE OF A REAL 2PCF PARFILE
# ==========================================
def test_unknown_2pcf_sections_and_keys_are_ignored(tmp_path, catalogs):
    """A file carrying the 2PCF's own blocks must still load.

    The two codes keep separate parameter files in the same format, and the
    [Catalog.*] and [Cosmology] blocks are copied between them. The 2PCF's
    versions carry density, mask, name and more, so a copied block has to load
    as it stands rather than needing the other code's keys stripped out first.
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
# COSMOLOGY BEYOND THE COMOVING DISTANCE
# ==========================================
def test_cosmology_extra_reports_what_create_cosmology_cannot_take(tmp_path, catalogs):
    """The inert 2PCF keys are validated and then reported, not dropped."""
    config = CatalogConfig.from_ini(_write_parfile(tmp_path, catalogs))

    # `cosmology` stays exactly the create_cosmology call payload: it is
    # splatted into it, so a stray key there would be a TypeError.
    assert set(config.cosmology) <= {"H0", "Om0", "Ob0", "Tcmb0", "Mnu", "name"}

    extra = config.cosmology_extra
    assert extra["ns"] == pytest.approx(0.96)
    assert extra["sigma8"] == pytest.approx(0.83)
    assert extra["Omega_L"] == pytest.approx(0.6825)
    assert extra["h"] == pytest.approx(0.6711)
    assert extra["w0"] == pytest.approx(-1.0)


def test_cosmology_extra_picks_up_keys_the_2pcf_block_lacks(tmp_path, catalogs):
    """`As` and `om_nu` have no 2PCF key; added to the block, they come through.

    A linear power spectrum needs both, and without them a code layered on top
    has to re-read the file for itself.
    """
    text = Path(_write_parfile(tmp_path, catalogs)).read_text()
    path = tmp_path / "with_as.ini"
    path.write_text(text.replace("\nsigma8 = 0.83\n",
                                 "\nsigma8 = 0.83\nAs = 2.13e-9\nom_nu = 0.0\n"))

    extra = CatalogConfig.from_ini(str(path)).cosmology_extra
    assert extra["As"] == pytest.approx(2.13e-9)
    assert extra["Omega_nu"] == pytest.approx(0.0)


def test_cosmology_extra_is_empty_when_the_block_carries_nothing_extra(tmp_path, catalogs):
    """It is optional: a minimal [Cosmology] simply produces nothing."""
    text = Path(_write_parfile(tmp_path, catalogs)).read_text()
    minimal = []
    for line in text.splitlines(keepends=True):
        key = line.split("=")[0].strip()
        if key in ("om_vac", "om_k", "om_radiation", "spectral_index",
                   "w_eos", "N_eff", "sigma8", "hubble"):
            continue
        minimal.append(line)
    path = tmp_path / "minimal_cosmo.ini"
    # `hubble` was dropped above but H0 needs it; put it back on its own.
    path.write_text("".join(minimal).replace("\nHubble = 100.0\n",
                                             "\nHubble = 100.0\nhubble = 0.6711\n"))

    config = CatalogConfig.from_ini(str(path))
    assert config.cosmology["H0"] == pytest.approx(67.11)
    assert config.cosmology_extra == {"h": pytest.approx(0.6711)}


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

    # Column mapping. Compared through coordinates(), because the two examples
    # express the same thing differently on purpose: the parfile spells the
    # random coordinate columns out (the 2PCF shape carries them per catalogue),
    # the YAML leaves the optional block out and inherits. The resolved names are
    # what anything downstream sees.
    assert from_ini.columns.coordinates(True) == from_yaml.columns.coordinates(True)
    assert from_ini.columns.coordinates(False) == from_yaml.columns.coordinates(False)
    ini_rest = asdict(from_ini.columns)
    yaml_rest = asdict(from_yaml.columns)
    for key in ("ra_random", "dec_random", "redshift_random"):
        ini_rest.pop(key), yaml_rest.pop(key)
    assert ini_rest == yaml_rest

    # Cosmology: the INI carries `name` from cosmology_ID, the YAML spells its
    # own label, so compare the numbers and check both name the same universe.
    for key in ("H0", "Om0", "Ob0", "Tcmb0"):
        assert from_ini.cosmology[key] == pytest.approx(from_yaml.cosmology[key]), key

    # Coordinate system. `input` and `frame` are dead settings (nothing reads
    # them), so neither example carries them and only the live pair is compared.
    for key in ("input", "ra_dec_unit", "distance_unit"):
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

    # The background beyond the distance: the INI derives it from the 2PCF
    # spelling, the YAML states it directly, and the two must agree.
    assert set(from_ini.cosmology_extra) == set(from_yaml.cosmology_extra)
    for key, value in from_ini.cosmology_extra.items():
        assert value == pytest.approx(from_yaml.cosmology_extra[key]), key
