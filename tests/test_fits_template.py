"""The template-driven FITS writer: what a derived catalogue inherits.

Writing a reconstructed catalogue with ``Table.from_pandas(df).write(path)``
keeps the numbers and throws away everything else the file carried -- units,
display formats, header keywords, the other HDUs. Passing a ``template`` builds
the output against an existing file instead, so a derived product stays
interchangeable with the one it came from.

The row count is deliberately different from the template's throughout: masking
a catalogue and writing it back against its own input is the case this exists
for, and a writer that quietly required matching lengths would be useless for it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from baorecon.io.backends import get_backend, write_like_template


TEMPLATE_ROWS = 60
OUTPUT_ROWS = 25


def _template(path, *, signed=True, extra_hdu=False):
    """A catalogue with units, a custom keyword and optionally a second table."""
    rng = np.random.default_rng(0)
    columns = [
        fits.Column(name="RA", format="D", unit="deg",
                    array=rng.uniform(0, 360, TEMPLATE_ROWS)),
        fits.Column(name="DEC", format="D", unit="deg",
                    array=rng.uniform(-90, 90, TEMPLATE_ROWS)),
        fits.Column(name="Z", format="E", array=rng.uniform(0.1, 0.8, TEMPLATE_ROWS)),
        fits.Column(name="ID", format="J",
                    array=np.arange(TEMPLATE_ROWS, dtype=np.int32)),
    ]
    table = fits.BinTableHDU.from_columns(columns, name="CATALOG")
    table.header["SURVEY"] = "TESTSURVEY"
    table.header["ZBIN"] = 3

    hdus = [fits.PrimaryHDU(), table]
    if extra_hdu:
        hdus.append(fits.BinTableHDU.from_columns(
            [fits.Column(name="NZ", format="D", array=np.linspace(0, 1, 10))],
            name="NZ_TABLE"))
    fits.HDUList(hdus).writeto(path, checksum=signed)
    return str(path)


def _frame(rows=OUTPUT_ROWS, *, drop=(), add=()):
    """An output frame: the template's columns, minus ``drop``, plus ``add``."""
    rng = np.random.default_rng(1)
    data = {
        "RA": rng.uniform(0, 360, rows),
        "DEC": rng.uniform(-90, 90, rows),
        "Z": rng.uniform(0.1, 0.8, rows),
        "ID": np.arange(rows, dtype=np.int32),
    }
    for name in drop:
        data.pop(name)
    for name in add:
        data[name] = rng.uniform(-1, 1, rows)
    return pd.DataFrame(data)


def _checksum_complaints(path):
    """Whatever astropy says when asked to verify the file's own checksums."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with fits.open(path, checksum=True) as hdul:
            for hdu in hdul:
                _ = hdu.data
    return [str(w.message) for w in caught
            if "hecksum" in str(w.message) or "atasum" in str(w.message)]


# ------------------------------------------------------------------ the schema

def test_inherits_column_order_formats_and_units(tmp_path):
    template = _template(tmp_path / "t.fits")
    out = tmp_path / "o.fits"
    # The frame deliberately lists its columns in a different order.
    frame = _frame()[["ID", "Z", "DEC", "RA"]]

    write_like_template(frame, str(out), template=template, hdu="CATALOG")

    with fits.open(out) as hdul:
        table = hdul["CATALOG"]
        assert [c.name for c in table.columns] == ["RA", "DEC", "Z", "ID"]
        assert [c.format for c in table.columns] == ["D", "D", "E", "J"]
        assert [c.unit for c in table.columns] == ["deg", "deg", None, None]


def test_inherits_header_keywords_but_not_structural_ones(tmp_path):
    template = _template(tmp_path / "t.fits")
    out = tmp_path / "o.fits"

    write_like_template(_frame(), str(out), template=template, hdu="CATALOG")

    with fits.open(out) as hdul:
        header = hdul["CATALOG"].header
        assert header["SURVEY"] == "TESTSURVEY"
        assert header["ZBIN"] == 3
        # Structural keywords describe the new table, not the template's.
        assert header["NAXIS2"] == OUTPUT_ROWS


def test_keeps_the_other_hdus(tmp_path):
    template = _template(tmp_path / "t.fits", extra_hdu=True)
    out = tmp_path / "o.fits"

    write_like_template(_frame(), str(out), template=template, hdu="CATALOG")

    with fits.open(out) as hdul:
        assert [h.name for h in hdul] == ["PRIMARY", "CATALOG", "NZ_TABLE"]
        assert len(hdul["NZ_TABLE"].data) == 10
        assert len(hdul["CATALOG"].data) == OUTPUT_ROWS


def test_row_count_need_not_match_the_template(tmp_path):
    template = _template(tmp_path / "t.fits")
    out = tmp_path / "o.fits"

    write_like_template(_frame(rows=7), str(out), template=template, hdu="CATALOG")

    with fits.open(out) as hdul:
        assert len(hdul["CATALOG"].data) == 7


def test_values_are_cast_to_the_template_dtype(tmp_path):
    """A float64 column written into a float32 TFORM must not lie about it."""
    template = _template(tmp_path / "t.fits")
    out = tmp_path / "o.fits"
    frame = _frame()
    assert frame["Z"].to_numpy().dtype == np.float64  # template says 'E'

    write_like_template(frame, str(out), template=template, hdu="CATALOG")

    with fits.open(out) as hdul:
        column = hdul["CATALOG"].columns["Z"]
        assert column.format == "E"
        np.testing.assert_allclose(
            hdul["CATALOG"].data["Z"], frame["Z"].to_numpy(), rtol=1e-6)


# -------------------------------------------------------------- the integrity pair

def test_checksums_are_recomputed_not_inherited(tmp_path):
    """A signed template yields a signed output that verifies at its own length."""
    template = _template(tmp_path / "t.fits", signed=True)
    out = tmp_path / "o.fits"

    write_like_template(_frame(), str(out), template=template, hdu="CATALOG")

    with fits.open(out) as hdul:
        assert "CHECKSUM" in hdul["CATALOG"].header
        assert "DATASUM" in hdul["CATALOG"].header
    assert _checksum_complaints(out) == []


def test_an_unsigned_template_does_not_acquire_checksums(tmp_path):
    template = _template(tmp_path / "t.fits", signed=False)
    out = tmp_path / "o.fits"

    write_like_template(_frame(), str(out), template=template, hdu="CATALOG")

    with fits.open(out) as hdul:
        assert "CHECKSUM" not in hdul["CATALOG"].header
        assert "DATASUM" not in hdul["CATALOG"].header


# ------------------------------------------------------------ the column-set policy

def test_extra_columns_are_appended_after_the_template_ones(tmp_path):
    template = _template(tmp_path / "t.fits")
    out = tmp_path / "o.fits"
    frame = _frame(add=("S_X", "S_Y", "S_Z"))

    write_like_template(frame, str(out), template=template, hdu="CATALOG")

    with fits.open(out) as hdul:
        table = hdul["CATALOG"]
        assert [c.name for c in table.columns] == [
            "RA", "DEC", "Z", "ID", "S_X", "S_Y", "S_Z"]
        # The template's own metadata is untouched by the additions.
        assert table.columns["RA"].unit == "deg"
        assert table.columns["S_X"].format == "D"
        np.testing.assert_allclose(table.data["S_X"], frame["S_X"].to_numpy())


def test_missing_columns_are_filled_at_the_template_dtype(tmp_path):
    template = _template(tmp_path / "t.fits")
    out = tmp_path / "o.fits"

    write_like_template(_frame(drop=("ID",)), str(out),
                        template=template, hdu="CATALOG")

    with fits.open(out) as hdul:
        table = hdul["CATALOG"]
        assert [c.name for c in table.columns] == ["RA", "DEC", "Z", "ID"]
        assert table.columns["ID"].format == "J"
        np.testing.assert_array_equal(table.data["ID"], np.zeros(OUTPUT_ROWS))


def test_strict_rejects_a_column_the_template_does_not_have(tmp_path):
    template = _template(tmp_path / "t.fits")
    with pytest.raises(KeyError, match="absent from the template"):
        write_like_template(_frame(add=("S_X",)), str(tmp_path / "o.fits"),
                            template=template, hdu="CATALOG", strict=True)


def test_strict_rejects_a_column_the_frame_does_not_supply(tmp_path):
    template = _template(tmp_path / "t.fits")
    with pytest.raises(KeyError, match="missing template columns"):
        write_like_template(_frame(drop=("Z",)), str(tmp_path / "o.fits"),
                            template=template, hdu="CATALOG", strict=True)


def test_strict_accepts_an_exact_match_at_a_different_length(tmp_path):
    """Strict constrains the column set, never the row count."""
    template = _template(tmp_path / "t.fits")
    out = tmp_path / "o.fits"

    write_like_template(_frame(rows=3), str(out),
                        template=template, hdu="CATALOG", strict=True)

    with fits.open(out) as hdul:
        assert len(hdul["CATALOG"].data) == 3


# ----------------------------------------------------------------- the backends

def test_fits_backend_without_a_template_is_unchanged(tmp_path):
    """The historical path: no template, no schema, and no new behaviour."""
    out = tmp_path / "o.fits"
    frame = _frame()

    get_backend(str(out)).write(frame, str(out))

    with fits.open(out) as hdul:
        assert set(hdul[1].columns.names) == set(frame.columns)
        assert hdul[1].columns["RA"].unit is None


def test_parquet_backend_ignores_the_template(tmp_path):
    # Gated, not module-level: the FITS template tests around it are the ones
    # that matter where pyarrow is absent -- EDEN 3.1, for one.
    pytest.importorskip("pyarrow", reason="optional: backs the parquet backend only")

    template = _template(tmp_path / "t.fits")
    out = tmp_path / "o.parquet"
    frame = _frame()

    get_backend(str(out)).write(frame, str(out), template=template, strict=True)

    back = pd.read_parquet(out)
    assert list(back.columns) == list(frame.columns)
    assert len(back) == OUTPUT_ROWS


# -------------------------------------------------------- the configuration path

class _Config:
    """Enough of CatalogConfig for Catalog.write_output to resolve a template."""

    def __init__(self, data_path, random_path, output):
        from baorecon.io.config import ColumnMapping

        self.data_path = str(data_path)
        self.random_path = str(random_path)
        self.data_hdu = "CATALOG"
        self.random_hdu = "CATALOG"
        self.catalog_format = None
        self.output = output
        self.columns = ColumnMapping(
            ra="RA", dec="DEC", redshift="Z",
            weight_data=None, weight_random=None,
            id_data="ID", id_random="ID")


def _catalog(tmp_path, output):
    from baorecon.io.catalog_io import Catalog

    data = _template(tmp_path / "d.fits")
    random = _template(tmp_path / "r.fits")
    catalog = Catalog(_Config(data, random, output))
    catalog.load()
    return catalog


@pytest.mark.parametrize("is_data", [True, False])
def test_template_input_resolves_to_the_tracers_own_catalogue(tmp_path, is_data):
    """Each tracer is written against the file it was read from, not the data's."""
    catalog = _catalog(tmp_path, {"template": "input"})
    out = tmp_path / "rec.fits"
    n = TEMPLATE_ROWS

    catalog.write_output(path=str(out), is_data=is_data,
                         reconstructed_radec=(np.zeros(n), np.ones(n)),
                         reconstructed_redshift=np.full(n, 0.5))

    with fits.open(out) as hdul:
        table = hdul["CATALOG"]
        assert [c.name for c in table.columns] == ["RA", "DEC", "Z", "ID"]
        assert table.columns["RA"].unit == "deg"
        assert table.header["SURVEY"] == "TESTSURVEY"
    assert _checksum_complaints(out) == []


def test_no_template_configured_leaves_the_writer_alone(tmp_path):
    catalog = _catalog(tmp_path, {})
    out = tmp_path / "rec.fits"
    n = TEMPLATE_ROWS

    catalog.write_output(path=str(out), is_data=True,
                         reconstructed_radec=(np.zeros(n), np.ones(n)))

    with fits.open(out) as hdul:
        assert hdul[1].columns["RA"].unit is None


def test_template_strict_from_the_configuration_rejects_displacements(tmp_path):
    """The setting a survey product wants: the output may not grow a column."""
    catalog = _catalog(tmp_path, {"template": "input", "template_strict": True})
    n = TEMPLATE_ROWS

    with pytest.raises(KeyError, match="absent from the template"):
        catalog.write_output(path=str(tmp_path / "rec.fits"), is_data=True,
                             reconstructed_radec=(np.zeros(n), np.ones(n)),
                             displacements=np.zeros((n, 3)))


# ------------------------------------------------------------- format mismatches

def test_a_non_fits_template_is_refused_by_name(tmp_path):
    """Parquet in, FITS out, template = input: the error must name the cause.

    Without the check astropy fails inside ``fits.open`` with "No SIMPLE card
    found", which says nothing about the configuration that produced it.
    """
    # The refusal itself is format-agnostic, but building a non-FITS template to
    # provoke it needs a parquet writer.
    pytest.importorskip("pyarrow", reason="optional: backs the parquet backend only")

    template = tmp_path / "t.parquet"
    _frame().to_parquet(template, index=False)

    with pytest.raises(ValueError, match="not FITS"):
        write_like_template(_frame(), str(tmp_path / "o.fits"),
                            template=str(template))


def test_parquet_output_says_it_is_ignoring_the_template(tmp_path, caplog):
    """The warning is attached to the backend's own logger.

    ``setup_logger`` sets ``propagate = False`` throughout the package, so
    caplog's root handler never sees these records; it has to be hooked onto
    the module logger directly.
    """
    pytest.importorskip("pyarrow", reason="optional: backs the parquet backend only")

    from baorecon.io.backends import parquet_backend

    template = _template(tmp_path / "t.fits")
    out = tmp_path / "o.parquet"

    parquet_backend.logger.addHandler(caplog.handler)
    try:
        get_backend(str(out)).write(_frame(), str(out), template=template)
    finally:
        parquet_backend.logger.removeHandler(caplog.handler)

    assert any("ignores the template" in record.message for record in caplog.records)
    assert len(pd.read_parquet(out)) == OUTPUT_ROWS
