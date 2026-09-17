import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from baorecon.io.backends import resolve_format
from baorecon.io.catalog_io import Catalog
from baorecon.io.config import ColumnMapping


# Mock minimale della configurazione delle colonne.
def DummyColumns(keep_cols=None, **overrides):
    """The real ColumnMapping, not a stand-in.

    This used to be a hand-written stub with the same attributes, which drifted
    the moment ColumnMapping grew a method -- the tests then exercised a shape
    the code no longer had.
    """
    return ColumnMapping(
        ra="RA", dec="DEC", redshift="Z",
        weight_data="WEIGHT", weight_random="WEIGHT",
        id_data="ID", id_random="ID",
        keep_cols=list(keep_cols) if keep_cols is not None else [],
        **overrides,
    )


# Mock minimale di CatalogConfig per evitare di caricare un vero file YAML.
class DummyConfig:
    def __init__(self, data_path, random_path, keep_cols=None, catalog_format=None,
                 columns=None):
        self.data_path = str(data_path)
        self.random_path = str(random_path)
        self.data_hdu = 1
        self.random_hdu = 1
        self.catalog_format = catalog_format
        self.columns = columns if columns is not None else DummyColumns(keep_cols)


def _make_data(size, seed):
    rng = np.random.default_rng(seed)
    return {
        "RA": rng.uniform(0, 360, size),
        "DEC": rng.uniform(-90, 90, size),
        "Z": rng.uniform(0.1, 0.8, size),
        "WEIGHT": np.ones(size),
        "ID": np.arange(size),
        "EXTRA": rng.uniform(0, 1, size),  # column not needed by the pipeline
    }


@pytest.fixture
def mock_catalogs_setup(tmp_path):
    """Crea due file FITS temporanei e restituisce la configurazione."""
    data_fpath = tmp_path / "mock_data.fits"
    random_fpath = tmp_path / "mock_random.fits"

    data_dict = _make_data(100, seed=1)
    random_dict = _make_data(200, seed=2)
    t_data = Table(data_dict)
    t_random = Table(random_dict)

    t_data.write(data_fpath, format="fits", overwrite=True)
    t_random.write(random_fpath, format="fits", overwrite=True)

    config = DummyConfig(data_fpath, random_fpath)
    return config, t_data, t_random


def test_catalog_load_and_extraction(mock_catalogs_setup):
    """Caricamento ed estrazione delle coordinate (percorso FITS)."""
    config, t_data, t_random = mock_catalogs_setup

    catalog = Catalog(config)
    (
        d_ra, d_dec, d_z, d_w, d_ids,
        r_ra, r_dec, r_z, r_w, r_ids,
    ) = catalog.get_positions_weights_ids()

    assert d_ra.shape == (100,)
    assert d_dec.shape == (100,)
    assert d_z.shape == (100,)
    assert r_ra.shape == (200,)
    assert len(d_w) == 100
    assert len(r_w) == 200

    assert np.isclose(d_ra[0], t_data["RA"][0])
    assert np.isclose(r_dec[0], t_random["DEC"][0])


def test_catalog_apply_mask(mock_catalogs_setup):
    """Applicazione di una maschera booleana."""
    config, _, _ = mock_catalogs_setup
    catalog = Catalog(config)

    catalog.load()
    mask_data = np.arange(len(catalog.data)) % 2 == 0
    initial_len = len(catalog.data)
    catalog.apply_mask(mask_data, is_data=True)

    assert len(catalog.data) == initial_len // 2
    # L'indice deve essere stato reimpostato dopo il filtro.
    assert list(catalog.data.index) == list(range(len(catalog.data)))


def test_catalog_build_output_table(mock_catalogs_setup):
    """Sovrascrittura di RA/DEC/Z con le coordinate ricostruite (ritorna DataFrame)."""
    config, _, _ = mock_catalogs_setup
    catalog = Catalog(config)
    catalog.load()

    size_d = len(catalog.data)
    mock_rec_radec = np.random.uniform(0, 360, size=(size_d, 2))
    mock_rec_radec[:, 1] = np.clip(mock_rec_radec[:, 1], -90, 90)
    mock_rec_z = np.random.uniform(0.1, 0.8, size=size_d)

    original_ra = catalog.data["RA"].iloc[0]
    original_dec = catalog.data["DEC"].iloc[0]
    original_z = catalog.data["Z"].iloc[0]

    output = catalog.build_output_table(
        is_data=True,
        reconstructed_radec=mock_rec_radec,
        reconstructed_redshift=mock_rec_z,
    )

    assert isinstance(output, pd.DataFrame)
    assert {"RA", "DEC", "Z"}.issubset(output.columns)

    assert not np.isclose(output["RA"].iloc[0], original_ra)
    assert not np.isclose(output["DEC"].iloc[0], original_dec)
    assert not np.isclose(output["Z"].iloc[0], original_z)

    np.testing.assert_allclose(output["RA"].to_numpy(), mock_rec_radec[:, 0])
    np.testing.assert_allclose(output["DEC"].to_numpy(), mock_rec_radec[:, 1])
    np.testing.assert_allclose(output["Z"].to_numpy(), mock_rec_z)


def test_column_pruning_drops_unneeded_columns(mock_catalogs_setup):
    """keep_cols attiva il pruning: le colonne extra non vengono caricate."""
    config, _, _ = mock_catalogs_setup
    config.columns.keep_cols = ["RA", "DEC", "Z", "WEIGHT"]

    catalog = Catalog(config)
    catalog.load()

    assert "EXTRA" not in catalog.data.columns
    # Le colonne di compute (incluso ID via id_data) sono sempre presenti.
    assert {"RA", "DEC", "Z", "WEIGHT", "ID"}.issubset(catalog.data.columns)


def test_legacy_reads_all_columns(mock_catalogs_setup):
    """Senza keep_cols si legge tutto (comportamento legacy)."""
    config, _, _ = mock_catalogs_setup  # keep_cols vuoto di default
    catalog = Catalog(config)
    catalog.load()
    assert "EXTRA" in catalog.data.columns


def test_parquet_roundtrip(tmp_path):
    """Lettura/estrazione dal backend Parquet (formato inferito dall'estensione)."""
    data_fpath = tmp_path / "mock_data.parquet"
    random_fpath = tmp_path / "mock_random.parquet"

    pd.DataFrame(_make_data(100, seed=1)).to_parquet(data_fpath, index=False)
    pd.DataFrame(_make_data(200, seed=2)).to_parquet(random_fpath, index=False)

    config = DummyConfig(data_fpath, random_fpath)
    catalog = Catalog(config)
    (
        d_ra, d_dec, d_z, d_w, d_ids,
        r_ra, r_dec, r_z, r_w, r_ids,
    ) = catalog.get_positions_weights_ids()

    assert d_ra.shape == (100,)
    assert r_ra.shape == (200,)


def test_fits_parquet_parity(tmp_path):
    """Gli stessi dati danno posizioni identiche da FITS e da Parquet."""
    data_dict = _make_data(50, seed=7)
    random_dict = _make_data(60, seed=8)

    fits_data = tmp_path / "d.fits"
    fits_random = tmp_path / "r.fits"
    Table(data_dict).write(fits_data, format="fits", overwrite=True)
    Table(random_dict).write(fits_random, format="fits", overwrite=True)

    pq_data = tmp_path / "d.parquet"
    pq_random = tmp_path / "r.parquet"
    pd.DataFrame(data_dict).to_parquet(pq_data, index=False)
    pd.DataFrame(random_dict).to_parquet(pq_random, index=False)

    cat_fits = Catalog(DummyConfig(fits_data, fits_random))
    cat_pq = Catalog(DummyConfig(pq_data, pq_random))

    fits_pos = cat_fits.get_positions_weights_ids()[0]
    pq_pos = cat_pq.get_positions_weights_ids()[0]
    np.testing.assert_allclose(fits_pos, pq_pos)


@pytest.mark.parametrize("ext", ["fits", "parquet"])
def test_write_output_roundtrip(tmp_path, ext):
    """write_output costruisce e scrive nel formato richiesto, rileggibile."""
    data_dict = _make_data(40, seed=3)
    random_dict = _make_data(40, seed=4)
    Table(data_dict).write(tmp_path / "d.fits", format="fits", overwrite=True)
    Table(random_dict).write(tmp_path / "r.fits", format="fits", overwrite=True)

    catalog = Catalog(DummyConfig(tmp_path / "d.fits", tmp_path / "r.fits"))
    catalog.load()

    n = len(catalog.data)
    rec_radec = np.random.uniform(0, 1, size=(n, 2))
    rec_z = np.random.uniform(0.1, 0.8, size=n)
    displacements = np.random.uniform(-1, 1, size=(n, 3))

    out_path = tmp_path / ("out_data." + ext)
    catalog.write_output(
        path=str(out_path),
        is_data=True,
        reconstructed_radec=rec_radec,
        reconstructed_redshift=rec_z,
        displacements=displacements,
    )

    assert out_path.exists()
    if ext == "parquet":
        back = pd.read_parquet(out_path)
    else:
        back = Table.read(out_path).to_pandas()

    assert {"RA", "DEC", "Z", "S_X", "S_Y", "S_Z"}.issubset(back.columns)
    np.testing.assert_allclose(np.asarray(back["RA"]), rec_radec[:, 0], rtol=1e-5)


def test_resolve_format_unknown_format_raises():
    with pytest.raises(ValueError, match="Unknown catalog format"):
        resolve_format("catalog.fits", fmt="hdf5")


def test_resolve_format_unknown_extension_raises():
    with pytest.raises(ValueError, match="Cannot infer catalog format"):
        resolve_format("catalog.csv")


# ==========================================
# PER-CATALOGUE COORDINATE COLUMNS
# ==========================================
def test_randoms_are_read_through_their_own_coordinate_columns(tmp_path):
    """The two catalogues may spell their coordinates differently."""
    data = _make_data(40, seed=1)
    randoms = _make_data(90, seed=2)
    # Same randoms, observed spelling.
    randoms = {"RIGHT_ASCENSION": randoms["RA"], "DECLINATION": randoms["DEC"],
               "SPE_Z": randoms["Z"], "WEIGHT": randoms["WEIGHT"], "ID": randoms["ID"]}

    data_path = tmp_path / "data.fits"
    random_path = tmp_path / "random.fits"
    Table(data).write(data_path, overwrite=True)
    Table(randoms).write(random_path, overwrite=True)

    columns = DummyColumns(ra_random="RIGHT_ASCENSION", dec_random="DECLINATION",
                           redshift_random="SPE_Z")
    catalog = Catalog(DummyConfig(data_path, random_path, columns=columns))
    catalog.load()

    (_, _, _, _, _, random_ra, random_dec, random_z, _, _) = \
        catalog.get_positions_weights_ids()
    np.testing.assert_allclose(random_ra, randoms["RIGHT_ASCENSION"], rtol=1e-6)
    np.testing.assert_allclose(random_z, randoms["SPE_Z"], rtol=1e-6)


def test_output_table_keeps_the_spelling_of_its_own_catalogue(tmp_path):
    """A random written back keeps RIGHT_ASCENSION, not the data's RA."""
    data = _make_data(40, seed=1)
    randoms = _make_data(90, seed=2)
    randoms = {"RIGHT_ASCENSION": randoms["RA"], "DECLINATION": randoms["DEC"],
               "SPE_Z": randoms["Z"], "WEIGHT": randoms["WEIGHT"], "ID": randoms["ID"]}

    data_path = tmp_path / "data.fits"
    random_path = tmp_path / "random.fits"
    Table(data).write(data_path, overwrite=True)
    Table(randoms).write(random_path, overwrite=True)

    columns = DummyColumns(ra_random="RIGHT_ASCENSION", dec_random="DECLINATION",
                           redshift_random="SPE_Z")
    catalog = Catalog(DummyConfig(data_path, random_path, columns=columns))
    catalog.load()

    n = len(catalog.random)
    shifted_ra = np.zeros(n)
    shifted_dec = np.ones(n)
    table = catalog.build_output_table(
        False, (shifted_ra, shifted_dec), reconstructed_redshift=None)

    assert "RIGHT_ASCENSION" in table.columns
    assert "RA" not in table.columns
    np.testing.assert_allclose(table["RIGHT_ASCENSION"].to_numpy(), shifted_ra)
