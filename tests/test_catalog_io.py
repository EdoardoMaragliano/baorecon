import numpy as np
import pytest
from astropy.table import Table

from baorecon.io.catalog_io import Catalog
from baorecon.io.config import CatalogConfig


# Mock minimale della classe ColumnsConfig per simulare la configurazione
class DummyColumns:
    ra = "RA"
    dec = "DEC"
    redshift = "Z"
    weight_data = "WEIGHT"
    weight_random = "WEIGHT"
    id_data = "ID"
    id_random = "ID"


# Mock minimale di CatalogConfig per evitare di dover caricare un vero file YAML
class DummyConfig:
    data_path = ""
    random_path = ""
    data_hdu = 1
    random_hdu = 1
    columns = DummyColumns()


@pytest.fixture
def mock_catalogs_setup(tmp_path):
    """Fixture che crea due file FITS temporanei e restituisce la configurazione."""
    data_fpath = tmp_path / "mock_data.fits"
    random_fpath = tmp_path / "mock_random.fits"
    
    # 1. Prepariamo i dati mock (100 galassie, 200 random)
    size_d = 100
    size_r = 200
    
    t_data = Table({
        "RA": np.random.uniform(0, 360, size_d),
        "DEC": np.random.uniform(-90, 90, size_d),
        "Z": np.random.uniform(0.1, 0.8, size_d),
        "WEIGHT": np.ones(size_d),
        "ID": np.arange(size_d)
    })
    
    t_random = Table({
        "RA": np.random.uniform(0, 360, size_r),
        "DEC": np.random.uniform(-90, 90, size_r),
        "Z": np.random.uniform(0.1, 0.8, size_r),
        "WEIGHT": np.ones(size_r),
        "ID": np.arange(size_r)
    })
    
    # Scrittura dei FITS temporanei
    t_data.write(data_fpath, format="fits", overwrite=True)
    t_random.write(random_fpath, format="fits", overwrite=True)
    
    # Configurazione fittizia che punta ai file temporanei
    config = DummyConfig()
    config.data_path = str(data_fpath)
    config.random_path = str(random_fpath)
    
    return config, t_data, t_random


def test_catalog_load_and_extraction(mock_catalogs_setup):
    """Testa che il caricamento e l'estrazione delle coordinate siano corretti."""
    config, t_data, t_random = mock_catalogs_setup
    
    # Inizializziamo il catalogo
    catalog = Catalog(config)
    
    # Estraiamo le matrici
    d_pos, d_w, d_ids, r_pos, r_w, r_ids = catalog.get_positions_weights_ids()
    
    # Verifichiamo le dimensioni delle matrici estratte
    assert d_pos.shape == (100, 3)
    assert r_pos.shape == (200, 3)
    assert len(d_w) == 100
    assert len(r_w) == 200
    
    # Verifichiamo la precisione dei dati estratti (RA della prima riga)
    assert np.isclose(d_pos[0, 0], t_data["RA"][0])
    assert np.isclose(r_pos[0, 1], t_random["DEC"][0])


def test_catalog_apply_mask(mock_catalogs_setup):
    """Testa l'applicazione delle maschere booleane."""
    config, _, _ = mock_catalogs_setup
    catalog = Catalog(config)
    
    # Creiamo una maschera booleana (es. teniamo solo le galassie con indice pari)
    catalog.load()  # Forza il caricamento preliminare
    mask_data = np.arange(len(catalog.data_table)) % 2 == 0
    
    initial_len = len(catalog.data_table)
    catalog.apply_mask(mask_data, is_data=True)
    
    # Verifica che la lunghezza sia stata dimezzata correttamente
    assert len(catalog.data_table) == initial_len // 2


def test_catalog_build_output_table(mock_catalogs_setup):
    """Testa la sovrascrittura delle coordinate RA/DEC/Z con quelle ricostruite."""
    config, _, _ = mock_catalogs_setup
    catalog = Catalog(config)
    catalog.load()
    
    # Prepariamo coordinate ricostruite fittizie (RA/DEC come array 2D)
    size_d = len(catalog.data_table)
    mock_rec_radec = np.random.uniform(0, 360, size=(size_d, 2))
    mock_rec_radec[:, 1] = np.clip(mock_rec_radec[:, 1], -90, 90)  # DEC in range
    mock_rec_z = np.random.uniform(0.1, 0.8, size=size_d)
    
    # Ricorda i valori originali
    original_ra = catalog.data_table["RA"][0]
    original_dec = catalog.data_table["DEC"][0]
    original_z = catalog.data_table["Z"][0]
    
    # Generiamo la tabella di output
    output_table = catalog.build_output_table(
        is_data=True,
        reconstructed_radec=mock_rec_radec,
        reconstructed_redshift=mock_rec_z
    )
    
    # Verifichiamo che le colonne originali siano state sovrascritte
    assert "RA" in output_table.colnames
    assert "DEC" in output_table.colnames
    assert "Z" in output_table.colnames
    
    # Verifichiamo che i valori siano cambiarti (nuovo != originale)
    assert not np.isclose(output_table["RA"][0], original_ra)
    assert not np.isclose(output_table["DEC"][0], original_dec)
    assert not np.isclose(output_table["Z"][0], original_z)
    
    # Verifichiamo che i nuovi valori corrispondano a quelli passati
    np.testing.assert_allclose(output_table["RA"], mock_rec_radec[:, 0])
    np.testing.assert_allclose(output_table["DEC"], mock_rec_radec[:, 1])
    np.testing.assert_allclose(output_table["Z"], mock_rec_z)