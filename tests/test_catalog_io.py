import numpy as np
import pytest
from astropy.table import Table

from zeldareco.io.catalog_io import Catalog
from zeldareco.io.config import CatalogConfig


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
    """Testa la corretta generazione della tabella di output con i campi REC_X, Y, Z."""
    config, _, _ = mock_catalogs_setup
    catalog = Catalog(config)
    catalog.load()
    
    # Prepariamo coordinate coordinate di ricostruzione fittizie (XYZ)
    size_d = len(catalog.data_table)
    mock_rec_xyz = np.random.uniform(-100, 100, size=(size_d, 3))
    
    # Generiamo la tabella di output
    output_table = catalog.build_output_table(is_data=True, reconstructed_xyz=mock_rec_xyz, xyz_prefix="REC")
    
    # Verifichiamo che le nuove colonne esistano e contengano i dati corretti
    assert "REC_X" in output_table.colnames
    assert "REC_Y" in output_table.colnames
    assert "REC_Z" in output_table.colnames
    np.testing.assert_allclose(output_table["REC_X"], mock_rec_xyz[:, 0])