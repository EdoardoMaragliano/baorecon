"""End-to-end BAO reconstruction pipeline driven by YAML configuration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy.table import Table

from zeldareco.BAOreconstruction.bao_reconstructor import BAOReconstructor
from zeldareco.io.catalog_io import Catalog
from zeldareco.io.config import CatalogConfig
from zeldareco.io.naming import NamingTokenizer
from zeldareco.utils.coordinates import create_cosmology, radec_z_to_xyz, xyz_to_radec_z
from zeldareco.utils.loggers import setup_logger

logger = setup_logger(__name__)


class ReconstructionPipeline:
    """High-level orchestrator that runs catalog I/O and BAO reconstruction."""

    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.config = CatalogConfig.from_yaml(config_file)
        self.catalog = Catalog(self.config)
        self.cosmology = create_cosmology(**self.config.cosmology)

        self.data_pos_radec: Optional[np.ndarray] = None
        self.random_pos_radec: Optional[np.ndarray] = None
        self.data_pos_xyz: Optional[np.ndarray] = None
        self.random_pos_xyz: Optional[np.ndarray] = None
        self.data_weights: Optional[np.ndarray] = None
        self.random_weights: Optional[np.ndarray] = None
        self.data_ids: Optional[np.ndarray] = None
        self.random_ids: Optional[np.ndarray] = None
        self.reconstructor: Optional[BAOReconstructor] = None
        self.data_rec_xyz: Optional[np.ndarray] = None
        self.random_rec_xyz: Optional[np.ndarray] = None
        self.data_rec_radec: Optional[np.ndarray] = None
        self.random_rec_radec: Optional[np.ndarray] = None
        self.data_rec_z: Optional[np.ndarray] = None
        self.random_rec_z: Optional[np.ndarray] = None

    def load_catalogs(self) -> None:
        """Load input FITS catalogs and extract raw arrays."""
        self.catalog.load()
        (
            self.data_pos_radec,
            self.data_weights,
            self.data_ids,
            self.random_pos_radec,
            self.random_weights,
            self.random_ids,
        ) = self.catalog.get_positions_weights_ids()

    def convert_to_xyz(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert RA/DEC/redshift coordinates to Cartesian coordinates."""
        if self.data_pos_radec is None or self.random_pos_radec is None:
            self.load_catalogs()

        coordinate_cfg = self.config.coordinate_system
        ra_dec_unit = coordinate_cfg.get("ra_dec_unit", "deg")
        frame = coordinate_cfg.get("frame", "icrs")
        distance_unit = coordinate_cfg.get("distance_unit", "Mpc/h")

        self.data_pos_xyz, _ = radec_z_to_xyz(
            self.data_pos_radec[:, 0],
            self.data_pos_radec[:, 1],
            self.data_pos_radec[:, 2],
            cosmo=self.cosmology,
            ra_dec_unit=ra_dec_unit,
            frame=frame,
            distance_unit=distance_unit,
        )
        self.random_pos_xyz, _ = radec_z_to_xyz(
            self.random_pos_radec[:, 0],
            self.random_pos_radec[:, 1],
            self.random_pos_radec[:, 2],
            cosmo=self.cosmology,
            ra_dec_unit=ra_dec_unit,
            frame=frame,
            distance_unit=distance_unit,
        )
        return self.data_pos_xyz, self.random_pos_xyz

    '''def convert_to_xyz(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert RA/DEC/redshift coordinates to Cartesian coordinates with memory efficiency."""
        if self.data_pos_radec is None or self.random_pos_radec is None:
            self.load_catalogs()

        coordinate_cfg = self.config.coordinate_system
        ra_dec_unit = coordinate_cfg.get("ra_dec_unit", "deg")
        distance_unit = coordinate_cfg.get("distance_unit", "Mpc/h")

        # --- OPTIMIZE MEMORY ---
        d_ra = np.ascontiguousarray(self.data_pos_radec[:, 0])
        d_dec = np.ascontiguousarray(self.data_pos_radec[:, 1])
        d_z = np.ascontiguousarray(self.data_pos_radec[:, 2])

        self.data_pos_xyz, _ = radec_z_to_xyz(
            d_ra, d_dec, d_z,
            cosmo=self.cosmology,
            ra_dec_unit=ra_dec_unit,
            distance_unit=distance_unit,
        )
        
        # Liberiamo subito i riferimenti temporanei per la RAM
        del d_ra, d_dec, d_z 

        # --- OTTIMIZZAZIONE MEMORIA RANDOM (37 MILIONI DI PUNTI) ---
        # Lo slicing transizionale [:, i] su 37 milioni di righe è lentissimo.
        # Trasformando i dati in vettori 1D contigui separati in memoria in un colpo solo,
        # np.interp e np.cos non dovranno fare salti di memoria giganti (cache misses).
        r_ra = np.ascontiguousarray(self.random_pos_radec[:, 0])
        r_dec = np.ascontiguousarray(self.random_pos_radec[:, 1])
        r_z = np.ascontiguousarray(self.random_pos_radec[:, 2])

        self.random_pos_xyz, _ = radec_z_to_xyz(
            r_ra, r_dec, r_z,
            cosmo=self.cosmology,
            ra_dec_unit=ra_dec_unit,
            distance_unit=distance_unit,
        )
        
        del r_ra, r_dec, r_z

        return self.data_pos_xyz, self.random_pos_xyz'''

    def build_reconstructor(self) -> BAOReconstructor:
        """Instantiate the BAO reconstructor using the configuration values."""
        if self.data_pos_xyz is None or self.random_pos_xyz is None:
            self.convert_to_xyz()

        reconstruction_cfg = self.config.reconstruction
        self.reconstructor = BAOReconstructor(
            data_pos=self.data_pos_xyz,
            random_pos=self.random_pos_xyz,
            data_weights=self.data_weights,
            random_weights=self.random_weights,
            data_ids=self.data_ids,
            RSDspace=reconstruction_cfg.get("RSDspace", "RealSpace"),
            nmesh=int(reconstruction_cfg.get("nmesh", 256)),
            boxsize=reconstruction_cfg.get("boxsize"),
            boxcentre=reconstruction_cfg.get("boxcentre"),
            padding=float(reconstruction_cfg.get("padding", 0.0)),
            los=reconstruction_cfg.get("los", "z"),
            R_sm=float(reconstruction_cfg.get("R_sm", 15.0)),
            pbc=bool(reconstruction_cfg.get("pbc", True)),
            rectype=reconstruction_cfg.get("rectype", "rec-sym"),
            f=float(reconstruction_cfg.get("f", 0.88)),
            bias=float(reconstruction_cfg.get("bias", 1.0)),
            MAS=reconstruction_cfg.get("MAS", "CIC"),
            dtype=np.float32,
            threshold_randoms=float(reconstruction_cfg.get("threshold_randoms", 0.7)),
            solver_type=reconstruction_cfg.get("solver_type", "ifft"),
            n_iterations=int(reconstruction_cfg.get("n_iterations", 3)),
        )
        return self.reconstructor

    def reconstruct(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run the BAO reconstruction and keep the reconstructed XYZ positions."""
        if self.reconstructor is None:
            self.build_reconstructor()

        assert self.reconstructor is not None
        self.data_rec_xyz, self.random_rec_xyz = self.reconstructor.run_reconstruction()
        return self.data_rec_xyz, self.random_rec_xyz

    def convert_back(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert reconstructed XYZ coordinates back to RA/DEC/redshift."""
        if self.data_rec_xyz is None or self.random_rec_xyz is None:
            self.reconstruct()

        logger.info("Converting back to RA/DEC/redshift...")
        coordinate_cfg = self.config.coordinate_system
        ra_dec_unit = coordinate_cfg.get("ra_dec_unit", "deg")
        frame = coordinate_cfg.get("frame", "icrs")
        distance_unit = coordinate_cfg.get("distance_unit", "Mpc/h")

        self.data_rec_radec, _, self.data_rec_z, _ = xyz_to_radec_z(
            self.data_rec_xyz,
            cosmo=self.cosmology,
            ra_dec_unit=ra_dec_unit,
            frame=frame,
            distance_unit=distance_unit,
        )
        self.random_rec_radec, _, self.random_rec_z, _ = xyz_to_radec_z(
            self.random_rec_xyz,
            cosmo=self.cosmology,
            ra_dec_unit=ra_dec_unit,
            frame=frame,
            distance_unit=distance_unit,
        )
        self.data_rec_radec = np.column_stack((self.data_rec_radec, self.data_rec_z))
        self.random_rec_radec = np.column_stack((self.random_rec_radec, self.random_rec_z))
        logger.info("Conversion complete.") 
        return self.data_rec_radec, self.random_rec_radec, self.data_rec_z, self.random_rec_z

    def apply_mask(self, data_mask: Optional[np.ndarray] = None, random_mask: Optional[np.ndarray] = None) -> None:
        """Apply boolean masks while keeping IDs aligned to the filtered catalogues."""
        if data_mask is not None:
            self.catalog.apply_mask(np.asarray(data_mask, dtype=bool), is_data=True)
        if random_mask is not None:
            self.catalog.apply_mask(np.asarray(random_mask, dtype=bool), is_data=False)

    def save_outputs(self) -> Tuple[str, str]:
        """Save reconstructed catalogs into the configured output folder."""
        if self.data_rec_xyz is None or self.random_rec_xyz is None:
            self.reconstruct()
        if self.data_rec_radec is None or self.random_rec_radec is None:
            self.convert_back()

        output_cfg = self.config.output
        output_folder = Path(output_cfg.get("folder", "."))
        output_folder.mkdir(parents=True, exist_ok=True)

        pattern = output_cfg.get("naming_pattern", "rec_{name}_{solver}_{z0:.2f}")
        base_name = NamingTokenizer.format_name(
            pattern,
            name=self.config.catalog_name or "catalog",
            solver=self.config.reconstruction.get("solver_type", "ifft"),
            nmesh=self.config.reconstruction.get("nmesh", 256),
            boxsize=self.config.reconstruction.get("boxsize", "auto"),
            z0=self.config.reconstruction.get("redshift", 1.0),
            H0=self.config.cosmology.get("H0", 67.11),
            Om0=self.config.cosmology.get("Om0", 0.3175),
        )

        data_table = self.catalog.build_output_table(
            is_data=True,
            reconstructed_xyz=self.data_rec_xyz,
            reconstructed_radec=self.data_rec_radec,
            reconstructed_redshift=self.data_rec_z,
        )
        random_table = self.catalog.build_output_table(
            is_data=False,
            reconstructed_xyz=self.random_rec_xyz,
            reconstructed_radec=self.random_rec_radec,
            reconstructed_redshift=self.random_rec_z,
        )

        data_path = str(output_folder / (base_name + "_data.fits"))
        random_path = str(output_folder / (base_name + "_random.fits"))
        data_table.write(data_path, overwrite=True)
        random_table.write(random_path, overwrite=True)

        if output_cfg.get("save_metadata", True):
            metadata_path = output_folder / (base_name + "_metadata.txt")
            metadata_path.write_text(str(asdict(self.config)), encoding="utf-8")

        logger.info("Saved outputs to {0} and {1}".format(data_path, random_path))
        return data_path, random_path

    def run(self) -> Tuple[str, str]:
        """Run the full reconstruction pipeline end-to-end."""
        self.load_catalogs()
        self.convert_to_xyz()
        self.reconstruct()
        self.convert_back()
        return self.save_outputs()
