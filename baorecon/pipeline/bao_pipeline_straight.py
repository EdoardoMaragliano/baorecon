"""Legacy end-to-end BAO reconstruction pipeline (kept for reference).

.. deprecated::
    This is the original, monolithic ``ReconstructionPipeline`` implementation,
    superseded by :mod:`baorecon.pipeline.bao_pipeline`. It is retained only as
    a historical/reference variant and is **not** exported by the package --
    ``baorecon.pipeline.ReconstructionPipeline`` points at the current module.

    Compared to the current pipeline it lacks:

    * GPU host-transfer handling (no device -> host coercion of grid outputs);
    * the staged, progressive-release ``run()`` that frees the solver/delta
      grids mid-run to cap peak memory;
    * the modular ``_prepare_output`` / ``_save_grids`` / ``_save_catalogs`` /
      ``_save_metadata`` save path.

    Use :mod:`baorecon.pipeline.bao_pipeline` for all new work.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pickle

import numpy as np
from astropy.io import fits

from baorecon.reconstruction.bao_reconstructor import BAOReconstructor
from baorecon.io.catalog_io import Catalog
from baorecon.io.config import CatalogConfig
from baorecon.io.naming import NamingTokenizer
from baorecon.utils.coordinates import create_cosmology, radec_z_to_xyz, xyz_to_radec_z
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)


def _coerce_nmesh(value):
    """Coerce a YAML ``nmesh`` entry to None, an int, or a list of three ints.

    Accepts a scalar (cubic grid) or a length-3 list/tuple (anisotropic grid);
    validation/broadcasting happens downstream in ``format_nmesh``.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]
    return int(value)


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


    def build_reconstructor(self) -> BAOReconstructor:
        """Instantiate the BAO reconstructor using the configuration values."""
        if self.data_pos_xyz is None or self.random_pos_xyz is None:
            self.convert_to_xyz()

        reconstruction_cfg = self.config.reconstruction

        # Grid resolution: `cellsize` (target cell size) is mutually exclusive
        # with `nmesh`/`boxsize`. When `cellsize` is given, leave `nmesh` unset
        # so it is derived per axis from the catalogue extent; otherwise fall
        # back to the historical 256^3 default.
        cellsize = reconstruction_cfg.get("cellsize")
        if cellsize is not None:
            cellsize = float(cellsize)
        nmesh_cfg = reconstruction_cfg.get("nmesh")
        if cellsize is None and nmesh_cfg is None:
            nmesh_cfg = 256

        self.reconstructor = BAOReconstructor(
            data_pos=self.data_pos_xyz,
            random_pos=self.random_pos_xyz,
            data_weights=self.data_weights,
            random_weights=self.random_weights,
            data_ids=self.data_ids,
            RSDspace=reconstruction_cfg.get("RSDspace", "RealSpace"),
            nmesh=_coerce_nmesh(nmesh_cfg),
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
            device=reconstruction_cfg.get("device", "cpu"),
            cellsize=cellsize,
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

        data_ra, data_dec, self.data_rec_z, _ = xyz_to_radec_z(
            self.data_rec_xyz,
            cosmo=self.cosmology,
            ra_dec_unit=ra_dec_unit,
            frame=frame,
            distance_unit=distance_unit,
        )
        self.data_rec_radec = np.column_stack((data_ra, data_dec))

        random_ra, random_dec, self.random_rec_z, _ = xyz_to_radec_z(
            self.random_rec_xyz,
            cosmo=self.cosmology,
            ra_dec_unit=ra_dec_unit,
            frame=frame,
            distance_unit=distance_unit,
        )
        self.random_rec_radec = np.column_stack((random_ra, random_dec))

        logger.info("Conversion complete.")
        return self.data_rec_radec, self.random_rec_radec, self.data_rec_z, self.random_rec_z

    def apply_mask(self, data_mask: Optional[np.ndarray] = None, random_mask: Optional[np.ndarray] = None) -> None:
        """Apply boolean masks while keeping IDs aligned to the filtered catalogues."""
        if data_mask is not None:
            self.catalog.apply_mask(np.asarray(data_mask, dtype=bool), is_data=True)
        if random_mask is not None:
            self.catalog.apply_mask(np.asarray(random_mask, dtype=bool), is_data=False)

    def save_outputs(self) -> Dict[str, str]:
        """Save configured outputs into the output folder."""
        if self.data_rec_xyz is None or self.random_rec_xyz is None:
            self.reconstruct()
        if self.data_rec_radec is None or self.random_rec_radec is None:
            self.convert_back()

        output_cfg = self.config.output
        save_options = set(output_cfg.get("save", ["catalogs"]))
        output_folder = Path(output_cfg.get("folder", "."))
        output_folder.mkdir(parents=True, exist_ok=True)

        pattern = output_cfg.get("naming_pattern", "rec_{name}_{solver}")
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

        saved_paths: Dict[str, str] = {}

        def _save_fits_image(data: np.ndarray, suffix: str) -> str:
            path = str(output_folder / (base_name + suffix))
            fits.writeto(path, data, overwrite=True, output_verify="silentfix")
            logger.info("Saved FITS image to {0}".format(path))
            return path

        if "catalogs" in save_options or "tracer_displacements" in save_options:
            data_displacements = None
            random_displacements = None
            if "tracer_displacements" in save_options:
                # The displacement vector 's' is defined as s = x_initial - x_reconstructed
                assert self.data_pos_xyz is not None and self.data_rec_xyz is not None
                assert self.random_pos_xyz is not None and self.random_rec_xyz is not None
                logger.info("Calculating tracer displacements from initial and reconstructed positions.")
                data_displacements = self.data_pos_xyz - self.data_rec_xyz
                random_displacements = self.random_pos_xyz - self.random_rec_xyz

            if "catalogs" in save_options:
                fmt = str(self.config.output.get("format", "fits")).lower()
                ext = "parquet" if fmt == "parquet" else "fits"
                data_path = str(output_folder / (base_name + "_data." + ext))
                random_path = str(output_folder / (base_name + "_random." + ext))
                self.catalog.write_output(
                    path=data_path,
                    is_data=True,
                    reconstructed_radec=self.data_rec_radec,
                    reconstructed_redshift=self.data_rec_z,
                    displacements=data_displacements,
                    fmt=fmt,
                )
                self.catalog.write_output(
                    path=random_path,
                    is_data=False,
                    reconstructed_radec=self.random_rec_radec,
                    reconstructed_redshift=self.random_rec_z,
                    displacements=random_displacements,
                    fmt=fmt,
                )
                saved_paths["data_catalog"] = data_path
                saved_paths["random_catalog"] = random_path
                logger.info("Saved catalogs to {0} and {1}".format(data_path, random_path))

        if "grid_potential" in save_options:
            assert self.reconstructor is not None and self.reconstructor.solver is not None
            potential = self.reconstructor.solver.potential
            if potential is not None:
                path = _save_fits_image(potential, "_potential.fits")
                saved_paths["grid_potential"] = path
            else:
                logger.warning("Potential not computed or available in solver. Skipping save.")

        if "grid_displacement" in save_options:
            assert self.reconstructor is not None and self.reconstructor.solver is not None
            displacement = self.reconstructor.solver.displacement
            if displacement is not None:
                path = _save_fits_image(displacement, "_displacement.fits")
                saved_paths["grid_displacement"] = path
            else:
                logger.warning("Displacement not computed or available in solver. Skipping save.")

        if "reconstructor_object" in save_options:
            path = str(output_folder / (base_name + "_reconstructor.pkl"))
            with open(path, "wb") as f:
                pickle.dump(self.reconstructor, f)
            saved_paths["reconstructor_object"] = path
            logger.info("Saved reconstructor object to {0}".format(path))

        if output_cfg.get("save_metadata", True):
            metadata_path = output_folder / (base_name + "_metadata.txt")
            metadata_path.write_text(str(asdict(self.config)), encoding="utf-8")
            saved_paths["metadata"] = str(metadata_path)

        logger.info("Saved outputs: {0}".format(list(saved_paths.keys())))
        return saved_paths

    def run(self) -> Dict[str, str]:
        """Run the full reconstruction pipeline end-to-end."""
        self.load_catalogs()
        self.convert_to_xyz()
        self.reconstruct()
        self.convert_back()
        return self.save_outputs()
