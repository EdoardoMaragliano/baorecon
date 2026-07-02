"""End-to-end BAO reconstruction pipeline driven by YAML configuration."""

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
from baorecon.utils.formatters import format_positions
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


def _to_host(array):
    """Return a host (NumPy) copy of ``array``, converting from CuPy if needed.

    On the GPU path the solver fields (notably the displacement grid) are CuPy
    arrays, which ``astropy.io.fits`` and the rest of the I/O layer cannot
    consume directly. NumPy inputs (and ``None``) are returned untouched.
    """
    if array is None:
        return None
    try:
        import cupy as cp

        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    except ImportError:
        pass
    return np.asarray(array)


class ReconstructionPipeline:
    """High-level orchestrator that runs catalog I/O and BAO reconstruction."""

    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.config = CatalogConfig.from_yaml(config_file)
        self.catalog = Catalog(self.config)
        self.cosmology = create_cosmology(**self.config.cosmology)

        # Single source of truth for the working precision (default float32).
        # Coordinate arrays are cast to this and it is forwarded to the
        # reconstructor: the coordinate helpers now preserve their input dtype,
        # so the pipeline must own the cast rather than relying on them.
        self.dtype = np.dtype(self.config.reconstruction.get("dtype", "float32")).type

        self.data_pos_ra: Optional[np.ndarray] = None
        self.data_pos_dec: Optional[np.ndarray] = None
        self.data_pos_z: Optional[np.ndarray] = None
        self.random_pos_ra: Optional[np.ndarray] = None
        self.random_pos_dec: Optional[np.ndarray] = None
        self.random_pos_z: Optional[np.ndarray] = None
        self.data_pos_xyz: Optional[np.ndarray] = None
        self.random_pos_xyz: Optional[np.ndarray] = None
        self.data_weights: Optional[np.ndarray] = None
        self.random_weights: Optional[np.ndarray] = None
        self.data_ids: Optional[np.ndarray] = None
        self.random_ids: Optional[np.ndarray] = None
        self.reconstructor: Optional[BAOReconstructor] = None
        self.data_rec_xyz: Optional[np.ndarray] = None
        self.random_rec_xyz: Optional[np.ndarray] = None
        self.data_rec_ra: Optional[np.ndarray] = None
        self.data_rec_dec: Optional[np.ndarray] = None
        self.random_rec_ra: Optional[np.ndarray] = None
        self.random_rec_dec: Optional[np.ndarray] = None
        self.data_rec_z: Optional[np.ndarray] = None
        self.random_rec_z: Optional[np.ndarray] = None

    def load_catalogs(self) -> None:
        """Load input FITS catalogs and extract raw arrays."""
        self.catalog.load()
        (
            self.data_pos_ra,
            self.data_pos_dec, 
            self.data_pos_z,
            self.data_weights,
            self.data_ids,
            self.random_pos_ra, 
            self.random_pos_dec, 
            self.random_pos_z,
            self.random_weights,
            self.random_ids,
        ) = self.catalog.get_positions_weights_ids(target_dtype=self.dtype)

    def convert_to_xyz(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert RA/DEC/redshift coordinates to Cartesian coordinates."""
        if self.data_pos_ra is None or self.random_pos_ra is None:
            self.load_catalogs()

        coordinate_cfg = self.config.coordinate_system
        ra_dec_unit = coordinate_cfg.get("ra_dec_unit", "deg")
        frame = coordinate_cfg.get("frame", "icrs")
        distance_unit = coordinate_cfg.get("distance_unit", "Mpc/h")

        data_xyz, _ = radec_z_to_xyz(
            self.data_pos_ra,
            self.data_pos_dec,
            self.data_pos_z,
            cosmo=self.cosmology,
            ra_dec_unit=ra_dec_unit,
            frame=frame,
            distance_unit=distance_unit,
        )
        random_xyz, _ = radec_z_to_xyz(
            self.random_pos_ra,
            self.random_pos_dec,
            self.random_pos_z,
            cosmo=self.cosmology,
            ra_dec_unit=ra_dec_unit,
            frame=frame,
            distance_unit=distance_unit,
        )
        # Positions are already loaded at the working precision (self.dtype) and
        # coordinates.py preserves it, so format_positions just validates shape.
        # Drop the raw RA/DEC/z now -- only needed to build the xyz arrays (the
        # random catalogue is the large one).
        self.data_pos_ra = self.data_pos_dec = self.data_pos_z = None
        self.random_pos_ra = self.random_pos_dec = self.random_pos_z = None

        self.data_pos_xyz = format_positions(data_xyz, dtype=self.dtype)
        self.random_pos_xyz = format_positions(random_xyz, dtype=self.dtype)
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
            dtype=self.dtype,
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

    def convert_back(self) -> Tuple[np.ndarray, ...]:
        """Convert reconstructed XYZ coordinates back to RA/DEC/redshift.

        RA and DEC are kept as separate 1D arrays (not stacked into an (N, 2)),
        so the large random catalogue avoids an extra full-size allocation.
        """
        if self.data_rec_xyz is None or self.random_rec_xyz is None:
            self.reconstruct()

        logger.info("Converting back to RA/DEC/redshift...")
        coordinate_cfg = self.config.coordinate_system
        ra_dec_unit = coordinate_cfg.get("ra_dec_unit", "deg")
        frame = coordinate_cfg.get("frame", "icrs")
        distance_unit = coordinate_cfg.get("distance_unit", "Mpc/h")

        self.data_rec_ra, self.data_rec_dec, self.data_rec_z, _ = xyz_to_radec_z(
            self.data_rec_xyz,
            cosmo=self.cosmology,
            ra_dec_unit=ra_dec_unit,
            frame=frame,
            distance_unit=distance_unit,
        )

        self.random_rec_ra, self.random_rec_dec, self.random_rec_z, _ = xyz_to_radec_z(
            self.random_rec_xyz,
            cosmo=self.cosmology,
            ra_dec_unit=ra_dec_unit,
            frame=frame,
            distance_unit=distance_unit,
        )

        logger.info("Conversion complete.")
        return (
            self.data_rec_ra, self.data_rec_dec, self.data_rec_z,
            self.random_rec_ra, self.random_rec_dec, self.random_rec_z,
        )

    def apply_mask(self, data_mask: Optional[np.ndarray] = None, random_mask: Optional[np.ndarray] = None) -> None:
        """Apply boolean masks while keeping IDs aligned to the filtered catalogues."""
        if data_mask is not None:
            self.catalog.apply_mask(np.asarray(data_mask, dtype=bool), is_data=True)
        if random_mask is not None:
            self.catalog.apply_mask(np.asarray(random_mask, dtype=bool), is_data=False)

    def _prepare_output(self) -> Tuple[Path, str, set]:
        """Resolve the output folder, base filename, and requested save options."""
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
        return output_folder, base_name, save_options

    def _save_grids(self, output_folder: Path, base_name: str, save_options: set) -> Dict[str, str]:
        """Save the grid potential/displacement and the pickled reconstructor.

        Called right after :meth:`reconstruct` so the heavy solver fields can be
        released immediately afterwards. On the GPU path the displacement grid is
        a CuPy array, so it is moved to host before being written to FITS.
        """
        saved_paths: Dict[str, str] = {}

        def _save_fits_image(data: np.ndarray, suffix: str) -> str:
            path = str(output_folder / (base_name + suffix))
            fits.writeto(path, _to_host(data), overwrite=True, output_verify="silentfix")
            logger.info("Saved FITS image to {0}".format(path))
            return path

        if "grid_potential" in save_options:
            assert self.reconstructor is not None and self.reconstructor.solver is not None
            potential = self.reconstructor.solver.potential
            if potential is not None:
                saved_paths["grid_potential"] = _save_fits_image(potential, "_potential.fits")
            else:
                logger.warning("Potential not computed or available in solver. Skipping save.")

        if "grid_displacement" in save_options:
            assert self.reconstructor is not None and self.reconstructor.solver is not None
            displacement = self.reconstructor.solver.displacement
            if displacement is not None:
                saved_paths["grid_displacement"] = _save_fits_image(displacement, "_displacement.fits")
            else:
                logger.warning("Displacement not computed or available in solver. Skipping save.")

        if "reconstructor_object" in save_options:
            path = str(output_folder / (base_name + "_reconstructor.pkl"))
            with open(path, "wb") as f:
                pickle.dump(self.reconstructor, f)
            saved_paths["reconstructor_object"] = path
            logger.info("Saved reconstructor object to {0}".format(path))

        return saved_paths

    def _save_catalogs(
        self, output_folder: Path, base_name: str, save_options: set, release: bool = False
    ) -> Dict[str, str]:
        """Save the reconstructed catalogues, one tracer type at a time.

        Data then randoms are processed sequentially: each catalogue's tracer
        displacement (``s = x_initial - x_reconstructed``, when requested) is
        computed, written, and -- when ``release`` is set -- its position arrays
        are dropped, before moving to the next. This keeps the two catalogues'
        transients from being live at once.
        ``release`` is used by :meth:`run`; :meth:`save_outputs` leaves it False
        so its step-by-step callers keep their intermediates.
        """
        saved_paths: Dict[str, str] = {}
        want_catalogs = "catalogs" in save_options
        want_displacements = "tracer_displacements" in save_options
        if not want_catalogs and not want_displacements:
            return saved_paths

        fmt = str(self.config.output.get("format", "fits")).lower()
        ext = "parquet" if fmt == "parquet" else "fits"

        def _process(is_data: bool) -> None:
            pos_xyz = self.data_pos_xyz if is_data else self.random_pos_xyz
            rec_xyz = self.data_rec_xyz if is_data else self.random_rec_xyz
            rec_ra = self.data_rec_ra if is_data else self.random_rec_ra
            rec_dec = self.data_rec_dec if is_data else self.random_rec_dec
            rec_z = self.data_rec_z if is_data else self.random_rec_z
            label = "data" if is_data else "random"

            # Displacements only feed the catalogue columns, so compute them only
            # when the catalogue is actually written.
            displacements = None
            if want_catalogs and want_displacements:
                assert pos_xyz is not None and rec_xyz is not None
                logger.info("Calculating %s tracer displacements.", label)
                displacements = pos_xyz - rec_xyz

            if want_catalogs:
                path = str(output_folder / (base_name + "_" + label + "." + ext))
                self.catalog.write_output(
                    path=path,
                    is_data=is_data,
                    reconstructed_radec=(rec_ra, rec_dec),
                    reconstructed_redshift=rec_z,
                    displacements=displacements,
                    fmt=fmt,
                )
                saved_paths[label + "_catalog"] = path
                logger.info("Saved %s catalog to %s", label, path)

            if release:
                # These arrays are now on disk; drop them before the next tracer.
                if is_data:
                    self.data_pos_xyz = self.data_rec_xyz = None
                    self.data_rec_ra = self.data_rec_dec = self.data_rec_z = None
                else:
                    self.random_pos_xyz = self.random_rec_xyz = None
                    self.random_rec_ra = self.random_rec_dec = self.random_rec_z = None

        _process(is_data=True)
        _process(is_data=False)
        return saved_paths

    def _save_metadata(self, output_folder: Path, base_name: str) -> Dict[str, str]:
        """Write the run configuration as a metadata sidecar file."""
        saved_paths: Dict[str, str] = {}
        if self.config.output.get("save_metadata", True):
            metadata_path = output_folder / (base_name + "_metadata.txt")
            metadata_path.write_text(str(asdict(self.config)), encoding="utf-8")
            saved_paths["metadata"] = str(metadata_path)
        return saved_paths

    def _release_reconstruction_grids(self) -> None:
        """Drop the solver and cached delta grids, then reclaim GPU memory.

        These are the largest arrays in the run (the displacement/potential
        fields and the overdensity grid) and are CuPy device arrays on the GPU
        path. Dropping the Python references lets CuPy's pool reclaim them; the
        pool is then returned to the driver so the freed device memory is
        actually available again.
        """
        if self.reconstructor is not None:
            self.reconstructor._solver = None
            density_manager = getattr(self.reconstructor, "_density_manager", None)
            if density_manager is not None:
                density_manager._delta_on_mesh = None

        if self.config.reconstruction.get("device", "cpu") != "gpu":
            return
        try:
            import cupy as cp
        except ImportError:
            return
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    def save_outputs(self) -> Dict[str, str]:
        """Save all configured outputs into the output folder (non-progressive).

        :meth:`run` instead interleaves saving with memory release; this method
        is kept for callers that drive the pipeline step by step and want every
        output written in one shot without freeing intermediate arrays.
        """
        if self.data_rec_xyz is None or self.random_rec_xyz is None:
            self.reconstruct()
        if self.data_rec_ra is None or self.random_rec_ra is None:
            self.convert_back()

        output_folder, base_name, save_options = self._prepare_output()

        saved_paths: Dict[str, str] = {}
        saved_paths.update(self._save_catalogs(output_folder, base_name, save_options))
        saved_paths.update(self._save_grids(output_folder, base_name, save_options))
        saved_paths.update(self._save_metadata(output_folder, base_name))

        logger.info("Saved outputs: {0}".format(list(saved_paths.keys())))
        return saved_paths

    def run(self) -> Dict[str, str]:
        """Run the full reconstruction pipeline with progressive memory release.

        Outputs are written as soon as their inputs are final, so the heavy
        solver grids and Cartesian position arrays can be dropped mid-run instead
        of being held until the end. On the GPU path the freed CuPy buffers are
        also returned to the device memory pool.
        """
        output_folder, base_name, save_options = self._prepare_output()
        saved_paths: Dict[str, str] = {}

        # 1. I/O + coordinate conversion.
        self.load_catalogs()
        self.convert_to_xyz()
        
        # 2. Reconstruction (builds the solver and the displacement grid).
        self.reconstruct()

        # 3. Save the grids + pickle, then release the solver/delta grids (the
        #    heaviest arrays -- CuPy device arrays on GPU).
        saved_paths.update(self._save_grids(output_folder, base_name, save_options))
        self._release_reconstruction_grids()
        # Tracer positions are only needed later for tracer displacements.
        if "tracer_displacements" not in save_options:
            self.data_pos_xyz = None
            self.random_pos_xyz = None

        # 4. Convert back to sky coordinates, then save catalogues. With
        #    release=True, _save_catalogs drops each catalogue's position and
        #    reconstructed arrays as soon as it is written.
        self.convert_back()
        saved_paths.update(
            self._save_catalogs(output_folder, base_name, save_options, release=True)
        )

        # 5. Metadata (cheap, no large arrays).
        saved_paths.update(self._save_metadata(output_folder, base_name))

        logger.info("Saved outputs: {0}".format(list(saved_paths.keys())))
        return saved_paths
