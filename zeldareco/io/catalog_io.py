"""FITS catalog I/O for BAO reconstruction pipelines."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table

from zeldareco.io.config import CatalogConfig
from zeldareco.utils.loggers import setup_logger

logger = setup_logger(__name__)


class Catalog:
    """Load, inspect, and filter data/random FITS catalogs."""

    def __init__(self, config: CatalogConfig) -> None:
        self.config = config
        self.data_table: Optional[Table] = None
        self.random_table: Optional[Table] = None

    def load(self) -> None:
        """Load both data and random FITS tables."""
        self.data_table = Table.read(self.config.data_path, hdu=self.config.data_hdu)
        self.random_table = Table.read(self.config.random_path, hdu=self.config.random_hdu)
        logger.info(
            "Loaded catalogs: data={0}, random={1}".format(len(self.data_table), len(self.random_table))
        )

    def _ensure_loaded(self) -> None:
        if self.data_table is None or self.random_table is None:
            self.load()

    def get_positions_weights_ids(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return data/random positions, weights, and IDs from the configured columns."""
        self._ensure_loaded()
        assert self.data_table is not None
        assert self.random_table is not None

        columns = self.config.columns

        data_pos = np.column_stack(
            (
                np.asarray(self.data_table[columns.ra]),
                np.asarray(self.data_table[columns.dec]),
                np.asarray(self.data_table[columns.redshift]),
            )
        )

        random_pos = np.column_stack(
            (
                np.asarray(self.random_table[columns.ra]),
                np.asarray(self.random_table[columns.dec]),
                np.asarray(self.random_table[columns.redshift]),
            )
        )

        if columns.weight_data is not None:
            data_weights = np.asarray(self.data_table[columns.weight_data], dtype=np.float32)
        else:
            data_weights = np.ones(len(self.data_table), dtype=np.float32)

        if columns.weight_random is not None:
            random_weights = np.asarray(self.random_table[columns.weight_random], dtype=np.float32)
        else:
            random_weights = np.ones(len(self.random_table), dtype=np.float32)

        data_ids = None
        if columns.id_data is not None:
            data_ids = np.asarray(self.data_table[columns.id_data])

        random_ids = None
        if columns.id_random is not None:
            random_ids = np.asarray(self.random_table[columns.id_random])

        return data_pos, data_weights, data_ids, random_pos, random_weights, random_ids


    def apply_mask(self, mask: np.ndarray, is_data: bool = True) -> None:
        """Apply a boolean mask while preserving the associated IDs."""
        self._ensure_loaded()
        table = self.data_table if is_data else self.random_table
        if table is None:
            raise RuntimeError("Catalog not loaded.")

        if mask.dtype != bool:
            mask = np.asarray(mask, dtype=bool)

        if len(mask) != len(table):
            raise ValueError("Mask length must match catalog length.")

        filtered = table[mask]
        if is_data:
            self.data_table = filtered
        else:
            self.random_table = filtered

        logger.info("Applied mask to {0} catalog: {1} -> {2}".format("data" if is_data else "random", len(table), len(filtered)))

    def build_output_table(
        self,
        is_data: bool,
        reconstructed_xyz: np.ndarray,
        reconstructed_radec: Optional[np.ndarray] = None,
        reconstructed_redshift: Optional[np.ndarray] = None,
        xyz_prefix: str = "REC",
    ) -> Table:
        """Return a table with original columns plus reconstructed coordinates."""
        self._ensure_loaded()
        table = self.data_table if is_data else self.random_table
        if table is None:
            raise RuntimeError("Catalog not loaded.")

        output = table.copy()
        output["{0}_X".format(xyz_prefix)] = np.asarray(reconstructed_xyz)[:, 0]
        output["{0}_Y".format(xyz_prefix)] = np.asarray(reconstructed_xyz)[:, 1]
        output["{0}_Z".format(xyz_prefix)] = np.asarray(reconstructed_xyz)[:, 2]

        if reconstructed_radec is not None:
            output["{0}_RA".format(xyz_prefix)] = np.asarray(reconstructed_radec)[:, 0]
            output["{0}_DEC".format(xyz_prefix)] = np.asarray(reconstructed_radec)[:, 1]
            if np.asarray(reconstructed_radec).shape[1] > 2:
                output["{0}_ZOBS".format(xyz_prefix)] = np.asarray(reconstructed_radec)[:, 2]

        if reconstructed_redshift is not None:
            output["{0}_ZCOSMO".format(xyz_prefix)] = np.asarray(reconstructed_redshift)

        return output
