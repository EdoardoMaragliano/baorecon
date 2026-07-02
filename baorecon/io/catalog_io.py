"""Catalog I/O facade for BAO reconstruction pipelines.

:class:`Catalog` is a thin facade over pluggable I/O backends (FITS, Parquet).
Catalogs are held internally as :class:`pandas.DataFrame` objects; the choice of
on-disk format is delegated to :mod:`baorecon.io.backends`. The compute side of
the pipeline only ever sees NumPy arrays (:meth:`get_positions_weights_ids`,
:meth:`apply_mask`), so it is decoupled from the underlying table type.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from baorecon.io.backends import get_backend
from baorecon.io.config import CatalogConfig
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)


class Catalog:
    """Load, inspect, and filter data/random catalogs (FITS or Parquet)."""

    def __init__(self, config: CatalogConfig) -> None:
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.random: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Backward-compatible aliases (catalogs are now DataFrames, not Tables).
    # ------------------------------------------------------------------
    @property
    def data_table(self) -> Optional[pd.DataFrame]:
        return self.data

    @property
    def random_table(self) -> Optional[pd.DataFrame]:
        return self.random

    def _required_columns(self, is_data: bool) -> Optional[List[str]]:
        """Return the columns to read, or ``None`` to read every column.

        When ``keep_cols`` is configured we read only the columns the pipeline
        actually needs: the coordinate/weight/id columns used for compute plus
        the ``keep_cols`` propagated to the output. With ``keep_cols`` empty we
        fall back to reading everything (legacy behaviour), so existing configs
        keep working.
        """
        cols = self.config.columns
        if not cols.keep_cols:
            return None

        needed: List[str] = [cols.ra, cols.dec, cols.redshift]
        weight = cols.weight_data if is_data else cols.weight_random
        identifier = cols.id_data if is_data else cols.id_random
        if weight is not None:
            needed.append(weight)
        if identifier is not None:
            needed.append(identifier)
        needed.extend(cols.keep_cols)

        # De-duplicate while preserving order.
        seen: set = set()
        unique = []
        for name in needed:
            if name not in seen:
                seen.add(name)
                unique.append(name)
        return unique

    def load(self) -> None:
        """Load both data and random catalogs into DataFrames."""
        fmt = self.config.catalog_format
        data_backend = get_backend(self.config.data_path, fmt)
        random_backend = get_backend(self.config.random_path, fmt)

        self.data = data_backend.read(
            self.config.data_path,
            hdu=self.config.data_hdu,
            columns=self._required_columns(is_data=True),
        )
        self.random = random_backend.read(
            self.config.random_path,
            hdu=self.config.random_hdu,
            columns=self._required_columns(is_data=False),
        )
        logger.info(
            "Loaded catalogs: data={0}, random={1}".format(len(self.data), len(self.random))
        )

    def _ensure_loaded(self) -> None:
        if self.data is None or self.random is None:
            self.load()

    def get_positions_weights_ids(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return data/random positions, weights, and IDs from the configured columns."""
        self._ensure_loaded()
        assert self.data is not None
        assert self.random is not None

        columns = self.config.columns

        data_pos = np.column_stack(
            (
                np.asarray(self.data[columns.ra]),
                np.asarray(self.data[columns.dec]),
                np.asarray(self.data[columns.redshift]),
            )
        )

        random_pos = np.column_stack(
            (
                np.asarray(self.random[columns.ra]),
                np.asarray(self.random[columns.dec]),
                np.asarray(self.random[columns.redshift]),
            )
        )

        if columns.weight_data is not None:
            data_weights = np.asarray(self.data[columns.weight_data], dtype=np.float32)
        else:
            data_weights = np.ones(len(self.data), dtype=np.float32)

        if columns.weight_random is not None:
            random_weights = np.asarray(self.random[columns.weight_random], dtype=np.float32)
        else:
            random_weights = np.ones(len(self.random), dtype=np.float32)

        data_ids = None
        if columns.id_data is not None:
            data_ids = np.asarray(self.data[columns.id_data])

        random_ids = None
        if columns.id_random is not None:
            random_ids = np.asarray(self.random[columns.id_random])

        return data_pos, data_weights, data_ids, random_pos, random_weights, random_ids

    def apply_mask(self, mask: np.ndarray, is_data: bool = True) -> None:
        """Apply a boolean mask while preserving the associated columns."""
        self._ensure_loaded()
        df = self.data if is_data else self.random
        if df is None:
            raise RuntimeError("Catalog not loaded.")

        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(df):
            raise ValueError("Mask length must match catalog length.")

        filtered = df.loc[mask].reset_index(drop=True)
        if is_data:
            self.data = filtered
        else:
            self.random = filtered

        logger.info(
            "Applied mask to {0} catalog: {1} -> {2}".format(
                "data" if is_data else "random", len(df), len(filtered)
            )
        )

    def build_output_table_verbose(
        self,
        is_data: bool,
        reconstructed_xyz: np.ndarray,
        reconstructed_radec: Optional[np.ndarray] = None,
        reconstructed_redshift: Optional[np.ndarray] = None,
        xyz_prefix: str = "REC",
    ) -> pd.DataFrame:
        """Return a DataFrame with original columns plus reconstructed coordinates."""
        self._ensure_loaded()
        df = self.data if is_data else self.random
        if df is None:
            raise RuntimeError("Catalog not loaded.")

        output = df.copy()
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

    def build_output_table(
        self,
        is_data: bool,
        reconstructed_radec: np.ndarray,
        reconstructed_redshift: Optional[np.ndarray] = None,
        displacements: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame where original coordinates are replaced by reconstructed ones."""
        self._ensure_loaded()
        df = self.data if is_data else self.random
        if df is None:
            raise RuntimeError("Catalog not loaded.")

        output = df.copy()

        # Fetch the original column names from the config.
        col_ra = self.config.columns.ra
        col_dec = self.config.columns.dec
        col_z = self.config.columns.redshift

        # Overwrite the original coordinates with the reconstructed ones.
        output[col_ra] = np.asarray(reconstructed_radec)[:, 0]
        output[col_dec] = np.asarray(reconstructed_radec)[:, 1]
        if reconstructed_redshift is not None:
            output[col_z] = np.asarray(reconstructed_redshift)

        if displacements is not None:
            if displacements.shape[0] != len(output):
                raise ValueError("Displacements array length must match table length.")
            if displacements.shape[1] != 3:
                raise ValueError("Displacements array must have shape (N, 3).")
            output["S_X"] = displacements[:, 0]
            output["S_Y"] = displacements[:, 1]
            output["S_Z"] = displacements[:, 2]

        return output

    def write_output(
        self,
        path: str,
        is_data: bool,
        reconstructed_radec: np.ndarray,
        reconstructed_redshift: Optional[np.ndarray] = None,
        displacements: Optional[np.ndarray] = None,
        fmt: Optional[str] = None,
    ) -> None:
        """Build the reconstructed catalog and write it to ``path``.

        The output format is taken from ``fmt`` when given, otherwise inferred
        from the ``path`` extension. Keeping construction and writing together
        means the column-building logic lives in exactly one place.
        """
        output = self.build_output_table(
            is_data=is_data,
            reconstructed_radec=reconstructed_radec,
            reconstructed_redshift=reconstructed_redshift,
            displacements=displacements,
        )
        get_backend(path, fmt).write(output, path)
        logger.info("Saved {0} catalog to {1}".format("data" if is_data else "random", path))
