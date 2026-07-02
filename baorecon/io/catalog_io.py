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
        """Load both data and random catalogs into DataFrames at the working dtype.

        Floating-point columns are downcast in place to the pipeline's working
        precision (``reconstruction.dtype``, default float32), so the resident
        catalogues -- the largest structures held for the whole run -- are not
        kept at the on-disk float64.
        """
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

        recon_cfg = getattr(self.config, "reconstruction", None) or {}
        target_dtype = np.dtype(recon_cfg.get("dtype", "float32"))
        self._cast_float_columns(self.data, target_dtype)
        self._cast_float_columns(self.random, target_dtype)

        logger.info(
            "Loaded catalogs: data={0}, random={1}".format(len(self.data), len(self.random))
        )

    @staticmethod
    def _cast_float_columns(df: pd.DataFrame, dtype: np.dtype) -> None:
        """Cast the floating-point columns of ``df`` to ``dtype`` in place.

        Only float columns are touched (integer ID columns keep their type). The
        cast is done column-by-column and reassigned, so each old (e.g. float64)
        column is released as its replacement is built: peak memory stays at
        roughly the frame plus one temporary column, rather than the full
        duplicate frame a whole-frame ``astype`` would allocate.
        """
        for col in df.columns:
            if df[col].dtype.kind == "f" and df[col].dtype != dtype:
                df[col] = df[col].astype(dtype, copy=False)

    def _ensure_loaded(self) -> None:
        if self.data is None or self.random is None:
            self.load()

    def get_positions_weights_ids(
        self, target_dtype=np.float32
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray],
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray],
    ]:
        """Return flat 1D data/random RA, DEC, z, weights, and IDs (avoids N x 3 allocations).

        Order: ``data_ra, data_dec, data_z, data_weights, data_ids,
        random_ra, random_dec, random_z, random_weights, random_ids``.
        """
        self._ensure_loaded()
        assert self.data is not None
        assert self.random is not None

        columns = self.config.columns

        data_pos = (
            np.asarray(self.data[columns.ra], dtype=target_dtype),
            np.asarray(self.data[columns.dec], dtype=target_dtype),
            np.asarray(self.data[columns.redshift], dtype=target_dtype),
        )

        random_pos = (
            np.asarray(self.random[columns.ra], dtype=target_dtype),
            np.asarray(self.random[columns.dec], dtype=target_dtype),
            np.asarray(self.random[columns.redshift], dtype=target_dtype),
        )

        if columns.weight_data is not None:
            data_weights = np.asarray(self.data[columns.weight_data], dtype=target_dtype)
        else:
            data_weights = np.ones(len(self.data), dtype=target_dtype)

        if columns.weight_random is not None:
            random_weights = np.asarray(self.random[columns.weight_random], dtype=target_dtype)
        else:
            random_weights = np.ones(len(self.random), dtype=target_dtype)

        data_ids = None
        if columns.id_data is not None:
            data_ids = np.asarray(self.data[columns.id_data])

        random_ids = None
        if columns.id_random is not None:
            random_ids = np.asarray(self.random[columns.id_random])

        return (*data_pos, data_weights, data_ids, *random_pos, random_weights, random_ids)

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

    def build_output_table(
        self,
        is_data: bool,
        reconstructed_radec,
        reconstructed_redshift: Optional[np.ndarray] = None,
        displacements: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame where original coordinates are replaced by reconstructed ones.

        ``reconstructed_radec`` may be an ``(N, 2)`` array or a ``(ra, dec)`` pair
        of 1D arrays. The pair form lets callers avoid materialising an
        intermediate ``(N, 2)`` copy of the (large) random catalogue.
        """
        self._ensure_loaded()
        df = self.data if is_data else self.random
        if df is None:
            raise RuntimeError("Catalog not loaded.")

        # Fetch the original column names from the config.
        col_ra = self.config.columns.ra
        col_dec = self.config.columns.dec
        col_z = self.config.columns.redshift

        # Overwrite the original coordinates with the reconstructed ones. Accept
        # either a (ra, dec) pair (no intermediate allocation) or an (N, 2) array.
        if isinstance(reconstructed_radec, (tuple, list)):
            rec_ra, rec_dec = reconstructed_radec
        else:
            radec = np.asarray(reconstructed_radec)
            rec_ra, rec_dec = radec[:, 0], radec[:, 1]
        df[col_ra] = np.asarray(rec_ra)
        df[col_dec] = np.asarray(rec_dec)
        if reconstructed_redshift is not None:
            df[col_z] = np.asarray(reconstructed_redshift)

        if displacements is not None:
            if displacements.shape[0] != len(df):
                raise ValueError("Displacements array length must match table length.")
            if displacements.shape[1] != 3:
                raise ValueError("Displacements array must have shape (N, 3).")
            df["S_X"] = displacements[:, 0]
            df["S_Y"] = displacements[:, 1]
            df["S_Z"] = displacements[:, 2]

        return df

    def write_output(
        self,
        path: str,
        is_data: bool,
        reconstructed_radec,
        reconstructed_redshift: Optional[np.ndarray] = None,
        displacements: Optional[np.ndarray] = None,
        fmt: Optional[str] = None,
    ) -> None:
        """Build the reconstructed catalog and write it to ``path``.

        ``reconstructed_radec`` is forwarded to :meth:`build_output_table` and may
        be an ``(N, 2)`` array or a ``(ra, dec)`` pair of 1D arrays. The output
        format is taken from ``fmt`` when given, otherwise inferred from the
        ``path`` extension. Keeping construction and writing together means the
        column-building logic lives in exactly one place.
        """
        output = self.build_output_table(
            is_data=is_data,
            reconstructed_radec=reconstructed_radec,
            reconstructed_redshift=reconstructed_redshift,
            displacements=displacements,
        )
        get_backend(path, fmt).write(output, path)
        logger.info("Saved {0} catalog to {1}".format("data" if is_data else "random", path))
