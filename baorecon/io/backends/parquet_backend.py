"""Parquet catalog backend.

Reads push the column selection down to the Parquet reader so only the needed
columns are decoded off disk; this is the main performance win for the
multi-million-row catalogs the pipeline targets.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from baorecon.io.backends.base import CatalogBackend
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)


class ParquetBackend(CatalogBackend):
    """Read/write Parquet catalogs as pandas DataFrames."""

    def read(
        self,
        path: str,
        hdu: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        # ``hdu`` is meaningless for Parquet and intentionally ignored.
        return pd.read_parquet(path, columns=columns, engine="pyarrow")

    def write(
        self,
        df: pd.DataFrame,
        path: str,
        template: Optional[str] = None,
        hdu: Optional[int] = None,
        strict: bool = False,
    ) -> None:
        # Parquet carries its schema in the frame itself, so there is nothing a
        # template could contribute; ``template``, ``hdu`` and ``strict`` are
        # accepted for signature compatibility and ignored. Said out loud,
        # because a configured template that quietly does nothing is worse than
        # one that does something unexpected.
        if template is not None:
            logger.warning(
                "Parquet output ignores the template %s: the format carries its "
                "schema in the frame itself, so there is nothing to inherit.",
                template,
            )
        df.to_parquet(path, index=False, engine="pyarrow")
