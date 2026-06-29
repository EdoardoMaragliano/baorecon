"""Parquet catalog backend.

Reads push the column selection down to the Parquet reader so only the needed
columns are decoded off disk; this is the main performance win for the
multi-million-row catalogs the pipeline targets.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from baorecon.io.backends.base import CatalogBackend


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

    def write(self, df: pd.DataFrame, path: str) -> None:
        df.to_parquet(path, index=False, engine="pyarrow")
