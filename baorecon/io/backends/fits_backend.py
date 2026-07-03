"""FITS catalog backend.

Reads use ``fitsio`` when available so that only the requested columns are
pulled off disk (true column pruning); otherwise they fall back to
``astropy``. Writes always go through ``astropy`` so the on-disk FITS stays
byte-compatible with the pre-refactor pipeline output.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from astropy.table import Table

from baorecon.io.backends.base import CatalogBackend

try:  # optional: enables real column-subset reads
    import fitsio

    _HAVE_FITSIO = True
except ImportError:  # pragma: no cover - exercised only without fitsio
    _HAVE_FITSIO = False


def _recarray_to_df(arr: np.ndarray) -> pd.DataFrame:
    """Convert a FITS structured array to a native-endian DataFrame.

    FITS stores big-endian data, which pandas refuses to operate on; each column
    is byte-swapped to the platform's native order on the way in.
    """
    data = {}
    for name in arr.dtype.names:
        col = arr[name]
        if col.dtype.byteorder not in ("=", "|"):
            col = col.astype(col.dtype.newbyteorder("="))
        data[name] = col
    return pd.DataFrame(data)


class FitsBackend(CatalogBackend):
    """Read/write FITS catalogs as pandas DataFrames."""

    def read(
        self,
        path: str,
        hdu: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        ext = 1 if hdu is None else hdu
        if _HAVE_FITSIO:
            arr = fitsio.read(path, ext=ext, columns=columns)
            return _recarray_to_df(np.atleast_1d(arr))

        table = Table.read(path, hdu=ext)
        if columns is not None:
            table = table[list(columns)]
        return table.to_pandas()

    def write(self, df: pd.DataFrame, path: str) -> None:
        Table.from_pandas(df).write(path, overwrite=True)
