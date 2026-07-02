"""Abstract catalog I/O backend.

A backend is responsible only for moving rows between disk and an in-memory
:class:`pandas.DataFrame`. All catalog manipulation logic (masking, building the
reconstructed output) lives in :class:`baorecon.io.catalog_io.Catalog`, which
treats backends as interchangeable readers/writers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd


class CatalogBackend(ABC):
    """Read and write catalogs as :class:`pandas.DataFrame` objects."""

    @abstractmethod
    def read(
        self,
        path: str,
        hdu: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Read ``path`` into a DataFrame.

        When ``columns`` is provided, only those columns are loaded (column
        pruning); ``None`` loads every column. ``hdu`` is honoured by formats
        that have the concept (FITS) and ignored otherwise.
        """

    @abstractmethod
    def write(self, df: pd.DataFrame, path: str) -> None:
        """Write ``df`` to ``path`` in this backend's format."""
