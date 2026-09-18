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
    def write(
        self,
        df: pd.DataFrame,
        path: str,
        template: Optional[str] = None,
        hdu: Optional[int] = None,
        strict: bool = False,
    ) -> None:
        """Write ``df`` to ``path`` in this backend's format.

        ``template`` names an existing file whose *schema* the output should
        reproduce: column order and per-column metadata, plus whatever else the
        format carries around the table. It is honoured by formats rich enough
        to have a schema worth preserving (FITS) and ignored otherwise, the way
        ``read`` treats ``hdu`` -- which here selects the table inside the
        template, and the table the output replaces.

        ``strict`` decides what happens when the frame and the template disagree
        on the column set. With ``False`` (the default) columns absent from the
        template are appended and columns the frame does not supply are filled,
        so a catalogue may gain derived columns or carry fewer than it read.
        With ``True`` either disagreement is an error, which is what a caller
        wants when the output has to stay substitutable for its input.

        Backends that ignore ``template`` ignore ``strict`` with it.
        """
