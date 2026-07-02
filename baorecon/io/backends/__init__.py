"""Pluggable catalog I/O backends, selected by format or file extension."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from baorecon.io.backends.base import CatalogBackend
from baorecon.io.backends.fits_backend import FitsBackend
from baorecon.io.backends.parquet_backend import ParquetBackend

# Canonical format name -> backend class.
_BACKENDS = {
    "fits": FitsBackend,
    "parquet": ParquetBackend,
}

# File extension -> canonical format name.
_EXTENSIONS = {
    ".fits": "fits",
    ".fit": "fits",
    ".parquet": "parquet",
    ".pq": "parquet",
}


def resolve_format(path: str, fmt: Optional[str] = None) -> str:
    """Return the canonical format for ``path``, honouring an explicit ``fmt``."""
    if fmt is not None:
        key = fmt.lower()
        if key not in _BACKENDS:
            raise ValueError(
                "Unknown catalog format '{0}'. Supported: {1}".format(
                    fmt, sorted(_BACKENDS)
                )
            )
        return key

    suffix = Path(path).suffix.lower()
    if suffix not in _EXTENSIONS:
        raise ValueError(
            "Cannot infer catalog format from '{0}'. Use a known extension "
            "({1}) or set the format explicitly.".format(path, sorted(_EXTENSIONS))
        )
    return _EXTENSIONS[suffix]


def get_backend(path: str, fmt: Optional[str] = None) -> CatalogBackend:
    """Return a backend instance for ``path`` (extension) or explicit ``fmt``."""
    return _BACKENDS[resolve_format(path, fmt)]()


__all__ = [
    "CatalogBackend",
    "FitsBackend",
    "ParquetBackend",
    "get_backend",
    "resolve_format",
]
