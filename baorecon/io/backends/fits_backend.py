"""FITS catalog backend.

Reads use ``fitsio`` when available so that only the requested columns are
pulled off disk (true column pruning); otherwise they fall back to
``astropy``. Writes always go through ``astropy`` so the on-disk FITS stays
byte-compatible with the pre-refactor pipeline output.

Writes come in two flavours. Without a template the frame is written as it
stands, which is the historical behaviour and loses everything FITS carries
besides the values: units, display formats, header keywords, extra HDUs. With a
template the output is built *against* an existing file -- normally the
catalogue the rows were read from -- and inherits its column order, ``TFORM``,
``TUNIT``, ``TDIM``, its non-structural header keywords, its primary HDU and
every other HDU it has. That matters wherever a derived catalogue has to remain
interchangeable with the one it came from, which for survey products is the
usual requirement rather than a special case.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from baorecon.io.backends.base import CatalogBackend

try:  # optional: enables real column-subset reads
    import fitsio

    _HAVE_FITSIO = True
except ImportError:  # pragma: no cover - exercised only without fitsio
    _HAVE_FITSIO = False


#: Structural keywords the new table header defines for itself.
_STRUCTURAL = (
    "XTENSION",
    "BITPIX",
    "NAXIS",
    "NAXIS1",
    "NAXIS2",
    "PCOUNT",
    "GCOUNT",
    "TFIELDS",
)

#: Per-column keyword prefixes, likewise rebuilt from the columns themselves.
_COLUMN_PREFIXES = ("TFORM", "TTYPE", "TUNIT", "TDIM")

#: The FITS integrity pair, which describes the *data* and so cannot be
#: inherited: ``DATASUM`` is a checksum of the data block and ``CHECKSUM`` one
#: of the whole HDU. Copied from a template with a different number of rows they
#: are simply wrong, and a product that fails its own checksum is worse than one
#: carrying none -- every verifier reports it as corrupt. They are recomputed on
#: the way out instead, and only for a template that had them.
_INTEGRITY = ("CHECKSUM", "DATASUM")


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


def _fill_for(template_hdu, name: str, nrows: int) -> np.ndarray:
    """A default column for a template column the frame did not supply."""
    try:
        dtype = np.asarray(template_hdu.data[name]).dtype
    except Exception:
        return np.zeros(nrows, dtype=float)

    if dtype.kind in ("i", "u", "f"):
        return np.zeros(nrows, dtype=dtype)
    if dtype.kind == "b":
        return np.zeros(nrows, dtype=bool)
    if dtype.kind in ("S", "U"):
        arr = np.empty(nrows, dtype=dtype)
        arr[:] = b"" if dtype.kind == "S" else ""
        return arr
    return np.zeros(nrows, dtype=dtype)


def _derive_format(arr: np.ndarray) -> str:
    """The ``TFORM`` astropy would give ``arr``, derived from its dtype alone.

    The sample is sliced to zero length first: the format follows from the
    dtype, and building it off the full column would copy every row of a
    multi-million-row catalogue to learn a two-character string.
    """
    empty = np.asarray(arr)[:0]
    table = Table({"c": empty})
    return fits.table_to_hdu(table).columns[0].format


def _columns_from_frame(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """``{name: array}``, native-endian, in the frame's own column order."""
    out: Dict[str, np.ndarray] = {}
    for name in df.columns:
        arr = np.asarray(df[name])
        if arr.dtype.byteorder not in ("=", "|"):
            arr = arr.astype(arr.dtype.newbyteorder("="))
        out[str(name)] = arr
    return out


def _resolve_ext(hdu: Optional[Union[int, str]]) -> Union[int, str]:
    """The extension to read from the template and to replace in the output."""
    return 1 if hdu is None else hdu


def write_like_template(
    df: pd.DataFrame,
    path: str,
    template: str,
    hdu: Optional[Union[int, str]] = None,
    strict: bool = False,
    overwrite: bool = True,
    memmap: bool = True,
) -> None:
    """Write ``df`` to ``path`` using ``template`` as the schema.

    The row count need not match the template -- only the column set does, and
    only under ``strict``. Masking a catalogue and writing it back against the
    file it came from is the case this is built for.
    """
    # A template is a schema, and a schema only transfers between files of the
    # same format. Reached through ``[Output] template = input`` with Parquet
    # inputs and a FITS output, astropy would otherwise fail deep inside
    # ``fits.open`` with "No SIMPLE card found", which says nothing about the
    # configuration that caused it.
    from baorecon.io.backends import resolve_format

    try:
        template_format = resolve_format(template)
    except ValueError:
        template_format = None
    if template_format != "fits":
        raise ValueError(
            "the FITS writer was given a template that is not FITS: {0}. "
            "A template carries a schema, so it has to be the same format as "
            "the output. With '[Output] template = input' this means the input "
            "catalogues and '[Output] format' must agree; set template = none "
            "to write without one.".format(template)
        )

    columns = _columns_from_frame(df)
    if not columns:
        raise ValueError("no columns to write")

    lengths = {len(v) for v in columns.values()}
    if len(lengths) != 1:
        raise ValueError(
            "all columns must share one length; got {0}".format(sorted(lengths))
        )
    nrows = lengths.pop()
    ext = _resolve_ext(hdu)

    with fits.open(template, memmap=memmap) as hdul:
        template_hdu = hdul[ext]
        template_columns = template_hdu.columns
        names = [c.name for c in template_columns]

        missing = [n for n in names if n not in columns]
        extra = [n for n in columns if n not in names]
        if strict:
            if missing:
                raise KeyError(
                    "missing template columns: {0}".format(missing)
                )
            if extra:
                raise KeyError(
                    "columns absent from the template: {0}".format(extra)
                )

        new_columns: List[fits.Column] = []

        # The template's own columns first, in the template's order, so a reader
        # that indexes by position still finds what it expects.
        for column in template_columns:
            name = column.name
            if name in columns:
                arr = columns[name]
                # Cast to the template dtype so the inherited TFORM stays truthful.
                try:
                    template_dtype = np.asarray(template_hdu.data[name]).dtype
                    if arr.dtype != template_dtype:
                        arr = arr.astype(template_dtype, copy=False)
                except Exception:
                    pass
            else:
                arr = _fill_for(template_hdu, name, nrows)

            new_columns.append(
                fits.Column(
                    name=name,
                    format=column.format,
                    unit=column.unit,
                    dim=column.dim,
                    array=arr,
                )
            )

        # Then whatever the frame added, with a format astropy derives from the
        # values -- there is no template metadata to inherit for these.
        for name in extra:
            new_columns.append(
                fits.Column(
                    name=name,
                    format=_derive_format(columns[name]),
                    array=columns[name],
                )
            )

        table = fits.BinTableHDU.from_columns(new_columns, name=template_hdu.name)

        for key, value in template_hdu.header.items():
            if (
                key in _STRUCTURAL
                or key in _INTEGRITY
                or key.startswith(_COLUMN_PREFIXES)
            ):
                continue
            if key not in table.header:
                table.header[key] = value

        # Recompute the integrity pair only for a product whose template carried
        # one, so a template without checksums does not acquire them here.
        signed = any(key in hdu_.header for hdu_ in hdul for key in _INTEGRITY)

        out = fits.HDUList([hdul[0].copy()])
        for index, hdu_ in enumerate(hdul[1:], start=1):
            is_target = (isinstance(ext, int) and index == ext) or (
                isinstance(ext, str) and hdu_.name == ext
            )
            out.append(table if is_target else hdu_.copy())
        out.writeto(path, overwrite=overwrite, checksum=signed)


class FitsBackend(CatalogBackend):
    """Read/write FITS catalogs as pandas DataFrames."""

    def read(
        self,
        path: str,
        hdu: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        ext = _resolve_ext(hdu)
        if _HAVE_FITSIO:
            arr = fitsio.read(path, ext=ext, columns=columns)
            return _recarray_to_df(np.atleast_1d(arr))

        table = Table.read(path, hdu=ext)
        if columns is not None:
            table = table[list(columns)]
        return table.to_pandas()

    def write(
        self,
        df: pd.DataFrame,
        path: str,
        template: Optional[str] = None,
        hdu: Optional[int] = None,
        strict: bool = False,
    ) -> None:
        if template is None:
            Table.from_pandas(df).write(path, overwrite=True)
            return
        write_like_template(df, path, template=template, hdu=hdu, strict=strict)
