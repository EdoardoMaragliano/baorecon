"""YAML and INI configuration parsing for BAO reconstruction pipelines.

Two front-ends produce the same :class:`CatalogConfig`:

* :meth:`CatalogConfig.from_yaml` -- the native format
  (``examples/bao_pipeline_example.yaml``).
* :meth:`CatalogConfig.from_ini` -- a 2PCF/Euclid-style parameter file
  (``examples/bao_pipeline_parfile.ini``), whose ``[Cosmology]`` block is the
  2PCF one verbatim so a single block can serve both codes.

:meth:`CatalogConfig.from_file` dispatches on the file extension. Everything
downstream of ``CatalogConfig`` is unaware of which format was read.
"""

from __future__ import annotations

import configparser
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# INI parsing helpers
# =============================================================================
#
# configparser hands back strings only, so every value has to be coerced
# explicitly -- notably booleans, where ``bool("false")`` is True.
#
# An option that is PRESENT BUT EMPTY means "null" (the INI spelling of YAML's
# ``key: null``); an option that is ABSENT is left out of the resulting dict, so
# the pipeline's own default applies. The two are not interchangeable: ``los =``
# selects the per-tracer radial line of sight, while omitting ``los`` leaves the
# plane-parallel ``'z'`` default in place. Only keys whose consumer accepts None
# are read with ``allow_null=True``; for the others an empty value is dropped
# rather than passed on as None, which would break e.g. ``Path(folder)``.

_MISSING = object()


def _raw(section, key: str):
    """Stripped raw value; ``None`` when empty, ``_MISSING`` when absent."""
    if key not in section:
        return _MISSING
    value = section[key].strip()
    return value if value else None


def _put(out: Dict[str, Any], section, key: str, cast: Callable[[str], Any],
         name: Optional[str] = None, allow_null: bool = False) -> None:
    """Read ``section[key]`` into ``out[name or key]``, coerced through ``cast``."""
    raw = _raw(section, key)
    if raw is _MISSING:
        return
    if raw is None:
        if allow_null:
            out[name or key] = None
        return
    out[name or key] = cast(raw)


def _optional(section, key: str) -> Optional[str]:
    """A plain optional string: ``None`` when absent or empty."""
    raw = _raw(section, key)
    return None if raw is _MISSING or raw is None else raw


def _required(section, key: str, section_name: str, filepath: str) -> str:
    """A mandatory string, with an error naming the offending file."""
    raw = _raw(section, key)
    if raw is _MISSING or raw is None:
        raise ValueError(
            "[{0}] {1} is required and must not be empty (in {2})".format(
                section_name, key, filepath)
        )
    return raw


def _to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in ("1", "yes", "true", "on"):
        return True
    if lowered in ("0", "no", "false", "off"):
        return False
    raise ValueError("expected a boolean (true/false), got {0!r}".format(value))


def _to_list(value: str) -> List[str]:
    """Split a comma-separated value, dropping empty entries."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _scalar_or_list(cast: Callable[[str], Any]) -> Callable[[str], Any]:
    """Parse a lone value as a scalar and a comma-separated one as a list.

    Mirrors the YAML form, where ``nmesh`` and ``boxsize`` are either a scalar
    (cubic/isotropic) or a length-3 sequence; validation happens downstream.
    """
    def parse(value: str):
        items = _to_list(value)
        return cast(items[0]) if len(items) == 1 else [cast(item) for item in items]
    return parse


# Parameters that FlatLambdaCDM fixes by construction, with the value it implies.
_FLAT_LCDM_FIXED = (("om_k", 0.0), ("om_radiation", 0.0), ("w_eos", -1.0))


def _validate_flat_lcdm(section, om_matter: Optional[float]) -> None:
    """Reject a ``[Cosmology]`` block this code cannot actually represent.

    The inert 2PCF keys are only inert because the reconstruction needs nothing
    but the comoving distance of a flat LCDM, which pins om_k, om_radiation and
    w_eos. A parfile that sets them elsewhere is not "extra detail we ignore" --
    it is a different cosmology, and dropping it silently would produce wrong
    distances with no warning anywhere.
    """
    for key, fixed in _FLAT_LCDM_FIXED:
        raw = _raw(section, key)
        if raw is _MISSING or raw is None:
            continue
        if not math.isclose(float(raw), fixed, abs_tol=1e-8):
            raise ValueError(
                "[Cosmology] {0} = {1} is incompatible with the flat LCDM this "
                "code integrates, which fixes {0} = {2}. Reconstruction would "
                "otherwise run with wrong comoving distances.".format(key, raw, fixed)
            )

    om_vac = _raw(section, "om_vac")
    if om_vac not in (_MISSING, None) and om_matter is not None:
        implied = 1.0 - om_matter
        if not math.isclose(float(om_vac), implied, abs_tol=1e-4):
            logger.warning(
                "[Cosmology] om_vac = %s, but flatness implies 1 - om_matter = %.6f. "
                "The parfile value is ignored; %.6f is used.",
                om_vac, implied, implied,
            )


def _cosmology_from_ini(section) -> Dict[str, Any]:
    """Translate the shared 2PCF ``[Cosmology]`` block to create_cosmology kwargs.

    The block keeps the 2PCF spelling verbatim, so nothing here matches the
    Python API by name, and ``H0`` is not stored at all: the parfile splits it
    into ``Hubble`` (100 by convention) and the dimensionless ``hubble``.

    Only the five parameters ``create_cosmology`` accepts are emitted -- it is
    called with ``**``, so any stray key would raise TypeError. The remaining
    2PCF keys are validated by :func:`_validate_flat_lcdm` and then dropped.
    """
    cosmology: Dict[str, Any] = {}

    hubble_h = _raw(section, "hubble")
    if hubble_h not in (_MISSING, None):
        hubble_100 = _raw(section, "Hubble")
        # `Hubble` is 100.0 in every 2PCF parfile; tolerate its absence.
        scale = 100.0 if hubble_100 in (_MISSING, None) else float(hubble_100)
        cosmology["H0"] = scale * float(hubble_h)

    _put(cosmology, section, "om_matter", float, name="Om0")
    _put(cosmology, section, "om_baryons", float, name="Ob0")
    _put(cosmology, section, "Tcmb", float, name="Tcmb0")
    _put(cosmology, section, "cosmology_ID", str, name="name")

    _validate_flat_lcdm(section, cosmology.get("Om0"))
    return cosmology


@dataclass
class ColumnMapping:
    """FITS column mapping for data and random catalogues."""

    ra: str
    dec: str
    redshift: str
    weight_data: Optional[str] = None
    weight_random: Optional[str] = None
    id_data: Optional[str] = None
    id_random: Optional[str] = None
    keep_cols: List[str] = field(default_factory=list)


@dataclass
class CatalogConfig:
    """Top-level configuration for the reconstruction pipeline."""

    data_path: str
    random_path: str
    columns: ColumnMapping
    coordinate_system: Dict[str, Any]
    cosmology: Dict[str, Any]
    reconstruction: Dict[str, Any]
    output: Dict[str, Any]
    catalog_name: Optional[str] = None
    data_hdu: int = 1
    random_hdu: int = 1
    catalog_format: Optional[str] = None
    masking: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not Path(self.data_path).exists():
            raise FileNotFoundError("Data catalog not found: {0}".format(self.data_path))
        if not Path(self.random_path).exists():
            raise FileNotFoundError("Random catalog not found: {0}".format(self.random_path))

    @classmethod
    def from_yaml(cls, filepath: str) -> "CatalogConfig":
        """Load configuration from a YAML file."""
        with open(filepath, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        if not isinstance(config, dict):
            raise ValueError("Invalid YAML configuration format.")

        columns_cfg = config.get("columns", {})
        coordinates_cfg = columns_cfg.get("coordinates", {})
        weights_cfg = columns_cfg.get("weights", {})
        ids_cfg = columns_cfg.get("ids", {})

        columns = ColumnMapping(
            ra=coordinates_cfg["ra"],
            dec=coordinates_cfg["dec"],
            redshift=coordinates_cfg["redshift"],
            weight_data=weights_cfg.get("data"),
            weight_random=weights_cfg.get("random"),
            id_data=ids_cfg.get("data"),
            id_random=ids_cfg.get("random"),
            keep_cols=list(columns_cfg.get("keep_cols", [])),
        )

        logger.info("Loaded reconstruction config from {0}".format(filepath))

        return cls(
            data_path=config["catalog"]["data_path"],
            random_path=config["catalog"]["random_path"],
            data_hdu=int(config["catalog"].get("data_hdu", 1)),
            random_hdu=int(config["catalog"].get("random_hdu", 1)),
            catalog_format=config["catalog"].get("format"),
            columns=columns,
            coordinate_system=dict(config.get("coordinate_system", {})),
            cosmology=dict(config.get("cosmology", {})),
            reconstruction=dict(config.get("reconstruction", {})),
            output=dict(config.get("output", {})),
            catalog_name=config.get("catalog_name"),
            masking=dict(config.get("masking", {})),
        )

    @classmethod
    def from_ini(cls, filepath: str) -> "CatalogConfig":
        """Load configuration from a 2PCF/Euclid-style INI parameter file.

        See ``examples/bao_pipeline_parfile.ini`` for the layout. Sections and
        keys the reconstruction does not consume are ignored rather than
        rejected -- a full 2PCF parfile's ``[2PCF]``, ``[Path]``,
        ``[Catalog.Reconstructed]``, ``density``, ``mask`` and so on -- so the
        same file can be handed to both codes.

        Raises
        ------
        ValueError
            If a required section or key is missing, if the coordinate columns
            disagree between the two catalogue sections, or if ``[Cosmology]``
            describes a cosmology that is not flat LCDM.
        """
        parser = configparser.ConfigParser()
        # Preserve key case: lower-casing would break both the 2PCF spellings
        # (Hubble, Tcmb) and the pipeline's own (RSDspace, R_sm, MAS).
        parser.optionxform = str
        with open(filepath, "r", encoding="utf-8") as handle:
            parser.read_file(handle)

        for required in ("Catalog.Galaxy", "Catalog.Random"):
            if not parser.has_section(required):
                raise ValueError(
                    "Missing [{0}] section in {1}".format(required, filepath)
                )

        galaxy = parser["Catalog.Galaxy"]
        random = parser["Catalog.Random"]
        cosmology_sec = parser["Cosmology"] if parser.has_section("Cosmology") else {}
        recon_sec = parser["Recon"] if parser.has_section("Recon") else {}
        output_sec = parser["Output"] if parser.has_section("Output") else {}

        # --- Columns -------------------------------------------------------
        # The config holds ONE set of coordinate column names for the run. The
        # parfile carries them per catalogue (the 2PCF/Euclid shape), so a
        # disagreement is rejected rather than silently resolved in favour of
        # one section. Omitting them from [Catalog.Random] inherits the galaxy
        # names, which is the common case.
        ra = _required(galaxy, "coord1", "Catalog.Galaxy", filepath)
        dec = _required(galaxy, "coord2", "Catalog.Galaxy", filepath)
        redshift = _required(galaxy, "coord3", "Catalog.Galaxy", filepath)
        for key, galaxy_value in (("coord1", ra), ("coord2", dec), ("coord3", redshift)):
            random_value = _optional(random, key)
            if random_value is not None and random_value != galaxy_value:
                raise ValueError(
                    "[Catalog.Random] {0} = {1!r} does not match [Catalog.Galaxy] "
                    "{0} = {2!r}; baorecon uses one set of coordinate columns for "
                    "both catalogues (in {3})".format(
                        key, random_value, galaxy_value, filepath)
                )

        keep_raw = _raw(galaxy, "keep_cols")
        keep_cols = [] if keep_raw in (_MISSING, None) else _to_list(keep_raw)

        columns = ColumnMapping(
            ra=ra,
            dec=dec,
            redshift=redshift,
            weight_data=_optional(galaxy, "weight"),
            weight_random=_optional(random, "weight"),
            id_data=_optional(galaxy, "id"),
            id_random=_optional(random, "id"),
            keep_cols=keep_cols,
        )

        # --- Coordinate system ---------------------------------------------
        # Global for the run, so it is declared once in [Catalog.Galaxy].
        coordinate_system: Dict[str, Any] = {}
        _put(coordinate_system, galaxy, "angle_units", str, name="ra_dec_unit")
        _put(coordinate_system, galaxy, "distance_unit", str)

        # --- Reconstruction -------------------------------------------------
        reconstruction: Dict[str, Any] = {}
        _put(reconstruction, recon_sec, "redshift", float)
        _put(reconstruction, recon_sec, "RSDspace", str)
        _put(reconstruction, recon_sec, "nmesh", _scalar_or_list(int))
        _put(reconstruction, recon_sec, "padding", float)
        _put(reconstruction, recon_sec, "R_sm", float)
        _put(reconstruction, recon_sec, "pbc", _to_bool)
        _put(reconstruction, recon_sec, "rectype", str)
        _put(reconstruction, recon_sec, "f", float)
        _put(reconstruction, recon_sec, "bias", float)
        _put(reconstruction, recon_sec, "MAS", str)
        _put(reconstruction, recon_sec, "threshold_randoms", float)
        _put(reconstruction, recon_sec, "solver_type", str)
        _put(reconstruction, recon_sec, "n_iterations", int)
        _put(reconstruction, recon_sec, "device", str)
        _put(reconstruction, recon_sec, "dtype", str)
        # These four accept null: an empty value is meaningful, not a no-op.
        # `cellsize`/`boxsize`/`boxcentre` empty -> derive from the catalogue;
        # `los` empty -> per-tracer radial line of sight.
        _put(reconstruction, recon_sec, "cellsize", float, allow_null=True)
        _put(reconstruction, recon_sec, "boxsize", _scalar_or_list(float), allow_null=True)
        _put(reconstruction, recon_sec, "boxcentre", _scalar_or_list(float), allow_null=True)
        _put(reconstruction, recon_sec, "los", str, allow_null=True)

        # --- Output ---------------------------------------------------------
        output: Dict[str, Any] = {}
        _put(output, output_sec, "folder", str)
        _put(output, output_sec, "naming_pattern", str)
        _put(output, output_sec, "format", str)
        _put(output, output_sec, "save_metadata", _to_bool)
        _put(output, output_sec, "save", _to_list)

        logger.info("Loaded reconstruction config from {0}".format(filepath))

        return cls(
            data_path=_required(galaxy, "filename", "Catalog.Galaxy", filepath),
            random_path=_required(random, "filename", "Catalog.Random", filepath),
            data_hdu=int(_optional(galaxy, "hdu") or 1),
            random_hdu=int(_optional(random, "hdu") or 1),
            catalog_format=_optional(galaxy, "format"),
            columns=columns,
            coordinate_system=coordinate_system,
            cosmology=_cosmology_from_ini(cosmology_sec),
            reconstruction=reconstruction,
            output=output,
            catalog_name=_optional(output_sec, "catalog_name"),
            masking={},
        )

    @classmethod
    def from_file(cls, filepath: str) -> "CatalogConfig":
        """Load configuration from a YAML or INI file, chosen by extension."""
        suffix = Path(filepath).suffix.lower()
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(filepath)
        if suffix in (".ini", ".par", ".parfile"):
            return cls.from_ini(filepath)
        raise ValueError(
            "Unsupported config extension {0!r} for {1}: expected one of "
            ".yaml, .yml, .ini, .par, .parfile".format(suffix, filepath)
        )

