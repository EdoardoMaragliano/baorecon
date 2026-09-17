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
from typing import Any, Callable, Dict, List, Optional, Tuple

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


# 2PCF/Euclid spelling of the coordinate system, as written into the FITS
# `COORD` header the LE3 parser reads, mapped to baorecon's internal names.
# "Pseudo-equatorial" is this lineage's term for RA/DEC/redshift (cf. the old
# Recon_challenge.GetData.pseudoequatorial_to_cartesian).
_INI_COORDINATE_INPUTS = {
    "PSEUDO_EQUATORIAL": "ra_dec_z",
    "CARTESIAN": "cartesian",
}


def _coordinate_input_from_ini(section, filepath: str) -> Optional[str]:
    """Translate the parfile's ``coordinates`` value to an internal input name.

    Validated here, at load time, rather than when the conversion runs: an
    invalid value would otherwise only surface after both catalogues have been
    read from disk.
    """
    raw = _raw(section, "coordinates")
    if raw is _MISSING or raw is None:
        return None
    key = raw.strip().upper()
    if key not in _INI_COORDINATE_INPUTS:
        raise ValueError(
            "[Catalog.Galaxy] coordinates = {0!r} is not a known coordinate "
            "system: expected one of {1} (in {2})".format(
                raw, ", ".join(sorted(_INI_COORDINATE_INPUTS)), filepath)
        )
    return _INI_COORDINATE_INPUTS[key]


#: Every key ``from_ini`` consumes from ``[Recon]``.
#:
#: That section is baorecon's alone -- the 2PCF parameter file has no
#: ``[Recon]`` -- so an unrecognised key in it is a mistake rather than a key
#: carried over from elsewhere, and is rejected. The blocks the two files share
#: (``[Catalog.*]``, ``[Cosmology]``) stay tolerant, because they are copied
#: between the files and the 2PCF's versions carry keys baorecon does not read
#: (``density``, ``mask``, ``name``, ``om_k``, ``N_eff`` ...).
#:
#: A downstream processing element with settings of its own therefore puts them
#: in its own section, not here: unknown sections are ignored by design, and
#: that is the extension point.
_RECON_KEYS = frozenset({
    "redshift", "RSDspace", "nmesh", "cellsize", "padding", "R_sm", "pbc",
    "align_cone", "rectype", "f", "bias", "MAS", "threshold_randoms",
    "solver_type", "n_iterations", "device", "dtype", "boxsize", "boxcentre",
    "los",
})


def _reject_unknown_recon_keys(section, filepath: str) -> None:
    """Refuse a ``[Recon]`` key nothing reads.

    Silently dropping one turns a typo -- ``smothing``, ``rec_type`` -- into a
    run that quietly used the default, which is the kind of mistake that only
    shows up in the results.
    """
    unknown = sorted(set(section.keys()) - _RECON_KEYS)
    if not unknown:
        return
    raise ValueError(
        "[Recon] key(s) nothing reads: {0} (in {1}). Known keys: {2}. "
        "Settings belonging to another code go in a section of their own; "
        "sections baorecon does not know are ignored.".format(
            ", ".join(repr(k) for k in unknown), filepath,
            ", ".join(sorted(_RECON_KEYS)))
    )


#: ``[Cosmology]`` keys that ``create_cosmology`` cannot take, and the
#: canonical name each is reported under in
#: :attr:`CatalogConfig.cosmology_extra`.
_COSMOLOGY_EXTRA = {
    "om_nu": "Omega_nu", "om_radiation": "Omega_r", "om_vac": "Omega_L",
    "om_k": "Omega_k", "spectral_index": "ns", "As": "As", "sigma8": "sigma8",
    "w_eos": "w0", "N_eff": "N_eff", "hubble": "h",
}


def _cosmology_extra_from_ini(section) -> Dict[str, Any]:
    """The rest of ``[Cosmology]``, for consumers that need more than a distance.

    ``cosmology`` carries exactly the five arguments ``create_cosmology``
    takes, because it is splatted into that call. Reconstruction needs nothing
    else -- the comoving distance is all it asks of the background. A code
    computing a linear power spectrum does need more, and until now those keys
    were validated and then dropped, so it had to re-read the file to get them.
    """
    extra: Dict[str, Any] = {}
    for key, name in _COSMOLOGY_EXTRA.items():
        raw = _raw(section, key)
        if raw in (_MISSING, None) or (isinstance(raw, str) and not raw.strip()):
            continue
        try:
            extra[name] = float(raw)
        except (TypeError, ValueError):
            logger.warning("[Cosmology] %s = %r is not a number; ignored.", key, raw)
    return extra


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
    """FITS column mapping for data and random catalogues.

    ``ra``/``dec``/``redshift`` name the coordinate columns. The randoms may
    name theirs differently through ``ra_random``/``dec_random``/
    ``redshift_random``; left unset they fall back to the data names, which is
    the common case and what every configuration written before those three
    fields existed means. Use :meth:`coordinates` rather than the attributes, so
    the fallback is applied in one place.
    """

    ra: str
    dec: str
    redshift: str
    ra_random: Optional[str] = None
    dec_random: Optional[str] = None
    redshift_random: Optional[str] = None
    weight_data: Optional[str] = None
    weight_random: Optional[str] = None
    id_data: Optional[str] = None
    id_random: Optional[str] = None
    keep_cols: List[str] = field(default_factory=list)

    def coordinates(self, is_data: bool = True) -> Tuple[str, str, str]:
        """``(ra, dec, redshift)`` column names of one of the two catalogues."""
        if is_data:
            return self.ra, self.dec, self.redshift
        return (self.ra_random or self.ra,
                self.dec_random or self.dec,
                self.redshift_random or self.redshift)


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
    #: The [Cosmology] parameters ``create_cosmology`` cannot take -- ``As``,
    #: ``ns``, ``Omega_nu``, ``sigma8`` and the rest -- under canonical names.
    #: Empty unless the parfile supplies them. Reconstruction never reads it: it
    #: is there for codes layered on top that need the background for more than
    #: a comoving distance. Last in the field order so no positional call moves.
    cosmology_extra: Dict[str, Any] = field(default_factory=dict)

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

        # The randoms may name their coordinates differently; a nested
        # `coordinates.random` block overrides, and its absence inherits.
        random_coords = coordinates_cfg.get("random", {})

        columns = ColumnMapping(
            ra=coordinates_cfg["ra"],
            dec=coordinates_cfg["dec"],
            redshift=coordinates_cfg["redshift"],
            ra_random=random_coords.get("ra"),
            dec_random=random_coords.get("dec"),
            redshift_random=random_coords.get("redshift"),
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
            cosmology_extra=dict(config.get("cosmology_extra", {})),
        )

    @classmethod
    def from_ini(cls, filepath: str) -> "CatalogConfig":
        """Load configuration from a 2PCF/Euclid-style INI parameter file.

        See ``examples/bao_pipeline_parfile.ini`` for the layout.

        Reconstruction and the 2PCF keep **separate** parameter files in the
        same format, with ``[Catalog.*]`` and ``[Cosmology]`` largely in common.
        Those blocks are meant to be copied across unchanged, and the 2PCF's
        carry keys this code does not read -- ``density``, ``mask``, ``name``,
        ``om_k``, ``N_eff`` -- so unknown keys there are ignored rather than
        rejected, and a copied block needs no stripping. Unknown sections are
        ignored for the same reason.

        ``[Recon]`` is the exception: it is baorecon's own section, absent from
        the 2PCF parfile, so a key nothing reads there is a typo rather than
        another code's setting, and is rejected. A code layered on top puts its
        settings in a section of its own.

        Raises
        ------
        ValueError
            If a required section or key is missing, if ``[Recon]`` carries a
            key nothing reads, or if ``[Cosmology]`` describes a cosmology that
            is not flat LCDM.
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
        if parser.has_section("Recon"):
            _reject_unknown_recon_keys(recon_sec, filepath)
        output_sec = parser["Output"] if parser.has_section("Output") else {}

        # --- Columns -------------------------------------------------------
        # The parfile carries the coordinate columns per catalogue (the
        # 2PCF/Euclid shape) and so does the config: a catalogue pair that
        # spells them differently -- observed data against randoms written to
        # another convention -- is expressible. Omitting them from
        # [Catalog.Random] inherits the galaxy names, which is the common case.
        ra = _required(galaxy, "coord1", "Catalog.Galaxy", filepath)
        dec = _required(galaxy, "coord2", "Catalog.Galaxy", filepath)
        redshift = _required(galaxy, "coord3", "Catalog.Galaxy", filepath)
        ra_random = _optional(random, "coord1")
        dec_random = _optional(random, "coord2")
        redshift_random = _optional(random, "coord3")

        keep_raw = _raw(galaxy, "keep_cols")
        keep_cols = [] if keep_raw in (_MISSING, None) else _to_list(keep_raw)

        columns = ColumnMapping(
            ra=ra,
            dec=dec,
            redshift=redshift,
            ra_random=ra_random,
            dec_random=dec_random,
            redshift_random=redshift_random,
            weight_data=_optional(galaxy, "weight"),
            weight_random=_optional(random, "weight"),
            id_data=_optional(galaxy, "id"),
            id_random=_optional(random, "id"),
            keep_cols=keep_cols,
        )

        # --- Coordinate system ---------------------------------------------
        # Global for the run, so it is declared once in [Catalog.Galaxy].
        coordinate_system: Dict[str, Any] = {}
        # `coordinates` is the 2PCF/Euclid spelling (PSEUDO_EQUATORIAL / CARTESIAN);
        # it selects what coord1/coord2/coord3 actually hold.
        coord_input = _coordinate_input_from_ini(galaxy, filepath)
        if coord_input is not None:
            coordinate_system["input"] = coord_input
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
        _put(reconstruction, recon_sec, "align_cone", _to_bool)
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
            cosmology_extra=_cosmology_extra_from_ini(cosmology_sec),
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


# =============================================================================
# Input coordinate system
# =============================================================================
#
# The pipeline accepts catalogues either as observed sky coordinates or as
# Cartesian positions. The reconstruction engine itself is Cartesian-native
# (``BAOReconstructor`` takes (N, 3) positions), so "cartesian" simply skips the
# conversion on the way in and the inverse conversion on the way out; the
# reconstructed catalogue is then written back in the same system it came in.

COORDINATE_INPUTS = ("ra_dec_z", "cartesian")
DEFAULT_COORDINATE_INPUT = "ra_dec_z"


def resolve_coordinate_input(coordinate_system: Dict[str, Any]) -> str:
    """Return the validated ``input`` coordinate system for a run.

    Unknown values are rejected rather than defaulted: silently falling back to
    ``ra_dec_z`` would reinterpret Cartesian columns as degrees and a redshift,
    which produces plausible-looking numbers instead of an error.
    """
    value = coordinate_system.get("input") or DEFAULT_COORDINATE_INPUT
    value = str(value).strip().lower()
    if value not in COORDINATE_INPUTS:
        raise ValueError(
            "Unknown coordinate input {0!r}: expected one of {1}".format(
                value, ", ".join(COORDINATE_INPUTS))
        )
    return value
