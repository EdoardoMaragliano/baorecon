"""YAML configuration parsing for BAO reconstruction pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)


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
