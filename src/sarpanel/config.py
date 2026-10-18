"""
Run configuration.

A run is defined by one YAML file, not by a dozen command-line flags. The file
is the reproducibility unit: it is hashed into the output provenance, so an
output can always be traced to the exact settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from .processing import ProcessingConfig

log = logging.getLogger(__name__)


@dataclass
class AOIConfig:
    source: str
    id_column: str | None = None
    min_area_m2: float | None = None
    layer: str | None = None
    where: str | None = None
    search_buffer_m: float = 10_000.0


@dataclass
class SearchConfig:
    start_date: str
    end_date: str
    beam_mode: str = "IW"
    polarization: str | None = "VV+VH"
    relative_orbits: Sequence[int] | None = None
    orbit_direction: str | None = None


@dataclass
class OutputConfig:
    root: str
    granularity: str = "polygon"
    min_valid_pixels: int = 4
    enl_trim_percentile: float | None = 2.0
    drop_layover_shadow: bool = True
    compression: str = "zstd"
    partition_by_date: bool = True


@dataclass
class RunConfig:
    name: str
    aoi: AOIConfig
    search: SearchConfig
    processing: ProcessingConfig
    output: OutputConfig
    n_shards: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        path = Path(path)
        with open(path) as handle:
            raw = yaml.safe_load(handle)
        return cls.from_dict(raw, source_path=path)

    @classmethod
    def from_dict(cls, raw: dict, source_path: Path | None = None) -> "RunConfig":
        missing = [k for k in ("name", "aoi", "search", "output") if k not in raw]
        if missing:
            raise ValueError(f"Config is missing required section(s): {missing}")

        cfg = cls(
            name=raw["name"],
            aoi=AOIConfig(**raw["aoi"]),
            search=SearchConfig(**raw["search"]),
            processing=ProcessingConfig(**raw.get("processing", {})),
            output=OutputConfig(**raw["output"]),
            n_shards=int(raw.get("n_shards", 1)),
        )
        cfg._source_path = source_path
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.output.granularity not in ("pixel", "polygon"):
            raise ValueError("output.granularity must be 'pixel' or 'polygon'")
        if self.search.start_date > self.search.end_date:
            raise ValueError("search.start_date is after search.end_date")
        if self.n_shards < 1:
            raise ValueError("n_shards must be at least 1")

        if self.processing.is_slc and not self.search.relative_orbits:
            log.warning(
                "SLC processing without a relative_orbits restriction. SLC "
                "granules are roughly five times the size of GRD, and stacking "
                "across tracks mixes viewing geometries. Consider restricting "
                "to specific tracks."
            )
        if self.output.granularity == "pixel":
            log.warning(
                "Pixel granularity writes one row per pixel per acquisition. "
                "Estimate the volume before launching a multi-year run."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "aoi": asdict(self.aoi),
            "search": asdict(self.search),
            "processing": {
                k: (list(v) if isinstance(v, (tuple, list)) else v)
                for k, v in asdict(self.processing).items()
            },
            "output": asdict(self.output),
            "n_shards": self.n_shards,
        }

    def checksum(self) -> str:
        """Stable hash of the settings that affect output content."""
        payload = self.as_dict()
        payload.pop("n_shards", None)  # parallelism does not change content
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def root(self) -> Path:
        return Path(self.output.root)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest" / f"{self.name}_scenes.parquet"

    @property
    def aoi_path(self) -> Path:
        return self.root / "manifest" / f"{self.name}_aoi.parquet"

    @property
    def observations_dir(self) -> Path:
        return self.root / "observations"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            yaml.safe_dump(self.as_dict(), handle, sort_keys=False)
        return path
