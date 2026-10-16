"""
Provenance capture.

Every output file carries, in its Parquet key-value metadata, enough
information to reconstruct how it was made: software versions, the DEM used,
the orbit file type, the configuration hash, and the AOI hash. This is the
difference between a result that can be defended in review and one that cannot.

Orbit file type is recorded deliberately. pyroSAR fetches orbit state vectors
during processing, and on a compute node with restricted egress or a cold cache
it can silently fall back from precise to restituted orbits. For SLC geometry
that matters, and a run where half the scenes used restituted orbits without
anyone noticing is a run that has to be repeated.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def software_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for module in ("numpy", "pandas", "geopandas", "rasterio", "pyarrow", "shapely"):
        versions[module] = _version(module)
    versions["pyroSAR"] = _version("pyroSAR")
    versions["asf_search"] = _version("asf_search")
    versions["snap"] = snap_version()
    versions["sarpanel"] = _version("sarpanel")
    return versions


def _version(module_name: str) -> str:
    try:
        from importlib.metadata import version

        return version(module_name)
    except Exception:
        try:
            module = __import__(module_name)
            return str(getattr(module, "__version__", "unknown"))
        except Exception:
            return "not installed"


def snap_version() -> str:
    """
    Read the SNAP version from the installation rather than assuming.

    SNAP's operator behaviour changes between releases, notably in terrain
    flattening and border noise removal, so the version is part of the
    scientific description of the output.
    """
    snap_home = os.environ.get("SNAP_HOME")
    if not snap_home:
        return "unknown"
    for candidate in ("VERSION.txt", "snap/VERSION.txt"):
        path = Path(snap_home) / candidate
        if path.exists():
            return path.read_text().strip()
    try:
        out = subprocess.run(
            [str(Path(snap_home) / "bin" / "gpt"), "-h"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        for line in (out.stdout or "").splitlines():
            if "SNAP" in line and any(c.isdigit() for c in line):
                return line.strip()
    except Exception:
        pass
    return "unknown"


def slurm_context() -> dict[str, str]:
    keys = [
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURMD_NODENAME",
    ]
    return {k: os.environ[k] for k in keys if k in os.environ}


def build_provenance(
    config,
    aoi_checksum: str,
    extra: dict | None = None,
) -> dict[str, str]:
    record = {
        "sarpanel_run_name": config.name,
        "config_checksum": config.checksum(),
        "aoi_checksum": aoi_checksum,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": json.dumps(config.as_dict(), sort_keys=True, default=str),
        "software": json.dumps(software_versions(), sort_keys=True),
        "scheduler": json.dumps(slurm_context(), sort_keys=True),
    }
    if extra:
        record.update({k: str(v) for k, v in extra.items()})
    return record


def write_parquet_with_provenance(
    df,
    path: str | Path,
    provenance: dict[str, str],
    compression: str = "zstd",
) -> Path:
    """
    Write a DataFrame to Parquet with provenance in the file-level metadata.

    Metadata travels with the file. A parquet copied out of the run directory
    to a laptop still knows which DEM and which SNAP version produced it.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)
    existing = table.schema.metadata or {}
    merged = dict(existing)
    merged.update({k.encode(): str(v).encode() for k, v in provenance.items()})
    table = table.replace_schema_metadata(merged)

    pq.write_table(table, path, compression=compression)
    return path


def read_provenance(path: str | Path) -> dict[str, str]:
    """Recover the provenance record from a written parquet."""
    import pyarrow.parquet as pq

    meta = pq.read_schema(str(path)).metadata or {}
    out = {}
    for key, value in meta.items():
        key_s = key.decode() if isinstance(key, bytes) else str(key)
        if key_s.startswith("pandas"):
            continue
        out[key_s] = value.decode() if isinstance(value, bytes) else str(value)
    return out
