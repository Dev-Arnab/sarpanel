"""
Scene discovery and the run manifest.

The catalog is queried exactly once, by a ``plan`` step, and the result is
written to disk as a manifest. Worker tasks read their assigned rows from that
manifest rather than repeating the query.

This replaces the common pattern of having every array task run the same search
and then take a slice of the result. That pattern is wrong in two ways: it
issues N identical queries against the catalog, and it assumes the result
ordering is stable across calls. It is not guaranteed to be, so tasks can
silently process the same scene twice or skip one entirely, and the failure is
invisible because every task reports success.

The manifest also records which AOIs each scene actually intersects. For a
sparse AOI set this is a large saving: a scene that touches four fields should
not trigger a full-scene read against three hundred thousand polygons.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape

from .aoi import AOISet

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "scene_id",
    "product_type",
    "platform",
    "beam_mode",
    "polarization",
    "start_time",
    "stop_time",
    "acquisition_date",
    "orbit_direction",
    "absolute_orbit",
    "relative_orbit",
    "path_number",
    "frame_number",
    "url",
    "size_mb",
    "footprint_wkt",
    "n_aoi_hit",
    "aoi_ids",
    "shard",
]


def search_scenes(
    aoi: AOISet,
    start_date: str,
    end_date: str,
    product_type: str = "SLC",
    beam_mode: str = "IW",
    polarization: str | None = "VV+VH",
    platforms: Sequence[str] | None = None,
    relative_orbits: Sequence[int] | None = None,
    orbit_direction: str | None = None,
    search_buffer_m: float = 10_000.0,
) -> pd.DataFrame:
    """
    Query ASF once and return a scene manifest as a DataFrame.

    Parameters
    ----------
    product_type
        ``SLC`` or ``GRD``. SLC preserves phase and permits explicit control of
        multilooking, coherence, and dual-pol covariance. GRD is already
        multilooked and roughly a fifth of the volume.
    relative_orbits
        Restrict to specific tracks. Strongly recommended for time series:
        observations from different tracks have different viewing geometry and
        should not be stacked without conditioning on it.
    """
    import asf_search as asf

    footprint = aoi.search_footprint(buffer_m=search_buffer_m)

    if platforms is None:
        # S1B failed in December 2021; S1C began contributing in 2025. Include
        # all three and let the date range do the filtering.
        platforms = _available_platforms(asf)

    kwargs = dict(
        platform=list(platforms),
        processingLevel=_processing_levels(asf, product_type),
        beamMode=[beam_mode],
        start=_parse_date(start_date),
        end=_parse_date(end_date),
        intersectsWith=footprint.wkt,
        maxResults=None,
    )
    if polarization:
        kwargs["polarization"] = [polarization]
    if relative_orbits:
        kwargs["relativeOrbit"] = list(relative_orbits)
    if orbit_direction:
        kwargs["flightDirection"] = orbit_direction.upper()

    log.info(
        "Querying ASF: %s %s %s to %s over %d AOI(s)",
        beam_mode,
        product_type,
        start_date,
        end_date,
        len(aoi),
    )
    results = list(asf.search(**kwargs))
    log.info("Catalog returned %d scenes", len(results))

    rows = [_scene_row(r) for r in results]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    # Deduplicate: ASF can return the same granule under multiple entries.
    df = df.drop_duplicates(subset="scene_id")
    # Sort by a stable key so shard assignment is reproducible across runs.
    df = df.sort_values(["start_time", "scene_id"]).reset_index(drop=True)
    return df


def attach_aoi_membership(
    manifest: pd.DataFrame, aoi: AOISet, max_ids_stored: int = 5000
) -> pd.DataFrame:
    """
    Record which AOIs each scene intersects, using a spatial index.

    Scenes touching no AOI are dropped. Where a scene intersects more AOIs than
    ``max_ids_stored`` the explicit list is omitted and the intersection is
    recomputed at extraction time, because storing millions of identifiers per
    row makes the manifest larger than the data it describes.
    """
    if manifest.empty:
        return manifest

    footprints = gpd.GeoSeries(
        [shape_from_wkt(w) for w in manifest["footprint_wkt"]], crs="EPSG:4326"
    )
    scenes = gpd.GeoDataFrame(
        manifest.copy(), geometry=footprints.values, crs="EPSG:4326"
    )

    joined = gpd.sjoin(
        aoi.gdf[["aoi_id", "geometry"]],
        scenes[["scene_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    grouped = joined.groupby("scene_id")["aoi_id"].agg(list)

    manifest = manifest.copy()
    manifest["n_aoi_hit"] = manifest["scene_id"].map(grouped.apply(len)).fillna(0).astype(int)
    manifest["aoi_ids"] = manifest["scene_id"].map(
        grouped.apply(lambda ids: ids if len(ids) <= max_ids_stored else None)
    )

    before = len(manifest)
    manifest = manifest[manifest["n_aoi_hit"] > 0].reset_index(drop=True)
    log.info(
        "AOI intersection: %d of %d scenes touch at least one AOI",
        len(manifest),
        before,
    )
    return manifest


def assign_shards(manifest: pd.DataFrame, n_shards: int) -> pd.DataFrame:
    """
    Assign each scene to a worker shard.

    Assignment is a deterministic function of the sorted position, so the same
    manifest always produces the same partition and a rerun of a single failed
    task processes exactly the scenes it processed before.

    Scenes are dealt round-robin rather than in contiguous blocks so that each
    worker receives a mix of dates and tracks. Contiguous blocks tend to give
    one worker an entire dense season while another idles.
    """
    if manifest.empty:
        return manifest
    if n_shards < 1:
        raise ValueError("n_shards must be at least 1")
    manifest = manifest.copy()
    manifest["shard"] = np.arange(len(manifest)) % n_shards
    counts = manifest["shard"].value_counts().sort_index()
    log.info(
        "Sharded %d scenes across %d workers (min %d, max %d per worker)",
        len(manifest),
        n_shards,
        int(counts.min()),
        int(counts.max()),
    )
    return manifest


def write_manifest(manifest: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = manifest.copy()
    if "aoi_ids" in out.columns:
        out["aoi_ids"] = out["aoi_ids"].apply(
            lambda v: list(v) if isinstance(v, (list, tuple)) else None
        )
    out.to_parquet(path, index=False)
    log.info("Wrote manifest with %d scenes to %s", len(out), path)
    return path


def read_manifest(path: str | Path, shard: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if shard is not None:
        if "shard" not in df.columns:
            raise KeyError("Manifest has no shard column; run the plan step first")
        df = df[df["shard"] == shard].reset_index(drop=True)
        log.info("Shard %d has %d scenes assigned", shard, len(df))
    return df


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def shape_from_wkt(wkt_text: str):
    from shapely import wkt as _wkt

    return _wkt.loads(wkt_text)


def _parse_date(value: str):
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _available_platforms(asf) -> list[str]:
    names = ["SENTINEL1A", "SENTINEL1B", "SENTINEL1C"]
    out = []
    for name in names:
        value = getattr(asf.PLATFORM, name, None)
        if value is not None:
            out.append(value)
        else:
            log.debug("Platform %s not available in this asf_search version", name)
    if not out:
        raise RuntimeError("No Sentinel-1 platforms available in asf_search")
    return out


def _processing_levels(asf, product_type: str) -> list[str]:
    """
    Resolve a product type to the constants the installed asf_search defines.

    Only defined constants are sent. An earlier version fell back to the
    attribute name as a literal string when a constant was missing, which
    injected an invalid ``"GRD"`` value into the query. There is no plain
    ``GRD`` product type in asf_search: Sentinel-1 ground range products are
    ``GRD_HD``, ``GRD_MD``, ``GRD_HS``, ``GRD_MS``, and ``GRD_FD``. An
    unrecognised level makes the catalog resolve the platform and level pair to
    an empty or wrong collection set, and the query silently returns nothing,
    which reads as an archive gap rather than a malformed request.
    """
    product_type = product_type.upper()
    if product_type == "SLC":
        names = ["SLC"]
    elif product_type.startswith("GRD"):
        # High resolution first: GRDH is the standard IW land product.
        names = ["GRD_HD", "GRD_HS", "GRD_MD", "GRD_MS", "GRD_FD"]
    else:
        raise ValueError(f"product_type must be SLC or GRD, got {product_type!r}")

    levels = []
    for name in names:
        value = getattr(getattr(asf, "PRODUCT_TYPE", None), name, None)
        if isinstance(value, str):
            levels.append(value)
        else:
            log.debug("PRODUCT_TYPE.%s not defined in this asf_search", name)

    if not levels:
        raise RuntimeError(
            f"This asf_search build defines no product type constants for "
            f"{product_type!r}. Run tools/debug_search.py to see what it does define."
        )
    log.debug("processingLevel resolved to %s", levels)
    return levels


def _scene_row(result) -> dict:
    p = dict(getattr(result, "properties", {}) or {})
    geom = getattr(result, "geometry", None)
    footprint = shape(geom).wkt if geom else None

    start = p.get("startTime") or ""
    return {
        "scene_id": _clean_scene_id(p.get("fileID") or p.get("sceneName") or ""),
        "granule_name": p.get("sceneName"),
        "product_type": p.get("processingLevel"),
        "platform": p.get("platform"),
        "beam_mode": p.get("beamModeType") or p.get("beamMode"),
        "polarization": p.get("polarization"),
        "start_time": start,
        "stop_time": p.get("stopTime"),
        "acquisition_date": start[:10] if start else None,
        "orbit_direction": (p.get("flightDirection") or "UNKNOWN").upper(),
        "absolute_orbit": _as_int(p.get("orbit")),
        "relative_orbit": _as_int(p.get("pathNumber")),
        "path_number": _as_int(p.get("pathNumber")),
        "frame_number": _as_int(p.get("frameNumber")),
        "url": p.get("url") or p.get("downloadUrl"),
        "size_mb": _as_float(p.get("bytes"), scale=1 / (1024 * 1024)),
        "footprint_wkt": footprint,
    }


def _clean_scene_id(file_id: str) -> str:
    """
    Strip the ASF product-type suffix but keep everything that identifies the
    acquisition. The original granule name is retained separately.
    """
    for tag in ("-SLC", "-GRD_HD", "-GRD_MD", "-GRD"):
        file_id = file_id.replace(tag, "")
    return file_id


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value, scale: float = 1.0):
    try:
        return float(value) * scale
    except (TypeError, ValueError):
        return None
