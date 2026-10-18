"""
Per-scene orchestration: download, geocode, extract, clean up.

Ordering matters here. The completion check happens *before* the download, not
after the geocode. Checking afterwards means a rerun pays the full download and
processing cost before discovering it has nothing to do, which for SLC granules
of several gigabytes turns a no-op rerun into hours of wasted allocation.

Each scene is independent and failures are isolated. A scene that fails is
recorded in the run status table with its error and the shard continues, because
a single corrupt granule should not take down a 400-task array.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import radiometry as rad
from .aoi import AOISet
from .extract import SceneContext, SceneReader, extract_scene
from .processing import (
    GeocodeCapabilities,
    geocode_scene,
    validate_geometry_bands,
    write_scene_subset,
)
from .provenance import build_provenance, write_parquet_with_provenance

log = logging.getLogger(__name__)


@dataclass
class SceneResult:
    scene_id: str
    status: str
    n_observations: int = 0
    output_path: str | None = None
    seconds: float = 0.0
    error: str | None = None


def output_path_for(config, scene_row) -> Path:
    """
    Deterministic, collision-free output path.

    The scene identifier is part of the filename. Naming outputs by date and
    orbit direction alone collides whenever two granules on the same date share
    a direction, which across any large AOI is routine. The collision is not
    merely a name clash: with a skip-if-exists check, the second granule is
    silently dropped and the run reports success.
    """
    directory = config.observations_dir
    if config.output.partition_by_date:
        date = str(scene_row["acquisition_date"])
        directory = directory / f"month={date[:7]}"
    return directory / f"{scene_row['scene_id']}.parquet"


def download_scene(url: str, dest: Path, session, max_retries: int = 4) -> Path:
    """Stream a granule to disk with resumable retries and atomic rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Archive already present: %s", dest.name)
        return dest

    partial = dest.with_suffix(dest.suffix + ".part")
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            with session.get(url, allow_redirects=True, stream=True, timeout=180) as resp:
                if resp.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {resp.status_code} from {url}")
                with open(partial, "wb") as handle:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            partial.replace(dest)
            return dest
        except Exception as exc:
            last_error = exc
            wait = min(60, 2**attempt)
            log.warning(
                "Download attempt %d of %d failed (%s); retrying in %ds",
                attempt,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)
    partial.unlink(missing_ok=True)
    raise RuntimeError(f"Download failed after {max_retries} attempts: {last_error}")


def process_scene(
    scene_row: pd.Series,
    aoi: AOISet,
    config,
    session,
    caps: GeocodeCapabilities,
    provenance: dict,
) -> SceneResult:
    scene_id = scene_row["scene_id"]
    started = time.time()
    out_path = output_path_for(config, scene_row)

    # Completion check first, before any transfer or processing.
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info("Skipping %s, output already present", scene_id)
        return SceneResult(scene_id, "skipped", output_path=str(out_path))

    work = config.work_dir
    archive_dir = work / "archives"
    geocode_dir = work / "geocoded" / scene_id
    archive = archive_dir / f"{scene_id}.zip"

    try:
        # Restrict processing to the part of the swath holding AOIs. Written
        # before the download so a scene with no overlap costs nothing.
        subset_vector = None
        if config.processing.subset_to_aoi and caps.supports("shapefile"):
            subset_vector = write_scene_subset(
                aoi=aoi,
                scene_footprint_wkt=scene_row.get("footprint_wkt"),
                path=work / "subsets" / f"{scene_id}.geojson",
                buffer_m=config.processing.subset_buffer_m,
            )
            if subset_vector is None:
                log.info("Scene %s does not overlap the AOI; skipping", scene_id)
                return SceneResult(scene_id, "no_overlap", seconds=time.time() - started)

        download_scene(scene_row["url"], archive, session)

        bands = geocode_scene(
            archive=archive,
            outdir=geocode_dir,
            cfg=config.processing,
            tmpdir=work / "tmp",
            caps=caps,
            subset_vector=subset_vector,
        )
        validate_geometry_bands(bands, require_incidence=True)

        if not config.processing.keep_archives:
            archive.unlink(missing_ok=True)

        looks = rad.nominal_looks(
            config.processing.target_spacing_m, config.processing.product_type
        )
        context = SceneContext(
            scene_id=scene_id,
            acquisition_date=str(scene_row["acquisition_date"]),
            start_time=str(scene_row["start_time"]),
            platform=str(scene_row.get("platform")),
            orbit_direction=str(scene_row.get("orbit_direction")),
            relative_orbit=_opt_int(scene_row.get("relative_orbit")),
            absolute_orbit=_opt_int(scene_row.get("absolute_orbit")),
            frame_number=_opt_int(scene_row.get("frame_number")),
            product_type=config.processing.product_type,
            looks=looks,
        )

        # Preload the full scene only when the AOI set is dense enough that
        # windowed reads would touch most of it anyway.
        preload = len(aoi) > 5000

        with SceneReader(bands, preload=preload) as reader:
            df = extract_scene(
                reader=reader,
                aoi=aoi,
                context=context,
                granularity=config.output.granularity,
                min_valid_pixels=config.output.min_valid_pixels,
                enl_trim_percentile=config.output.enl_trim_percentile,
                drop_layover_shadow=config.output.drop_layover_shadow,
            )

        if df.empty:
            _cleanup(geocode_dir)
            return SceneResult(
                scene_id, "no_data", seconds=time.time() - started
            )

        scene_provenance = dict(provenance)
        scene_provenance["scene_id"] = scene_id
        scene_provenance["processing_seconds"] = str(bands.get("processing_seconds"))
        write_parquet_with_provenance(
            df, out_path, scene_provenance, compression=config.output.compression
        )

        _cleanup(geocode_dir)
        elapsed = time.time() - started
        log.info(
            "Completed %s: %d observations in %.1f min",
            scene_id,
            len(df),
            elapsed / 60,
        )
        return SceneResult(
            scene_id, "ok", len(df), str(out_path), elapsed
        )

    except Exception as exc:
        _cleanup(geocode_dir)
        log.error("Scene %s failed: %s", scene_id, exc)
        log.debug(traceback.format_exc())
        return SceneResult(
            scene_id,
            "failed",
            seconds=time.time() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _cleanup(directory: Path) -> None:
    import shutil

    shutil.rmtree(directory, ignore_errors=True)


def _opt_int(value):
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def run_shard(config, shard: int) -> pd.DataFrame:
    """Process every scene assigned to one shard and return a status table."""
    from .aoi import load_aoi
    from .auth import build_session
    from .manifest import read_manifest
    from .processing import prepare_environment

    env_info = prepare_environment(tmpdir=config.work_dir / "tmp")
    manifest = read_manifest(config.manifest_path, shard=shard)
    if manifest.empty:
        log.warning("Shard %d has no scenes assigned", shard)
        return pd.DataFrame()

    aoi = load_aoi(
        config.aoi_path if config.aoi_path.exists() else config.aoi.source,
        id_column=None if config.aoi_path.exists() else config.aoi.id_column,
    )
    session = build_session()
    caps = GeocodeCapabilities()
    log.info("pyroSAR capabilities: %s", caps.report())

    provenance = build_provenance(
        config, aoi.checksum(), extra={"java_heap_gb": env_info["java_heap_gb"]}
    )

    results = []
    for _, scene_row in manifest.iterrows():
        results.append(
            process_scene(scene_row, aoi, config, session, caps, provenance)
        )

    status = pd.DataFrame([r.__dict__ for r in results])
    status_path = config.root / "status" / f"shard_{shard:05d}.parquet"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status.to_parquet(status_path, index=False)

    counts = status["status"].value_counts().to_dict()
    log.info("Shard %d finished: %s", shard, counts)
    return status
