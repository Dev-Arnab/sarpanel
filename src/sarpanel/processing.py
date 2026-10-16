"""
SNAP processing via pyroSAR, for both GRD and SLC input.

Geometry bands
--------------
The projected local incidence angle is requested as an *output band* of the
Range-Doppler Terrain Correction operator, via pyroSAR's ``export_extra``. SNAP
then writes it as a GeoTIFF on exactly the same grid as the backscatter bands,
so sampling it is an array slice with no coordinate transformation involved.

The alternative approach, reading ``tie_point_grids/incident_angle`` out of the
SNAP intermediate directory, is incorrect. The tie point grid lives in the
pre-terrain-correction raster geometry, whereas the output is on a map grid with
different dimensions and origin. Indexing one with the other's row and column
produces values in a plausible range that are not spatially correct, and
clipping the indices to the array bounds hides the mismatch rather than raising.
Because the intermediate directory is also shared between concurrent tasks, a
glob over it can return a different scene's grid entirely.

Three angle bands are available and they are not interchangeable:

``incidenceAngleFromEllipsoid``
    Ellipsoid geometry only. Ignores terrain. Adequate for flat ground.
``localIncidenceAngle``
    Angle between the radar look vector and the DEM surface normal, in the
    original radar geometry.
``projectedLocalIncidenceAngle``
    The same quantity projected into the map grid. This is the one to use as a
    per-observation covariate, and the one that matters as soon as the terrain
    is not flat.

Radiometric terrain flattening
------------------------------
With ``terrain_flattening`` enabled the output is gamma0 rather than sigma0,
which removes most of the topographic modulation of backscatter. The local
incidence angle is still recorded, because residual angle dependence in
vegetation is real and is better handled as a modelled covariate than assumed
away.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

# Bands requested from Terrain Correction. Each becomes a co-registered GeoTIFF.
DEFAULT_EXPORT_EXTRA = [
    "projectedLocalIncidenceAngle",
    "incidenceAngleFromEllipsoid",
    "localIncidenceAngle",
    "layoverShadowMask",
    "DEM",
]

# Suffix fragments used to identify each written band on disk.
BAND_PATTERNS = {
    "inc_angle_projected": "projectedLocalIncidenceAngle",
    "inc_angle_ellipsoid": "incidenceAngleFromEllipsoid",
    "inc_angle_local": "localIncidenceAngle",
    "layover_shadow": "layoverShadowMask",
    "elevation": "DEM",
}


@dataclass
class ProcessingConfig:
    """Everything that determines the radiometric content of the output."""

    product_type: str = "SLC"
    target_crs: str = "EPSG:32616"
    target_spacing_m: float = 10.0
    polarizations: Sequence[str] = ("VV", "VH")
    terrain_flattening: bool = True
    speckle_filter: str | None = None
    dem: str = "Copernicus 30m Global DEM"
    remove_thermal_noise: bool = True
    remove_border_noise: bool = True
    scaling: str = "linear"
    export_extra: Sequence[str] = field(default_factory=lambda: list(DEFAULT_EXPORT_EXTRA))
    subswaths: Sequence[str] | None = None
    keep_archives: bool = False

    # SNAP's tile cache is the dominant memory consumer, and by default it
    # grows toward the JVM heap limit. Terrain-Flattening in particular will
    # expand until something kills it. Capping the cache well below the heap
    # leaves room for the operator's own working set, which is what prevents
    # the SIGKILL that presents as "process return code -9".
    snap_tile_cache_gb: int | None = None
    snap_threads: int | None = None

    # Restrict processing to the part of the scene that contains AOIs.
    #
    # Terrain-Flattening dominates runtime and its cost scales with processed
    # area, not with how many polygons are wanted. Left unrestricted it works
    # over the entire 250 x 170 km swath, which for a sparse AOI means most of
    # the compute produces nothing that is ever read.
    #
    # The buffer is generous on purpose. Terrain-Flattening needs surrounding
    # terrain to compute the illuminated area for pixels near the edge, so a
    # subset cut tight to the polygons would bias the outermost pixels. Several
    # kilometres costs little and keeps the edge well outside the AOI.
    subset_to_aoi: bool = True
    subset_buffer_m: float = 5000.0

    def __post_init__(self):
        if self.scaling.lower() != "linear":
            raise ValueError(
                "scaling must be 'linear'. Decibel conversion is applied at "
                "aggregation time, not at geocoding time, because averaging in "
                "the log domain is biased and the bias is correlated with "
                "within-polygon heterogeneity."
            )
        if self.target_crs.upper() in ("EPSG:4326", "4326"):
            raise ValueError(
                "target_crs must be a projected CRS. In EPSG:4326 the units of "
                "target_spacing_m are degrees, which makes the requested "
                "resolution ambiguous and produces anisotropic pixels that vary "
                "with latitude. Use the appropriate UTM zone."
            )

    @property
    def is_slc(self) -> bool:
        return self.product_type.upper() == "SLC"


class GeocodeCapabilities:
    """
    Introspects the installed pyroSAR to see which arguments it accepts.

    pyroSAR's geocode signature has changed across releases (``tr`` versus
    ``spacing``, presence of ``export_extra``, ``terrainFlattening``,
    ``refarea``). Rather than pinning one version and failing loudly on
    others, the supported set is detected once and unsupported requests are
    reported explicitly instead of being silently dropped.
    """

    def __init__(self):
        from pyroSAR.snap.util import geocode

        self._geocode = geocode
        self.params = set(signature(geocode).parameters)

    def supports(self, name: str) -> bool:
        return name in self.params

    def build_kwargs(self, cfg: ProcessingConfig, infile: str, outdir: str,
                     tmpdir: str, subset_vector: str | None = None) -> dict:
        kwargs: dict = {
            "infile": infile,
            "outdir": outdir,
            "t_srs": cfg.target_crs,
            "polarizations": list(cfg.polarizations),
            "scaling": cfg.scaling,
            "tmpdir": tmpdir,
        }

        # Output pixel spacing: argument renamed between releases.
        if self.supports("tr"):
            kwargs["tr"] = cfg.target_spacing_m
        elif self.supports("spacing"):
            kwargs["spacing"] = cfg.target_spacing_m
        else:
            raise RuntimeError("Installed pyroSAR exposes neither 'tr' nor 'spacing'")

        for name, value in [
            ("removeS1ThermalNoise", cfg.remove_thermal_noise),
            ("removeS1BorderNoise", cfg.remove_border_noise),
            ("terrainFlattening", cfg.terrain_flattening),
            ("demName", cfg.dem),
            ("export_extra", list(cfg.export_extra)),
            ("speckleFilter", cfg.speckle_filter or False),
        ]:
            if self.supports(name):
                kwargs[name] = value
            elif value:
                log.warning(
                    "Installed pyroSAR does not support %r; this setting has no "
                    "effect and the output will differ from the configuration",
                    name,
                )

        if cfg.is_slc and cfg.subswaths and self.supports("subswaths"):
            kwargs["subswaths"] = list(cfg.subswaths)

        if subset_vector:
            if self.supports("shapefile"):
                kwargs["shapefile"] = subset_vector
            else:
                log.warning(
                    "Installed pyroSAR does not accept 'shapefile'; the full "
                    "scene will be processed and Terrain-Flattening will cost "
                    "several times more than necessary"
                )

        # Bound SNAP's tile cache. Left alone it grows toward the JVM heap
        # limit, and Terrain-Flattening will expand until the scheduler or the
        # OOM killer stops it, which surfaces as "process return code -9".
        gpt_args = _gpt_resource_args(cfg)
        if gpt_args:
            if self.supports("gpt_args"):
                kwargs["gpt_args"] = gpt_args
            else:
                log.warning(
                    "Installed pyroSAR does not accept 'gpt_args'; SNAP's tile "
                    "cache cannot be capped and memory use will be unbounded"
                )

        # gamma0 versus sigma0 reference area
        if self.supports("refarea"):
            kwargs["refarea"] = "gamma0" if cfg.terrain_flattening else "sigma0"

        return kwargs

    def report(self) -> dict:
        return {
            "export_extra": self.supports("export_extra"),
            "terrain_flattening": self.supports("terrainFlattening"),
            "subswaths": self.supports("subswaths"),
            "refarea": self.supports("refarea"),
            "gpt_args": self.supports("gpt_args"),
            "shapefile": self.supports("shapefile"),
            "spacing_arg": "tr" if self.supports("tr") else "spacing",
        }

    def __call__(self, **kwargs):
        return self._geocode(**kwargs)


def _gpt_resource_args(cfg: ProcessingConfig) -> list[str]:
    """
    Build the ``gpt`` resource flags: ``-c`` tile cache, ``-q`` thread count.

    Defaults are derived from the scheduler allocation when not set explicitly.
    The cache is held to roughly a third of the allocation, which is
    deliberately conservative: SNAP's operators need working memory beyond the
    cache, and the JVM heap is separately sized at about half the allocation,
    so a cache near the heap limit leaves nothing for the operator itself.
    """
    args: list[str] = []

    cache_gb = cfg.snap_tile_cache_gb
    if cache_gb is None:
        allocated_mb = _slurm_memory_mb()
        cache_gb = max(2, int(allocated_mb / 3 / 1024))
    args += ["-c", f"{cache_gb}G"]

    threads = cfg.snap_threads
    if threads is None:
        threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or None
    if threads:
        args += ["-q", str(threads)]

    return args


def prepare_environment(tmpdir: str | Path | None = None, mem_fraction: float = 0.5) -> dict:
    """
    Configure the JVM before SNAP starts.

    The heap is sized as a fraction of the scheduler's memory allocation rather
    than hardcoded. Setting ``-Xmx`` close to the full allocation is a common
    cause of jobs being killed: the JVM will happily expand to its limit, and
    the Python process still has to hold a scene-sized array afterwards, so the
    two together exceed the cgroup limit and the task dies without a traceback.
    """
    allocated_mb = _slurm_memory_mb()
    heap_gb = max(4, int((allocated_mb * mem_fraction) / 1024))

    tmpdir = Path(tmpdir or os.environ.get("TMPDIR") or "/tmp")
    tmpdir.mkdir(parents=True, exist_ok=True)

    java_opts = f"-Djava.awt.headless=true -Xmx{heap_gb}G -Djava.io.tmpdir={tmpdir}"
    os.environ["_JAVA_OPTIONS"] = java_opts
    os.environ["SNAP_JAVA_OPTS"] = java_opts
    os.environ["JAVA_TOOL_OPTIONS"] = f"-Djava.io.tmpdir={tmpdir}"
    os.environ["TMPDIR"] = str(tmpdir)

    under_scheduler = "SLURM_JOB_ID" in os.environ
    if not under_scheduler:
        log.warning(
            "No SLURM job detected. Development and login nodes cap per-user "
            "memory and CPU time well below what Terrain-Flattening needs on a "
            "full scene, and the process will be killed with return code -9. "
            "Submit with sbatch instead."
        )

    info = {
        "allocated_mb": allocated_mb,
        "java_heap_gb": heap_gb,
        "tmpdir": str(tmpdir),
        "under_scheduler": under_scheduler,
    }
    log.info(
        "JVM heap %d GB of %d MB allocated, tmpdir %s", heap_gb, allocated_mb, tmpdir
    )
    return info


def _slurm_memory_mb() -> int:
    for var in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        raw = os.environ.get(var)
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if var == "SLURM_MEM_PER_CPU":
            value *= int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        return value
    try:  # fall back to the cgroup limit, then to total RAM
        import psutil

        return int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        return 32 * 1024


def geocode_scene(
    archive: Path,
    outdir: Path,
    cfg: ProcessingConfig,
    tmpdir: Path,
    caps: GeocodeCapabilities | None = None,
    subset_vector: str | None = None,
) -> dict:
    """
    Run the SNAP graph for one scene and return the paths of the written bands.

    Each scene is given its own subdirectory of ``tmpdir``. SNAP writes large
    intermediate products there, and sharing one directory between concurrent
    tasks is how intermediates from different scenes get mixed together.
    """
    caps = caps or GeocodeCapabilities()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scene_tmp = Path(tmpdir) / f"snap_{archive.stem}_{os.getpid()}"
    scene_tmp.mkdir(parents=True, exist_ok=True)

    kwargs = caps.build_kwargs(cfg, str(archive), str(outdir), str(scene_tmp),
                               subset_vector=subset_vector)
    started = time.time()
    try:
        caps(**kwargs)
    finally:
        shutil.rmtree(scene_tmp, ignore_errors=True)

    elapsed = time.time() - started
    bands = discover_bands(outdir, cfg.polarizations)
    log.info(
        "Geocoded %s in %.1f min: %d backscatter band(s), incidence angle %s",
        archive.stem,
        elapsed / 60,
        len(bands["backscatter"]),
        "present" if bands.get("inc_angle_projected") else "MISSING",
    )
    bands["processing_seconds"] = elapsed
    return bands


def discover_bands(outdir: Path, polarizations: Sequence[str]) -> dict:
    """
    Map the GeoTIFFs SNAP wrote to their semantic roles.

    Band naming varies with the operator chain, so matching is on substrings
    rather than exact filenames. The auxiliary bands are matched first and
    excluded from the backscatter set, otherwise a file named
    ``..._VV_..._localIncidenceAngle.tif`` would be picked up as VV backscatter.
    """
    outdir = Path(outdir)
    tifs = sorted(p for p in outdir.rglob("*.tif") if p.is_file())

    result: dict = {"backscatter": {}, "all_files": [str(p) for p in tifs]}
    auxiliary: set[Path] = set()

    for role, pattern in BAND_PATTERNS.items():
        match = next((p for p in tifs if pattern.lower() in p.name.lower()), None)
        if match:
            result[role] = match
            auxiliary.add(match)
        else:
            result[role] = None

    for pol in polarizations:
        token = pol.upper()
        match = next(
            (
                p
                for p in tifs
                if p not in auxiliary
                and (f"_{token}_" in p.name.upper() or p.stem.upper().endswith(token))
            ),
            None,
        )
        if match:
            result["backscatter"][token] = match

    missing = [p for p in polarizations if p.upper() not in result["backscatter"]]
    if missing:
        log.warning("No output band found for polarization(s): %s", ", ".join(missing))

    return result


def validate_geometry_bands(bands: dict, require_incidence: bool = True) -> None:
    """
    Fail with the real cause when the SNAP graph did not produce output.

    Order matters. pyroSAR catches a ``gpt`` crash internally, logs it, and
    returns normally, so the caller sees an output directory with no bands
    rather than an exception. Checking for the incidence angle first would then
    report a missing angle band, which points at ``export_extra`` and the DEM
    and sends the reader in entirely the wrong direction. The absence of every
    band means the graph died, and that is what gets reported.
    """
    if not bands.get("backscatter"):
        raise RuntimeError(
            "SNAP produced no backscatter bands for this scene, which means the "
            "processing graph failed rather than that a band is missing.\n\n"
            "Look further up the log for the gpt error. If it reads 'process "
            "return code -9', the process was killed by the operating system, "
            "almost always for memory:\n"
            "  - Run under sbatch rather than on a login or development node. "
            "Development nodes cap per-user memory and CPU time, and "
            "Terrain-Flattening exceeds those caps on a full scene.\n"
            "  - Request more memory, or lower 'snap_tile_cache_gb' in the "
            "config so SNAP's cache leaves room for the operator's working "
            "set.\n"
            "  - Terrain-Flattening is the most memory-hungry step; setting "
            "'terrain_flattening: false' produces sigma0 instead of gamma0 and "
            "will confirm whether memory is the constraint."
        )

    if require_incidence and bands.get("inc_angle_projected") is None:
        written = [Path(p).name for p in bands.get("all_files", [])][:8]
        raise RuntimeError(
            "Backscatter was written but projectedLocalIncidenceAngle was not.\n"
            f"  bands found: {written}\n\n"
            "Check that 'sarpanel doctor' reports export_extra: True and that "
            "the DEM was reachable during processing. Without the angle band "
            "there is no correct fallback, so the run stops here rather than "
            "filling the column with nulls."
        )


def write_scene_subset(
    aoi,
    scene_footprint_wkt: str | None,
    path: Path,
    buffer_m: float = 5000.0,
    max_vertices: int = 200,
) -> str | None:
    """
    Write the processing subset for one scene as a vector file.

    The subset is the AOI, buffered, clipped to the scene footprint. Clipping
    matters when the AOI is much larger than any single scene: passing the whole
    AOI would ask SNAP to subset to a region mostly outside the data, which it
    handles poorly.

    The geometry is simplified to a convex hull past ``max_vertices``. SNAP
    subsets to a bounding region rather than a precise mask, so detail beyond a
    coarse outline buys nothing and slows graph parsing. Returns ``None`` when
    there is no overlap, which tells the caller to skip the scene.
    """
    import geopandas as gpd
    from shapely import wkt as _wkt

    metric = aoi.gdf.estimate_utm_crs()
    buffered = union_all_compat(aoi.gdf.to_crs(metric).buffer(buffer_m))
    geom = gpd.GeoSeries([buffered], crs=metric).to_crs("EPSG:4326").iloc[0]

    if scene_footprint_wkt:
        footprint = _wkt.loads(scene_footprint_wkt)
        if not geom.intersects(footprint):
            return None
        geom = geom.intersection(footprint)
        if geom.is_empty:
            return None

    geom = _simplify_for_snap(geom, max_vertices)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs="EPSG:4326").to_file(
        path, driver="GeoJSON"
    )
    return str(path)


def union_all_compat(geometries):
    """GeoPandas renamed ``unary_union`` to ``union_all()`` in 1.0."""
    geoms = getattr(geometries, "geometry", geometries)
    method = getattr(geoms, "union_all", None)
    return method() if callable(method) else geoms.unary_union


def _vertex_count(geom) -> int:
    if hasattr(geom, "geoms"):
        return sum(_vertex_count(g) for g in geom.geoms)
    if geom.geom_type == "Polygon":
        return len(geom.exterior.coords) + sum(len(r.coords) for r in geom.interiors)
    return len(getattr(geom, "coords", []))


def _simplify_for_snap(geom, max_vertices: int):
    """
    Reduce a subset geometry to something SNAP parses quickly.

    A convex hull alone is not enough. Buffering rounds every corner into an
    arc of many points, and the hull of a long buffered strip keeps all of
    them, so the vertex count barely moves. The hull is therefore simplified
    with a growing tolerance, and if that still does not fit, the bounding box
    is used.

    Over-covering is the safe direction. SNAP subsets to a region and the
    extraction step masks to the true polygons afterwards, so a generous subset
    costs some compute but cannot lose data. An under-covering subset would
    silently clip fields.
    """
    if _vertex_count(geom) <= max_vertices:
        return geom

    geom = geom.convex_hull
    if _vertex_count(geom) <= max_vertices:
        return geom

    # Tolerance in degrees: 0.001 is roughly 100 m at these latitudes.
    for tolerance in (0.001, 0.005, 0.01, 0.05):
        simplified = geom.simplify(tolerance, preserve_topology=True)
        if not simplified.is_empty and _vertex_count(simplified) <= max_vertices:
            return simplified

    log.debug("Subset geometry still complex after simplification; using envelope")
    return geom.envelope
