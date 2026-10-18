"""
Area-of-interest handling.

An AOI set is always represented internally as a GeoDataFrame with a stable
string column ``aoi_id`` and a polygon geometry in EPSG:4326. Everything
downstream (search footprint, pixel extraction, output partitioning) keys off
``aoi_id``, so a run over one field and a run over 300,000 fields differ only in
the number of rows.

Accepted inputs
---------------
* any vector file readable by GeoPandas (.gpkg, .geojson, .shp, .parquet, ...)
* a single bounding box as "minx,miny,maxx,maxy" in EPSG:4326
* a list of bounding boxes
* a WKT string
* an existing GeoDataFrame (in-process use)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
from shapely import wkt as shapely_wkt
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"


def union_all(geometries) -> BaseGeometry:
    """
    Dissolve a GeoSeries or GeoDataFrame into one geometry.

    GeoPandas renamed the ``unary_union`` property to a ``union_all()`` method
    in 1.0, deprecating the old name. Clusters commonly pin an older release
    alongside a conda GDAL stack, so both spellings are supported rather than
    forcing an upgrade of a working geospatial environment.
    """
    geoms = getattr(geometries, "geometry", geometries)
    method = getattr(geoms, "union_all", None)
    if callable(method):
        return method()
    return geoms.unary_union


@dataclass(frozen=True)
class AOISet:
    """A validated collection of one or more analysis polygons."""

    gdf: gpd.GeoDataFrame
    id_column: str = "aoi_id"

    def __len__(self) -> int:
        return len(self.gdf)

    @property
    def ids(self) -> list[str]:
        return self.gdf[self.id_column].tolist()

    def to_crs(self, crs) -> gpd.GeoDataFrame:
        return self.gdf.to_crs(crs)

    def search_footprint(self, buffer_m: float = 10_000.0) -> BaseGeometry:
        """
        Dissolved, buffered footprint in EPSG:4326 for the catalog query.

        The buffer is applied in a projected CRS so it is a true metric
        distance, then reprojected back to geographic coordinates. For very
        large or scattered AOI sets the convex hull of the union is used
        instead of the union itself, because ASF's ``intersectsWith`` rejects
        pathologically complex WKT.
        """
        metric = self.gdf.estimate_utm_crs()
        geom = union_all(self.gdf.to_crs(metric).buffer(buffer_m))
        footprint = gpd.GeoSeries([geom], crs=metric).to_crs(WGS84).iloc[0]

        # ASF's CMR backend rejects WKT beyond a few thousand vertices.
        n_coords = _count_coords(footprint)
        if n_coords > 500:
            log.info(
                "AOI footprint has %d vertices; simplifying to convex hull for search",
                n_coords,
            )
            footprint = footprint.convex_hull
        return footprint

    def subset(self, ids: Iterable[str]) -> "AOISet":
        wanted = set(ids)
        sub = self.gdf[self.gdf[self.id_column].isin(wanted)].reset_index(drop=True)
        return AOISet(sub, self.id_column)

    def intersecting(self, geom: BaseGeometry) -> "AOISet":
        """AOIs whose geometry intersects ``geom`` (both in EPSG:4326)."""
        hit = self.gdf[self.gdf.intersects(geom)].reset_index(drop=True)
        return AOISet(hit, self.id_column)

    def checksum(self) -> str:
        """Stable hash of the AOI geometries, recorded in run provenance."""
        h = hashlib.sha256()
        for aoi_id, geom in zip(self.gdf[self.id_column], self.gdf.geometry):
            h.update(str(aoi_id).encode())
            h.update(geom.wkb)
        return h.hexdigest()[:16]


def _count_coords(geom: BaseGeometry) -> int:
    if hasattr(geom, "geoms"):
        return sum(_count_coords(g) for g in geom.geoms)
    if geom.geom_type == "Polygon":
        return len(geom.exterior.coords) + sum(len(r.coords) for r in geom.interiors)
    return len(getattr(geom, "coords", []))


def _parse_bbox(text: str) -> BaseGeometry:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"bbox must be 'minx,miny,maxx,maxy' in EPSG:4326, got {text!r}"
        )
    minx, miny, maxx, maxy = (float(p) for p in parts)
    if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
        raise ValueError(f"bbox {text!r} is not a valid EPSG:4326 extent")
    return box(minx, miny, maxx, maxy)


def load_aoi(
    source,
    id_column: str | None = None,
    min_area_m2: float | None = None,
    layer: str | None = None,
    where: str | None = None,
) -> AOISet:
    """
    Build an :class:`AOISet` from any supported input.

    Parameters
    ----------
    source
        Path to a vector file, a bbox string, a list of bbox strings, a WKT
        string, a shapely geometry, or a GeoDataFrame.
    id_column
        Column to use as the stable AOI identifier. If omitted, the first of
        ``aoi_id``, ``farm_name``, ``field_id``, ``id``, ``name`` present is
        used; failing that, sequential identifiers are generated.
    min_area_m2
        Drop polygons smaller than this, measured in a projected CRS. Useful
        because sub-pixel polygons produce empty extractions and waste a
        scene-sized read.
    where
        Optional pandas query applied to the attribute table before use.
    """
    gdf = _to_gdf(source, layer=layer)

    if gdf.crs is None:
        raise ValueError(
            "AOI source has no CRS. Assign one explicitly rather than assuming "
            "EPSG:4326; a wrong CRS here silently produces an empty search."
        )

    if where:
        gdf = gdf.query(where).reset_index(drop=True)
        log.info("AOI filter %r left %d features", where, len(gdf))

    gdf = gdf[~gdf.geometry.isna()].reset_index(drop=True)
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(
        drop=True
    )
    if gdf.empty:
        raise ValueError("No polygon features found in AOI source")

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        log.warning("Repairing %d invalid geometries with buffer(0)", int(invalid.sum()))
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    metric = gdf.estimate_utm_crs()
    area_m2 = gdf.to_crs(metric).area
    gdf["aoi_area_m2"] = area_m2.values

    if min_area_m2 is not None:
        keep = gdf["aoi_area_m2"] >= min_area_m2
        log.info(
            "Area filter >= %.0f m2: %d of %d AOIs retained",
            min_area_m2,
            int(keep.sum()),
            len(gdf),
        )
        gdf = gdf[keep].reset_index(drop=True)
        if gdf.empty:
            raise ValueError("No AOIs remain after the area filter")

    resolved = _resolve_id_column(gdf, id_column)
    gdf["aoi_id"] = gdf[resolved].astype(str).str.strip().str.replace(r"\s+", "_", regex=True)

    dupes = gdf["aoi_id"].duplicated()
    if dupes.any():
        log.warning(
            "%d duplicate AOI ids in column %r; disambiguating with a suffix",
            int(dupes.sum()),
            resolved,
        )
        counts = gdf.groupby("aoi_id").cumcount()
        gdf.loc[counts > 0, "aoi_id"] = (
            gdf.loc[counts > 0, "aoi_id"] + "__" + counts[counts > 0].astype(str)
        )

    gdf = gdf.to_crs(WGS84)
    keep_cols = ["aoi_id", "aoi_area_m2", "geometry"]
    extra = [c for c in gdf.columns if c not in keep_cols and c != resolved]
    gdf = gdf[keep_cols + extra]

    log.info("Loaded %d AOI polygon(s)", len(gdf))
    return AOISet(gdf.reset_index(drop=True))


def _resolve_id_column(gdf: gpd.GeoDataFrame, id_column: str | None) -> str:
    if id_column:
        if id_column not in gdf.columns:
            raise KeyError(
                f"id_column {id_column!r} not in AOI source. "
                f"Available: {sorted(c for c in gdf.columns if c != 'geometry')}"
            )
        return id_column
    for candidate in ("aoi_id", "farm_name", "field_id", "field_name", "id", "name"):
        if candidate in gdf.columns:
            log.info("Using %r as the AOI identifier column", candidate)
            return candidate
    gdf["_generated_id"] = [f"aoi_{i:06d}" for i in range(len(gdf))]
    log.info("No identifier column found; generated sequential AOI ids")
    return "_generated_id"


def _to_gdf(source, layer: str | None = None) -> gpd.GeoDataFrame:
    if isinstance(source, gpd.GeoDataFrame):
        return source.copy()

    if isinstance(source, BaseGeometry):
        return gpd.GeoDataFrame(geometry=[source], crs=WGS84)

    if isinstance(source, (list, tuple)):
        geoms = [_coerce_scalar(s) for s in source]
        return gpd.GeoDataFrame(
            {"aoi_id": [f"aoi_{i:04d}" for i in range(len(geoms))]},
            geometry=geoms,
            crs=WGS84,
        )

    if isinstance(source, dict):  # GeoJSON-like mapping
        return gpd.GeoDataFrame(geometry=[shape(source)], crs=WGS84)

    if isinstance(source, (str, Path)):
        text = str(source)
        path = Path(text)
        if path.exists():
            if path.suffix.lower() == ".parquet":
                return gpd.read_parquet(path)
            return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        return gpd.GeoDataFrame(geometry=[_coerce_scalar(text)], crs=WGS84)

    raise TypeError(f"Unsupported AOI source type: {type(source)!r}")


def _coerce_scalar(value) -> BaseGeometry:
    if isinstance(value, BaseGeometry):
        return value
    text = str(value).strip()
    if text.upper().startswith(("POLYGON", "MULTIPOLYGON")):
        return shapely_wkt.loads(text)
    return _parse_bbox(text)


def write_aoi(aoi: AOISet, path: str | Path) -> Path:
    """Persist the resolved AOI set so a run is reproducible from disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    aoi.gdf.to_parquet(path, index=False)
    return path
