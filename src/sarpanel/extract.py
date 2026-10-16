"""
Polygon-native extraction of analysis-ready observations.

This is the part of the package that differs from existing SAR toolchains.
Rather than writing image products and leaving the user to sample them, each
(polygon, acquisition) pair becomes a row with the backscatter estimate, the
acquisition geometry that produced it, and the information needed to weight it
in a statistical model.

Two output granularities are supported:

``pixel``
    One row per pixel per polygon per acquisition. Large, but necessary for
    within-field spatial analysis such as yield zone delineation.
``polygon``
    One row per polygon per acquisition, with the mean computed in linear power,
    the empirical ENL, the valid pixel count, and the resulting standard error.
    Typically two to three orders of magnitude smaller.

Both can be written from a single read of the scene.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import box

from . import radiometry as rad
from .aoi import AOISet

log = logging.getLogger(__name__)

# Layover/shadow mask bit meanings from SNAP's Terrain Correction operator.
LAYOVER_SHADOW_VALID = 0
LAYOVER_BIT = 1
SHADOW_BIT = 2


@dataclass
class SceneContext:
    """Immutable metadata carried onto every row extracted from one scene."""

    scene_id: str
    acquisition_date: str
    start_time: str
    platform: str
    orbit_direction: str
    relative_orbit: int | None
    absolute_orbit: int | None
    frame_number: int | None
    product_type: str
    looks: dict

    def as_row_fields(self) -> dict:
        out = {
            "scene_id": self.scene_id,
            "date": self.acquisition_date,
            "start_time": self.start_time,
            "platform": self.platform,
            "orbit_direction": self.orbit_direction,
            "relative_orbit": self.relative_orbit,
            "absolute_orbit": self.absolute_orbit,
            "frame_number": self.frame_number,
            "product_type": self.product_type,
        }
        out.update(
            {
                "looks_nominal": self.looks.get("looks_nominal"),
                "looks_range": self.looks.get("looks_range"),
                "looks_azimuth": self.looks.get("looks_azimuth"),
            }
        )
        return out


class SceneReader:
    """
    Holds one scene's bands open and serves windowed reads.

    Reading per-polygon windows from an open dataset is faster than loading the
    full scene into memory once the AOI set is sparse, and it removes the
    scene-sized allocation that competes with the JVM for the memory
    allocation. For dense AOI sets covering most of the scene, ``preload``
    reverts to a single full read.
    """

    def __init__(self, bands: dict, preload: bool = False):
        self.paths = bands
        self.preload = preload
        self._datasets: dict[str, rasterio.DatasetReader] = {}
        self._arrays: dict[str, np.ndarray] = {}
        self.transform = None
        self.crs = None
        self.shape = None
        self.nodata = None

    def __enter__(self) -> "SceneReader":
        for pol, path in self.paths["backscatter"].items():
            self._open(f"bs_{pol}", path)
        for role in ("inc_angle_projected", "layover_shadow", "elevation"):
            if self.paths.get(role):
                self._open(role, self.paths[role])

        if not self._datasets:
            raise RuntimeError("No readable bands for this scene")

        reference = next(iter(self._datasets.values()))
        self.transform = reference.transform
        self.crs = reference.crs
        self.shape = (reference.height, reference.width)
        self.nodata = reference.nodata

        self._check_alignment()
        if self.preload:
            for key, ds in self._datasets.items():
                self._arrays[key] = ds.read(1).astype(np.float32)
        return self

    def __exit__(self, *exc):
        for ds in self._datasets.values():
            ds.close()
        self._datasets.clear()
        self._arrays.clear()

    def _open(self, key: str, path) -> None:
        self._datasets[key] = rasterio.open(path)

    def _check_alignment(self) -> None:
        """
        Verify every band shares one grid.

        This is the assertion that makes the export_extra approach safe. If the
        incidence angle band were on a different grid than the backscatter, as
        it is when read from the SNAP tie point grids, this raises instead of
        producing silently misaligned values.
        """
        for key, ds in self._datasets.items():
            if (ds.height, ds.width) != self.shape:
                raise ValueError(
                    f"Band {key!r} has shape {(ds.height, ds.width)} but the "
                    f"reference band has {self.shape}. Bands must share one grid."
                )
            if not _transforms_close(ds.transform, self.transform):
                raise ValueError(
                    f"Band {key!r} has a different geotransform than the "
                    f"reference band. Bands must share one grid."
                )

    def read_window(self, key: str, window: Window) -> np.ndarray | None:
        if key not in self._datasets:
            return None
        if self.preload:
            r0 = int(window.row_off)
            c0 = int(window.col_off)
            return self._arrays[key][
                r0 : r0 + int(window.height), c0 : c0 + int(window.width)
            ]
        return self._datasets[key].read(1, window=window).astype(np.float32)

    @property
    def polarizations(self) -> list[str]:
        return [k[3:] for k in self._datasets if k.startswith("bs_")]


def _transforms_close(a, b, tol: float = 1e-6) -> bool:
    return all(abs(x - y) < tol for x, y in zip(a[:6], b[:6]))


def polygon_window(geometry, reader: SceneReader, pad: int = 1) -> Window | None:
    """
    Pixel window covering a polygon, clamped to the raster.

    Uses rasterio's own window arithmetic rather than manual row/column
    algebra. Hand-rolled versions of this calculation tend to be subtly wrong
    at the raster edge or with a negative y resolution, and the error is a
    one-pixel offset that is invisible in aggregate statistics but corrupts
    any per-pixel spatial analysis.
    """
    minx, miny, maxx, maxy = geometry.bounds
    win = from_bounds(minx, miny, maxx, maxy, transform=reader.transform)

    row_off = int(math.floor(win.row_off)) - pad
    col_off = int(math.floor(win.col_off)) - pad
    height = int(math.ceil(win.height)) + 2 * pad
    width = int(math.ceil(win.width)) + 2 * pad

    row_off_clamped = max(0, row_off)
    col_off_clamped = max(0, col_off)
    row_end = min(reader.shape[0], row_off + height)
    col_end = min(reader.shape[1], col_off + width)

    if row_end <= row_off_clamped or col_end <= col_off_clamped:
        return None

    return Window(
        col_off=col_off_clamped,
        row_off=row_off_clamped,
        width=col_end - col_off_clamped,
        height=row_end - row_off_clamped,
    )


def extract_scene(
    reader: SceneReader,
    aoi: AOISet,
    context: SceneContext,
    granularity: str = "polygon",
    min_valid_pixels: int = 4,
    enl_trim_percentile: float | None = 2.0,
    drop_layover_shadow: bool = True,
) -> pd.DataFrame:
    """
    Extract every AOI that intersects this scene.

    Returns an empty DataFrame rather than raising when no AOI yields valid
    pixels, since a scene clipping the edge of the AOI set legitimately
    produces nothing.
    """
    if granularity not in ("pixel", "polygon"):
        raise ValueError("granularity must be 'pixel' or 'polygon'")

    aoi_projected = aoi.gdf.to_crs(reader.crs)
    scene_bounds = box(*rasterio.transform.array_bounds(*reader.shape, reader.transform))
    candidates = aoi_projected[aoi_projected.intersects(scene_bounds)]

    log.info(
        "Scene %s: %d of %d AOIs intersect the footprint",
        context.scene_id,
        len(candidates),
        len(aoi_projected),
    )

    rows: list = []
    hit = 0
    for record in candidates.itertuples():
        result = _extract_one(
            reader=reader,
            geometry=record.geometry,
            aoi_id=record.aoi_id,
            context=context,
            granularity=granularity,
            min_valid_pixels=min_valid_pixels,
            enl_trim_percentile=enl_trim_percentile,
            drop_layover_shadow=drop_layover_shadow,
        )
        if result is not None:
            rows.append(result)
            hit += 1

    log.info("Scene %s: %d AOIs produced valid observations", context.scene_id, hit)
    if not rows:
        return pd.DataFrame()

    if granularity == "pixel":
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(rows)


def _extract_one(
    reader: SceneReader,
    geometry,
    aoi_id: str,
    context: SceneContext,
    granularity: str,
    min_valid_pixels: int,
    enl_trim_percentile: float | None,
    drop_layover_shadow: bool,
):
    window = polygon_window(geometry, reader)
    if window is None:
        return None

    win_transform = window_transform(window, reader.transform)
    out_shape = (int(window.height), int(window.width))

    inside = geometry_mask(
        [geometry],
        transform=win_transform,
        out_shape=out_shape,
        invert=True,
        all_touched=False,
    )
    if not inside.any():
        # Polygon smaller than a pixel or falling between pixel centres.
        inside = geometry_mask(
            [geometry],
            transform=win_transform,
            out_shape=out_shape,
            invert=True,
            all_touched=True,
        )
        if not inside.any():
            return None

    pol_arrays: dict[str, np.ndarray] = {}
    valid = inside.copy()
    for pol in reader.polarizations:
        arr = reader.read_window(f"bs_{pol}", window)
        if arr is None:
            continue
        pol_arrays[pol] = arr
        finite = np.isfinite(arr) & (arr > 0)
        if reader.nodata is not None and np.isfinite(reader.nodata):
            finite &= arr != reader.nodata
        # Every polarization must be valid at a pixel for it to be usable,
        # so that co-polar ratios are computed on a consistent sample.
        valid &= finite

    if not pol_arrays:
        return None

    inc = reader.read_window("inc_angle_projected", window)
    if inc is not None:
        valid &= np.isfinite(inc) & (inc > 0)

    lsm = reader.read_window("layover_shadow", window)
    if lsm is not None and drop_layover_shadow:
        valid &= lsm.astype(np.int32) == LAYOVER_SHADOW_VALID

    n_valid = int(valid.sum())
    if n_valid < min_valid_pixels:
        return None

    rows_idx, cols_idx = np.where(valid)
    if granularity == "pixel":
        return _pixel_rows(
            reader, window, rows_idx, cols_idx, pol_arrays, inc, lsm, aoi_id, context
        )
    return _polygon_row(
        reader,
        valid,
        pol_arrays,
        inc,
        lsm,
        inside,
        aoi_id,
        context,
        n_valid,
        enl_trim_percentile,
    )


def _pixel_rows(reader, window, rows_idx, cols_idx, pol_arrays, inc, lsm, aoi_id, context):
    abs_rows = rows_idx + int(window.row_off)
    abs_cols = cols_idx + int(window.col_off)
    xs, ys = rasterio.transform.xy(reader.transform, abs_rows, abs_cols)

    data = {
        "aoi_id": aoi_id,
        "x": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
    }
    for pol, arr in pol_arrays.items():
        linear = arr[rows_idx, cols_idx].astype(np.float32)
        data[f"{pol}_linear"] = linear
        data[f"{pol}_db"] = rad.to_db(linear).astype(np.float32)
    if inc is not None:
        data["inc_angle"] = inc[rows_idx, cols_idx].astype(np.float32)
    if lsm is not None:
        data["layover_shadow"] = lsm[rows_idx, cols_idx].astype(np.int8)

    df = pd.DataFrame(data)
    for key, value in context.as_row_fields().items():
        df[key] = value
    return df


def _polygon_row(
    reader,
    valid,
    pol_arrays,
    inc,
    lsm,
    inside,
    aoi_id,
    context,
    n_valid,
    enl_trim_percentile,
):
    row: dict = {"aoi_id": aoi_id, "n_pixels_valid": n_valid}
    row["n_pixels_in_polygon"] = int(inside.sum())
    row["valid_fraction"] = n_valid / max(1, int(inside.sum()))

    enl_values = []
    for pol, arr in pol_arrays.items():
        sample = arr[valid].astype(np.float64)
        mean_linear = float(sample.mean())
        enl = rad.estimate_enl(sample, trim_percentile=enl_trim_percentile)
        enl_values.append(enl)

        row[f"{pol}_mean_linear"] = mean_linear
        row[f"{pol}_mean_db"] = float(rad.to_db(mean_linear))
        row[f"{pol}_median_db"] = float(rad.to_db(np.median(sample)))
        row[f"{pol}_p10_db"] = float(rad.to_db(np.percentile(sample, 10)))
        row[f"{pol}_p90_db"] = float(rad.to_db(np.percentile(sample, 90)))
        row[f"{pol}_cv"] = rad.coefficient_of_variation(sample)
        row[f"{pol}_enl"] = enl
        row[f"{pol}_se_db"] = rad.speckle_standard_error_db(n_valid, enl)

    if "VV" in pol_arrays and "VH" in pol_arrays:
        vv = pol_arrays["VV"][valid].astype(np.float64)
        vh = pol_arrays["VH"][valid].astype(np.float64)
        row["VH_VV_ratio_db"] = float(rad.to_db(vh.mean() / vv.mean()))
        # Radar Vegetation Index for dual-pol data.
        row["rvi_dual"] = float(np.mean(4.0 * vh / (vv + vh)))

    if inc is not None:
        angles = inc[valid].astype(np.float64)
        row["inc_angle_mean"] = float(angles.mean())
        row["inc_angle_std"] = float(angles.std(ddof=1)) if angles.size > 1 else 0.0
        row["inc_angle_min"] = float(angles.min())
        row["inc_angle_max"] = float(angles.max())
    else:
        row["inc_angle_mean"] = float("nan")

    if lsm is not None:
        flagged = lsm[inside].astype(np.int32)
        row["layover_shadow_fraction"] = float(np.mean(flagged != LAYOVER_SHADOW_VALID))

    # The empirical ENL, averaged across polarizations, is the effective look
    # count actually achieved. Comparing it with looks_nominal quantifies how
    # much independence was lost to processing correlation and scene texture.
    finite_enl = [e for e in enl_values if np.isfinite(e)]
    row["enl_empirical"] = float(np.mean(finite_enl)) if finite_enl else float("nan")

    row.update(context.as_row_fields())
    if np.isfinite(row["enl_empirical"]) and row.get("looks_nominal"):
        row["enl_efficiency"] = row["enl_empirical"] / row["looks_nominal"]
    return row
