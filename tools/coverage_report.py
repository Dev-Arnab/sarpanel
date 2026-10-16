#!/usr/bin/env python3
"""
Report per-field observation coverage from a scene manifest.

Run after 'plan'. It reads the manifest and the resolved AOI and reports how
many acquisitions each field actually receives, by year and by track.

This exists because coverage is not uniform across an AOI that spans several
frames, and the non-uniformity is easy to miss. A field near a frame edge can
receive a fraction of what a field in the centre gets, and after the loss of
Sentinel-1B in December 2021 some fields fall out of coverage entirely for
months at a time.

For a panel model this is not a data-quality nuisance to be cleaned up. It is
missingness with a known, non-random structure: it depends on position relative
to the frame and on the acquisition era. Reporting it before modelling is
cheaper than discovering it as an unexplained gap afterwards.

    python tools/coverage_report.py configs/michigan_all_fields.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def bootstrap() -> None:
    try:
        import sarpanel  # noqa: F401
        return
    except ImportError:
        pass
    for candidate in (Path(__file__).resolve().parents[1] / "src", Path.cwd() / "src"):
        if (candidate / "sarpanel" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return
    raise SystemExit("Could not import sarpanel; run from the repo root or set PYTHONPATH=src")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--csv", default=None, help="write the per-field table here")
    args = ap.parse_args()

    bootstrap()
    from sarpanel.config import RunConfig
    from sarpanel.manifest import shape_from_wkt

    config = RunConfig.from_yaml(args.config)
    if not config.manifest_path.exists():
        print(f"No manifest at {config.manifest_path}")
        print(f"Run first:  python -m sarpanel.cli plan {args.config}")
        return 1

    manifest = pd.read_parquet(config.manifest_path)
    aoi = gpd.read_parquet(config.aoi_path)
    print(f"\nManifest: {len(manifest):,} scenes,  AOI: {len(aoi):,} fields")

    # Rebuild the field-by-scene incidence from the stored footprints.
    scenes = gpd.GeoDataFrame(
        manifest[["scene_id", "acquisition_date", "relative_orbit"]].copy(),
        geometry=[shape_from_wkt(w) for w in manifest["footprint_wkt"]],
        crs="EPSG:4326",
    )
    hits = gpd.sjoin(
        aoi[["aoi_id", "geometry"]], scenes, how="inner", predicate="intersects"
    )
    hits["year"] = hits["acquisition_date"].str[:4]

    per_field = hits.groupby("aoi_id").size()
    uncovered = set(aoi["aoi_id"]) - set(per_field.index)

    print("\nObservations per field, whole archive")
    for q in (0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00):
        print(f"  p{int(q * 100):03d}  {per_field.quantile(q):>7,.0f}")
    if uncovered:
        print(f"\n  {len(uncovered)} field(s) receive ZERO observations:")
        for name in sorted(uncovered)[:10]:
            print(f"    {name}")

    ratio = per_field.max() / max(1, per_field.min())
    print(f"\n  best-covered field sees {ratio:.1f}x what the worst-covered field sees")
    if ratio > 2:
        print("  Coverage is strongly non-uniform. Field-level observation count "
              "belongs in the model as a covariate or a weight, not as an "
              "afterthought.")

    # ---- by year -----------------------------------------------------------
    by_year = hits.groupby(["year", "aoi_id"]).size().rename("n").reset_index()
    summary = by_year.groupby("year")["n"].agg(["median", "min", "max"])
    summary["fields_with_zero"] = [
        len(aoi) - by_year[by_year["year"] == y]["aoi_id"].nunique()
        for y in summary.index
    ]

    print("\nPer-field observations by year")
    print(f"  {'year':6} {'median':>7} {'min':>6} {'max':>6}  {'fields with none':>17}")
    for year, row in summary.iterrows():
        flag = "  <-- S1A only" if year >= "2022" else ""
        print(f"  {year:6} {row['median']:>7,.0f} {row['min']:>6,.0f} "
              f"{row['max']:>6,.0f}  {int(row['fields_with_zero']):>17,}{flag}")

    # ---- by track ----------------------------------------------------------
    if hits["relative_orbit"].nunique() > 1:
        print("\nFields covered by each track")
        for track, group in hits.groupby("relative_orbit"):
            n_fields = group["aoi_id"].nunique()
            print(f"  track {int(track):>3}   {n_fields:>4} of {len(aoi):,} fields   "
                  f"{len(group):,} field-observations")
        both = (
            hits.groupby("aoi_id")["relative_orbit"].nunique().eq(2).sum()
        )
        print(f"  {both:,} field(s) seen by both tracks")
        print("  Only these have within-field incidence angle contrast between "
              "tracks; fields on a single track see one viewing geometry.")

    # ---- longest gap -------------------------------------------------------
    print("\nLongest gap between consecutive observations, per field")
    hits_sorted = hits.sort_values(["aoi_id", "acquisition_date"])
    gaps = []
    for aoi_id, group in hits_sorted.groupby("aoi_id"):
        dates = pd.to_datetime(group["acquisition_date"].unique())
        if len(dates) < 2:
            continue
        gaps.append((aoi_id, int(np.diff(np.sort(dates)).max() / np.timedelta64(1, "D"))))
    if gaps:
        gap_series = pd.Series(dict(gaps))
        print(f"  median {gap_series.median():.0f} days   "
              f"p95 {gap_series.quantile(0.95):.0f} days   "
              f"max {gap_series.max():.0f} days")
        worst = gap_series.idxmax()
        print(f"  worst: {worst} with a {gap_series.max():.0f} day gap")

    if args.csv:
        table = by_year.pivot(index="aoi_id", columns="year", values="n").fillna(0)
        table["total"] = table.sum(axis=1)
        table.to_csv(args.csv)
        print(f"\nPer-field table written to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
