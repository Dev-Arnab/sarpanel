#!/usr/bin/env python3
"""
Inspect an AOI file and preview what a run over it would involve.

Run this on a login node before submitting anything. It answers the questions
that determine whether the run will work at all: which identifier column to
use, whether the area filter will find its column, how complex the search
footprint is, and roughly how much data the date range implies.

    python tools/inspect_aoi.py \
        /mnt/research/bassolab/ysz_pipeline_rj/geopandas_data/combined_all_fields.geojson

Add --search to also hit the catalog and count scenes. That requires
credentials and takes a minute or two over a large AOI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd


def _bootstrap_sarpanel() -> None:
    """
    Make ``sarpanel`` importable whether or not it has been pip-installed.

    Tries the installed package first, then a few likely locations relative to
    this file. Running from a copied ``tools/`` directory without the rest of
    the repository is a common way to arrive here, so the failure message says
    exactly what to do rather than just reporting a missing module.
    """
    try:
        import sarpanel  # noqa: F401
        return
    except ImportError:
        pass

    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "src",        # repo/tools/inspect_aoi.py -> repo/src
        here.parent / "src",            # tools/ and src/ side by side
        here.parents[1],                # flat layout, package at repo root
        Path.cwd() / "src",
        Path.cwd(),
    ]
    for candidate in candidates:
        if (candidate / "sarpanel" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            try:
                import sarpanel  # noqa: F401
                return
            except ImportError:
                sys.path.pop(0)

    searched = "\n".join(f"    {c}" for c in candidates)
    raise SystemExit(
        "Could not import sarpanel. Looked in:\n"
        f"{searched}\n\n"
        "Fix with either:\n"
        "    cd /path/to/sarpanel && pip install -e .\n"
        "or run from the repository root:\n"
        "    PYTHONPATH=src python tools/inspect_aoi.py <geojson>\n\n"
        "The repository must contain src/sarpanel/ alongside tools/."
    )


def human(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:,.1f}{unit}" if unit else f"{n:,.0f}"
        n /= 1000
    return f"{n:,.1f}T"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--search", action="store_true", help="also query ASF for scene counts")
    ap.add_argument("--start-date", default="2016-01-01")
    ap.add_argument("--end-date", default="2023-12-01")
    ap.add_argument("--product-type", default="SLC", choices=["SLC", "GRD"])
    ap.add_argument("--min-area-m2", type=float, default=12140.0, help="default is 3 acres")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    print(f"Reading {path} ({path.stat().st_size / 1e6:,.1f} MB on disk)")
    gdf = gpd.read_file(path)

    print(f"\nFeatures      {len(gdf):,}")
    print(f"CRS           {gdf.crs}")
    print(f"Geometry      {dict(gdf.geometry.geom_type.value_counts())}")

    if gdf.crs is None:
        print("\n  PROBLEM: no CRS. sarpanel will refuse this rather than guess,")
        print("  because guessing EPSG:4326 on a projected file yields an empty search.")

    # ---- columns -----------------------------------------------------------
    attrs = [c for c in gdf.columns if c != "geometry"]
    print(f"\nColumns ({len(attrs)}):")
    for col in attrs:
        n_unique = gdf[col].nunique(dropna=True)
        sample = gdf[col].dropna().head(2).tolist()
        flag = ""
        if n_unique == len(gdf):
            flag = "  <- unique, usable as aoi_id"
        elif col.lower() in ("year", "season", "crop_year"):
            flag = "  <- multi-year file?"
        print(f"  {col:24} {str(gdf[col].dtype):10} {n_unique:>8,} unique  {sample}{flag}")

    # ---- the old script's assumptions --------------------------------------
    print("\nChecks against the old pipeline's assumptions:")
    import re

    year_match = re.search(r"\d{4}", path.name)
    if year_match:
        print(f"  filename year       found: {year_match.group(0)}")
    else:
        print("  filename year       ABSENT -> the old script raises RuntimeError here")

    if "acres" in gdf.columns:
        n_pass = int((gdf["acres"] >= 3.0).sum())
        print(f"  'acres' column      present, {n_pass:,} of {len(gdf):,} pass >= 3 acres")
    else:
        print("  'acres' column      ABSENT -> the old script raises KeyError here")

    # ---- area in a projected CRS -------------------------------------------
    metric = gdf.estimate_utm_crs()
    area_m2 = gdf.to_crs(metric).area
    print(f"\nArea (computed in {metric.to_string()}):")
    for q in (0.05, 0.25, 0.50, 0.75, 0.95):
        print(f"  p{int(q * 100):02d}  {area_m2.quantile(q):>12,.0f} m2  "
              f"({area_m2.quantile(q) / 4046.86:>8,.1f} acres)")
    n_keep = int((area_m2 >= args.min_area_m2).sum())
    print(f"  >= {args.min_area_m2:,.0f} m2: {n_keep:,} of {len(gdf):,} fields retained")

    # Pixels per field at 10 m drives how reliable the ENL estimate will be.
    px = area_m2 / 100.0
    print("\nPixels per field at 10 m spacing:")
    print(f"  median {px.median():,.0f}   p05 {px.quantile(0.05):,.0f}   "
          f"p95 {px.quantile(0.95):,.0f}")
    thin = int((px < 16).sum())
    if thin:
        print(f"  {thin:,} fields have fewer than 16 pixels; ENL is unreliable below that")

    # ---- search footprint complexity ---------------------------------------
    _bootstrap_sarpanel()
    from sarpanel.aoi import load_aoi, union_all

    id_col = next(
        (c for c in ("aoi_id", "field_id", "farm_name", "id", "name") if c in gdf.columns),
        None,
    )
    aoi = load_aoi(gdf, id_column=id_col, min_area_m2=args.min_area_m2)

    footprint = aoi.search_footprint(buffer_m=10_000)
    minx, miny, maxx, maxy = footprint.bounds
    print("\nSearch footprint (10 km buffer, dissolved):")
    print(f"  bbox        {minx:.3f},{miny:.3f},{maxx:.3f},{maxy:.3f}")
    print(f"  WKT length  {len(footprint.wkt):,} chars")
    print(f"  AOI hash    {aoi.checksum()}")

    raw_union = union_all(gdf.to_crs(metric).buffer(10_000))
    raw_wkt_len = len(
        gpd.GeoSeries([raw_union], crs=metric).to_crs("EPSG:4326").iloc[0].wkt
    )
    print(f"  un-simplified union WKT would be {raw_wkt_len:,} chars", end="")
    print("  <- likely rejected by CMR" if raw_wkt_len > 100_000 else "")

    # ---- optional catalog query -------------------------------------------
    if args.search:
        from sarpanel.manifest import attach_aoi_membership, search_scenes

        print(f"\nQuerying ASF for {args.product_type} between "
              f"{args.start_date} and {args.end_date} ...")
        manifest = search_scenes(
            aoi=aoi,
            start_date=args.start_date,
            end_date=args.end_date,
            product_type=args.product_type,
        )
        if manifest.empty:
            print("  no scenes returned")
            return 0
        manifest = attach_aoi_membership(manifest, aoi)
        gb = manifest["size_mb"].fillna(0).sum() / 1024

        print(f"\n  scenes        {len(manifest):,}")
        print(f"  download      {gb:,.1f} GB")
        print(f"  tracks        {dict(manifest.groupby('relative_orbit').size())}")
        print(f"  directions    {dict(manifest.groupby('orbit_direction').size())}")
        per_year = manifest["acquisition_date"].str[:4].value_counts().sort_index()
        print(f"  per year      {dict(per_year)}")

        obs = len(manifest) * manifest["n_aoi_hit"].mean()
        print(f"\n  approx polygon-level observations: {human(obs)}")
        print(f"  approx pixel-level rows:           "
              f"{human(obs * px[px >= args.min_area_m2 / 100].median())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
