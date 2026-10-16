#!/usr/bin/env python3
"""
Inspect what the catalog actually returns, rather than guessing at filters.

The first bisect showed that adding a platform filter takes the result count
from several hundred to zero. That means either the platform values being sent
do not match what the archive holds for these granules, or the unfiltered
results are not the products we assume they are.

Both questions are answered by reading the properties of the unfiltered
results, which is what this does.

    python tools/debug_search2.py
"""

from __future__ import annotations

import collections
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
for noisy in ("asf_search", "root", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

for candidate in (Path(__file__).resolve().parents[1] / "src", Path.cwd() / "src"):
    if (candidate / "sarpanel" / "__init__.py").exists():
        sys.path.insert(0, str(candidate))
        break

import asf_search as asf  # noqa: E402

from sarpanel.aoi import load_aoi  # noqa: E402

FIELDS = "/mnt/research/bassolab/ysz_pipeline_rj/geopandas_data/combined_all_fields.geojson"


def main() -> int:
    print("Platform constants this asf_search defines")
    for name in ("SENTINEL1", "SENTINEL1A", "SENTINEL1B", "SENTINEL1C"):
        value = getattr(asf.PLATFORM, name, None)
        print(f"  PLATFORM.{name:12} = {value!r}")

    aoi_one = load_aoi(FIELDS, id_column="farm_name", min_area_m2=12140,
                       where="farm_name == '520_Pitsilo'")
    aoi_all = load_aoi(FIELDS, id_column="farm_name", min_area_m2=12140)

    wkt_one = aoi_one.search_footprint(5000).wkt
    wkt_all = aoi_all.search_footprint(10000).wkt

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Unfiltered results over ONE field, 2022-06-01 to 2022-07-31")
    print("=" * 70)
    results = list(asf.search(intersectsWith=wkt_one, start="2022-06-01",
                              end="2022-07-31", maxResults=None))
    print(f"  total granules: {len(results)}")
    summarise(results)

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Same footprint and dates, platform filter added")
    print("=" * 70)
    for name in ("SENTINEL1", "SENTINEL1A", "SENTINEL1B"):
        value = getattr(asf.PLATFORM, name, None)
        if value is None:
            continue
        n = len(list(asf.search(intersectsWith=wkt_one, start="2022-06-01",
                                end="2022-07-31", platform=[value],
                                maxResults=None)))
        print(f"  platform=[{value!r}]  ->  {n}")

    # Literal strings, bypassing the constants entirely.
    for literal in ("Sentinel-1A", "SENTINEL-1A", "S1A", "Sentinel-1"):
        try:
            n = len(list(asf.search(intersectsWith=wkt_one, start="2022-06-01",
                                    end="2022-07-31", platform=[literal],
                                    maxResults=None)))
            print(f"  platform=[{literal!r}]  ->  {n}")
        except Exception as exc:
            print(f"  platform=[{literal!r}]  ->  ERROR {type(exc).__name__}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Does the ONE field differ from the FULL AOI? (same dates)")
    print("=" * 70)
    s1a = getattr(asf.PLATFORM, "SENTINEL1A")
    s1b = getattr(asf.PLATFORM, "SENTINEL1B", None)
    plats = [p for p in (s1a, s1b) if p]
    for label, wkt in (("one field ", wkt_one), ("full AOI  ", wkt_all)):
        for level in ("SLC", "GRD_HD"):
            n = len(list(asf.search(
                intersectsWith=wkt, start="2022-06-01", end="2022-07-31",
                platform=plats, beamMode=["IW"],
                processingLevel=[getattr(asf.PRODUCT_TYPE, level)],
                maxResults=None)))
            print(f"  {label} {level:7}  ->  {n}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Known-good control: full AOI, 2020, SLC (this returned 118 before)")
    print("=" * 70)
    n = len(list(asf.search(
        intersectsWith=wkt_all, start="2020-01-01", end="2020-12-31",
        platform=plats, beamMode=["IW"],
        processingLevel=[asf.PRODUCT_TYPE.SLC],
        polarization=["VV+VH"], flightDirection="ASCENDING",
        maxResults=None)))
    print(f"  full AOI 2020 SLC ASC VV+VH  ->  {n}")

    n = len(list(asf.search(
        intersectsWith=wkt_all, start="2020-06-01", end="2020-07-31",
        platform=plats, beamMode=["IW"],
        processingLevel=[asf.PRODUCT_TYPE.GRD_HD],
        maxResults=None)))
    print(f"  full AOI 2020 Jun-Jul GRD_HD ->  {n}")

    n = len(list(asf.search(
        intersectsWith=wkt_one, start="2020-06-01", end="2020-07-31",
        platform=plats, beamMode=["IW"],
        processingLevel=[asf.PRODUCT_TYPE.GRD_HD],
        maxResults=None)))
    print(f"  one field 2020 Jun-Jul GRD_HD->  {n}")

    return 0


def summarise(results) -> None:
    if not results:
        print("  (nothing to summarise)")
        return
    fields = ("platform", "processingLevel", "beamModeType", "polarization",
              "flightDirection")
    counters = {f: collections.Counter() for f in fields}
    for r in results:
        p = r.properties
        for f in fields:
            counters[f][p.get(f)] += 1

    for f in fields:
        print(f"\n  {f}:")
        for value, n in counters[f].most_common(12):
            print(f"    {str(value):32} {n:>5}")

    print("\n  first 3 granule names:")
    for r in results[:3]:
        p = r.properties
        print(f"    {p.get('sceneName')}")
        print(f"      platform={p.get('platform')!r}  "
              f"level={p.get('processingLevel')!r}  "
              f"start={p.get('startTime')}")


if __name__ == "__main__":
    sys.exit(main())
