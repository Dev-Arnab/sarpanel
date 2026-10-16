#!/usr/bin/env python3
"""
Isolate why a catalog query returns nothing.

Bisects the search parameters one at a time against a known-good AOI and date
range, so the offending term is identified rather than guessed at. Run it when
'plan' reports zero scenes for a search you expect to succeed.

    python tools/debug_search.py

Every Sentinel-1 acquisition yields both an SLC and a GRD product, so an SLC
result with no matching GRD result over the same footprint and dates is a query
problem, never an archive gap.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# asf_search logs the full query at INFO, which buries the result.
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
START, END = "2022-06-01", "2022-07-31"


def count(label: str, **kwargs) -> int:
    try:
        n = len(list(asf.search(**kwargs)))
    except Exception as exc:
        print(f"  {label:44} ERROR  {type(exc).__name__}: {exc}")
        return -1
    flag = "" if n else "   <-- empty"
    print(f"  {label:44} {n:>5}{flag}")
    return n


def main() -> int:
    print("\nWhat this asf_search build actually defines")
    print(f"  asf_search version: {getattr(asf, '__version__', 'unknown')}")
    product_types = sorted(
        a for a in dir(asf.PRODUCT_TYPE) if not a.startswith("_")
    )
    print(f"  PRODUCT_TYPE attributes ({len(product_types)}):")
    for name in product_types:
        value = getattr(asf.PRODUCT_TYPE, name)
        if isinstance(value, str):
            print(f"    {name:26} = {value!r}")
    for name in ("GRD", "GRD_HD", "GRD_MD", "SLC"):
        exists = hasattr(asf.PRODUCT_TYPE, name)
        print(f"  PRODUCT_TYPE.{name:8} {'defined' if exists else 'NOT DEFINED'}")

    platforms = [a for a in ("SENTINEL1A", "SENTINEL1B", "SENTINEL1C")
                 if hasattr(asf.PLATFORM, a)]
    print(f"  PLATFORM defined: {platforms}")

    aoi = load_aoi(FIELDS, id_column="farm_name", min_area_m2=12140,
                   where="farm_name == '520_Pitsilo'")
    wkt = aoi.search_footprint(5000).wkt

    base = dict(
        intersectsWith=wkt,
        start=START,
        end=END,
        maxResults=None,
    )
    s1 = [getattr(asf.PLATFORM, p) for p in platforms]

    print(f"\nOne field, {START} to {END}")

    print("\n  Baseline (what should definitely work)")
    count("SLC, IW, VV+VH", **base, platform=s1, beamMode=["IW"],
          polarization=["VV+VH"], processingLevel=[asf.PRODUCT_TYPE.SLC])

    print("\n  Progressively loosening the GRD query")
    count("nothing but the footprint and dates", **base)
    count("platform only", **base, platform=s1)
    count("platform + IW", **base, platform=s1, beamMode=["IW"])

    grd_hd = getattr(asf.PRODUCT_TYPE, "GRD_HD", "GRD_HD")
    count("+ processingLevel GRD_HD", **base, platform=s1,
          beamMode=["IW"], processingLevel=[grd_hd])
    count("+ processingLevel GRD_HD + VV+VH", **base, platform=s1,
          beamMode=["IW"], processingLevel=[grd_hd], polarization=["VV+VH"])

    print("\n  The exact list sarpanel currently sends")
    current = []
    for name in ("GRD_HD", "GRD_MD", "GRD"):
        value = getattr(asf.PRODUCT_TYPE, name, None)
        current.append(value if isinstance(value, str) else name)
    print(f"    processingLevel = {current}")
    count("sarpanel's list, no polarization", **base, platform=s1,
          beamMode=["IW"], processingLevel=current)
    count("sarpanel's list + VV+VH", **base, platform=s1,
          beamMode=["IW"], processingLevel=current, polarization=["VV+VH"])

    print("\n  Each GRD product type on its own")
    for name in [p for p in product_types if p.startswith("GRD")]:
        value = getattr(asf.PRODUCT_TYPE, name)
        if isinstance(value, str):
            count(f"processingLevel={value}", **base, platform=s1,
                  beamMode=["IW"], processingLevel=[value])

    print("\n  What polarizations exist for GRD here")
    results = list(asf.search(**base, platform=s1, beamMode=["IW"],
                              processingLevel=[grd_hd]))
    if results:
        seen = {}
        for r in results:
            p = r.properties
            key = (p.get("processingLevel"), p.get("polarization"))
            seen[key] = seen.get(key, 0) + 1
        for (level, pol), n in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"    processingLevel={level!r:12} polarization={pol!r:12} {n:>4}")
    else:
        print("    no GRD_HD results to inspect")

    return 0


if __name__ == "__main__":
    sys.exit(main())
