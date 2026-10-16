#!/usr/bin/env python3
"""
Verify a completed run, with emphasis on the acquisition geometry.

Run this after the smoke test and again after stage 2. It checks the things
that distinguish a working pipeline from one that produced plausible-looking
numbers, and it is deliberately opinionated about what "working" means.

    python tools/check_output.py configs/smoke_test.yaml

The single most important check is whether the incidence angle varies across
relative orbits. A constant angle, or one that is identical between tracks,
means the value is not genuinely per-observation and the export_extra band is
not being read. That is exactly the failure the old tie-point-grid approach
produced: numbers in a plausible range that were not spatially correct.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


PASS, FAIL, WARN = "  PASS ", "  FAIL ", "  warn "


def check(label: str, condition: bool, detail: str = "", warn_only: bool = False) -> bool:
    marker = PASS if condition else (WARN if warn_only else FAIL)
    print(f"{marker} {label}{('  ' + detail) if detail else ''}")
    return condition or warn_only


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--parquet", default=None, help="check one file instead of the whole run")
    args = ap.parse_args()

    bootstrap()
    from sarpanel.config import RunConfig
    from sarpanel.provenance import read_provenance

    config = RunConfig.from_yaml(args.config)

    if args.parquet:
        files = [Path(args.parquet)]
    else:
        files = sorted(config.observations_dir.rglob("*.parquet"))
    if not files:
        print(f"No parquet files under {config.observations_dir}")
        print("Has the run finished? Try: sarpanel status " + args.config)
        return 1

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"\nRun: {config.name}")
    print(f"  files          {len(files):,}")
    print(f"  observations   {len(df):,}")
    print(f"  fields         {df['aoi_id'].nunique():,}")
    print(f"  dates          {df['date'].nunique():,}  "
          f"({df['date'].min()} to {df['date'].max()})")

    ok = True

    # ---- incidence angle ---------------------------------------------------
    print("\nIncidence angle")
    if "inc_angle_mean" not in df.columns:
        print(f"{FAIL} column absent entirely")
        return 1

    inc = df["inc_angle_mean"]
    n_null = int(inc.isna().sum())
    ok &= check("column populated", n_null == 0, f"{n_null:,} null of {len(df):,}")
    ok &= check(
        "values in the Sentinel-1 IW range",
        bool(inc.min() > 25 and inc.max() < 50),
        f"observed {inc.min():.1f} to {inc.max():.1f} degrees",
    )

    # The decisive test. Different tracks view the same ground at different
    # angles, so a per-observation angle must differ between them.
    if "relative_orbit" in df.columns and df["relative_orbit"].nunique() > 1:
        by_track = df.groupby("relative_orbit")["inc_angle_mean"].agg(["mean", "std", "count"])
        print("\n  by track:")
        for track, row in by_track.iterrows():
            print(f"    track {int(track):>3}   mean {row['mean']:5.2f}   "
                  f"sd {row['std']:5.2f}   n {int(row['count']):,}")
        spread = by_track["mean"].max() - by_track["mean"].min()
        ok &= check(
            "angle differs between tracks",
            spread > 0.5,
            f"spread {spread:.2f} degrees",
        )
    else:
        print(f"{WARN} only one track present; cannot test between-track variation")

    # Within a single field the angle should be near-constant on a given date
    # but differ across dates from different tracks.
    per_field = df.groupby("aoi_id")["inc_angle_mean"].agg(["min", "max", "nunique"])
    varying = int((per_field["max"] - per_field["min"] > 0.5).sum())
    ok &= check(
        "angle varies within a field across dates",
        varying > 0,
        f"{varying:,} of {len(per_field):,} fields",
        warn_only=df.get("relative_orbit", pd.Series()).nunique() <= 1,
    )

    if "inc_angle_std" in df.columns:
        across = df["inc_angle_std"].median()
        ok &= check(
            "within-field angle spread is small",
            bool(across < 2.0),
            f"median sd {across:.3f} degrees across the polygon",
        )

    # ---- looks and ENL -----------------------------------------------------
    print("\nLooks and speckle")
    if "enl_empirical" in df.columns:
        enl = df["enl_empirical"].dropna()
        ok &= check("ENL estimated", len(enl) > 0, f"{len(enl):,} of {len(df):,} rows")
        if len(enl):
            ok &= check(
                "ENL is physically plausible",
                bool(enl.quantile(0.05) > 0.5 and enl.quantile(0.95) < 200),
                f"p05 {enl.quantile(0.05):.2f}  median {enl.median():.2f}  "
                f"p95 {enl.quantile(0.95):.2f}",
            )
        if "looks_nominal" in df.columns and "enl_efficiency" in df.columns:
            eff = df["enl_efficiency"].dropna()
            if len(eff):
                print(f"    nominal looks {df['looks_nominal'].iloc[0]:.2f}   "
                      f"efficiency median {eff.median():.2f}")
                ok &= check(
                    "empirical ENL below nominal, as expected",
                    bool(eff.median() < 1.5),
                    "oversampling and texture always reduce independence",
                    warn_only=True,
                )

    if "n_pixels_valid" in df.columns:
        n = df["n_pixels_valid"]
        ok &= check(
            "pixel counts are sane",
            bool(n.min() >= config.output.min_valid_pixels),
            f"min {n.min():,}  median {n.median():,.0f}  max {n.max():,}",
        )

    # ---- backscatter -------------------------------------------------------
    print("\nBackscatter")
    for pol in ("VV", "VH"):
        col = f"{pol}_mean_db"
        if col not in df.columns:
            continue
        v = df[col].dropna()
        lo, hi = (-25, 0) if pol == "VV" else (-32, -5)
        ok &= check(
            f"{pol} in a plausible range",
            bool(v.quantile(0.01) > lo and v.quantile(0.99) < hi),
            f"p01 {v.quantile(0.01):.1f}  median {v.median():.1f}  p99 {v.quantile(0.99):.1f} dB",
        )
        se = df.get(f"{pol}_se_db")
        if se is not None:
            print(f"    {pol} standard error: median {se.median():.3f} dB  "
                  f"range {se.min():.3f} to {se.max():.3f}")

    if {"VV_mean_db", "VH_mean_db"} <= set(df.columns):
        ok &= check(
            "VH is below VV",
            bool((df["VH_mean_db"] < df["VV_mean_db"]).mean() > 0.95),
            "cross-pol returns less power than co-pol for vegetation",
        )

    # ---- provenance --------------------------------------------------------
    print("\nProvenance")
    prov = read_provenance(files[0])
    for key in ("config_checksum", "aoi_checksum", "sarpanel_run_name"):
        ok &= check(f"{key} recorded", key in prov, prov.get(key, ""))
    import json
    software = json.loads(prov.get("software", "{}"))
    snap = software.get("snap", "unknown")
    ok &= check("SNAP version recorded", snap != "unknown", snap,
                warn_only=True)
    print(f"    pyroSAR {software.get('pyroSAR')}   rasterio {software.get('rasterio')}")

    # ---- summary -----------------------------------------------------------
    print("\n" + ("All checks passed." if ok else
                  "One or more checks FAILED. Do not scale up until resolved."))
    if not ok:
        print("If the incidence angle checks failed, confirm 'export_extra: True' "
              "in 'sarpanel doctor' and look for projectedLocalIncidenceAngle "
              "among the GeoTIFFs SNAP wrote.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
