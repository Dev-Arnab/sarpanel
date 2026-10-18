#!/usr/bin/env python3
"""
Compare two runs of the same scenes, field by field and date by date.

Written to calibrate the processing subset buffer, and reusable afterwards as
the benchmark evidence for any claim that an optimisation left the science
unchanged.

A subset that speeds up Terrain-Flattening is only worth having if the
backscatter it produces matches the unsubsetted result. Terrain-Flattening
normalises by the illuminated area computed from surrounding terrain, so a
subset cut too close to the polygons biases the normalisation, and the bias
appears as a near-constant multiplicative offset on every polarisation at once.

    python tools/compare_runs.py \\
        /mnt/gs21/scratch/bassolab/sarpanel_smoke/observations_nosubset \\
        /mnt/gs21/scratch/bassolab/sarpanel_smoke/observations \\
        --labels "no subset" "5 km buffer"
"""

from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import pandas as pd

# A systematic offset above this is treated as a processing artefact rather
# than numerical noise. Chosen well below the ~1 dB seasonal signal a crop
# time series carries, and well above grid-resampling differences.
TOLERANCE_DB = 0.05

KEYS = ["aoi_id", "date"]
COMPARE = [
    "inc_angle_mean",
    "inc_angle_std",
    "n_pixels_valid",
    "enl_empirical",
    "VV_mean_db",
    "VH_mean_db",
    "VH_VV_ratio_db",
]


def load(directory: str) -> pd.DataFrame:
    files = sorted(glob.glob(f"{directory}/**/*.parquet", recursive=True))
    if not files:
        raise SystemExit(f"No parquet files under {directory}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(KEYS).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--labels", nargs=2, default=["baseline", "candidate"])
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_DB)
    args = ap.parse_args()

    a, b = load(args.baseline), load(args.candidate)
    lab_a, lab_b = args.labels

    merged = a.merge(b, on=KEYS, suffixes=("_a", "_b"))
    print(f"\n{lab_a}: {len(a)} rows   {lab_b}: {len(b)} rows   matched: {len(merged)}")
    if merged.empty:
        raise SystemExit("No rows matched on aoi_id and date")

    print(f"\n{'column':18} {'mean diff':>11} {'median':>9} {'max abs':>9} "
          f"{'sd':>8}  verdict")
    print("-" * 72)

    problems = []
    for col in COMPARE:
        ca, cb = f"{col}_a", f"{col}_b"
        if ca not in merged or cb not in merged:
            continue
        diff = (merged[cb] - merged[ca]).astype(float)
        diff = diff[np.isfinite(diff)]
        if diff.empty:
            continue

        is_db = col.endswith("_db")
        systematic = abs(diff.mean())
        if is_db:
            bad = systematic > args.tolerance
            verdict = "OFFSET" if bad else "ok"
            if bad:
                problems.append((col, diff.mean()))
        else:
            verdict = ""

        print(f"{col:18} {diff.mean():>11.4f} {diff.median():>9.4f} "
              f"{diff.abs().max():>9.4f} {diff.std():>8.4f}  {verdict}")

    # A shared offset across polarisations points at the radiometric
    # normalisation rather than at geometry or resampling.
    if {"VV_mean_db_a", "VH_mean_db_a"} <= set(merged.columns):
        dvv = (merged["VV_mean_db_b"] - merged["VV_mean_db_a"]).mean()
        dvh = (merged["VH_mean_db_b"] - merged["VH_mean_db_a"]).mean()
        dratio = abs(dvv - dvh)
        print(f"\nVV offset {dvv:+.4f} dB, VH offset {dvh:+.4f} dB, "
              f"difference {dratio:.4f} dB")
        if abs(dvv) > args.tolerance and dratio < args.tolerance:
            print(
                "  Both polarisations shifted by nearly the same amount, so the\n"
                "  VH/VV ratio is preserved. That is the signature of the\n"
                "  gamma0 normalisation changing, not of a geometry or masking\n"
                "  error. Terrain-Flattening divides by an illuminated area\n"
                "  derived from surrounding terrain, and a subset cut too close\n"
                "  to the polygons truncates that neighbourhood.\n"
                "  Remedy: raise subset_buffer_m and compare again."
            )
        elif abs(dvv) > args.tolerance:
            print(
                "  The polarisations shifted by different amounts, so the ratio\n"
                "  changed too. That points at masking or geometry rather than\n"
                "  at radiometric normalisation."
            )

    # Per-scene detail: a single bad scene and a uniform bias need different fixes.
    if "VV_mean_db_a" in merged:
        print("\nPer-date VV offset")
        per_date = (
            (merged["VV_mean_db_b"] - merged["VV_mean_db_a"])
            .groupby(merged["date"])
            .agg(["mean", "std", "count"])
        )
        for date, row in per_date.iterrows():
            sd = 0.0 if pd.isna(row["std"]) else row["std"]
            print(f"  {date}  {row['mean']:+.4f} dB  (sd {sd:.4f}, n {int(row['count'])})")
        spread = per_date["mean"].max() - per_date["mean"].min()
        print(f"  spread across dates: {spread:.4f} dB")
        if spread < args.tolerance:
            print("  A uniform offset across every date is a systematic bias, "
                  "not a per-scene failure.")

    print()
    if problems:
        print("VERDICT: the subset changed the radiometry. Do not scale up.")
        for col, value in problems:
            print(f"  {col} shifted by {value:+.4f} dB")
        print("\n  Raise subset_buffer_m (try 20000), rerun, and compare again.")
        print("  If the offset shrinks with a larger buffer, it is an edge")
        print("  effect in Terrain-Flattening and the buffer simply needs to be")
        print("  wide enough. If it does not shrink, subsetting interacts with")
        print("  the normalisation in a way that cannot be tuned away, and")
        print("  terrain_flattening and subset_to_aoi should not be combined.")
        return 1

    print("VERDICT: radiometry preserved within tolerance.")
    print(f"  All decibel columns agree to within {args.tolerance} dB on average.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
