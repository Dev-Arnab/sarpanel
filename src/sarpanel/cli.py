"""
Command line interface.

Five verbs:

``plan``       query the catalog once, write the scene manifest
``run``        process the scenes assigned to one shard
``doctor``     report the environment and what the installed stack can do
``status``     summarise a run in progress or finished
``aggregate``  combine per-scene outputs into one analysis table

The plan/run split exists so that catalog access happens once on a login node
and the compute nodes only read a file. It also makes the set of scenes a run
will process inspectable before any allocation is spent.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    logging.getLogger("fiona").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # asf_search echoes the entire query, including the full AOI WKT, at INFO,
    # which buries the result line. Its warnings are still shown.
    logging.getLogger("asf_search").setLevel(
        logging.DEBUG if verbose else logging.WARNING
    )


def cmd_plan(args) -> int:
    from .aoi import load_aoi, write_aoi
    from .config import RunConfig
    from .manifest import (
        assign_shards,
        attach_aoi_membership,
        search_scenes,
        write_manifest,
    )

    config = RunConfig.from_yaml(args.config)
    if args.n_shards:
        config.n_shards = args.n_shards

    aoi = load_aoi(
        config.aoi.source,
        id_column=config.aoi.id_column,
        min_area_m2=config.aoi.min_area_m2,
        layer=config.aoi.layer,
        where=config.aoi.where,
    )
    write_aoi(aoi, config.aoi_path)

    manifest = search_scenes(
        aoi=aoi,
        start_date=config.search.start_date,
        end_date=config.search.end_date,
        product_type=config.processing.product_type,
        beam_mode=config.search.beam_mode,
        polarization=config.search.polarization,
        relative_orbits=config.search.relative_orbits,
        orbit_direction=config.search.orbit_direction,
        search_buffer_m=config.aoi.search_buffer_m,
    )
    if manifest.empty:
        print("No scenes matched the search. Check the dates, AOI, and product type.")
        return 1

    manifest = attach_aoi_membership(manifest, aoi)
    manifest = assign_shards(manifest, config.n_shards)
    write_manifest(manifest, config.manifest_path)
    config.write(config.root / "manifest" / f"{config.name}_config.yaml")

    total_gb = manifest["size_mb"].fillna(0).sum() / 1024
    by_track = manifest.groupby("relative_orbit").size().sort_values(ascending=False)

    print()
    print(f"Run:            {config.name}")
    print(f"AOI polygons:   {len(aoi):,}")
    print(f"Scenes:         {len(manifest):,}")
    start = manifest["acquisition_date"].min()
    end = manifest["acquisition_date"].max()
    per_shard = len(manifest) / config.n_shards
    print(f"Date range:     {start} to {end}")
    print(f"Tracks:         {dict(by_track.head(8))}")
    print(f"Download:       {total_gb:,.1f} GB")
    print(f"Shards:         {config.n_shards} ({per_shard:.1f} scenes each)")
    print(f"Manifest:       {config.manifest_path}")
    print()
    print(
        "Submit with:    "
        f"sbatch --array=0-{config.n_shards - 1} "
        f"slurm/run_shard.sb {args.config}"
    )
    return 0


def cmd_run(args) -> int:
    from .config import RunConfig
    from .runner import run_shard

    config = RunConfig.from_yaml(args.config)
    shard = args.shard
    if shard is None:
        import os

        shard = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    if not config.manifest_path.exists():
        print(f"No manifest at {config.manifest_path}. Run 'sarpanel plan' first.")
        return 1

    status = run_shard(config, shard)
    if status.empty:
        return 0
    failed = int((status["status"] == "failed").sum())
    return 1 if failed and args.strict else 0


def cmd_doctor(args) -> int:
    """
    Report what the installed environment can actually do.

    The single most common cause of a wrong SAR product is a silently
    unsupported processing option. This surfaces those before a run rather
    than after.
    """
    from .provenance import software_versions

    print("Software versions")
    for key, value in software_versions().items():
        print(f"  {key:16} {value}")

    print("\nEarthdata credentials")
    try:
        from .auth import resolve_credentials

        creds = resolve_credentials()
        print(f"  found, from {creds.source}")
    except Exception as exc:
        print(f"  NOT AVAILABLE: {exc}")
        creds = None

    if creds is not None:
        # Locating credentials is not the same as their working. The catalog
        # query is unauthenticated, so a successful 'plan' proves nothing; the
        # first real test would otherwise be at download, after an allocation
        # has already been spent.
        try:
            from .auth import build_session

            build_session(creds)
            print("  VERIFIED against urs.earthdata.nasa.gov")
        except Exception as exc:
            print("  REJECTED by urs.earthdata.nasa.gov\n")
            for line in str(exc).splitlines():
                print(f"    {line}")

    print("\npyroSAR geocode capabilities")
    try:
        from .processing import GeocodeCapabilities

        caps = GeocodeCapabilities()
        report = caps.report()
        for key, value in report.items():
            print(f"  {key:20} {value}")
        if not report["export_extra"]:
            print(
                "\n  WARNING: this pyroSAR does not support 'export_extra'.\n"
                "  The projected local incidence angle cannot be written as a\n"
                "  terrain-corrected band, and there is no correct fallback.\n"
                "  Upgrade pyroSAR before running."
            )
        if not report["terrain_flattening"]:
            print("\n  WARNING: no terrainFlattening support; output will be sigma0.")
    except Exception as exc:
        print(f"  pyroSAR not importable: {exc}")

    print("\nSNAP")
    import os

    snap_home = os.environ.get("SNAP_HOME")
    print(f"  SNAP_HOME  {snap_home or 'not set'}")
    if snap_home:
        gpt = Path(snap_home) / "bin" / "gpt"
        print(f"  gpt        {'found' if gpt.exists() else 'MISSING at ' + str(gpt)}")
    return 0


def cmd_status(args) -> int:
    import pandas as pd

    from .config import RunConfig

    config = RunConfig.from_yaml(args.config)
    status_dir = config.root / "status"
    files = sorted(status_dir.glob("shard_*.parquet"))
    if not files:
        print(f"No status files under {status_dir}")
        return 1

    status = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    counts = status["status"].value_counts()

    print(f"\nRun {config.name}: {len(status):,} scenes across {len(files)} shard(s)")
    for key, value in counts.items():
        print(f"  {key:10} {value:>8,}")
    print(f"  observations {int(status['n_observations'].sum()):>6,}")

    ok = status[status["status"] == "ok"]
    if len(ok):
        print(f"  median scene time  {ok['seconds'].median() / 60:.1f} min")

    failures = status[status["status"] == "failed"]
    if len(failures):
        print(f"\nFailures ({len(failures)}):")
        for error, group in failures.groupby(failures["error"].str.slice(0, 80)):
            print(f"  {len(group):>4}x  {error}")
    return 0


def cmd_aggregate(args) -> int:
    """Combine per-scene parquets into one table, preserving provenance."""
    import pandas as pd

    from .config import RunConfig
    from .provenance import read_provenance, write_parquet_with_provenance

    config = RunConfig.from_yaml(args.config)
    files = sorted(config.observations_dir.rglob("*.parquet"))
    if not files:
        print(f"No observations under {config.observations_dir}")
        return 1

    frames = [pd.read_parquet(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)

    sort_keys = [k for k in ("aoi_id", "date", "relative_orbit") if k in combined.columns]
    combined = combined.sort_values(sort_keys).reset_index(drop=True)

    provenance = read_provenance(files[0])
    provenance["aggregated_from_n_scenes"] = str(len(files))

    out = Path(args.output or config.root / f"{config.name}_observations.parquet")
    write_parquet_with_provenance(combined, out, provenance, config.output.compression)

    print(f"Combined {len(files):,} scene files")
    print(f"  observations   {len(combined):,}")
    print(f"  AOIs           {combined['aoi_id'].nunique():,}")
    print(f"  dates          {combined['date'].nunique():,}")
    if "enl_empirical" in combined:
        print(f"  median ENL     {combined['enl_empirical'].median():.2f}")
    if "inc_angle_mean" in combined:
        lo, hi = combined["inc_angle_mean"].min(), combined["inc_angle_mean"].max()
        print(f"  incidence      {lo:.1f} to {hi:.1f} degrees")
    print(f"  written to     {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sarpanel",
        description="Analysis-ready Sentinel-1 observations over polygons, "
        "with acquisition geometry and speckle statistics retained.",
    )
    parser.add_argument("--version", action="version", version=f"sarpanel {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="query the catalog and write the scene manifest")
    p.add_argument("config", help="path to the run YAML")
    p.add_argument("--n-shards", type=int, default=None)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="process one shard of the manifest")
    p.add_argument("config")
    p.add_argument("--shard", type=int, default=None, help="defaults to SLURM_ARRAY_TASK_ID")
    p.add_argument("--strict", action="store_true", help="exit non-zero if any scene fails")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("doctor", help="report environment and stack capabilities")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("status", help="summarise a run")
    p.add_argument("config")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("aggregate", help="combine per-scene outputs")
    p.add_argument("config")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_aggregate)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
