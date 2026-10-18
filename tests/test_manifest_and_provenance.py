"""
Manifest and provenance tests.

The sharding tests encode the property that motivated the plan/run split:
partitioning must be a deterministic function of the manifest, so that a rerun
of one failed task processes exactly the scenes it processed before, with no
overlap and no gaps.
"""

import pandas as pd
import pytest

from sarpanel.aoi import load_aoi
from sarpanel.config import RunConfig
from sarpanel.manifest import assign_shards, attach_aoi_membership
from sarpanel.provenance import (
    build_provenance,
    read_provenance,
    write_parquet_with_provenance,
)
from sarpanel.runner import output_path_for


def fake_manifest(n=37):
    return pd.DataFrame(
        {
            "scene_id": [f"S1A_SCENE_{i:04d}" for i in range(n)],
            "acquisition_date": ["2022-06-15"] * n,
            "start_time": [f"2022-06-15T2{i % 10}:00:00Z" for i in range(n)],
            "relative_orbit": [26 if i % 2 else 121 for i in range(n)],
            "orbit_direction": ["ASCENDING"] * n,
            "size_mb": [4200.0] * n,
        }
    )


def test_shards_partition_without_overlap_or_gaps():
    manifest = assign_shards(fake_manifest(37), n_shards=8)
    assigned = [set(manifest[manifest["shard"] == s]["scene_id"]) for s in range(8)]

    union = set().union(*assigned)
    assert len(union) == 37
    for i in range(8):
        for j in range(i + 1, 8):
            assert not assigned[i] & assigned[j]


def test_sharding_is_deterministic():
    a = assign_shards(fake_manifest(37), 8)
    b = assign_shards(fake_manifest(37), 8)
    pd.testing.assert_series_equal(a["shard"], b["shard"])


def test_shards_are_balanced():
    manifest = assign_shards(fake_manifest(100), n_shards=7)
    counts = manifest["shard"].value_counts()
    assert counts.max() - counts.min() <= 1


def test_more_shards_than_scenes_is_allowed():
    manifest = assign_shards(fake_manifest(3), n_shards=10)
    assert manifest["shard"].nunique() == 3
    assert manifest["shard"].max() < 10


def test_zero_shards_raises():
    with pytest.raises(ValueError, match="at least 1"):
        assign_shards(fake_manifest(5), 0)


def test_aoi_membership_drops_scenes_touching_nothing():
    aoi = load_aoi("-84.6,42.6,-84.5,42.7")
    manifest = pd.DataFrame(
        {
            "scene_id": ["hit", "miss"],
            "footprint_wkt": [
                "POLYGON ((-85 42, -84 42, -84 43, -85 43, -85 42))",
                "POLYGON ((10 10, 11 10, 11 11, 10 11, 10 10))",
            ],
        }
    )
    result = attach_aoi_membership(manifest, aoi)
    assert list(result["scene_id"]) == ["hit"]
    assert result.iloc[0]["n_aoi_hit"] == 1


def test_output_paths_do_not_collide_for_same_date_and_direction(tmp_path):
    """
    Two granules from the same date and orbit direction is routine across any
    large AOI. Naming outputs by date and direction alone would make the second
    look already-complete and silently drop it.
    """
    cfg = RunConfig.from_dict(
        {
            "name": "t",
            "aoi": {"source": "-84.6,42.6,-84.5,42.7"},
            "search": {"start_date": "2022-05-01", "end_date": "2022-09-30"},
            "processing": {"target_crs": "EPSG:32616"},
            "output": {"root": str(tmp_path)},
        }
    )
    a = pd.Series(
        {"scene_id": "S1A_IW_SLC_20220615T234041_A", "acquisition_date": "2022-06-15"}
    )
    b = pd.Series(
        {"scene_id": "S1A_IW_SLC_20220615T234106_A", "acquisition_date": "2022-06-15"}
    )
    assert output_path_for(cfg, a) != output_path_for(cfg, b)


def test_provenance_survives_a_parquet_roundtrip(tmp_path):
    cfg = RunConfig.from_dict(
        {
            "name": "prov_test",
            "aoi": {"source": "-84.6,42.6,-84.5,42.7"},
            "search": {"start_date": "2022-05-01", "end_date": "2022-09-30"},
            "processing": {"target_crs": "EPSG:32616", "target_spacing_m": 10.0},
            "output": {"root": str(tmp_path)},
        }
    )
    aoi = load_aoi(cfg.aoi.source)
    record = build_provenance(cfg, aoi.checksum())

    df = pd.DataFrame({"aoi_id": ["a"], "VV_mean_db": [-12.3], "enl_empirical": [4.1]})
    path = write_parquet_with_provenance(df, tmp_path / "obs.parquet", record)

    recovered = read_provenance(path)
    assert recovered["config_checksum"] == cfg.checksum()
    assert recovered["aoi_checksum"] == aoi.checksum()
    assert recovered["sarpanel_run_name"] == "prov_test"
    assert "software" in recovered

    # The data itself must be unchanged by the metadata attachment.
    pd.testing.assert_frame_equal(pd.read_parquet(path), df)


# ---------------------------------------------------------------------------
# catalog query construction
# ---------------------------------------------------------------------------


class FakeProductType:
    """Mimics asf_search.PRODUCT_TYPE, which has no plain 'GRD' constant."""

    SLC = "SLC"
    GRD_HD = "GRD_HD"
    GRD_MD = "GRD_MD"
    GRD_HS = "GRD_HS"
    GRD_MS = "GRD_MS"
    GRD_FD = "GRD_FD"
    OCN = "OCN"
    RAW = "RAW"


class FakeASF:
    PRODUCT_TYPE = FakeProductType


def test_processing_levels_never_emits_an_undefined_type():
    """
    An invalid processingLevel makes the catalog resolve to an empty or wrong
    collection set and return zero results, which is indistinguishable from an
    archive gap. Only constants the installed asf_search defines may be sent.
    """
    from sarpanel.manifest import _processing_levels

    defined = {
        v for k, v in vars(FakeProductType).items()
        if not k.startswith("_") and isinstance(v, str)
    }

    for product_type in ("GRD", "GRDH", "grd"):
        levels = _processing_levels(FakeASF, product_type)
        assert levels, f"{product_type} resolved to nothing"
        assert "GRD" not in levels, "plain 'GRD' is not a valid product type"
        assert set(levels) <= defined, f"undefined level in {levels}"

    assert _processing_levels(FakeASF, "SLC") == ["SLC"]


def test_grd_prefers_high_resolution_first():
    from sarpanel.manifest import _processing_levels

    assert _processing_levels(FakeASF, "GRD")[0] == "GRD_HD"


def test_missing_constants_are_skipped_not_passed_through():
    class Sparse:
        class PRODUCT_TYPE:
            SLC = "SLC"
            GRD_HD = "GRD_HD"

    from sarpanel.manifest import _processing_levels

    assert _processing_levels(Sparse, "GRD") == ["GRD_HD"]


def test_no_defined_levels_raises_a_useful_error():
    class Empty:
        class PRODUCT_TYPE:
            SLC = "SLC"

    from sarpanel.manifest import _processing_levels

    with pytest.raises(RuntimeError, match="no product type constants"):
        _processing_levels(Empty, "GRD")


def test_unknown_product_type_rejected():
    from sarpanel.manifest import _processing_levels

    with pytest.raises(ValueError, match="SLC or GRD"):
        _processing_levels(FakeASF, "OCN")


# ---------------------------------------------------------------------------
# SNAP failure reporting and memory controls
# ---------------------------------------------------------------------------


def test_snap_crash_reports_the_real_cause_not_a_missing_angle_band():
    """
    pyroSAR catches a gpt crash internally and returns normally, so the caller
    sees an empty output directory. Reporting that as a missing incidence angle
    band points at export_extra and the DEM, which is the wrong diagnosis.
    """
    from sarpanel.processing import validate_geometry_bands

    crashed = {"backscatter": {}, "inc_angle_projected": None, "all_files": []}
    with pytest.raises(RuntimeError, match="processing graph failed"):
        validate_geometry_bands(crashed)

    # The memory hint must be present, since -9 is the common case.
    try:
        validate_geometry_bands(crashed)
    except RuntimeError as exc:
        assert "-9" in str(exc)
        assert "sbatch" in str(exc)
        assert "export_extra" not in str(exc).split("Terrain-Flattening")[0]


def test_missing_angle_reported_only_when_backscatter_exists():
    from sarpanel.processing import validate_geometry_bands

    partial = {
        "backscatter": {"VV": "/tmp/vv.tif"},
        "inc_angle_projected": None,
        "all_files": ["/tmp/vv.tif"],
    }
    with pytest.raises(RuntimeError, match="projectedLocalIncidenceAngle"):
        validate_geometry_bands(partial)


def test_complete_output_passes():
    from sarpanel.processing import validate_geometry_bands

    good = {
        "backscatter": {"VV": "/tmp/vv.tif", "VH": "/tmp/vh.tif"},
        "inc_angle_projected": "/tmp/inc.tif",
        "all_files": [],
    }
    validate_geometry_bands(good)


def test_tile_cache_is_bounded_well_below_the_allocation(monkeypatch):
    """
    SNAP's tile cache grows toward the heap limit if uncapped, leaving the
    operator no working memory. It is held near a third of the allocation.
    """
    from sarpanel.processing import ProcessingConfig, _gpt_resource_args

    monkeypatch.setenv("SLURM_MEM_PER_NODE", "96000")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    args = _gpt_resource_args(ProcessingConfig(target_crs="EPSG:32616"))

    assert "-c" in args and "-q" in args
    cache_gb = int(args[args.index("-c") + 1].rstrip("G"))
    assert 2 <= cache_gb <= 96 / 2, "cache must stay well under the allocation"
    assert args[args.index("-q") + 1] == "8"


def test_explicit_tile_cache_overrides_the_derived_value(monkeypatch):
    from sarpanel.processing import ProcessingConfig, _gpt_resource_args

    monkeypatch.setenv("SLURM_MEM_PER_NODE", "96000")
    cfg = ProcessingConfig(target_crs="EPSG:32616", snap_tile_cache_gb=6, snap_threads=2)
    assert _gpt_resource_args(cfg) == ["-c", "6G", "-q", "2"]


# ---------------------------------------------------------------------------
# AOI subsetting
# ---------------------------------------------------------------------------


def test_subset_is_clipped_to_the_scene_footprint(tmp_path):
    """
    Passing the whole AOI when it is larger than the scene asks SNAP to subset
    to a region mostly outside the data. The subset is intersected with the
    footprint so it always lies within the swath.
    """
    import geopandas as gpd
    from shapely.geometry import box

    from sarpanel.aoi import load_aoi
    from sarpanel.processing import write_scene_subset

    aoi = load_aoi(["-85.0,42.2,-84.9,42.3", "-84.2,43.6,-84.1,43.7"])
    footprint = box(-85.2, 42.0, -84.8, 42.5).wkt  # covers only the first

    path = write_scene_subset(aoi, footprint, tmp_path / "s.geojson", buffer_m=1000)
    assert path is not None

    written = gpd.read_file(path).geometry.iloc[0]
    scene = box(-85.2, 42.0, -84.8, 42.5)
    assert scene.buffer(1e-9).contains(written), "subset must lie inside the footprint"


def test_no_overlap_returns_none_so_the_scene_can_be_skipped(tmp_path):
    from shapely.geometry import box

    from sarpanel.aoi import load_aoi
    from sarpanel.processing import write_scene_subset

    aoi = load_aoi("-85.0,42.2,-84.9,42.3")
    far_away = box(10.0, 10.0, 11.0, 11.0).wkt
    assert write_scene_subset(aoi, far_away, tmp_path / "s.geojson") is None


def test_subset_buffer_is_applied_in_metres(tmp_path):
    import geopandas as gpd

    from sarpanel.aoi import load_aoi
    from sarpanel.processing import write_scene_subset

    aoi = load_aoi("-85.0,42.2,-84.9,42.3")
    path = write_scene_subset(aoi, None, tmp_path / "s.geojson", buffer_m=5000)

    written = gpd.read_file(path)
    metric = written.estimate_utm_crs()
    grown = written.to_crs(metric).area.iloc[0]
    original = aoi.gdf.to_crs(metric).area.iloc[0]
    assert grown > original * 2, "a 5 km buffer must enlarge a ~8 km field substantially"


def test_complex_aoi_is_simplified(tmp_path):
    """SNAP subsets to a region, so detail beyond a coarse outline is wasted."""
    import geopandas as gpd

    from sarpanel.aoi import load_aoi
    from sarpanel.processing import _vertex_count, write_scene_subset

    boxes = [f"{-85 + i * 0.02:.3f},42.6,{-85 + i * 0.02 + 0.01:.3f},42.61"
             for i in range(120)]
    aoi = load_aoi(boxes)
    path = write_scene_subset(aoi, None, tmp_path / "s.geojson", buffer_m=2000)
    written = gpd.read_file(path).geometry.iloc[0]
    assert _vertex_count(written) <= 200
