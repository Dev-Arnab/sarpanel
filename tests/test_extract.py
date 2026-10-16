"""
Geometry and extraction tests.

These use small synthetic rasters with analytically known answers. The point is
to catch the off-by-one and misalignment errors that are invisible in aggregate
statistics but corrupt per-pixel analysis.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import geopandas as gpd

from sarpanel.aoi import load_aoi
from sarpanel.extract import SceneContext, SceneReader, extract_scene, polygon_window


def make_aoi(*geometries, ids=None):
    """Build an AOISet from projected geometries, declaring the CRS explicitly."""
    ids = ids or [f"field_{i}" for i in range(len(geometries))]
    gdf = gpd.GeoDataFrame({"aoi_id": ids}, geometry=list(geometries), crs=CRS)
    return load_aoi(gdf)


CRS = "EPSG:32616"
RES = 10.0
ORIGIN_X, ORIGIN_Y = 500_000.0, 4_500_000.0


def write_band(path, array, nodata=None):
    transform = from_origin(ORIGIN_X, ORIGIN_Y, RES, RES)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array.astype("float32"), 1)
    return path


@pytest.fixture
def scene(tmp_path):
    """A 100x100 scene with a known 20 dB square patch and a known angle ramp."""
    n = 100
    vv = np.full((n, n), 0.1, dtype=np.float32)
    vv[20:40, 30:50] = 1.0  # bright square, rows 20-39, cols 30-49
    vh = vv * 0.2

    # Incidence angle ramps across range (columns), 30 to 45 degrees.
    inc = np.tile(np.linspace(30.0, 45.0, n, dtype=np.float32), (n, 1))
    lsm = np.zeros((n, n), dtype=np.float32)

    bands = {
        "backscatter": {
            "VV": write_band(tmp_path / "vv.tif", vv),
            "VH": write_band(tmp_path / "vh.tif", vh),
        },
        "inc_angle_projected": write_band(tmp_path / "inc.tif", inc),
        "layover_shadow": write_band(tmp_path / "lsm.tif", lsm),
    }
    return bands


@pytest.fixture
def context():
    return SceneContext(
        scene_id="S1A_TEST",
        acquisition_date="2022-06-15",
        start_time="2022-06-15T23:40:41Z",
        platform="Sentinel-1A",
        orbit_direction="ASCENDING",
        relative_orbit=99,
        absolute_orbit=12345,
        frame_number=500,
        product_type="SLC",
        looks={"looks_nominal": 4.35, "looks_range": 4.35, "looks_azimuth": 1.0},
    )


def polygon_over_bright_patch():
    """Polygon covering exactly the bright square, rows 20-39, cols 30-49."""
    minx = ORIGIN_X + 30 * RES
    maxx = ORIGIN_X + 50 * RES
    maxy = ORIGIN_Y - 20 * RES
    miny = ORIGIN_Y - 40 * RES
    return box(minx, miny, maxx, maxy)


def test_polygon_window_locates_correct_pixels(scene):
    with SceneReader(scene) as reader:
        win = polygon_window(polygon_over_bright_patch(), reader, pad=0)
        assert (int(win.row_off), int(win.col_off)) == (20, 30)
        assert (int(win.height), int(win.width)) == (20, 20)


def test_window_is_clamped_at_raster_edge(scene):
    """A polygon hanging off the corner must not produce negative offsets."""
    edge = box(ORIGIN_X - 500, ORIGIN_Y - 200, ORIGIN_X + 200, ORIGIN_Y + 500)
    with SceneReader(scene) as reader:
        win = polygon_window(edge, reader)
        assert win.row_off >= 0 and win.col_off >= 0
        assert win.row_off + win.height <= reader.shape[0]
        assert win.col_off + win.width <= reader.shape[1]


def test_polygon_entirely_outside_returns_none(scene):
    far = box(ORIGIN_X + 10_000, ORIGIN_Y - 12_000, ORIGIN_X + 11_000, ORIGIN_Y - 11_000)
    with SceneReader(scene) as reader:
        assert polygon_window(far, reader) is None


def test_extracted_values_match_the_synthetic_patch(scene, context, tmp_path):
    aoi = make_aoi(polygon_over_bright_patch())
    with SceneReader(scene) as reader:
        df = extract_scene(reader, aoi, context, granularity="polygon")

    assert len(df) == 1
    row = df.iloc[0]
    assert row["n_pixels_valid"] == 400  # 20 x 20
    assert row["VV_mean_linear"] == pytest.approx(1.0, rel=1e-5)
    assert row["VV_mean_db"] == pytest.approx(0.0, abs=1e-4)
    # VH is exactly 0.2 * VV, so the ratio must be 10*log10(0.2)
    assert row["VH_VV_ratio_db"] == pytest.approx(10 * np.log10(0.2), abs=1e-4)


def test_incidence_angle_is_read_from_the_matching_grid(scene, context):
    """
    The angle ramps 30 to 45 across 100 columns. Columns 30-49 therefore span
    a known sub-range, and the mean must match it. This is the test that would
    fail if the angle band came from a different grid than the backscatter.
    """
    aoi = make_aoi(polygon_over_bright_patch())
    with SceneReader(scene) as reader:
        df = extract_scene(reader, aoi, context, granularity="polygon")

    expected = np.linspace(30.0, 45.0, 100)[30:50].mean()
    assert df.iloc[0]["inc_angle_mean"] == pytest.approx(expected, abs=1e-3)
    assert df.iloc[0]["inc_angle_min"] == pytest.approx(np.linspace(30, 45, 100)[30], abs=1e-3)
    assert df.iloc[0]["inc_angle_max"] == pytest.approx(np.linspace(30, 45, 100)[49], abs=1e-3)


def test_misaligned_band_raises_rather_than_silently_resampling(tmp_path, scene):
    """
    A band on a different grid must raise. This is the guard against the
    tie-point-grid failure mode, where an angle array of different dimensions
    is indexed with the output raster's row and column.
    """
    small = np.full((50, 50), 35.0, dtype=np.float32)
    scene = dict(scene)
    scene["inc_angle_projected"] = write_band(tmp_path / "inc_small.tif", small)
    with pytest.raises(ValueError, match="shape"):
        with SceneReader(scene):
            pass


def test_pixel_granularity_returns_one_row_per_pixel(scene, context):
    aoi = make_aoi(polygon_over_bright_patch())
    with SceneReader(scene) as reader:
        df = extract_scene(reader, aoi, context, granularity="pixel")
    assert len(df) == 400
    assert {"x", "y", "VV_db", "inc_angle"} <= set(df.columns)
    assert df["VV_linear"].min() == pytest.approx(1.0)


def test_layover_shadow_pixels_are_excluded(tmp_path, scene, context):
    lsm = np.zeros((100, 100), dtype=np.float32)
    lsm[20:30, 30:50] = 1.0  # flag half the patch as layover
    scene = dict(scene)
    scene["layover_shadow"] = write_band(tmp_path / "lsm2.tif", lsm)

    aoi = make_aoi(polygon_over_bright_patch())
    with SceneReader(scene) as reader:
        df = extract_scene(reader, aoi, context, granularity="polygon")
    assert df.iloc[0]["n_pixels_valid"] == 200
    assert df.iloc[0]["layover_shadow_fraction"] == pytest.approx(0.5)


def test_nodata_pixels_are_excluded(tmp_path, context):
    n = 60
    vv = np.full((n, n), 1.0, dtype=np.float32)
    vv[:, :30] = -9999.0
    vh = np.full((n, n), 0.2, dtype=np.float32)
    inc = np.full((n, n), 38.0, dtype=np.float32)
    bands = {
        "backscatter": {
            "VV": write_band(tmp_path / "a.tif", vv, nodata=-9999.0),
            "VH": write_band(tmp_path / "b.tif", vh, nodata=-9999.0),
        },
        "inc_angle_projected": write_band(tmp_path / "c.tif", inc),
    }
    poly = box(ORIGIN_X, ORIGIN_Y - 40 * RES, ORIGIN_X + 40 * RES, ORIGIN_Y)
    aoi = make_aoi(poly)
    with SceneReader(bands) as reader:
        df = extract_scene(reader, aoi, context, granularity="polygon")
    # 40x40 window, but the left 30 columns are nodata -> 40 rows x 10 cols
    assert df.iloc[0]["n_pixels_valid"] == 400
    assert df.iloc[0]["valid_fraction"] == pytest.approx(0.25)
