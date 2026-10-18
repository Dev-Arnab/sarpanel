"""
AOI ingestion and configuration tests.

The AOI tests cover the requirement that one polygon and many polygons follow
the same code path, differing only in row count.
"""

import geopandas as gpd
import pytest
from shapely.geometry import box

from sarpanel.aoi import load_aoi
from sarpanel.config import RunConfig
from sarpanel.processing import ProcessingConfig


def test_single_bbox_string():
    aoi = load_aoi("-84.60,42.68,-84.55,42.72")
    assert len(aoi) == 1
    assert aoi.gdf.crs.to_epsg() == 4326
    assert aoi.ids == ["aoi_000000"]


def test_list_of_bboxes():
    aoi = load_aoi(["-84.6,42.6,-84.5,42.7", "-84.4,42.6,-84.3,42.7"])
    assert len(aoi) == 2
    assert len(set(aoi.ids)) == 2


def test_wkt_string():
    aoi = load_aoi("POLYGON ((-84.6 42.6, -84.5 42.6, -84.5 42.7, -84.6 42.7, -84.6 42.6))")
    assert len(aoi) == 1


def test_malformed_bbox_raises():
    with pytest.raises(ValueError, match="minx,miny,maxx,maxy"):
        load_aoi("-84.6,42.6,-84.5")


def test_inverted_bbox_raises():
    with pytest.raises(ValueError, match="valid EPSG:4326 extent"):
        load_aoi("-84.5,42.7,-84.6,42.6")


def test_geodataframe_input_is_reprojected():
    gdf = gpd.GeoDataFrame(
        {"field_id": ["a", "b"]},
        geometry=[
            box(500000, 4500000, 500500, 4500500),
            box(501000, 4500000, 501500, 4500500),
        ],
        crs="EPSG:32616",
    )
    aoi = load_aoi(gdf)
    assert aoi.gdf.crs.to_epsg() == 4326
    assert aoi.ids == ["a", "b"]


def test_missing_crs_raises_rather_than_assuming():
    """
    Assuming EPSG:4326 for a CRS-less file silently produces an empty catalog
    search, which looks like 'no data available' rather than a configuration
    error.
    """
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs=None)
    with pytest.raises(ValueError, match="no CRS"):
        load_aoi(gdf)


def test_area_filter_uses_projected_area():
    gdf = gpd.GeoDataFrame(
        {"id": ["big", "small"]},
        geometry=[
            box(500000, 4500000, 500500, 4500500),   # 250,000 m2
            box(502000, 4500000, 502010, 4500010),   # 100 m2
        ],
        crs="EPSG:32616",
    )
    aoi = load_aoi(gdf, min_area_m2=1000)
    assert aoi.ids == ["big"]


def test_duplicate_ids_are_disambiguated():
    gdf = gpd.GeoDataFrame(
        {"name": ["north", "north"]},
        geometry=[
            box(500000, 4500000, 500500, 4500500),
            box(501000, 4500000, 501500, 4500500),
        ],
        crs="EPSG:32616",
    )
    aoi = load_aoi(gdf, id_column="name")
    assert len(set(aoi.ids)) == 2


def test_id_column_missing_raises_with_available_columns():
    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    with pytest.raises(KeyError, match="Available"):
        load_aoi(gdf, id_column="nonexistent")


def test_search_footprint_covers_all_polygons():
    aoi = load_aoi(["-84.6,42.6,-84.5,42.7", "-84.0,42.6,-83.9,42.7"])
    footprint = aoi.search_footprint(buffer_m=1000)
    for geom in aoi.gdf.geometry:
        assert footprint.contains(geom)


def test_search_footprint_is_simplified_when_complex():
    """Scattered AOI sets must not produce WKT the catalog will reject."""
    boxes = [
        f"{-85 + i * 0.05:.3f},42.6,{-85 + i * 0.05 + 0.01:.3f},42.61" for i in range(300)
    ]
    aoi = load_aoi(boxes)
    footprint = aoi.search_footprint()
    assert len(footprint.wkt) < 60_000


def test_checksum_is_stable_and_geometry_sensitive():
    a = load_aoi("-84.6,42.6,-84.5,42.7")
    b = load_aoi("-84.6,42.6,-84.5,42.7")
    c = load_aoi("-84.6,42.6,-84.5,42.8")
    assert a.checksum() == b.checksum()
    assert a.checksum() != c.checksum()


def test_one_and_many_polygons_share_the_schema():
    one = load_aoi("-84.6,42.6,-84.5,42.7")
    many = load_aoi([f"-84.{i},42.6,-84.{i-1},42.7" for i in range(9, 4, -1)])
    assert list(one.gdf.columns) == list(many.gdf.columns)
    assert len(one) == 1 and len(many) == 5


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_geographic_crs_is_rejected():
    """
    In EPSG:4326 the units of target_spacing_m are degrees, making the
    requested resolution ambiguous and latitude-dependent.
    """
    with pytest.raises(ValueError, match="projected CRS"):
        ProcessingConfig(target_crs="EPSG:4326")


def test_db_scaling_is_rejected_at_geocode_time():
    with pytest.raises(ValueError, match="linear"):
        ProcessingConfig(scaling="dB")


def test_config_roundtrip_and_checksum(tmp_path):
    raw = {
        "name": "t",
        "aoi": {"source": "-84.6,42.6,-84.5,42.7"},
        "search": {"start_date": "2022-05-01", "end_date": "2022-09-30"},
        "processing": {"target_crs": "EPSG:32616", "product_type": "SLC"},
        "output": {"root": str(tmp_path)},
        "n_shards": 4,
    }
    cfg = RunConfig.from_dict(raw)
    path = cfg.write(tmp_path / "cfg.yaml")
    reloaded = RunConfig.from_yaml(path)
    assert reloaded.checksum() == cfg.checksum()

    # Shard count is parallelism, not content, so it must not change the hash.
    raw["n_shards"] = 64
    assert RunConfig.from_dict(raw).checksum() == cfg.checksum()

    # Spacing does change content.
    raw["processing"]["target_spacing_m"] = 20.0
    assert RunConfig.from_dict(raw).checksum() != cfg.checksum()


def test_reversed_dates_raise(tmp_path):
    raw = {
        "name": "t",
        "aoi": {"source": "-84.6,42.6,-84.5,42.7"},
        "search": {"start_date": "2022-09-30", "end_date": "2022-05-01"},
        "processing": {"target_crs": "EPSG:32616"},
        "output": {"root": str(tmp_path)},
    }
    with pytest.raises(ValueError, match="after"):
        RunConfig.from_dict(raw)


# ---------------------------------------------------------------------------
# geopandas version compatibility
# ---------------------------------------------------------------------------


def test_union_all_works_on_both_geopandas_generations():
    """
    GeoPandas 1.0 replaced the ``unary_union`` property with a ``union_all()``
    method. Clusters often pin an older release alongside a conda GDAL stack,
    so both spellings must work. This simulates each generation with a stub
    exposing only one of the two.
    """
    from shapely.geometry import box as _box

    from sarpanel.aoi import union_all

    parts = [_box(0, 0, 1, 1), _box(2, 0, 3, 1)]
    expected_area = 2.0

    class NewStyle:
        def union_all(self):
            from shapely.ops import unary_union as _u

            return _u(parts)

    class OldStyle:
        @property
        def unary_union(self):
            from shapely.ops import unary_union as _u

            return _u(parts)

    assert union_all(NewStyle()).area == pytest.approx(expected_area)
    assert union_all(OldStyle()).area == pytest.approx(expected_area)


def test_union_all_accepts_a_geodataframe_directly():
    gdf = gpd.GeoDataFrame(
        geometry=[box(500000, 4500000, 500500, 4500500),
                  box(501000, 4500000, 501500, 4500500)],
        crs="EPSG:32616",
    )
    from sarpanel.aoi import union_all

    assert union_all(gdf).area == pytest.approx(500 * 500 * 2)


def test_search_footprint_on_real_geopandas():
    """End to end through whichever geopandas is actually installed."""
    aoi = load_aoi(["-84.6,42.6,-84.5,42.7", "-84.0,42.6,-83.9,42.7"])
    footprint = aoi.search_footprint(buffer_m=1000)
    assert footprint.is_valid and not footprint.is_empty
    for geom in aoi.gdf.geometry:
        assert footprint.contains(geom)


# ---------------------------------------------------------------------------
# credential handling
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch, tmp_path):
    """
    Isolate credential resolution from the machine running the tests.

    Without this, these tests pass or fail depending on whether the developer
    happens to have ~/.earthdata_token or ~/.netrc on disk. A suite whose
    result depends on the absence of a file in someone's home directory is not
    telling you anything reliable. Both sources are redirected to paths under
    tmp_path that do not exist, so each test states its own preconditions.
    """
    import sarpanel.auth as auth

    monkeypatch.setattr(auth, "TOKEN_FILE", tmp_path / "no_such_token")
    monkeypatch.setenv("NETRC", str(tmp_path / "no_such_netrc"))
    for var in ("EARTHDATA_TOKEN", "ASF_TOKEN", "EARTHDATA_USERNAME",
                "EARTHDATA_PASSWORD", "ASF_USERNAME", "ASF_PASSWORD"):
        monkeypatch.delenv(var, raising=False)


def test_credentials_are_stripped_of_whitespace(monkeypatch):
    """
    Values sourced from a file or set with $(cat ...) carry a trailing newline.
    URS rejects those without explanation, which presents as a wrong password.
    """
    from sarpanel.auth import resolve_credentials

    for var in ("EARTHDATA_TOKEN", "ASF_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EARTHDATA_USERNAME", "someuser\n")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "  secret  \n")

    creds = resolve_credentials()
    assert creds.username == "someuser"
    assert creds.password == "secret"


def test_token_is_preferred_over_password(monkeypatch):
    from sarpanel.auth import resolve_credentials

    monkeypatch.setenv("EARTHDATA_TOKEN", " tok123\n")
    monkeypatch.setenv("EARTHDATA_USERNAME", "someuser")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "secret")

    creds = resolve_credentials()
    assert creds.token == "tok123"
    assert creds.username is None


def test_blank_token_falls_through_to_password(monkeypatch):
    """An exported-but-empty variable must not shadow working credentials."""
    from sarpanel.auth import resolve_credentials

    monkeypatch.setenv("EARTHDATA_TOKEN", "   ")
    monkeypatch.delenv("ASF_TOKEN", raising=False)
    monkeypatch.setenv("EARTHDATA_USERNAME", "someuser")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "secret")

    creds = resolve_credentials()
    assert creds.token is None
    assert creds.username == "someuser"


def test_credentials_never_appear_in_repr(monkeypatch):
    """Secrets must not leak into logs or tracebacks."""
    from sarpanel.auth import Credentials

    creds = Credentials(username="someuser", password="hunter2", source="test")
    assert "hunter2" not in repr(creds)
    assert "redacted" in repr(creds)

    creds = Credentials(token="tok123", source="test")
    assert "tok123" not in repr(creds)


def test_token_file_is_used_when_env_is_absent(tmp_path, monkeypatch):
    """
    Batch jobs do not reliably inherit exported variables, and a login shell
    may re-source a profile holding stale credentials. A token on disk works in
    both contexts, which an environment-only token does not.
    """
    import sarpanel.auth as auth

    for var in ("EARTHDATA_TOKEN", "ASF_TOKEN", "EARTHDATA_USERNAME",
                "EARTHDATA_PASSWORD", "ASF_USERNAME", "ASF_PASSWORD"):
        monkeypatch.delenv(var, raising=False)

    token_file = tmp_path / ".earthdata_token"
    token_file.write_text("  abc123token\n")
    token_file.chmod(0o600)
    monkeypatch.setattr(auth, "TOKEN_FILE", token_file)
    monkeypatch.setenv("NETRC", str(tmp_path / "nonexistent"))

    creds = auth.resolve_credentials()
    assert creds.token == "abc123token"
    assert "token file" in creds.source


def test_token_file_accepts_a_sourceable_key_value_line(tmp_path, monkeypatch):
    """The same file can be sourced by a shell or read here."""
    import sarpanel.auth as auth

    for var in ("EARTHDATA_TOKEN", "ASF_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    token_file = tmp_path / ".earthdata_token"
    token_file.write_text("EARTHDATA_TOKEN='abc123token'\n")
    token_file.chmod(0o600)
    monkeypatch.setattr(auth, "TOKEN_FILE", token_file)

    assert auth.resolve_credentials().token == "abc123token"


def test_world_readable_token_file_is_refused(tmp_path, monkeypatch):
    import sarpanel.auth as auth

    token_file = tmp_path / ".earthdata_token"
    token_file.write_text("abc123token\n")
    token_file.chmod(0o644)
    monkeypatch.setattr(auth, "TOKEN_FILE", token_file)
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)

    with pytest.raises(PermissionError, match="chmod 600"):
        auth.resolve_credentials()


def test_env_token_still_takes_precedence_over_the_file(tmp_path, monkeypatch):
    import sarpanel.auth as auth

    token_file = tmp_path / ".earthdata_token"
    token_file.write_text("filetoken\n")
    token_file.chmod(0o600)
    monkeypatch.setattr(auth, "TOKEN_FILE", token_file)
    monkeypatch.setenv("EARTHDATA_TOKEN", "envtoken")

    assert auth.resolve_credentials().token == "envtoken"


def test_token_file_beats_a_stale_environment_password(tmp_path, monkeypatch):
    """The exact failure seen under sbatch: a stale password must not win."""
    import sarpanel.auth as auth

    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    monkeypatch.delenv("ASF_TOKEN", raising=False)
    monkeypatch.setenv("EARTHDATA_USERNAME", "rjayne12")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "stale_password")

    token_file = tmp_path / ".earthdata_token"
    token_file.write_text("goodtoken\n")
    token_file.chmod(0o600)
    monkeypatch.setattr(auth, "TOKEN_FILE", token_file)

    creds = auth.resolve_credentials()
    assert creds.token == "goodtoken"
    assert creds.username is None
