"""Integration tests using a synthetic LAZ file.

Creates a small LAS file with known point positions in EPSG:4326
(lat/lon/elevation) to test the full query pipeline without needing
real LiDAR data.
"""

import tempfile
from pathlib import Path

import laspy
import numpy as np
import pytest

from elidar.core.query import query_lidar
from elidar.core.types import QueryOptions, QueryType


@pytest.fixture
def synthetic_las_path() -> Path:
    """Create a synthetic LAS file with known points in EPSG:4326.

    Uses the LLA_ prefix convention so no DEM or CRS VLR is needed.
    Points form a small hill centered at (38.828, -104.906) with a
    known peak at z=2476.0 meters.
    """
    center_lat = 38.828
    center_lon = -104.906
    n_points = 500
    rng = np.random.RandomState(42)

    # Generate points in a grid around center
    lat = center_lat + rng.uniform(-0.0005, 0.0005, n_points)
    lon = center_lon + rng.uniform(-0.0005, 0.0005, n_points)

    # Elevation: gaussian hill centered at (center_lat, center_lon)
    dist = np.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)
    z = 2450.0 + 26.0 * np.exp(-(dist / 0.0002) ** 2)

    # Force exact peak at a known location
    peak_idx = 0
    lat[peak_idx] = center_lat
    lon[peak_idx] = center_lon
    z[peak_idx] = 2476.0

    # Create LAS file
    header = laspy.LasHeader(point_format=0, version="1.2")
    header.offsets = [np.min(lat), np.min(lon), np.min(z)]
    header.scales = [1e-7, 1e-7, 0.001]

    las = laspy.LasData(header)
    las.x = lat
    las.y = lon
    las.z = z

    tmp = tempfile.NamedTemporaryFile(suffix=".las", prefix="LLA_test_", delete=False)
    las.write(tmp.name)
    tmp.close()

    yield Path(tmp.name)

    Path(tmp.name).unlink(missing_ok=True)


class TestPeakQuery:
    def test_finds_peak(self, synthetic_las_path: Path) -> None:
        result = query_lidar(
            query_lat=38.828,
            query_lon=-104.906,
            query_type=QueryType.PEAK,
            las_path=synthetic_las_path,
            options=QueryOptions(clean_data=False),
        )

        assert result is not None
        assert abs(result.elevation_m - 2476.0) < 0.01
        assert abs(result.location_deg[0] - 38.828) < 0.001
        assert abs(result.location_deg[1] - (-104.906)) < 0.001
        assert result.query_type == QueryType.PEAK

    def test_elevation_ft_conversion(self, synthetic_las_path: Path) -> None:
        result = query_lidar(
            query_lat=38.828,
            query_lon=-104.906,
            query_type=QueryType.PEAK,
            las_path=synthetic_las_path,
            options=QueryOptions(clean_data=False),
        )

        assert result is not None
        expected_ft = 2476.0 * 3.2808398950131233
        assert abs(result.elevation_ft - expected_ft) < 0.01


class TestPointQuery:
    def test_finds_closest_point(self, synthetic_las_path: Path) -> None:
        result = query_lidar(
            query_lat=38.828,
            query_lon=-104.906,
            query_type=QueryType.POINT,
            las_path=synthetic_las_path,
            options=QueryOptions(clean_data=False),
        )

        assert result is not None
        # Should find the peak point since it's at the exact center
        assert abs(result.elevation_m - 2476.0) < 0.01
        assert result.query_type == QueryType.POINT

    def test_off_center_query(self, synthetic_las_path: Path) -> None:
        result = query_lidar(
            query_lat=38.8282,
            query_lon=-104.9062,
            query_type=QueryType.POINT,
            las_path=synthetic_las_path,
            options=QueryOptions(clean_data=False),
        )

        assert result is not None
        # Should find some point near the query, not necessarily the peak
        assert result.elevation_m > 2440.0
        assert result.query_type == QueryType.POINT


class TestQueryEdgeCases:
    def test_point_outside_data(self, synthetic_las_path: Path) -> None:
        result = query_lidar(
            query_lat=40.0,
            query_lon=-100.0,
            query_type=QueryType.PEAK,
            las_path=synthetic_las_path,
            options=QueryOptions(clean_data=False),
        )
        assert result is None

    def test_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            query_lidar(
                query_lat=38.0,
                query_lon=-105.0,
                query_type=QueryType.PEAK,
                las_path="/nonexistent/file.laz",
            )

    def test_center_mode(self, synthetic_las_path: Path) -> None:
        result = query_lidar(
            query_lat=0.0,
            query_lon=0.0,
            query_type=QueryType.PEAK,
            las_path=synthetic_las_path,
            options=QueryOptions(clean_data=False, center=True),
        )
        assert result is not None
        assert result.elevation_m > 2440.0
