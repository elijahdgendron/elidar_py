"""Core LiDAR query engine -- port of queryLidar.m.

Loads a LAS/LAZ file, resolves the CRS, reprojects to lat/lon, filters
and cleans the point cloud, then executes one of three query modes:
  - peak: find the local maximum elevation
  - point: find the closest LiDAR return to the query location
  - search: interactive 3D point picking (Phase 2)
"""

from __future__ import annotations

import logging
from pathlib import Path

import laspy
import numpy as np

from elidar.core import cleaning
from elidar.core.crs import make_transformer, resolve_crs
from elidar.core.types import (
    M2FT,
    CrsInfo,
    PointCloudData,
    QueryOptions,
    QueryResult,
    QueryType,
)

logger = logging.getLogger(__name__)


def query_lidar(
    query_lat: float,
    query_lon: float,
    query_type: QueryType | str,
    las_path: str | Path,
    options: QueryOptions | None = None,
) -> QueryResult | None:
    """Query a LiDAR dataset at or around a geographic point.

    :param query_lat: Latitude of query in degrees (-90 to 90).
    :param query_lon: Longitude of query in degrees (-180 to 180).
    :param query_type: Query algorithm ("peak", "point", or "search").
    :param las_path: Path to LAS/LAZ file.
    :param options: Query configuration. Uses defaults if None.
    :return: QueryResult with elevation and position, or None if cancelled/no data.
    :raises FileNotFoundError: If the LAS file does not exist.
    :raises ValueError: If CRS cannot be determined.
    """
    if options is None:
        options = QueryOptions()

    if isinstance(query_type, str):
        query_type = QueryType(query_type.lower())

    las_path = Path(las_path)
    if not las_path.exists():
        raise FileNotFoundError(f"LAS file not found: {las_path}")
    if las_path.suffix.lower() not in (".las", ".laz"):
        raise ValueError(f"Expected .las or .laz file, got: {las_path.suffix}")

    crs_info = resolve_crs(las_path, dem_path=options.dem_path, force_unit=options.force_unit)
    logger.info(
        "CRS resolved: source=%s, unit=%s, geographic=%s",
        crs_info.source,
        crs_info.unit_name,
        crs_info.is_geographic,
    )

    cloud = _read_point_cloud(las_path, filter_ground=options.filter_ground)
    logger.info("Loaded %d points from %s", cloud.n_points, las_path.name)

    cloud = _reproject(cloud, crs_info)

    if options.center:
        query_lat = float(np.mean(cloud.lat))
        query_lon = float(np.mean(cloud.lon))
        logger.info("Using dataset center: lat=%.6f, lon=%.6f", query_lat, query_lon)

    cloud = _spatial_filter(cloud, query_lat, query_lon, options.query_radius_deg)
    if cloud.n_points == 0:
        logger.warning("Query point is not within the LAS data extent")
        return None

    logger.info("Points within query radius: %d", cloud.n_points)

    if options.clean_data and not crs_info.is_geographic:
        # MATLAB uses 0.01 meters for voxel dedup; scale to native units
        voxel_m = 0.01
        voxel_native = voxel_m / crs_info.unit_to_meters if crs_info.is_foot else voxel_m
        cloud = cleaning.clean(
            cloud,
            num_neighbors=options.filter_num_neighbors,
            std_ratio=options.filter_threshold,
            voxel_size=voxel_native,
        )
        logger.info("After cleaning: %d points", cloud.n_points)

    if query_type == QueryType.PEAK:
        return _query_peak(cloud, crs_info)
    elif query_type == QueryType.POINT:
        return _query_point(cloud, crs_info, query_lat, query_lon)
    elif query_type == QueryType.SEARCH:
        return _query_search(cloud, crs_info, options)
    else:
        raise ValueError(f"Unknown query type: {query_type}")


def _read_point_cloud(
    las_path: Path,
    filter_ground: bool = False,
) -> PointCloudData:
    """Read a LAS/LAZ file into a PointCloudData.

    :param las_path: Path to the file.
    :param filter_ground: If True, keep only classification == 2 points.
    :return: PointCloudData with x, y, z (and optionally classification/intensity).
    """
    las = laspy.read(las_path)

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)

    classification = None
    if hasattr(las, "classification"):
        classification = np.asarray(las.classification, dtype=np.uint8)

    intensity = None
    if hasattr(las, "intensity"):
        intensity = np.asarray(las.intensity, dtype=np.uint16)

    cloud = PointCloudData(
        x=x, y=y, z=z,
        classification=classification,
        intensity=intensity,
    )

    if filter_ground and classification is not None:
        ground_mask = classification == 2
        cloud = cloud.subset(ground_mask)
        logger.info(
            "Ground filter: %d -> %d points",
            len(x),
            cloud.n_points,
        )

    return cloud


def _reproject(cloud: PointCloudData, crs_info: CrsInfo) -> PointCloudData:
    """Reproject point cloud coordinates to lat/lon.

    :param cloud: Point cloud in native CRS.
    :param crs_info: Resolved CRS information.
    :return: Point cloud with lat/lon populated.
    """
    transformer = make_transformer(crs_info)

    if transformer is None:
        cloud.lat = cloud.x.copy()
        cloud.lon = cloud.y.copy()
    else:
        # pyproj with always_xy=True: input is (x, y), output is (lon, lat)
        lon, lat = transformer.transform(cloud.x, cloud.y)
        cloud.lat = lat
        cloud.lon = lon

    return cloud


def _spatial_filter(
    cloud: PointCloudData,
    query_lat: float,
    query_lon: float,
    radius_deg: float,
) -> PointCloudData:
    """Filter point cloud to points within a radius of the query location.

    Uses simple Euclidean distance in degree space, matching the MATLAB version.

    :param cloud: Point cloud with lat/lon populated.
    :param query_lat: Query latitude in degrees.
    :param query_lon: Query longitude in degrees.
    :param radius_deg: Search radius in degrees.
    :return: Filtered point cloud.
    """
    dist = np.sqrt((cloud.lat - query_lat) ** 2 + (cloud.lon - query_lon) ** 2)
    mask = dist <= radius_deg
    return cloud.subset(mask)


def _elevation_to_result(
    z_native: float,
    x_local: float,
    y_local: float,
    lat: float,
    lon: float,
    crs_info: CrsInfo,
    query_type: QueryType,
) -> QueryResult:
    """Convert a native-unit elevation to a QueryResult with both m and ft.

    :param z_native: Elevation in the file's native unit.
    :param x_local: X coordinate in the file's native CRS.
    :param y_local: Y coordinate in the file's native CRS.
    :param lat: Latitude in degrees.
    :param lon: Longitude in degrees.
    :param crs_info: CRS info for unit conversion.
    :param query_type: The query mode that produced this result.
    :return: QueryResult with elevation in both meters and feet.
    """
    if crs_info.is_foot:
        elevation_ft = float(z_native)
        elevation_m = float(z_native) * crs_info.unit_to_meters
    else:
        elevation_m = float(z_native)
        elevation_ft = float(z_native) * M2FT

    return QueryResult(
        elevation_m=elevation_m,
        elevation_ft=elevation_ft,
        location_local=(float(x_local), float(y_local)),
        location_deg=(float(lat), float(lon)),
        query_type=query_type,
    )


def _query_peak(cloud: PointCloudData, crs_info: CrsInfo) -> QueryResult:
    """Find the point with the highest elevation.

    :param cloud: Filtered and cleaned point cloud.
    :param crs_info: CRS info for unit conversion.
    :return: QueryResult for the highest point.
    """
    ix = int(np.argmax(cloud.z))
    logger.info(
        "Peak found: z=%.4f at lat=%.6f, lon=%.6f",
        cloud.z[ix],
        cloud.lat[ix],
        cloud.lon[ix],
    )

    return _elevation_to_result(
        z_native=cloud.z[ix],
        x_local=cloud.x[ix],
        y_local=cloud.y[ix],
        lat=cloud.lat[ix],
        lon=cloud.lon[ix],
        crs_info=crs_info,
        query_type=QueryType.PEAK,
    )


def _query_point(
    cloud: PointCloudData,
    crs_info: CrsInfo,
    query_lat: float,
    query_lon: float,
) -> QueryResult:
    """Find the closest LiDAR return to the query location.

    :param cloud: Filtered and cleaned point cloud.
    :param crs_info: CRS info for unit conversion.
    :param query_lat: Query latitude in degrees.
    :param query_lon: Query longitude in degrees.
    :return: QueryResult for the closest point.
    """
    dist = np.sqrt((cloud.lat - query_lat) ** 2 + (cloud.lon - query_lon) ** 2)
    ix = int(np.argmin(dist))
    logger.info(
        "Closest point: z=%.4f at lat=%.6f, lon=%.6f (dist=%.8f deg)",
        cloud.z[ix],
        cloud.lat[ix],
        cloud.lon[ix],
        dist[ix],
    )

    return _elevation_to_result(
        z_native=cloud.z[ix],
        x_local=cloud.x[ix],
        y_local=cloud.y[ix],
        lat=cloud.lat[ix],
        lon=cloud.lon[ix],
        crs_info=crs_info,
        query_type=QueryType.POINT,
    )


def _query_search(
    cloud: PointCloudData,
    crs_info: CrsInfo,
    options: QueryOptions,
) -> QueryResult | None:
    """Interactive 3D point selection with zoom loop.

    :param cloud: Filtered and cleaned point cloud.
    :param crs_info: CRS info for unit conversion and display.
    :param options: Query options including zoom scale and colorscale.
    :return: QueryResult for the selected point, or None if cancelled.
    """
    try:
        from elidar.viz.scatter3d import interactive_search
    except ImportError:
        raise ImportError(
            "Search mode requires visualization dependencies. "
            "Install with: pip install elidar[viz]"
        )

    pick = interactive_search(cloud, crs_info, options)
    if pick is None:
        return None

    return _elevation_to_result(
        z_native=pick.z,
        x_local=pick.x_local,
        y_local=pick.y_local,
        lat=pick.lat,
        lon=pick.lon,
        crs_info=crs_info,
        query_type=QueryType.SEARCH,
    )
