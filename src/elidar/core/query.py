"""Core LiDAR query engine -- port of queryLidar.m.

Loads one or more LAS/LAZ files, resolves each file's CRS independently,
reprojects to lat/lon, normalizes elevation to meters, filters and cleans
the merged point cloud, then executes one of three query modes:
  - peak: find the local maximum elevation
  - point: find the closest LiDAR return to the query location
  - search: interactive 3D point picking
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
    las_path: str | Path | list[str | Path],
    options: QueryOptions | None = None,
) -> QueryResult | None:
    """Query one or more LiDAR datasets at or around a geographic point.

    When multiple files are provided, each is loaded and reprojected
    independently (they may have different CRS), then merged before
    querying. Elevations are normalized to meters internally.

    :param query_lat: Latitude of query in degrees (-90 to 90).
    :param query_lon: Longitude of query in degrees (-180 to 180).
    :param query_type: Query algorithm ("peak", "point", or "search").
    :param las_path: Path(s) to LAS/LAZ file(s). Single path or list.
    :param options: Query configuration. Uses defaults if None.
    :return: QueryResult with elevation and position, or None if cancelled/no data.
    :raises FileNotFoundError: If any LAS file does not exist.
    :raises ValueError: If CRS cannot be determined for any file.
    """
    if options is None:
        options = QueryOptions()

    if isinstance(query_type, str):
        query_type = QueryType(query_type.lower())

    paths = _normalize_paths(las_path)

    cloud, display_crs = _load_and_merge(paths, options)
    logger.info("Total loaded: %d points from %d file(s)", cloud.n_points, len(paths))

    if options.center:
        query_lat = float(np.mean(cloud.lat))
        query_lon = float(np.mean(cloud.lon))
        logger.info("Using dataset center: lat=%.6f, lon=%.6f", query_lat, query_lon)

    cloud = _spatial_filter(cloud, query_lat, query_lon, options.query_radius_deg)
    if cloud.n_points == 0:
        logger.warning("Query point is not within the LAS data extent")
        return None

    logger.info("Points within query radius: %d", cloud.n_points)

    if options.clean_data:
        cloud = cleaning.clean(
            cloud,
            num_neighbors=options.filter_num_neighbors,
            std_ratio=options.filter_threshold,
        )
        logger.info("After cleaning: %d points", cloud.n_points)

    if query_type == QueryType.PEAK:
        return _query_peak(cloud, display_crs)
    elif query_type == QueryType.POINT:
        return _query_point(cloud, display_crs, query_lat, query_lon)
    elif query_type == QueryType.SEARCH:
        return _query_search(cloud, display_crs, options)
    else:
        raise ValueError(f"Unknown query type: {query_type}")


def _normalize_paths(las_path: str | Path | list[str | Path]) -> list[Path]:
    """Validate and normalize file path(s).

    :param las_path: Single path or list of paths.
    :return: List of validated Path objects.
    """
    if isinstance(las_path, (str, Path)):
        paths = [Path(las_path)]
    else:
        paths = [Path(p) for p in las_path]

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"LAS file not found: {p}")
        if p.suffix.lower() not in (".las", ".laz"):
            raise ValueError(f"Expected .las or .laz file, got: {p.suffix}")

    return paths


def _load_and_merge(
    paths: list[Path],
    options: QueryOptions,
) -> tuple[PointCloudData, CrsInfo]:
    """Load one or more LAS/LAZ files and merge into a single cloud.

    Each file is read with its own CRS, reprojected to lat/lon, and
    elevation is normalized to meters. The merged cloud uses lat/lon as
    its x/y coordinates.

    :param paths: List of LAS/LAZ file paths.
    :param options: Query options (for filter_ground, dem_path, force_unit).
    :return: Tuple of (merged PointCloudData in meters, CrsInfo for display).
    """
    all_lat: list[np.ndarray] = []
    all_lon: list[np.ndarray] = []
    all_z_m: list[np.ndarray] = []
    all_cls: list[np.ndarray] = []
    all_int: list[np.ndarray] = []
    has_cls = True
    has_int = True
    single_crs: CrsInfo | None = None

    for p in paths:
        crs_info = resolve_crs(p, dem_path=options.dem_path, force_unit=options.force_unit)
        logger.info("  %s: CRS=%s, unit=%s", p.name, crs_info.source, crs_info.unit_name)

        cloud = _read_point_cloud(p, filter_ground=options.filter_ground)
        logger.info("  %s: %d points loaded", p.name, cloud.n_points)

        cloud = _reproject(cloud, crs_info)

        # Normalize Z to meters
        if crs_info.is_foot:
            z_m = cloud.z.astype(np.float64) * crs_info.unit_to_meters
        else:
            z_m = cloud.z.astype(np.float64)

        all_lat.append(cloud.lat)
        all_lon.append(cloud.lon)
        all_z_m.append(z_m)

        if cloud.classification is not None:
            all_cls.append(cloud.classification)
        else:
            has_cls = False
        if cloud.intensity is not None:
            all_int.append(cloud.intensity)
        else:
            has_int = False

        # For single-file case, keep the original CRS for display
        if len(paths) == 1:
            single_crs = crs_info

    lat = np.concatenate(all_lat)
    lon = np.concatenate(all_lon)
    z_m = np.concatenate(all_z_m)

    merged = PointCloudData(
        x=lat,
        y=lon,
        z=z_m,
        lat=lat,
        lon=lon,
        classification=np.concatenate(all_cls) if has_cls else None,
        intensity=np.concatenate(all_int) if has_int else None,
    )

    # Z is always meters after merge; track original unit for axis labeling
    original_is_foot = single_crs.is_foot if single_crs is not None else False
    display_crs = CrsInfo(
        crs=None,
        unit_name="metre",
        is_geographic=True,
        source=single_crs.source if single_crs is not None else "merged",
    )
    display_crs._original_is_foot = original_is_foot

    return merged, display_crs


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
    z_value: float,
    lat: float,
    lon: float,
    crs_info: CrsInfo,
    query_type: QueryType,
) -> QueryResult:
    """Convert an elevation to a QueryResult with both m and ft.

    For single-file queries the z_value is in the file's native unit.
    For multi-file merges the z_value is already in meters.

    :param z_value: Elevation value.
    :param lat: Latitude in degrees.
    :param lon: Longitude in degrees.
    :param crs_info: CRS info for unit interpretation and display.
    :param query_type: The query mode that produced this result.
    :return: QueryResult with elevation in both meters and feet.
    """
    if crs_info.is_foot:
        elevation_ft = float(z_value)
        elevation_m = float(z_value) * crs_info.unit_to_meters
    else:
        elevation_m = float(z_value)
        elevation_ft = float(z_value) * M2FT

    return QueryResult(
        elevation_m=elevation_m,
        elevation_ft=elevation_ft,
        location_local=(float(lat), float(lon)),
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
        z_value=cloud.z[ix],
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
        z_value=cloud.z[ix],
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
        z_value=pick.z,
        lat=pick.lat,
        lon=pick.lon,
        crs_info=crs_info,
        query_type=QueryType.SEARCH,
    )
