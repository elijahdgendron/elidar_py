"""Point cloud denoising and downsampling using Open3D.

Replaces MATLAB's pcdenoise (statistical outlier removal) and
pcdownsample (voxel grid averaging).

Since the merged point cloud uses lat/lon for x/y, all cleaning
operations convert to local meters first so that distances and
voxel grids are physically meaningful.
"""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from elidar.core.types import PointCloudData

logger = logging.getLogger(__name__)

_M_PER_DEG_LAT = 111_320.0


def clean(
    cloud: PointCloudData,
    num_neighbors: int = 20,
    std_ratio: float = 1.0,
    voxel_size: float = 0.01,
) -> PointCloudData:
    """Run full cleaning pipeline: denoise then downsample.

    Internally converts lat/lon to local meters so distances are
    physically meaningful, then maps results back to the original
    point indices.

    :param cloud: Input point cloud (x/y may be lat/lon degrees).
    :param num_neighbors: Neighbor count for denoising.
    :param std_ratio: Threshold for denoising.
    :param voxel_size: Voxel size in meters for downsampling.
    :return: Cleaned and downsampled point cloud.
    """
    if cloud.n_points == 0:
        return cloud

    local_xyz = _to_local_meters(cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(local_xyz)

    # Statistical outlier removal
    _, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=num_neighbors,
        std_ratio=std_ratio,
    )
    n_removed = cloud.n_points - len(inlier_idx)
    logger.debug("Denoised: removed %d / %d points", n_removed, cloud.n_points)

    # Subset for downsampling
    inlier_mask = np.zeros(cloud.n_points, dtype=bool)
    inlier_mask[inlier_idx] = True
    local_xyz = local_xyz[inlier_mask]

    # Voxel deduplication in metric space
    vx = (local_xyz[:, 0] / voxel_size).astype(np.int64)
    vy = (local_xyz[:, 1] / voxel_size).astype(np.int64)
    vz = (local_xyz[:, 2] / voxel_size).astype(np.int64)
    _, unique_in_inliers = np.unique(
        np.column_stack([vx, vy, vz]), axis=0, return_index=True,
    )

    # Map back to original indices
    inlier_original_idx = np.where(inlier_mask)[0]
    keep_idx = inlier_original_idx[unique_in_inliers]

    final_mask = np.zeros(cloud.n_points, dtype=bool)
    final_mask[keep_idx] = True

    logger.debug(
        "Downsampled: %d -> %d points (voxel_size=%.4f m)",
        len(inlier_idx),
        final_mask.sum(),
        voxel_size,
    )

    return cloud.subset(final_mask)


def _to_local_meters(cloud: PointCloudData) -> NDArray[np.float64]:
    """Convert point cloud x/y (lat/lon degrees) + z (meters) to local meters.

    Uses the cloud center as the origin. Returns an (N, 3) array in meters.

    :param cloud: Point cloud where x=lat, y=lon (degrees), z=meters.
    :return: (N, 3) array of (east_m, north_m, z_m).
    """
    center_lat = float(np.mean(cloud.lat)) if len(cloud.lat) > 0 else float(np.mean(cloud.x))
    center_lon = float(np.mean(cloud.lon)) if len(cloud.lon) > 0 else float(np.mean(cloud.y))
    m_per_deg_lon = _M_PER_DEG_LAT * np.cos(np.radians(center_lat))

    lat = cloud.lat if len(cloud.lat) > 0 else cloud.x
    lon = cloud.lon if len(cloud.lon) > 0 else cloud.y

    north_m = (lat - center_lat) * _M_PER_DEG_LAT
    east_m = (lon - center_lon) * m_per_deg_lon
    z_m = cloud.z.astype(np.float64)

    return np.column_stack([east_m, north_m, z_m])
