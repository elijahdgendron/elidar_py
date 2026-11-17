"""Point cloud denoising and downsampling using Open3D.

Replaces MATLAB's pcdenoise (statistical outlier removal) and
pcdownsample (voxel grid averaging).
"""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from elidar.core.types import PointCloudData

logger = logging.getLogger(__name__)


def denoise(
    cloud: PointCloudData,
    num_neighbors: int = 20,
    std_ratio: float = 1.0,
) -> PointCloudData:
    """Remove statistical outliers from a point cloud.

    Equivalent to MATLAB's pcdenoise(ptCloud, 'NumNeighbors', N, 'Threshold', T).
    For each point, computes the mean distance to its k nearest neighbors.
    Points whose mean distance exceeds (global_mean + std_ratio * global_std)
    are removed.

    :param cloud: Input point cloud.
    :param num_neighbors: Number of neighbors for distance calculation.
    :param std_ratio: Standard deviation multiplier for the threshold.
    :return: Cleaned point cloud with outliers removed.
    """
    if cloud.n_points == 0:
        return cloud

    pcd = _to_open3d(cloud)
    _, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=num_neighbors,
        std_ratio=std_ratio,
    )

    mask = np.zeros(cloud.n_points, dtype=bool)
    mask[inlier_indices] = True

    n_removed = cloud.n_points - len(inlier_indices)
    logger.debug("Denoised: removed %d / %d points", n_removed, cloud.n_points)

    return cloud.subset(mask)


def downsample(
    cloud: PointCloudData,
    voxel_size: float = 0.01,
) -> PointCloudData:
    """Downsample a point cloud using voxel grid averaging.

    Equivalent to MATLAB's pcdownsample(ptCloud, 'gridAverage', voxelSize).
    Divides space into a voxel grid and replaces all points in each voxel
    with their centroid.

    :param cloud: Input point cloud.
    :param voxel_size: Edge length of each voxel in the same units as the cloud.
    :return: Downsampled point cloud.
    """
    if cloud.n_points == 0:
        return cloud

    pcd = _to_open3d(cloud)
    _, _, indices_map = pcd.voxel_down_sample_and_trace(voxel_size)

    keep = set()
    for voxel_indices in indices_map:
        indices = list(voxel_indices)
        if indices:
            keep.add(indices[0])

    mask = np.zeros(cloud.n_points, dtype=bool)
    mask[list(keep)] = True

    logger.debug(
        "Downsampled: %d -> %d points (voxel_size=%.4f)",
        cloud.n_points,
        mask.sum(),
        voxel_size,
    )

    return cloud.subset(mask)


def clean(
    cloud: PointCloudData,
    num_neighbors: int = 20,
    std_ratio: float = 1.0,
    voxel_size: float = 0.01,
) -> PointCloudData:
    """Run full cleaning pipeline: denoise then downsample.

    :param cloud: Input point cloud.
    :param num_neighbors: Neighbor count for denoising.
    :param std_ratio: Threshold for denoising.
    :param voxel_size: Voxel size for downsampling.
    :return: Cleaned and downsampled point cloud.
    """
    cloud = denoise(cloud, num_neighbors=num_neighbors, std_ratio=std_ratio)
    cloud = downsample(cloud, voxel_size=voxel_size)
    return cloud


def _to_open3d(cloud: PointCloudData) -> o3d.geometry.PointCloud:
    """Convert a PointCloudData to an Open3D PointCloud.

    :param cloud: ELiDAR point cloud.
    :return: Open3D point cloud with XYZ positions.
    """
    points = np.column_stack([
        cloud.x.astype(np.float64),
        cloud.y.astype(np.float64),
        cloud.z.astype(np.float64),
    ])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd
