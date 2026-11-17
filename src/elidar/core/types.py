"""Data types for ELiDAR query options and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class QueryType(str, Enum):
    """LiDAR query algorithm."""

    SEARCH = "search"
    PEAK = "peak"
    POINT = "point"


class ColorScale(str, Enum):
    """Color mapping for 3D scatter plots."""

    JET = "jet"
    TURBO = "turbo"
    WINTER = "winter"
    GREYSCALE = "greyscale"
    BR = "br"


M2FT = 3.2808398950131233

ASPRS_CLASSIFICATION_NAMES: dict[int, str] = {
    0: "Never Classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Low Point (Noise)",
    8: "Model Key-point",
    9: "Water",
    10: "Rail",
    11: "Road Surface",
    12: "Overlap",
    13: "Wire - Guard",
    14: "Wire - Conductor",
    15: "Transmission Tower",
    16: "Wire - Connector",
    17: "Bridge Deck",
    18: "High Noise",
}

DEFAULT_CLASS_CMAPS: dict[int, ColorScale] = {
    0: ColorScale.GREYSCALE,
    1: ColorScale.GREYSCALE,
    2: ColorScale.TURBO,
    7: ColorScale.WINTER,
    18: ColorScale.WINTER,
}
DEFAULT_CMAP = ColorScale.JET
US_SURVEY_FOOT_TO_M = 0.30480060960121924
INTL_FOOT_TO_M = 0.3048


@dataclass
class CrsInfo:
    """Resolved coordinate reference system information.

    :param crs: pyproj CRS object, or None if data is already in lat/lon.
    :param unit_name: Linear unit name from the CRS (e.g. "metre", "US survey foot").
    :param is_geographic: True if data is already in lat/lon (no projection needed).
    :param source: Where the CRS was resolved from, for diagnostics.
    """

    crs: object | None
    unit_name: str
    is_geographic: bool
    source: Literal["laz_vlr", "dem", "override", "lla_convention"]

    @property
    def is_foot(self) -> bool:
        """Whether the native unit is any kind of foot."""
        return "foot" in self.unit_name.lower() or "ft" in self.unit_name.lower()

    @property
    def unit_to_meters(self) -> float:
        """Conversion factor from native unit to meters."""
        lower = self.unit_name.lower()
        if "survey" in lower or "ftus" in lower:
            return US_SURVEY_FOOT_TO_M
        if "foot" in lower or "ft" in lower:
            return INTL_FOOT_TO_M
        return 1.0


@dataclass
class QueryOptions:
    """Configuration for a LiDAR query.

    :param clean_data: Run statistical outlier removal and downsampling.
    :param colorscale: Color mapping for search mode plots.
    :param filter_ground: Filter to classification == 2 (ground) only.
    :param query_radius_deg: Initial search radius around query point in degrees.
    :param zoom_scale: Factor to reduce point cloud radius on each zoom iteration.
    :param force_unit: Override detected unit ("ft", "m", or None for auto).
    :param filter_num_neighbors: Neighbor count for statistical outlier removal.
    :param filter_threshold: Std ratio threshold for statistical outlier removal.
    :param display_lla: Show lat/lon on plot axes instead of local coordinates.
    :param center: Use dataset center as query point (dev testing).
    :param dem_path: Path to DEM file for CRS fallback. Now optional.
    """

    clean_data: bool = True
    colorscale: ColorScale = ColorScale.JET
    filter_ground: bool = False
    query_radius_deg: float = 0.0008
    zoom_scale: float = 0.5
    force_unit: str | None = None
    filter_num_neighbors: int = 20
    filter_threshold: float = 1.0
    display_lla: bool = True
    center: bool = False
    dem_path: str | None = None


@dataclass
class QueryResult:
    """Result from a LiDAR query.

    :param elevation_m: Elevation in meters.
    :param elevation_ft: Elevation in feet.
    :param location_local: Position in the file's native CRS as (x, y).
    :param location_deg: Position in lat/lon degrees as (lat, lon).
    :param query_type: The query algorithm that produced this result.
    """

    elevation_m: float
    elevation_ft: float
    location_local: tuple[float, float]
    location_deg: tuple[float, float]
    query_type: QueryType
    classification: int | None = None


@dataclass
class PointCloudData:
    """In-memory point cloud extracted from a LAS/LAZ file.

    Coordinates are in the file's native CRS until reprojected.

    :param x: X coordinates in native CRS.
    :param y: Y coordinates in native CRS.
    :param z: Z coordinates (elevation) in native CRS units.
    :param lat: Latitude in degrees (populated after reprojection).
    :param lon: Longitude in degrees (populated after reprojection).
    :param classification: LiDAR classification codes, if available.
    :param intensity: Return intensity values, if available.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    lat: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    lon: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    classification: NDArray[np.uint8] | None = None
    intensity: NDArray[np.uint16] | None = None

    @property
    def n_points(self) -> int:
        """Number of points in the cloud."""
        return len(self.x)

    def subset(self, mask: NDArray[np.bool_]) -> PointCloudData:
        """Return a new PointCloudData containing only points where mask is True.

        :param mask: Boolean array of length n_points.
        :return: Filtered point cloud.
        """
        return PointCloudData(
            x=self.x[mask],
            y=self.y[mask],
            z=self.z[mask],
            lat=self.lat[mask] if len(self.lat) > 0 else np.array([]),
            lon=self.lon[mask] if len(self.lon) > 0 else np.array([]),
            classification=self.classification[mask] if self.classification is not None else None,
            intensity=self.intensity[mask] if self.intensity is not None else None,
        )
