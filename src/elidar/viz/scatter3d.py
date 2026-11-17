"""Interactive 3D scatter plot with point picking and zoom loop.

Replaces the MATLAB search mode (scatter3 + datacursormode + MFquestdlg)
with a pyvista-based workflow:

  1. Render the point cloud as a 3D scatter plot colored by elevation
  2. User clicks on a point (point picking)
  3. User presses a key: Z=Zoom, D/Enter=Done, Escape=Cancel
  4. If Zoom: crop point cloud around picked point, re-render
  5. If Done: return the picked point
  6. If Cancel: return None

When the point count exceeds DISPLAY_THRESHOLD, a random subsample is
shown for fast interaction. The user zooms in until the full local
point set is small enough to display directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pyvista as pv

from elidar.core.crs import CrsInfo, make_transformer
from elidar.core.types import ColorScale, M2FT, PointCloudData, QueryOptions
from elidar.viz.colormaps import get_colormap

logger = logging.getLogger(__name__)

DISPLAY_THRESHOLD = 50_000


@dataclass
class PickResult:
    """Result from interactive point selection.

    :param z: Elevation in meters.
    :param x_local: X coordinate (lat after merge).
    :param y_local: Y coordinate (lon after merge).
    :param lat: Latitude in degrees.
    :param lon: Longitude in degrees.
    :param index: Index into the point cloud arrays.
    """

    z: float
    x_local: float
    y_local: float
    lat: float
    lon: float
    index: int


class _UserAction(Enum):
    ZOOM = auto()
    DONE = auto()
    CANCEL = auto()
    NONE = auto()


class _PickState:
    """Mutable state shared between the plotter callbacks."""

    def __init__(self) -> None:
        self.picked_point: np.ndarray | None = None
        self.action: _UserAction = _UserAction.NONE

    def reset_for_iteration(self) -> None:
        self.picked_point = None
        self.action = _UserAction.NONE


def interactive_search(
    cloud: PointCloudData,
    crs_info: CrsInfo,
    options: QueryOptions,
) -> PickResult | None:
    """Run the interactive 3D point selection with zoom loop.

    :param cloud: Filtered and cleaned point cloud with lat/lon populated.
    :param crs_info: CRS info for display and unit labeling.
    :param options: Query options (colorscale, zoom_scale, display_lla).
    :return: PickResult for the selected point, or None if cancelled.
    """
    state = _PickState()

    zoom_radius = options.query_radius_deg / 2
    working_cloud = cloud

    while True:
        state.reset_for_iteration()

        is_subsampled = working_cloud.n_points > DISPLAY_THRESHOLD
        if is_subsampled:
            display_cloud = _subsample(working_cloud, DISPLAY_THRESHOLD)
        else:
            display_cloud = working_cloud

        display_x, display_y = _show_plotter(
            display_cloud, crs_info, options, state, is_subsampled,
        )

        if state.action in (_UserAction.CANCEL, _UserAction.NONE):
            logger.info("Search cancelled")
            return None

        if state.picked_point is None:
            logger.warning("No point selected, cancelling")
            return None

        pick_idx = _find_picked_index(
            display_cloud, state.picked_point, display_x, display_y,
        )

        if state.action == _UserAction.DONE and not is_subsampled:
            logger.info(
                "Point selected: z=%.4f at lat=%.6f, lon=%.6f",
                display_cloud.z[pick_idx],
                display_cloud.lat[pick_idx],
                display_cloud.lon[pick_idx],
            )
            return PickResult(
                z=float(display_cloud.z[pick_idx]),
                x_local=float(display_cloud.x[pick_idx]),
                y_local=float(display_cloud.y[pick_idx]),
                lat=float(display_cloud.lat[pick_idx]),
                lon=float(display_cloud.lon[pick_idx]),
                index=pick_idx,
            )

        if state.action == _UserAction.DONE and is_subsampled:
            # User pressed Done but we're still subsampled -- treat as zoom
            state.action = _UserAction.ZOOM

        # ZOOM: crop the FULL working cloud around picked point in lat/lon space
        zoom_radius *= options.zoom_scale
        clat = display_cloud.lat[pick_idx]
        clon = display_cloud.lon[pick_idx]
        dist = np.sqrt(
            (working_cloud.lat - clat) ** 2 + (working_cloud.lon - clon) ** 2
        )
        mask = dist <= zoom_radius
        working_cloud = working_cloud.subset(mask)

        if working_cloud.n_points == 0:
            logger.warning("Zoom resulted in empty point cloud")
            return None

        logger.info(
            "Zoomed: radius=%.6f deg, %d points remaining",
            zoom_radius,
            working_cloud.n_points,
        )


def _subsample(cloud: PointCloudData, n: int) -> PointCloudData:
    """Randomly subsample a point cloud to at most n points.

    :param cloud: Input point cloud.
    :param n: Maximum number of points to keep.
    :return: Subsampled point cloud.
    """
    if cloud.n_points <= n:
        return cloud
    rng = np.random.default_rng(42)
    indices = rng.choice(cloud.n_points, size=n, replace=False)
    mask = np.zeros(cloud.n_points, dtype=bool)
    mask[indices] = True
    return cloud.subset(mask)


def _show_plotter(
    cloud: PointCloudData,
    crs_info: CrsInfo,
    options: QueryOptions,
    state: _PickState,
    is_subsampled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Create and show the pyvista plotter, blocking until the user acts.

    :param cloud: Point cloud to display (may be a subsample).
    :param crs_info: CRS info for axis labels.
    :param options: Display options.
    :param state: Mutable state for pick tracking.
    :param is_subsampled: Whether the displayed cloud is a subsample.
    :return: (display_x, display_y) arrays in local meters.
    """
    center_lat = float(np.mean(cloud.lat))
    center_lon = float(np.mean(cloud.lon))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(center_lat))

    plot_x = (cloud.lat - center_lat) * m_per_deg_lat
    plot_y = (cloud.lon - center_lon) * m_per_deg_lon
    plot_z = cloud.z.astype(np.float64)

    points = np.column_stack([plot_x, plot_y, plot_z])
    polydata = pv.PolyData(points)
    polydata["elevation"] = plot_z

    cmap = get_colormap(options.colorscale)

    plotter = pv.Plotter(title="ELiDAR Search")

    plotter.add_points(
        polydata,
        scalars="elevation",
        cmap=cmap,
        point_size=5,
        render_points_as_spheres=True,
    )

    plotter.add_axes(
        xlabel="South-North (m)",
        ylabel="West-East (m)",
        zlabel="Altitude (m)",
    )

    plotter.camera_position = "xz"
    plotter.camera.elevation = 25
    plotter.camera.azimuth = -60

    _add_instructions(plotter, cloud.n_points, is_subsampled)

    def on_pick(point: np.ndarray) -> None:
        state.picked_point = point
        real_lat = point[0] / m_per_deg_lat + center_lat
        real_lon = point[1] / m_per_deg_lon + center_lon
        z_m = point[2]
        _update_status(plotter, real_lat, real_lon, z_m)

    plotter.enable_point_picking(
        callback=on_pick,
        left_clicking=True,
        picker="point",
        show_point=True,
        color="red",
        point_size=15,
        show_message=False,
        tolerance=0.025,
    )

    def on_zoom() -> None:
        if state.picked_point is not None:
            state.action = _UserAction.ZOOM
            plotter.close()
        else:
            _flash_message(plotter, "Pick a point first, then press Z")

    def on_done() -> None:
        if state.picked_point is not None:
            state.action = _UserAction.DONE
            plotter.close()
        elif is_subsampled:
            _flash_message(plotter, "Zoom in further before selecting (Z)")
        else:
            _flash_message(plotter, "Pick a point first, then press D")

    def on_cancel() -> None:
        state.action = _UserAction.CANCEL
        plotter.close()

    plotter.add_key_event("z", on_zoom)
    plotter.add_key_event("d", on_done)
    plotter.add_key_event("Return", on_done)
    plotter.add_key_event("Escape", on_cancel)

    plotter.show()

    return plot_x, plot_y


def _add_instructions(plotter: pv.Plotter, n_points: int, is_subsampled: bool) -> None:
    """Add instruction text overlay to the plotter."""
    if is_subsampled:
        mode_text = "CLICK TO DOWNSELECT"
        mode_color = "red"
        hint = f"Showing {n_points:,} of many -- zoom in to see all points"
    else:
        mode_text = "CLICK TO SELECT"
        mode_color = "green"
        hint = f"{n_points:,} points (full resolution)"

    plotter.add_text(
        mode_text,
        position="upper_right",
        font_size=14,
        color=mode_color,
        shadow=True,
        name="mode",
    )

    instructions = (
        f"{hint}\n"
        "Z = Zoom around selection\n"
        "D / Enter = Done (accept selection)\n"
        "Escape = Cancel"
    )
    plotter.add_text(
        instructions,
        position="upper_left",
        font_size=10,
        color="white",
        shadow=True,
        name="instructions",
    )


def _update_status(plotter: pv.Plotter, lat: float, lon: float, z_m: float) -> None:
    """Update the status text after a point is picked."""
    elev_str = f"{z_m:.2f} m ({z_m * M2FT:.2f} ft)"
    status = f"Selected: {lat:.6f}°N, {lon:.6f}°W  Elev: {elev_str}"
    plotter.add_text(
        status,
        position="lower_left",
        font_size=10,
        color="yellow",
        shadow=True,
        name="status",
    )


def _flash_message(plotter: pv.Plotter, message: str) -> None:
    """Show a temporary warning message."""
    plotter.add_text(
        message,
        position="lower_right",
        font_size=10,
        color="red",
        shadow=True,
        name="warning",
    )


def _find_picked_index(
    cloud: PointCloudData,
    picked_display: np.ndarray,
    display_x: np.ndarray,
    display_y: np.ndarray,
) -> int:
    """Find the index of the picked point in the point cloud.

    :param cloud: The point cloud being displayed.
    :param picked_display: The 3D coordinates from the plotter (local_m_x, local_m_y, z).
    :param display_x: The x display array (local meters, same length as cloud).
    :param display_y: The y display array (local meters, same length as cloud).
    :return: Index into the cloud arrays.
    """
    dist = (
        (display_x - picked_display[0]) ** 2
        + (display_y - picked_display[1]) ** 2
        + (cloud.z - picked_display[2]) ** 2
    )
    return int(np.argmin(dist))
