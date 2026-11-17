"""Interactive 3D scatter plot with point picking and zoom loop.

Replaces the MATLAB search mode (scatter3 + datacursormode + MFquestdlg)
with a pyvista-based workflow:

  1. Render the point cloud as a 3D scatter plot colored by elevation
  2. User clicks on a point (point picking)
  3. User presses a key: Z=Zoom, D/Enter=Done, Escape=Cancel
  4. If Zoom: crop point cloud around picked point, re-render
  5. If Done: return the picked point
  6. If Cancel: return None
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


@dataclass
class PickResult:
    """Result from interactive point selection.

    :param z: Elevation in the file's native unit.
    :param x_local: X coordinate in the file's native CRS.
    :param y_local: Y coordinate in the file's native CRS.
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

    # Zoom radius in degrees (cloud coordinates are always lat/lon after merge)
    zoom_radius = options.query_radius_deg / 2

    working_cloud = cloud

    while True:
        state.reset_for_iteration()

        _show_plotter(working_cloud, crs_info, options, state)

        if state.action == _UserAction.CANCEL or state.action == _UserAction.NONE:
            logger.info("Search cancelled")
            return None

        if state.picked_point is None:
            logger.warning("No point selected, cancelling")
            return None

        # Find the index of the picked point in the working cloud
        pick_idx = _find_picked_index(working_cloud, state.picked_point)

        if state.action == _UserAction.DONE:
            logger.info(
                "Point selected: z=%.4f at lat=%.6f, lon=%.6f",
                working_cloud.z[pick_idx],
                working_cloud.lat[pick_idx],
                working_cloud.lon[pick_idx],
            )
            return PickResult(
                z=float(working_cloud.z[pick_idx]),
                x_local=float(working_cloud.x[pick_idx]),
                y_local=float(working_cloud.y[pick_idx]),
                lat=float(working_cloud.lat[pick_idx]),
                lon=float(working_cloud.lon[pick_idx]),
                index=pick_idx,
            )

        # ZOOM: crop around picked point in lat/lon space
        zoom_radius *= options.zoom_scale
        clat = working_cloud.lat[pick_idx]
        clon = working_cloud.lon[pick_idx]
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


def _show_plotter(
    cloud: PointCloudData,
    crs_info: CrsInfo,
    options: QueryOptions,
    state: _PickState,
) -> None:
    """Create and show the pyvista plotter, blocking until the user acts.

    :param cloud: Point cloud to display.
    :param crs_info: CRS info for axis labels.
    :param options: Display options.
    :param state: Mutable state for pick tracking.
    """
    # Display is always lat/lon since the merge step normalizes coordinates
    plot_x = cloud.lat.copy()
    plot_y = cloud.lon.copy()
    x_label = "Latitude (deg)"
    y_label = "Longitude (deg)"
    # Z is always meters internally; label based on original data unit
    original_ft = getattr(crs_info, "_original_is_foot", False)
    z_label = "Altitude (m)"

    points = np.column_stack([plot_x, plot_y, cloud.z.astype(np.float64)])
    polydata = pv.PolyData(points)
    polydata["elevation"] = cloud.z.astype(np.float64)

    cmap = get_colormap(options.colorscale)

    plotter = pv.Plotter(title="ELiDAR Search")

    plotter.add_points(
        polydata,
        scalars="elevation",
        cmap=cmap,
        point_size=5,
        render_points_as_spheres=True,
    )

    plotter.add_axes_at_origin(labels_off=True)
    plotter.add_axes(
        xlabel=x_label,
        ylabel=y_label,
        zlabel=z_label,
    )

    _add_instructions(plotter, cloud.n_points)

    # Point picking callback
    def on_pick(point: np.ndarray) -> None:
        state.picked_point = point
        _update_status(plotter, point, crs_info)

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

    # Key event callbacks
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


def _add_instructions(plotter: pv.Plotter, n_points: int) -> None:
    """Add instruction text overlay to the plotter."""
    instructions = (
        f"Points: {n_points:,}\n"
        "Click a point to select it\n"
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


def _update_status(plotter: pv.Plotter, point: np.ndarray, crs_info: CrsInfo) -> None:
    """Update the status text after a point is picked."""
    z_m = point[2]
    elev_str = f"{z_m:.2f} m ({z_m * M2FT:.2f} ft)"

    status = f"Selected: ({point[0]:.6f}, {point[1]:.6f})  Elev: {elev_str}"
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
    picked_point: np.ndarray,
) -> int:
    """Find the index of the picked point in the point cloud.

    The display coordinates are (lat, lon, z) matching the cloud's arrays.

    :param cloud: The point cloud being displayed.
    :param picked_point: The 3D coordinates from the plotter (lat, lon, z).
    :return: Index into the cloud arrays.
    """
    dist = (
        (cloud.lat - picked_point[0]) ** 2
        + (cloud.lon - picked_point[1]) ** 2
        + (cloud.z - picked_point[2]) ** 2
    )
    return int(np.argmin(dist))
