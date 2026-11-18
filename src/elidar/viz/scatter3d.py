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
from elidar.core.types import ColorScale, DEFAULT_CMAP, DEFAULT_CLASS_CMAPS, M2FT, PointCloudData, QueryOptions, SURFACE_CLASSES
from elidar.viz.colormaps import get_colormap, stretch_colormap

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
    classification: int | None = None


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
        scalar_bar_args={
            "title": "Elevation (m)",
            "n_labels": 2,
            "fmt": "%.0f",
        },
    )

    plotter.add_axes(
        xlabel="N",
        ylabel="E",
        zlabel="Up",
        line_width=2,
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
        elev_str = f"{z_m:.2f} m ({z_m * M2FT:.2f} ft)"
        _update_status(plotter, real_lat, real_lon, elev_str)

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


def _update_status(plotter: pv.Plotter, lat: float, lon: float, elev_str: str, cls_name: str = "") -> None:
    """Update the status text after a point is picked."""
    cls_part = f"[{cls_name}]  " if cls_name else ""
    status = f"{cls_part}{lat:.6f}°N, {lon:.6f}°W  {elev_str}"
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
    visible_mask: np.ndarray | None = None,
) -> int:
    """Find the index of the picked point in the point cloud.

    :param cloud: The point cloud being displayed.
    :param picked_display: The 3D coordinates from the plotter (local_m_x, local_m_y, z).
    :param display_x: The x display array (local meters, same length as cloud).
    :param display_y: The y display array (local meters, same length as cloud).
    :param visible_mask: Boolean mask of visible points. If provided, only
        visible points are considered for matching.
    :return: Index into the cloud arrays.
    """
    dist = (
        (display_x - picked_display[0]) ** 2
        + (display_y - picked_display[1]) ** 2
        + (cloud.z - picked_display[2]) ** 2
    )
    if visible_mask is not None:
        dist = np.where(visible_mask, dist, np.inf)
    return int(np.argmin(dist))


_M_PER_DEG_LAT = 111_320.0


class EmbeddedSearch:
    """Manages the search workflow inside an embedded pyvista plotter.

    Unlike the standalone ``interactive_search()`` which blocks in a loop,
    this class is event-driven: the GUI calls ``load()``, then ``zoom()``
    / ``done()`` / ``cancel()`` in response to button clicks or key presses.

    :param plotter: A pyvista plotter (``pyvistaqt.QtInteractor`` or ``pv.Plotter``).
    :param on_status: Callback invoked with a status string on each pick / zoom.
    :param on_result: Callback invoked with the final ``PickResult`` on done.
    """

    def __init__(
        self,
        plotter: pv.Plotter,
        on_status: callable | None = None,
        on_result: callable | None = None,
    ) -> None:
        self.plotter = plotter
        self.on_status = on_status
        self.on_result = on_result

        self._working_cloud: PointCloudData | None = None
        self._display_cloud: PointCloudData | None = None
        self._display_x: np.ndarray | None = None
        self._display_y: np.ndarray | None = None
        self._picked_point: np.ndarray | None = None
        self._pick_idx: int | None = None
        self._selected_lat: float | None = None
        self._selected_lon: float | None = None
        self._selected_z: float | None = None
        self._zoom_radius: float = 0.0
        self._options: QueryOptions | None = None
        self._crs_info: CrsInfo | None = None
        self._is_subsampled: bool = False
        self._center_lat: float = 0.0
        self._center_lon: float = 0.0
        self._m_per_deg_lon: float = 0.0
        self._active: bool = False
        self._max_points: int = DISPLAY_THRESHOLD
        self._display_unit: str = "m"
        self._unit_scale: float = 1.0
        self._z_exaggeration: float = 1.0
        self._color_stretch: float = 1.0
        self._point_size: int = 5
        self._font_color: str = "black"
        self._base_plot_z: np.ndarray | None = None
        self._polydatas: list[pv.PolyData] = []
        self._class_cmaps: dict[int, ColorScale] = {}
        self._class_actors: dict[int, object] = {}
        self._visible_classes: set[int] | None = None
        self._marker_actor: object | None = None
        self._polydata: pv.PolyData | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_subsampled(self) -> bool:
        return self._is_subsampled

    @property
    def visible_point_count(self) -> int:
        """Number of visible points in the working cloud."""
        wc = self._working_cloud
        if wc is None:
            return 0
        if self._visible_classes is not None and wc.classification is not None:
            return int(np.isin(wc.classification, list(self._visible_classes)).sum())
        return wc.n_points

    def load(
        self,
        cloud: PointCloudData,
        crs_info: CrsInfo,
        options: QueryOptions,
        display_unit: str = "m",
        unit_scale: float = 1.0,
        class_cmaps: dict[int, ColorScale] | None = None,
        visible_classes: set[int] | None = None,
        max_points: int = DISPLAY_THRESHOLD,
    ) -> None:
        """Load a point cloud and render it.

        :param cloud: Cleaned point cloud with lat/lon populated.
        :param crs_info: CRS info for display.
        :param options: Query options.
        :param display_unit: Unit label ("ft", "ftUS", "m").
        :param unit_scale: Meters-to-display-unit factor.
        :param class_cmaps: Per-classification colormap overrides.
        :param visible_classes: Set of classification codes to display. None = all.
        :param max_points: Maximum points per classification to display.
        """
        self._working_cloud = cloud
        self._original_cloud = cloud
        self._options = options
        self._crs_info = crs_info
        # Zoom radius in meters for circular (not elliptical) cropping.
        center_lat = float(np.mean(cloud.lat))
        m_per_deg_lon = _M_PER_DEG_LAT * np.cos(np.radians(center_lat))
        lat_span_m = float(cloud.lat.max() - cloud.lat.min()) * _M_PER_DEG_LAT
        lon_span_m = float(cloud.lon.max() - cloud.lon.min()) * m_per_deg_lon
        data_radius_m = max(lat_span_m, lon_span_m) / 2
        config_radius_m = options.query_radius_deg / 2 * _M_PER_DEG_LAT
        self._zoom_radius_m = min(config_radius_m, data_radius_m)
        self._picked_point = None
        self._pick_idx = None
        self._selected_lat = None
        self._selected_lon = None
        self._selected_z = None
        self._active = True
        self._max_points = max_points
        self._preserve_camera = False
        self._display_unit = display_unit
        self._unit_scale = unit_scale
        self._class_cmaps = class_cmaps if class_cmaps is not None else {}
        self._visible_classes = visible_classes
        self._render()

    def zoom(self, scale: float | None = None) -> None:
        """Zoom around the currently picked/selected point.

        :param scale: Radius multiplier (0-1). None uses options.zoom_scale.
        """
        if not self._active or self._selected_lat is None:
            return

        if scale is None:
            scale = self._options.zoom_scale
        self._zoom_radius_m *= scale
        clat = self._selected_lat
        clon = self._selected_lon
        # Distance in meters so the crop is a true circle on the ground
        m_per_deg_lon = _M_PER_DEG_LAT * np.cos(np.radians(clat))
        dlat_m = (self._working_cloud.lat - clat) * _M_PER_DEG_LAT
        dlon_m = (self._working_cloud.lon - clon) * m_per_deg_lon
        dist_m = np.sqrt(dlat_m ** 2 + dlon_m ** 2)
        mask = dist_m <= self._zoom_radius_m
        self._working_cloud = self._working_cloud.subset(mask)

        if self._working_cloud.n_points == 0:
            self._emit_status("Zoom resulted in empty cloud")
            self.cancel()
            return

        logger.info(
            "Zoomed: radius=%.1f m, %d points remaining",
            self._zoom_radius_m,
            self._working_cloud.n_points,
        )

        self._preserve_camera = True
        self._render()
        self._restore_selection()

    def find_peak(self) -> None:
        """Select the highest visible point and highlight it."""
        if not self._active or self._display_cloud is None:
            return

        cloud = self._display_cloud

        # Build visible mask
        visible_mask = None
        if self._visible_classes is not None and cloud.classification is not None:
            visible_mask = np.isin(cloud.classification, list(self._visible_classes))
            if not np.any(visible_mask):
                self._emit_status("No visible points to search")
                return

        z = cloud.z.copy()
        if visible_mask is not None:
            z[~visible_mask] = -np.inf
        idx = int(np.argmax(z))

        # Set as the current selection
        self._pick_idx = idx
        self._selected_lat = float(cloud.lat[idx])
        self._selected_lon = float(cloud.lon[idx])
        self._selected_z = float(cloud.z[idx])

        # Build a fake display-space picked point for consistency
        s = self._unit_scale
        self._picked_point = np.array([
            self._display_x[idx],
            self._display_y[idx],
            cloud.z[idx] * s * self._z_exaggeration,
        ])

        self._show_selection_marker()

        # Emit status
        from elidar.core.types import ASPRS_CLASSIFICATION_NAMES
        cls_name = ""
        if cloud.classification is not None:
            cls_name = ASPRS_CLASSIFICATION_NAMES.get(
                int(cloud.classification[idx]), f"Class {cloud.classification[idx]}"
            )
        z_m = self._selected_z
        if self._display_unit != "m":
            elev_display = f"{z_m * self._unit_scale:.2f} {self._display_unit}"
        else:
            elev_display = f"{z_m:.2f} m"
        elev_log = f"{z_m:.2f} m ({z_m * M2FT:.2f} ft)"
        _update_status(self.plotter, self._selected_lat, self._selected_lon, elev_display, cls_name)
        self._emit_status(
            f"Peak found: {cls_name}  {self._selected_lat:.6f}°N, "
            f"{self._selected_lon:.6f}°W  Elev: {elev_log}"
        )

    def find_saddle(self) -> None:
        """Fit a quadratic surface and select the saddle point."""
        if not self._active or self._display_cloud is None:
            return

        cloud = self._display_cloud

        # Get visible points in local meters (always work in meters)
        visible_mask = None
        if self._visible_classes is not None and cloud.classification is not None:
            visible_mask = np.isin(cloud.classification, list(self._visible_classes))
            if not np.any(visible_mask):
                self._emit_status("No visible points for saddle fit")
                return

        # Local meter coordinates (unscaled by display unit)
        x_m = (cloud.lat - self._center_lat) * _M_PER_DEG_LAT
        y_m = (cloud.lon - self._center_lon) * self._m_per_deg_lon
        z_m = cloud.z.astype(np.float64)

        if visible_mask is not None:
            x_m = x_m[visible_mask]
            y_m = y_m[visible_mask]
            z_m = z_m[visible_mask]

        if len(z_m) < 50:
            self._emit_status("Too few points for saddle fit (need >= 50)")
            return

        # Fit z = ax² + bxy + cy² + dx + ey + f
        A = np.column_stack([x_m**2, x_m * y_m, y_m**2, x_m, y_m, np.ones_like(x_m)])
        coeffs, _, _, _ = np.linalg.lstsq(A, z_m, rcond=None)
        a, b, c, d, e, f_coeff = coeffs

        # Hessian: [[2a, b], [b, 2c]]
        det_H = 4 * a * c - b ** 2
        if det_H >= 0:
            self._emit_status("No saddle found — surface is a bowl or dome here. Try a different area.")
            return

        # Critical point (gradient = 0)
        H = np.array([[2 * a, b], [b, 2 * c]])
        try:
            critical = np.linalg.solve(H, np.array([-d, -e]))
        except np.linalg.LinAlgError:
            self._emit_status("Saddle fit failed — degenerate surface")
            return

        cx_m, cy_m = critical[0], critical[1]

        # Check critical point is within data extent
        margin = 0.1 * max(x_m.max() - x_m.min(), y_m.max() - y_m.min())
        if (cx_m < x_m.min() - margin or cx_m > x_m.max() + margin or
                cy_m < y_m.min() - margin or cy_m > y_m.max() + margin):
            self._emit_status("Saddle point is outside the visible area. Try panning or zooming differently.")
            return

        # Saddle elevation from the fit
        saddle_z_m = float(
            a * cx_m**2 + b * cx_m * cy_m + c * cy_m**2 + d * cx_m + e * cy_m + f_coeff
        )
        saddle_lat = cx_m / _M_PER_DEG_LAT + self._center_lat
        saddle_lon = cy_m / self._m_per_deg_lon + self._center_lon

        # Find nearest LiDAR point to the saddle in the full display cloud
        all_x_m = (cloud.lat - self._center_lat) * _M_PER_DEG_LAT
        all_y_m = (cloud.lon - self._center_lon) * self._m_per_deg_lon
        dist = (all_x_m - cx_m) ** 2 + (all_y_m - cy_m) ** 2
        if visible_mask is not None:
            dist = np.where(visible_mask, dist, np.inf)
        nearest_idx = int(np.argmin(dist))

        # Set as current selection
        self._pick_idx = nearest_idx
        self._selected_lat = float(cloud.lat[nearest_idx])
        self._selected_lon = float(cloud.lon[nearest_idx])
        self._selected_z = float(cloud.z[nearest_idx])

        s = self._unit_scale
        self._picked_point = np.array([
            self._display_x[nearest_idx],
            self._display_y[nearest_idx],
            cloud.z[nearest_idx] * s * self._z_exaggeration,
        ])

        # Show the fitted surface as a transparent mesh
        self._show_saddle_surface(x_m, y_m, coeffs)
        self._show_selection_marker()
        self._show_selection_status()

        # Emit detailed saddle info
        if self._display_unit != "m":
            fit_elev = f"{saddle_z_m * self._unit_scale:.2f} {self._display_unit}"
            near_elev = f"{self._selected_z * self._unit_scale:.2f} {self._display_unit}"
        else:
            fit_elev = f"{saddle_z_m:.2f} m"
            near_elev = f"{self._selected_z:.2f} m"

        self._emit_status(
            f"Saddle fit: {saddle_lat:.6f}°N, {saddle_lon:.6f}°W  "
            f"Fit elev: {fit_elev}  |  "
            f"Nearest point elev: {near_elev}"
        )

    def _show_saddle_surface(
        self,
        x_m: np.ndarray,
        y_m: np.ndarray,
        coeffs: np.ndarray,
    ) -> None:
        """Render the fitted quadratic surface as a transparent mesh."""
        a, b, c, d, e, f_coeff = coeffs
        s = self._unit_scale

        # Create a grid over the data extent
        n_grid = 40
        xi = np.linspace(x_m.min(), x_m.max(), n_grid)
        yi = np.linspace(y_m.min(), y_m.max(), n_grid)
        xx, yy = np.meshgrid(xi, yi)
        zz = a * xx**2 + b * xx * yy + c * yy**2 + d * xx + e * yy + f_coeff

        # Convert to display coordinates
        px = xx.ravel() * s
        py = yy.ravel() * s
        pz = zz.ravel() * s * self._z_exaggeration

        grid = pv.StructuredGrid()
        grid.points = np.column_stack([px, py, pz])
        grid.dimensions = [n_grid, n_grid, 1]

        self.plotter.add_mesh(
            grid,
            color="cyan",
            opacity=0.3,
            show_scalar_bar=False,
            name="saddle_surface",
        )
        self.plotter.render()

    def _restore_selection(self) -> None:
        """Re-highlight the persisted selection after a re-render."""
        if self._selected_lat is None:
            return

        cloud = self._display_cloud
        if cloud is None or cloud.n_points == 0:
            return

        # Find the closest point in the new display cloud
        dist = (
            (cloud.lat - self._selected_lat) ** 2
            + (cloud.lon - self._selected_lon) ** 2
            + (cloud.z - self._selected_z) ** 2
        )
        idx = int(np.argmin(dist))
        self._pick_idx = idx

        s = self._unit_scale
        self._picked_point = np.array([
            self._display_x[idx],
            self._display_y[idx],
            cloud.z[idx] * s * self._z_exaggeration,
        ])

        self._show_selection_marker()
        self._show_selection_status()

    def _show_selection_status(self) -> None:
        """Update the yellow status text for the current selection."""
        if self._selected_lat is None or self._display_cloud is None:
            return

        from elidar.core.types import ASPRS_CLASSIFICATION_NAMES
        cloud = self._display_cloud
        cls_name = ""
        if cloud.classification is not None and self._pick_idx is not None:
            cls_code = int(cloud.classification[self._pick_idx])
            cls_name = ASPRS_CLASSIFICATION_NAMES.get(cls_code, f"Class {cls_code}")

        z_m = self._selected_z
        if self._display_unit != "m":
            elev_display = f"{z_m * self._unit_scale:.2f} {self._display_unit}"
        else:
            elev_display = f"{z_m:.2f} m"
        _update_status(self.plotter, self._selected_lat, self._selected_lon, elev_display, cls_name)

    def _show_selection_marker(self) -> None:
        """Show a marker at the current _selected_lat/lon/z."""
        if self._selected_lat is None:
            return

        # Remove old marker if it exists
        if self._marker_actor is not None:
            self.plotter.remove_actor(self._marker_actor)
            self._marker_actor = None

        s = self._unit_scale
        mx = (self._selected_lat - self._center_lat) * _M_PER_DEG_LAT * s
        my = (self._selected_lon - self._center_lon) * self._m_per_deg_lon * s
        mz = self._selected_z * s * self._z_exaggeration

        marker = pv.PolyData(np.array([[mx, my, mz]]))
        self._marker_actor = self.plotter.add_points(
            marker,
            color="magenta",
            point_size=10,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
        self.plotter.render()

    def done(self) -> PickResult | None:
        """Accept the current selection.

        :return: PickResult, or None if subsampled or no selection.
        """
        if not self._active or self._picked_point is None:
            return None

        if self._is_subsampled:
            self._emit_status("Still subsampled — zooming instead")
            self.zoom()
            return None

        cloud = self._display_cloud
        idx = self._pick_idx
        cls = int(cloud.classification[idx]) if cloud.classification is not None else None
        result = PickResult(
            z=float(cloud.z[idx]),
            x_local=float(cloud.x[idx]),
            y_local=float(cloud.y[idx]),
            lat=float(cloud.lat[idx]),
            lon=float(cloud.lon[idx]),
            index=idx,
            classification=cls,
        )
        logger.info(
            "Point selected: z=%.4f at lat=%.6f, lon=%.6f",
            result.z, result.lat, result.lon,
        )
        if self.on_result:
            self.on_result(result)

        # Re-render the full original cloud with the selected point highlighted
        self._working_cloud = self._original_cloud
        self._z_exaggeration = 1.0
        self._render()
        self._add_selection_marker(result)

        return result

    def _add_selection_marker(self, result: PickResult) -> None:
        """Add a bright marker and annotation at the selected point."""
        s = self._unit_scale
        mx = (result.lat - self._center_lat) * _M_PER_DEG_LAT * s
        my = (result.lon - self._center_lon) * self._m_per_deg_lon * s
        mz = result.z * s * self._z_exaggeration

        marker = pv.PolyData(np.array([[mx, my, mz]]))

        if self._display_unit != "m":
            elev_str = f"{result.z * M2FT:.1f} ft"
        else:
            elev_str = f"{result.z:.1f} m"

        from elidar.core.types import ASPRS_CLASSIFICATION_NAMES
        cls_name = ""
        if result.classification is not None:
            cls_name = ASPRS_CLASSIFICATION_NAMES.get(result.classification, f"Class {result.classification}")

        label = f" {cls_name}  {elev_str}" if cls_name else f" {elev_str}"
        marker["labels"] = [label]

        self.plotter.add_points(
            marker,
            color="magenta",
            point_size=10,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
        self.plotter.add_point_labels(
            marker,
            "labels",
            font_size=14,
            text_color="magenta",
            bold=True,
            shadow=True,
            shape=None,
            show_points=False,
        )
        self.plotter.render()

    def set_max_points(self, max_points: int) -> None:
        """Update the max points per class and re-render.

        :param max_points: Maximum points per classification to display.
        """
        self._max_points = max_points
        if self._active and self._working_cloud is not None:
            self._render()

    def cancel(self) -> None:
        """Clear state and blank the viewer."""
        self._active = False
        self._original_cloud = None
        self._working_cloud = None
        self._display_cloud = None
        self._picked_point = None
        self._pick_idx = None
        self._selected_lat = None
        self._selected_lon = None
        self._selected_z = None
        self._marker_actor = None
        self._polydata = None
        self._polydatas = []
        self._class_actors = {}
        self._base_plot_z = None
        self.plotter.clear()
        self.plotter.render()

    def set_z_exaggeration(self, factor: float) -> None:
        """Update the vertical exaggeration factor and re-render in place.

        :param factor: Multiplier for the Z axis (1.0 = no exaggeration).
        """
        self._z_exaggeration = max(factor, 0.01)
        if self._base_plot_z is None:
            return

        new_z = self._base_plot_z * factor
        offset = 0
        for pd in self._polydatas:
            n = pd.n_points
            pts = pd.points.copy()
            pts[:, 2] = new_z[offset:offset + n]
            pd.points = pts
            offset += n
        self.plotter.reset_camera()
        self.plotter.render()

    def set_color_stretch(self, stretch: float) -> None:
        """Update color stretch and re-render.

        :param stretch: Values > 1 give more differentiation at high elevations.
        """
        self._color_stretch = stretch
        if self._active and self._working_cloud is not None:
            self._render()

    def set_class_visible(self, cls: int, visible: bool) -> None:
        """Toggle visibility of a classification's points.

        :param cls: Classification code.
        :param visible: Whether to show or hide.
        """
        if self._visible_classes is None:
            self._visible_classes = set()
        if visible:
            self._visible_classes.add(cls)
        else:
            self._visible_classes.discard(cls)
        actor = self._class_actors.get(cls)
        if actor is not None:
            actor.SetVisibility(visible)
            self.plotter.render()

    def set_point_size(self, size: int) -> None:
        """Update point size for all actors in place.

        :param size: Point size in pixels.
        """
        self._point_size = size
        for actor in self._class_actors.values():
            actor.GetProperty().SetPointSize(size)
        self.plotter.render()

    def set_background(self, color: str) -> None:
        """Change the background color and adapt text colors.

        :param color: Background color name ("black", "gray", "white").
        """
        self._font_color = "white" if color == "black" else "black"
        self.plotter.set_background(color)
        if self._active and self._working_cloud is not None:
            self._render()

    def set_class_cmaps(self, class_cmaps: dict[int, ColorScale]) -> None:
        """Update colormaps and re-render.

        :param class_cmaps: Per-classification colormap mapping.
        """
        self._class_cmaps = class_cmaps
        if self._active and self._working_cloud is not None:
            self._render()

    def set_display_unit(self, display_unit: str, unit_scale: float) -> None:
        """Switch display unit and re-render.

        :param display_unit: Unit label ("ft", "ftUS", "m").
        :param unit_scale: Meters-to-display-unit factor.
        """
        self._display_unit = display_unit
        self._unit_scale = unit_scale
        if self._active and self._working_cloud is not None:
            self._render()

    def _render(self) -> None:
        """Re-render the current working cloud into the plotter.

        Each classification is subsampled independently (up to
        DISPLAY_THRESHOLD each) so minority classes like Ground are
        always well-represented. Visibility is set per-actor so
        toggling is instant with no re-render.
        """
        wc = self._working_cloud
        s = self._unit_scale
        # Short label for colorbar; full label is in _display_unit for results
        unit = "ft" if "ft" in self._display_unit else "m"

        # Center on surface classes (Ground + Snow/Ice) if available so
        # the view frames the terrain of interest, not the full tile.
        if wc.classification is not None:
            surface_mask = np.isin(wc.classification, list(SURFACE_CLASSES))
            if np.any(surface_mask):
                self._center_lat = float(np.mean(wc.lat[surface_mask]))
                self._center_lon = float(np.mean(wc.lon[surface_mask]))
            else:
                self._center_lat = float(np.mean(wc.lat))
                self._center_lon = float(np.mean(wc.lon))
        else:
            self._center_lat = float(np.mean(wc.lat))
            self._center_lon = float(np.mean(wc.lon))
        self._m_per_deg_lon = _M_PER_DEG_LAT * np.cos(np.radians(self._center_lat))

        self.plotter.disable_picking()
        self.plotter.clear()
        self._polydatas = []
        self._class_actors = {}

        has_classes = (
            wc.classification is not None
            and len(self._class_cmaps) > 0
        )

        if has_classes:
            from elidar.core.types import ASPRS_CLASSIFICATION_NAMES

            unique_classes = sorted(int(c) for c in np.unique(wc.classification))

            # Subsample each class independently, then concatenate
            class_clouds: list[tuple[int, PointCloudData]] = []
            any_subsampled = False
            for cls in unique_classes:
                mask = wc.classification == cls
                cls_cloud = wc.subset(mask)
                if cls_cloud.n_points > self._max_points:
                    cls_cloud = _subsample(cls_cloud, self._max_points)
                    any_subsampled = True
                class_clouds.append((cls, cls_cloud))

            self._is_subsampled = any_subsampled

            # Merge into display_cloud for pick-index mapping
            all_arrays = {
                "x": [], "y": [], "z": [], "lat": [], "lon": [], "cls": [],
            }
            for _, cc in class_clouds:
                all_arrays["x"].append(cc.x)
                all_arrays["y"].append(cc.y)
                all_arrays["z"].append(cc.z)
                all_arrays["lat"].append(cc.lat)
                all_arrays["lon"].append(cc.lon)
                all_arrays["cls"].append(cc.classification)

            self._display_cloud = PointCloudData(
                x=np.concatenate(all_arrays["x"]),
                y=np.concatenate(all_arrays["y"]),
                z=np.concatenate(all_arrays["z"]),
                lat=np.concatenate(all_arrays["lat"]),
                lon=np.concatenate(all_arrays["lon"]),
                classification=np.concatenate(all_arrays["cls"]),
            )
            cloud = self._display_cloud

            self._display_x = (cloud.lat - self._center_lat) * _M_PER_DEG_LAT * s
            self._display_y = (cloud.lon - self._center_lon) * self._m_per_deg_lon * s
            self._base_plot_z = cloud.z.astype(np.float64) * s
            plot_z = self._base_plot_z * self._z_exaggeration

            z_min = float(self._base_plot_z.min())
            z_max = float(self._base_plot_z.max())

            # Determine which surface classes are present for colorbars
            present_surface = sorted(sc for sc in SURFACE_CLASSES if sc in unique_classes)
            # Primary gets title + labels; secondary gets just the gradient
            primary_cls = present_surface[0] if present_surface else unique_classes[0]

            offset = 0
            for cls, cls_cloud in class_clouds:
                n = cls_cloud.n_points
                px = self._display_x[offset:offset + n]
                py = self._display_y[offset:offset + n]
                pz = plot_z[offset:offset + n]
                elev = self._base_plot_z[offset:offset + n]
                offset += n

                cmap_enum = self._class_cmaps.get(cls, DEFAULT_CMAP)
                cmap = get_colormap(cmap_enum)
                if self._color_stretch != 1.0:
                    cmap = stretch_colormap(cmap, self._color_stretch)

                pts = np.column_stack([px, py, pz])
                pd = pv.PolyData(pts)
                pd["elevation"] = elev
                self._polydatas.append(pd)

                add_kwargs: dict = {
                    "scalars": "elevation",
                    "cmap": cmap,
                    "clim": [z_min, z_max],
                    "point_size": self._point_size,
                    "render_points_as_spheres": True,
                }
                dual = len(present_surface) >= 2
                fc = self._font_color
                if cls == primary_cls:
                    title = f"   Elev ({unit})" if dual else f"Elev ({unit})"
                    add_kwargs["scalar_bar_args"] = {
                        "title": title,
                        "title_font_size": 10,
                        "vertical": True,
                        "n_labels": 0 if dual else 2,
                        "fmt": "%.0f",
                        "label_font_size": 10,
                        "color": fc,
                        "position_x": 0.89 if dual else 0.93,
                        "position_y": 0.05,
                        "width": 0.03,
                        "height": 0.8,
                    }
                elif cls in present_surface:
                    add_kwargs["scalar_bar_args"] = {
                        "title": " ",
                        "title_font_size": 10,
                        "vertical": True,
                        "n_labels": 2,
                        "fmt": "%.0f",
                        "label_font_size": 10,
                        "color": fc,
                        "position_x": 0.93,
                        "position_y": 0.05,
                        "width": 0.03,
                        "height": 0.8,
                    }
                else:
                    add_kwargs["show_scalar_bar"] = False

                actor = self.plotter.add_points(pd, **add_kwargs)
                self._class_actors[cls] = actor

                visible = (
                    self._visible_classes is None or cls in self._visible_classes
                )
                actor.SetVisibility(visible)
        else:
            self._is_subsampled = wc.n_points > self._max_points
            if self._is_subsampled:
                self._display_cloud = _subsample(wc, self._max_points)
            else:
                self._display_cloud = wc

            cloud = self._display_cloud
            self._display_x = (cloud.lat - self._center_lat) * _M_PER_DEG_LAT * s
            self._display_y = (cloud.lon - self._center_lon) * self._m_per_deg_lon * s
            self._base_plot_z = cloud.z.astype(np.float64) * s
            plot_z = self._base_plot_z * self._z_exaggeration

            cmap = get_colormap(self._options.colorscale)
            if self._color_stretch != 1.0:
                cmap = stretch_colormap(cmap, self._color_stretch)
            pts = np.column_stack([self._display_x, self._display_y, plot_z])
            pd = pv.PolyData(pts)
            pd["elevation"] = self._base_plot_z
            self._polydatas.append(pd)

            actor = self.plotter.add_points(
                pd,
                scalars="elevation",
                cmap=cmap,
                point_size=self._point_size,
                render_points_as_spheres=True,
                scalar_bar_args={
                    "title": f"Elevation ({unit})",
                    "n_labels": 2,
                    "fmt": "%.0f",
                },
            )
            self._class_actors[-1] = actor

        self.plotter.add_axes(
            xlabel="N",
            ylabel="E",
            zlabel="Up",
            line_width=2,
            color=self._font_color,
        )

        self.plotter.enable_point_picking(
            callback=self._on_pick,
            left_clicking=True,
            picker="point",
            show_point=True,
            color="red",
            point_size=15,
            show_message=False,
            tolerance=0.025,
        )

        if self._preserve_camera:
            # Compare the user's current camera distance to what a re-fit
            # would produce. Keep the tighter of the two so we auto-zoom
            # when the data shrinks a lot but don't zoom OUT if the user
            # was already in close.
            cam = self.plotter.camera
            old_pos = np.array(cam.position)
            old_fp = np.array(cam.focal_point)
            old_up = cam.up
            old_dist = np.linalg.norm(old_pos - old_fp)
            direction = (old_pos - old_fp) / (old_dist + 1e-12)

            self.plotter.reset_camera()
            fit_pos = np.array(self.plotter.camera.position)
            fit_fp = np.array(self.plotter.camera.focal_point)
            fit_dist = np.linalg.norm(fit_pos - fit_fp)

            # Use whichever is closer (tighter view)
            use_dist = min(old_dist, fit_dist)
            cam.focal_point = tuple(fit_fp)
            cam.position = tuple(fit_fp + direction * use_dist)
            cam.up = old_up
            self._preserve_camera = False
        else:
            # Default: frame on surface classes with isometric view
            surface_bounds = []
            for sc in SURFACE_CLASSES:
                actor = self._class_actors.get(sc)
                if actor is not None:
                    surface_bounds.append(actor.GetBounds())
            if surface_bounds:
                merged = [
                    min(b[0] for b in surface_bounds),
                    max(b[1] for b in surface_bounds),
                    min(b[2] for b in surface_bounds),
                    max(b[3] for b in surface_bounds),
                    min(b[4] for b in surface_bounds),
                    max(b[5] for b in surface_bounds),
                ]
                self.plotter.reset_camera(bounds=merged)
            else:
                self.plotter.reset_camera()
            self.plotter.camera_position = "xz"
            self.plotter.camera.elevation = 25
            self.plotter.camera.azimuth = -60
        self.plotter.render()

    def _on_pick(self, point: np.ndarray) -> None:
        """Callback when the user clicks a point in the embedded view."""
        if self._display_cloud is None or self._display_cloud.n_points == 0:
            return
        self._picked_point = point

        # Only match against visible classes
        visible_mask = None
        cloud = self._display_cloud
        if self._visible_classes is not None and cloud.classification is not None:
            visible_mask = np.isin(cloud.classification, list(self._visible_classes))
            if not np.any(visible_mask):
                return

        self._pick_idx = _find_picked_index(
            cloud, point, self._display_x, self._display_y,
            visible_mask=visible_mask,
        )

        s = self._unit_scale
        real_lat = point[0] / (_M_PER_DEG_LAT * s) + self._center_lat
        real_lon = point[1] / (self._m_per_deg_lon * s) + self._center_lon
        z_m = point[2] / (s * self._z_exaggeration)

        # Persist selection for re-highlighting after zoom
        self._selected_lat = real_lat
        self._selected_lon = real_lon
        self._selected_z = z_m

        # Classification name
        from elidar.core.types import ASPRS_CLASSIFICATION_NAMES
        cls_name = ""
        if cloud.classification is not None:
            cls_code = int(cloud.classification[self._pick_idx])
            cls_name = ASPRS_CLASSIFICATION_NAMES.get(cls_code, f"Class {cls_code}")

        if self._display_unit != "m":
            elev_display = f"{z_m * self._unit_scale:.2f} {self._display_unit}"
        else:
            elev_display = f"{z_m:.2f} m"
        elev_log = f"{z_m:.2f} m ({z_m * M2FT:.2f} ft)"
        _update_status(self.plotter, real_lat, real_lon, elev_display, cls_name)

        status = f"Selected: {cls_name}  {real_lat:.6f}°N, {real_lon:.6f}°W  Elev: {elev_log}"
        self._emit_status(status)

    def _emit_status(self, msg: str) -> None:
        if self.on_status:
            self.on_status(msg)
