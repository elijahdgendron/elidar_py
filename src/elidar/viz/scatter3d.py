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


def _update_status(plotter: pv.Plotter, lat: float, lon: float, elev_str: str) -> None:
    """Update the status text after a point is picked."""
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
        self._zoom_radius: float = 0.0
        self._options: QueryOptions | None = None
        self._crs_info: CrsInfo | None = None
        self._is_subsampled: bool = False
        self._center_lat: float = 0.0
        self._center_lon: float = 0.0
        self._m_per_deg_lon: float = 0.0
        self._active: bool = False
        self._display_feet: bool = False
        self._unit_scale: float = 1.0
        self._z_exaggeration: float = 1.0
        self._point_size: int = 5
        self._base_plot_z: np.ndarray | None = None
        self._polydatas: list[pv.PolyData] = []
        self._class_cmaps: dict[int, ColorScale] = {}
        self._class_actors: dict[int, object] = {}
        self._visible_classes: set[int] | None = None
        self._polydata: pv.PolyData | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_subsampled(self) -> bool:
        return self._is_subsampled

    def load(
        self,
        cloud: PointCloudData,
        crs_info: CrsInfo,
        options: QueryOptions,
        display_feet: bool = False,
        class_cmaps: dict[int, ColorScale] | None = None,
        visible_classes: set[int] | None = None,
    ) -> None:
        """Load a point cloud and render it.

        :param cloud: Cleaned point cloud with lat/lon populated.
        :param crs_info: CRS info for display.
        :param options: Query options.
        :param display_feet: If True, display in feet instead of meters.
        :param class_cmaps: Per-classification colormap overrides.
        :param visible_classes: Set of classification codes to display. None = all.
        """
        self._working_cloud = cloud
        self._options = options
        self._crs_info = crs_info
        # Use the actual data extent if smaller than the configured radius,
        # so the first zoom immediately tightens the view.
        lat_span = float(cloud.lat.max() - cloud.lat.min())
        lon_span = float(cloud.lon.max() - cloud.lon.min())
        data_radius = max(lat_span, lon_span) / 2
        self._zoom_radius = min(options.query_radius_deg / 2, data_radius)
        self._picked_point = None
        self._pick_idx = None
        self._active = True
        self._display_feet = display_feet
        self._unit_scale = M2FT if display_feet else 1.0
        self._class_cmaps = class_cmaps if class_cmaps is not None else {}
        self._visible_classes = visible_classes
        self._render()

    def zoom(self) -> None:
        """Zoom around the currently picked point."""
        if not self._active or self._picked_point is None:
            return

        idx = self._pick_idx
        cloud = self._display_cloud

        self._zoom_radius *= self._options.zoom_scale
        clat = cloud.lat[idx]
        clon = cloud.lon[idx]
        dist = np.sqrt(
            (self._working_cloud.lat - clat) ** 2
            + (self._working_cloud.lon - clon) ** 2
        )
        mask = dist <= self._zoom_radius
        self._working_cloud = self._working_cloud.subset(mask)

        if self._working_cloud.n_points == 0:
            self._emit_status("Zoom resulted in empty cloud")
            self.cancel()
            return

        logger.info(
            "Zoomed: radius=%.6f deg, %d points remaining",
            self._zoom_radius,
            self._working_cloud.n_points,
        )

        self._picked_point = None
        self._pick_idx = None
        self._render()

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
        return result

    def cancel(self) -> None:
        """Clear state and blank the viewer."""
        self._active = False
        self._working_cloud = None
        self._display_cloud = None
        self._picked_point = None
        self._pick_idx = None
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
        self._z_exaggeration = factor
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

    def set_class_cmaps(self, class_cmaps: dict[int, ColorScale]) -> None:
        """Update colormaps and re-render.

        :param class_cmaps: Per-classification colormap mapping.
        """
        self._class_cmaps = class_cmaps
        if self._active and self._working_cloud is not None:
            self._render()

    def set_display_feet(self, display_feet: bool) -> None:
        """Switch display unit and re-render.

        :param display_feet: True for feet, False for meters.
        """
        self._display_feet = display_feet
        self._unit_scale = M2FT if display_feet else 1.0
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
        unit = "ft" if self._display_feet else "m"

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
                if cls_cloud.n_points > DISPLAY_THRESHOLD:
                    cls_cloud = _subsample(cls_cloud, DISPLAY_THRESHOLD)
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
                if cls == primary_cls:
                    add_kwargs["scalar_bar_args"] = {
                        "title": f"Elev ({unit})",
                        "title_font_size": 10,
                        "vertical": True,
                        "n_labels": 0 if dual else 2,
                        "fmt": "%.0f",
                        "label_font_size": 10,
                        "position_x": 0.80 if dual else 0.88,
                        "position_y": 0.05,
                        "width": 0.04,
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
                        "position_x": 0.88,
                        "position_y": 0.05,
                        "width": 0.04,
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
            self._is_subsampled = wc.n_points > DISPLAY_THRESHOLD
            if self._is_subsampled:
                self._display_cloud = _subsample(wc, DISPLAY_THRESHOLD)
            else:
                self._display_cloud = wc

            cloud = self._display_cloud
            self._display_x = (cloud.lat - self._center_lat) * _M_PER_DEG_LAT * s
            self._display_y = (cloud.lon - self._center_lon) * self._m_per_deg_lon * s
            self._base_plot_z = cloud.z.astype(np.float64) * s
            plot_z = self._base_plot_z * self._z_exaggeration

            cmap = get_colormap(self._options.colorscale)
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

        # Frame camera on surface class actors (Ground + Snow/Ice)
        surface_bounds = []
        for sc in SURFACE_CLASSES:
            actor = self._class_actors.get(sc)
            if actor is not None:
                surface_bounds.append(actor.GetBounds())
        if surface_bounds:
            # Merge bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
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
        self._picked_point = point
        self._pick_idx = _find_picked_index(
            self._display_cloud, point, self._display_x, self._display_y,
        )

        s = self._unit_scale
        real_lat = point[0] / (_M_PER_DEG_LAT * s) + self._center_lat
        real_lon = point[1] / (self._m_per_deg_lon * s) + self._center_lon
        z_m = point[2] / s

        elev_str = f"{z_m:.2f} m ({z_m * M2FT:.2f} ft)"
        _update_status(self.plotter, real_lat, real_lon, elev_str)

        status = f"Selected: {real_lat:.6f}°N, {real_lon:.6f}°W  Elev: {elev_str}"
        self._emit_status(status)

    def _emit_status(self, msg: str) -> None:
        if self.on_status:
            self.on_status(msg)
