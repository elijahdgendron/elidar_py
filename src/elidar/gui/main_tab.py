"""Main analysis tab for the ELiDAR GUI."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from elidar.core.types import ColorScale, M2FT, QueryOptions, QueryResult, QueryType
from elidar.viz.scatter3d import EmbeddedSearch, PickResult

logger = logging.getLogger(__name__)


class _LoadWorker(QThread):
    """Background thread for loading and processing LiDAR data."""

    finished = pyqtSignal(object, object)  # (cloud, crs_info)
    error = pyqtSignal(str)

    def __init__(
        self,
        lat: float,
        lon: float,
        las_paths: list[str],
        options: QueryOptions,
    ) -> None:
        super().__init__()
        self.lat = lat
        self.lon = lon
        self.las_paths = las_paths
        self.options = options

    def run(self) -> None:
        try:
            from elidar.core.query import _load_and_merge, _normalize_paths, _spatial_filter
            from elidar.core import cleaning
            import numpy as np

            paths = _normalize_paths(
                self.las_paths if len(self.las_paths) > 1 else self.las_paths[0]
            )
            cloud, crs_info = _load_and_merge(paths, self.options)

            lat, lon = self.lat, self.lon
            if self.options.center:
                lat = float(np.mean(cloud.lat))
                lon = float(np.mean(cloud.lon))

            cloud = _spatial_filter(cloud, lat, lon, self.options.query_radius_deg)

            if cloud.n_points == 0:
                self.error.emit("Query point is not within the LAS data extent")
                return

            if self.options.clean_data:
                cloud = cleaning.clean(
                    cloud,
                    num_neighbors=self.options.filter_num_neighbors,
                    std_ratio=self.options.filter_threshold,
                )

            self.finished.emit(cloud, crs_info)
        except Exception as e:
            self.error.emit(str(e))


class _QueryWorker(QThread):
    """Background thread for non-interactive queries (peak/point)."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        lat: float,
        lon: float,
        query_type: QueryType,
        las_paths: list[str],
        options: QueryOptions,
    ) -> None:
        super().__init__()
        self.lat = lat
        self.lon = lon
        self.query_type = query_type
        self.las_paths = las_paths
        self.options = options

    def run(self) -> None:
        try:
            from elidar.core.query import query_lidar

            result = query_lidar(
                query_lat=self.lat,
                query_lon=self.lon,
                query_type=self.query_type,
                las_path=self.las_paths if len(self.las_paths) > 1 else self.las_paths[0],
                options=self.options,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainTab(QWidget):
    """Primary analysis tab with embedded 3D viewer."""

    def __init__(self, settings: QSettings) -> None:
        super().__init__()
        self.settings = settings
        self._worker: QThread | None = None
        self._last_result = None
        self._tools_tab = None

        layout = QHBoxLayout(self)

        # Left panel: controls (narrow)
        left = QWidget()
        left.setMaximumWidth(340)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(self._build_file_group())
        left_layout.addWidget(self._build_query_group())
        left_layout.addWidget(self._build_options_group())
        left_layout.addWidget(self._build_action_group())
        left_layout.addStretch()

        # Right panel: 3D view on top, results/prominence on bottom
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 3D viewer area with placeholder
        self.viewer_stack = QStackedWidget()
        self.viewer_stack.setMinimumHeight(300)

        # Page 0: placeholder
        placeholder = QFrame()
        placeholder.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        placeholder.setStyleSheet(
            "QFrame { background: #1a1a2e; border: 1px solid #444; border-radius: 4px; }"
        )
        ph_layout = QVBoxLayout(placeholder)
        ph_label = QLabel("3D LiDAR Viewer\n\nAdd files and click Analyze\nto load point cloud data")
        ph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_label.setStyleSheet("color: #888; font-size: 14px; border: none; background: transparent;")
        ph_layout.addWidget(ph_label)
        self.viewer_stack.addWidget(placeholder)

        # Page 1: pyvista plotter
        self.plotter = QtInteractor(self)
        self.viewer_stack.addWidget(self.plotter)

        self.embedded_search = EmbeddedSearch(
            self.plotter,
            on_status=self._on_search_status,
            on_result=self._on_search_result,
        )

        # Search control buttons
        btn_bar = QHBoxLayout()
        self.zoom_btn = QPushButton("Zoom (Z)")
        self.zoom_btn.clicked.connect(self._on_zoom)
        self.zoom_btn.setEnabled(False)
        btn_bar.addWidget(self.zoom_btn)

        self.done_btn = QPushButton("Done (D)")
        self.done_btn.clicked.connect(self._on_done)
        self.done_btn.setEnabled(False)
        btn_bar.addWidget(self.done_btn)

        self.cancel_btn = QPushButton("Cancel (Esc)")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)
        btn_bar.addWidget(self.cancel_btn)

        # Mode label
        self.mode_label = QLabel("")
        self.mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_label.setStyleSheet("font-weight: bold; font-size: 13px;")

        # Bottom: results log
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        self.results_text.setStyleSheet("font-family: monospace; font-size: 11px;")

        # Exaggeration slider (vertical, right of viewer)
        exag_col = QVBoxLayout()
        exag_label = QLabel("Z\nExag")
        exag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        exag_label.setStyleSheet("font-size: 10px;")
        exag_col.addWidget(exag_label)

        self.exag_slider = QSlider(Qt.Orientation.Vertical)
        self.exag_slider.setRange(10, 200)
        self.exag_slider.setValue(10)
        self.exag_slider.setTickPosition(QSlider.TickPosition.TicksRight)
        self.exag_slider.setTickInterval(10)
        self.exag_slider.setFixedWidth(30)
        self.exag_slider.valueChanged.connect(self._on_exag_changed)
        exag_col.addWidget(self.exag_slider)

        self.exag_value_label = QLabel("1.0x")
        self.exag_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.exag_value_label.setStyleSheet("font-size: 10px;")
        exag_col.addWidget(self.exag_value_label)

        # Viewer row: stack + slider
        viewer_row = QHBoxLayout()
        viewer_row.addWidget(self.viewer_stack, stretch=1)
        viewer_row.addLayout(exag_col)

        # Assemble right side
        right_layout.addWidget(self.mode_label)
        right_layout.addLayout(viewer_row, stretch=1)
        right_layout.addLayout(btn_bar)
        right_layout.addWidget(self.results_text)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([300, 600])
        layout.addWidget(splitter)

        # Keyboard shortcuts on the plotter widget
        self.plotter.add_key_event("z", self._on_zoom)
        self.plotter.add_key_event("d", self._on_done)
        self.plotter.add_key_event("Return", self._on_done)
        self.plotter.add_key_event("Escape", self._on_cancel)

    def set_tools_tab(self, tools_tab: object) -> None:
        """Set the tools tab reference for forwarding results."""
        self._tools_tab = tools_tab

    def _build_file_group(self) -> QGroupBox:
        group = QGroupBox("Data Files")
        layout = QVBoxLayout(group)

        btn_row = QHBoxLayout()
        self.add_files_btn = QPushButton("Add Files...")
        self.add_files_btn.clicked.connect(self._on_add_files)
        btn_row.addWidget(self.add_files_btn)

        self.add_folder_btn = QPushButton("Add Folder...")
        self.add_folder_btn.clicked.connect(self._on_add_folder)
        btn_row.addWidget(self.add_folder_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear_files)
        btn_row.addWidget(self.clear_btn)
        layout.addLayout(btn_row)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(80)
        layout.addWidget(self.file_list)

        dem_row = QHBoxLayout()
        dem_row.addWidget(QLabel("DEM:"))
        self.dem_edit = QLineEdit()
        self.dem_edit.setPlaceholderText("Optional — only if LAZ has no CRS")
        dem_row.addWidget(self.dem_edit)
        self.dem_browse_btn = QPushButton("...")
        self.dem_browse_btn.setMaximumWidth(30)
        self.dem_browse_btn.clicked.connect(self._on_browse_dem)
        dem_row.addWidget(self.dem_browse_btn)
        layout.addLayout(dem_row)

        return group

    def _build_query_group(self) -> QGroupBox:
        group = QGroupBox("Query Location")
        form = QFormLayout(group)

        self.guess_center_check = QCheckBox("Use tile center")
        self.guess_center_check.toggled.connect(self._on_guess_center_toggled)
        form.addRow(self.guess_center_check)

        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setRange(-90.0, 90.0)
        self.lat_spin.setDecimals(6)
        self.lat_spin.setValue(float(self.settings.value("last_lat", 48.7768)))
        form.addRow("Latitude:", self.lat_spin)

        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setRange(-180.0, 180.0)
        self.lon_spin.setDecimals(6)
        self.lon_spin.setValue(float(self.settings.value("last_lon", -121.8145)))
        form.addRow("Longitude:", self.lon_spin)

        self.query_type_combo = QComboBox()
        for qt in QueryType:
            self.query_type_combo.addItem(qt.value.capitalize(), qt)
        self.query_type_combo.setCurrentIndex(0)
        form.addRow("Query Type:", self.query_type_combo)

        return group

    def _on_guess_center_toggled(self, checked: bool) -> None:
        self.lat_spin.setEnabled(not checked)
        self.lon_spin.setEnabled(not checked)

    def _build_options_group(self) -> QGroupBox:
        group = QGroupBox("Options")
        form = QFormLayout(group)

        self.clean_check = QCheckBox()
        self.clean_check.setChecked(True)
        form.addRow("Clean Data:", self.clean_check)

        self.ground_check = QCheckBox()
        form.addRow("Ground Only:", self.ground_check)

        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.0001, 0.1)
        self.radius_spin.setDecimals(4)
        self.radius_spin.setSingleStep(0.0001)
        self.radius_spin.setValue(0.002)
        form.addRow("Search Radius (deg):", self.radius_spin)

        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(0.1, 0.9)
        self.zoom_spin.setDecimals(1)
        self.zoom_spin.setSingleStep(0.1)
        self.zoom_spin.setValue(0.5)
        form.addRow("Zoom Scale:", self.zoom_spin)

        self.color_combo = QComboBox()
        for cs in ColorScale:
            self.color_combo.addItem(cs.value.capitalize(), cs)
        self.color_combo.setCurrentText("Turbo")
        form.addRow("Color Scale:", self.color_combo)

        self.unit_combo = QComboBox()
        self.unit_combo.addItem("Auto", None)
        self.unit_combo.addItem("Feet", "ft")
        self.unit_combo.addItem("Meters", "m")
        form.addRow("Force Unit:", self.unit_combo)

        self.display_unit_combo = QComboBox()
        self.display_unit_combo.addItem("Meters", False)
        self.display_unit_combo.addItem("Feet", True)
        form.addRow("Display In:", self.display_unit_combo)

        return group

    def _build_action_group(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setMinimumHeight(36)
        self.analyze_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.analyze_btn.clicked.connect(self._on_analyze)
        layout.addWidget(self.analyze_btn)

        return widget

    # --- File slots ---

    def _on_add_files(self) -> None:
        last_dir = self.settings.value("last_dir", os.path.expanduser("~"))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select LAS/LAZ Files", str(last_dir),
            "LiDAR Files (*.las *.laz);;All Files (*)",
        )
        for f in files:
            if not self._file_in_list(f):
                self.file_list.addItem(f)
        if files:
            self.settings.setValue("last_dir", str(Path(files[0]).parent))

    def _on_add_folder(self) -> None:
        last_dir = self.settings.value("last_dir", os.path.expanduser("~"))
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", str(last_dir))
        if folder:
            self.settings.setValue("last_dir", folder)
            for f in Path(folder).glob("*.la[sz]"):
                if not self._file_in_list(str(f)):
                    self.file_list.addItem(str(f))

    def _on_clear_files(self) -> None:
        self.file_list.clear()

    def _on_browse_dem(self) -> None:
        last_dir = self.settings.value("last_dir", os.path.expanduser("~"))
        f, _ = QFileDialog.getOpenFileName(
            self, "Select DEM File", str(last_dir),
            "DEM Files (*.tif *.img);;All Files (*)",
        )
        if f:
            self.dem_edit.setText(f)

    def _file_in_list(self, path: str) -> bool:
        for i in range(self.file_list.count()):
            if self.file_list.item(i).text() == path:
                return True
        return False

    def _get_las_paths(self) -> list[str]:
        return [self.file_list.item(i).text() for i in range(self.file_list.count())]

    # --- Analyze ---

    def _get_options(self) -> QueryOptions:
        return QueryOptions(
            clean_data=self.clean_check.isChecked(),
            colorscale=self.color_combo.currentData(),
            filter_ground=self.ground_check.isChecked(),
            query_radius_deg=self.radius_spin.value(),
            zoom_scale=self.zoom_spin.value(),
            force_unit=self.unit_combo.currentData(),
            dem_path=self.dem_edit.text().strip() or None,
            center=self.guess_center_check.isChecked(),
        )

    def _on_analyze(self) -> None:
        paths = self._get_las_paths()
        if not paths:
            QMessageBox.warning(self, "No Files", "Add at least one LAS/LAZ file.")
            return

        lat = self.lat_spin.value()
        lon = self.lon_spin.value()
        query_type = self.query_type_combo.currentData()

        if not self.guess_center_check.isChecked():
            self.settings.setValue("last_lat", lat)
            self.settings.setValue("last_lon", lon)

        options = self._get_options()

        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("Loading...")
        self.results_text.append(f"Running {query_type.value} query...\n")

        if query_type == QueryType.SEARCH:
            self._worker = _LoadWorker(lat, lon, paths, options)
            self._worker.finished.connect(self._on_data_loaded)
            self._worker.error.connect(self._on_query_error)
            self._worker.start()
        else:
            self._worker = _QueryWorker(lat, lon, query_type, paths, options)
            self._worker.finished.connect(self._on_query_finished)
            self._worker.error.connect(self._on_query_error)
            self._worker.start()

    def _on_data_loaded(self, cloud: object, crs_info: object) -> None:
        """Search mode: data loaded, render into embedded viewer."""
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze")

        options = self._get_options()
        display_feet = self.display_unit_combo.currentData()
        self.viewer_stack.setCurrentIndex(1)
        self.embedded_search.load(cloud, crs_info, options, display_feet=display_feet)

        self._set_search_controls(True)
        self._update_mode_label()
        self.results_text.append(f"Loaded {cloud.n_points:,} points. Click to select.\n")

    def _on_query_finished(self, result: object) -> None:
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze")
        self._display_result(result)

    def _on_query_error(self, msg: str) -> None:
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze")
        self.results_text.append(f"ERROR: {msg}\n")
        logger.error("Query failed: %s", msg)

    # --- Search controls ---

    def _set_search_controls(self, enabled: bool) -> None:
        self.zoom_btn.setEnabled(enabled)
        self.done_btn.setEnabled(enabled)
        self.cancel_btn.setEnabled(enabled)

    def _update_mode_label(self) -> None:
        if not self.embedded_search.is_active:
            self.mode_label.setText("")
            return
        if self.embedded_search.is_subsampled:
            self.mode_label.setText("CLICK TO DOWNSELECT")
            self.mode_label.setStyleSheet(
                "font-weight: bold; font-size: 13px; color: red;"
            )
        else:
            self.mode_label.setText("CLICK TO SELECT OR DOWNSELECT")
            self.mode_label.setStyleSheet(
                "font-weight: bold; font-size: 13px; color: green;"
            )

    def _on_zoom(self) -> None:
        self.embedded_search.zoom()
        self._update_mode_label()

    def _on_done(self) -> None:
        result = self.embedded_search.done()
        if result is not None:
            self._set_search_controls(False)
            self._update_mode_label()

    def _on_cancel(self) -> None:
        self.embedded_search.cancel()
        self._set_search_controls(False)
        self.mode_label.setText("")
        self.viewer_stack.setCurrentIndex(0)
        self.results_text.append("Search cancelled.\n")

    def _on_exag_changed(self, value: int) -> None:
        factor = value / 10.0
        self.exag_value_label.setText(f"{factor:.1f}x")
        self.embedded_search.set_z_exaggeration(factor)

    # --- Search callbacks ---

    def _on_search_status(self, msg: str) -> None:
        self.results_text.append(msg + "\n")

    def _on_search_result(self, pick: PickResult) -> None:
        result = QueryResult(
            elevation_m=pick.z,
            elevation_ft=pick.z * M2FT,
            location_local=(pick.x_local, pick.y_local),
            location_deg=(pick.lat, pick.lon),
            query_type=QueryType.SEARCH,
        )
        self._display_result(result)

    # --- Results ---

    def _display_result(self, result: object) -> None:
        if result is None:
            self.results_text.append("No result (point not in data or cancelled)\n")
            return

        self._last_result = result
        self.results_text.append(
            f"--- {result.query_type.value.upper()} ---\n"
            f"  Elevation: {result.elevation_ft:,.2f} ft  ({result.elevation_m:,.2f} m)\n"
            f"  Location:  {result.location_deg[0]:.6f}°N, {result.location_deg[1]:.6f}°W\n"
        )
        if self._tools_tab is not None:
            self._tools_tab.set_last_result_ft(result.elevation_ft)

