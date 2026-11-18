"""Main analysis tab for the ELiDAR GUI."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QSettings, QThread, QTimer, pyqtSignal
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
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from elidar.core.types import (
    ASPRS_CLASSIFICATION_NAMES,
    ColorScale,
    DEFAULT_CLASS_CMAPS,
    DEFAULT_CMAP,
    M2FT,
    M2FT_SURVEY,
    QueryOptions,
    QueryResult,
    QueryType,
    SURFACE_CLASSES,
)
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

            if cloud.n_points == 0:
                self.error.emit("No points remaining after cleaning")
                return

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


class _LogSignalBridge(QObject):
    """Thread-safe bridge: emits a signal so the append happens on the main thread."""

    message = pyqtSignal(str)


class _LogHandler(logging.Handler):
    """Routes log records to a QTextEdit widget via a thread-safe signal."""

    def __init__(self, text_widget: QTextEdit) -> None:
        super().__init__()
        self._bridge = _LogSignalBridge()
        self._bridge.message.connect(text_widget.append)
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._bridge.message.emit(msg)


class MainTab(QWidget):
    """Primary analysis tab with embedded 3D viewer."""

    def __init__(self, settings: QSettings) -> None:
        super().__init__()
        self.settings = settings
        self._worker: QThread | None = None
        self._last_result = None
        self._last_crs_name: str = ""
        self._last_select_type: str = ""
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

        # Page 0: placeholder with background image
        bg_path = Path(__file__).parent / "resources" / "background.png"
        placeholder = QFrame()
        placeholder.setObjectName("placeholder")
        placeholder.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        placeholder.setStyleSheet(
            f"QFrame#placeholder {{ "
            f"  border-image: url({bg_path.as_posix()}) 0 0 0 0 stretch stretch; "
            f"  border: 1px solid #444; border-radius: 4px; "
            f"}}"
        )
        ph_layout = QVBoxLayout(placeholder)
        # Dark wash over the image to dull it
        wash = QWidget()
        wash.setStyleSheet("background: rgba(0, 0, 0, 120); border: none;")
        wash_layout = QVBoxLayout(wash)
        # Text with transparent grey box
        overlay = QLabel("3D LiDAR Viewer\n\nAdd files and click Analyze\nto load point cloud data")
        overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay.setStyleSheet(
            "color: white; font-size: 16px; font-weight: bold; "
            "background: rgba(80, 80, 80, 150); border: none; border-radius: 8px; "
            "padding: 20px;"
        )
        wash_layout.addStretch()
        wash_layout.addWidget(overlay, alignment=Qt.AlignmentFlag.AlignCenter)
        wash_layout.addStretch()
        ph_layout.setContentsMargins(0, 0, 0, 0)
        ph_layout.addWidget(wash)
        self.viewer_stack.addWidget(placeholder)

        # Page 1: pyvista plotter
        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
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

        self.fine_zoom_btn = QPushButton("Fine (F)")
        self.fine_zoom_btn.clicked.connect(self._on_fine_zoom)
        self.fine_zoom_btn.setEnabled(False)
        btn_bar.addWidget(self.fine_zoom_btn)

        self.done_btn = QPushButton("Done (D)")
        self.done_btn.clicked.connect(self._on_done)
        self.done_btn.setEnabled(False)
        btn_bar.addWidget(self.done_btn)

        self.cancel_btn = QPushButton("Cancel (Esc)")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)
        btn_bar.addWidget(self.cancel_btn)

        self.screenshot_btn = QPushButton("Screenshot")
        self.screenshot_btn.clicked.connect(self._on_screenshot)
        btn_bar.addWidget(self.screenshot_btn)

        # Mode label
        self.mode_label = QLabel("")
        self.mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_label.setStyleSheet("font-weight: bold; font-size: 13px;")

        # --- Result fields ---
        result_header = QLabel("Result")
        result_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #ccc;")

        result_row1 = QHBoxLayout()
        result_row1.setContentsMargins(0, 0, 0, 0)

        self.result_class_edit = QLineEdit()
        self.result_class_edit.setReadOnly(True)
        self.result_class_edit.setPlaceholderText("—")
        result_row1.addWidget(QLabel("Class:"))
        result_row1.addWidget(self.result_class_edit)

        self.result_elev_edit = QLineEdit()
        self.result_elev_edit.setReadOnly(True)
        self.result_elev_edit.setPlaceholderText("—")
        result_row1.addWidget(QLabel("Elev:"))
        result_row1.addWidget(self.result_elev_edit)

        self.result_lat_edit = QLineEdit()
        self.result_lat_edit.setReadOnly(True)
        self.result_lat_edit.setPlaceholderText("—")
        result_row1.addWidget(QLabel("Lat:"))
        result_row1.addWidget(self.result_lat_edit)

        self.result_lon_edit = QLineEdit()
        self.result_lon_edit.setReadOnly(True)
        self.result_lon_edit.setPlaceholderText("—")
        result_row1.addWidget(QLabel("Lon:"))
        result_row1.addWidget(self.result_lon_edit)

        result_row2 = QHBoxLayout()
        result_row2.setContentsMargins(0, 0, 0, 0)

        self.result_code_edit = QLineEdit()
        self.result_code_edit.setReadOnly(True)
        self.result_code_edit.setPlaceholderText("—")
        result_row2.addWidget(QLabel("Code:"))
        result_row2.addWidget(self.result_code_edit)

        self.result_units_edit = QLineEdit()
        self.result_units_edit.setReadOnly(True)
        self.result_units_edit.setPlaceholderText("—")
        result_row2.addWidget(QLabel("Units:"))
        result_row2.addWidget(self.result_units_edit)

        self.result_crs_edit = QLineEdit()
        self.result_crs_edit.setReadOnly(True)
        self.result_crs_edit.setPlaceholderText("—")
        result_row2.addWidget(QLabel("CRS:"))
        result_row2.addWidget(self.result_crs_edit)

        self.result_type_edit = QLineEdit()
        self.result_type_edit.setReadOnly(True)
        self.result_type_edit.setPlaceholderText("—")
        result_row2.addWidget(QLabel("Type:"))
        result_row2.addWidget(self.result_type_edit)

        # --- Log ---
        log_header = QLabel("Log")
        log_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #ccc;")

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(120)
        self.results_text.setStyleSheet("font-family: monospace; font-size: 11px;")

        # Route elidar.* log messages to the GUI log widget
        gui_handler = _LogHandler(self.results_text)
        gui_handler.setLevel(logging.INFO)
        logging.getLogger("elidar").addHandler(gui_handler)

        # Vertical sliders to the right of the viewer
        sliders = QHBoxLayout()

        # Z exaggeration slider
        exag_col = QVBoxLayout()
        exag_label = QLabel("Z\nExag")
        exag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        exag_label.setStyleSheet("font-size: 10px;")
        exag_col.addWidget(exag_label)

        self.exag_slider = QSlider(Qt.Orientation.Vertical)
        self.exag_slider.setRange(10, 100)
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
        sliders.addLayout(exag_col)

        # Color stretch slider
        stretch_col = QVBoxLayout()
        stretch_label = QLabel("Color\nFocus")
        stretch_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stretch_label.setStyleSheet("font-size: 10px;")
        stretch_col.addWidget(stretch_label)

        self.stretch_slider = QSlider(Qt.Orientation.Vertical)
        self.stretch_slider.setRange(10, 50)
        self.stretch_slider.setValue(10)
        self.stretch_slider.setTickPosition(QSlider.TickPosition.TicksRight)
        self.stretch_slider.setTickInterval(5)
        self.stretch_slider.setFixedWidth(30)
        self.stretch_slider.valueChanged.connect(self._on_stretch_changed)
        stretch_col.addWidget(self.stretch_slider)

        self.stretch_value_label = QLabel("1.0x")
        self.stretch_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stretch_value_label.setStyleSheet("font-size: 10px;")
        stretch_col.addWidget(self.stretch_value_label)
        sliders.addLayout(stretch_col)

        # Viewer row: stack + sliders
        viewer_row = QHBoxLayout()
        viewer_row.addWidget(self.viewer_stack, stretch=1)
        viewer_row.addLayout(sliders)

        # Toolbar above the plot: View group | mode label | Util group
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(4)

        # View group
        view_group = QGroupBox("View")
        view_group.setStyleSheet("QGroupBox { padding-top: 14px; }")
        view_lay = QHBoxLayout(view_group)
        view_lay.setContentsMargins(4, 2, 4, 2)

        self.view_iso_btn = QPushButton("3D")
        self.view_iso_btn.setToolTip("Isometric view")
        self.view_iso_btn.clicked.connect(lambda: self._set_camera_view("iso"))
        view_lay.addWidget(self.view_iso_btn)

        self.view_top_btn = QPushButton("Top")
        self.view_top_btn.setToolTip("Top-down view")
        self.view_top_btn.clicked.connect(lambda: self._set_camera_view("top"))
        view_lay.addWidget(self.view_top_btn)

        toolbar.addWidget(view_group)
        toolbar.addWidget(self.mode_label, stretch=1)

        # Util group
        util_group = QGroupBox("Util")
        util_group.setStyleSheet("QGroupBox { padding-top: 14px; }")
        util_lay = QHBoxLayout(util_group)
        util_lay.setContentsMargins(4, 2, 4, 2)

        self.find_peak_btn = QPushButton("Find Peak")
        self.find_peak_btn.setEnabled(False)
        self.find_peak_btn.clicked.connect(self._on_find_peak)
        util_lay.addWidget(self.find_peak_btn)

        self.find_saddle_btn = QPushButton("Auto Saddle")
        self.find_saddle_btn.setToolTip(
            "Find key col via minimax path (auto-detects peaks)"
        )
        self.find_saddle_btn.setEnabled(False)
        self.find_saddle_btn.clicked.connect(self._on_find_saddle)
        util_lay.addWidget(self.find_saddle_btn)

        self.select_peaks_btn = QPushButton("Manual Saddle")
        self.select_peaks_btn.setToolTip(
            "Click two points as peak references, then click again to find saddle"
        )
        self.select_peaks_btn.setCheckable(True)
        self.select_peaks_btn.setEnabled(False)
        self.select_peaks_btn.toggled.connect(self._on_select_peaks_toggled)
        util_lay.addWidget(self.select_peaks_btn)

        toolbar.addWidget(util_group)

        # State for manual peak selection
        self._manual_peak_a: int | None = None
        self._manual_peak_b: int | None = None
        self._peak_select_mode: bool = False

        # Assemble right side
        right_layout.addLayout(toolbar)
        right_layout.addSpacing(4)
        right_layout.addLayout(viewer_row, stretch=1)
        right_layout.addLayout(btn_bar)
        right_layout.addWidget(result_header)
        right_layout.addLayout(result_row1)
        right_layout.addLayout(result_row2)
        right_layout.addWidget(log_header)
        right_layout.addWidget(self.results_text)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([300, 600])
        layout.addWidget(splitter)

        # Keyboard shortcuts on the plotter widget
        self.plotter.add_key_event("z", self._on_zoom)
        self.plotter.add_key_event("f", self._on_fine_zoom)
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


        return group

    def _build_query_group(self) -> QGroupBox:
        group = QGroupBox("Query Options")
        form = QFormLayout(group)

        self.guess_center_check = QCheckBox("Use tile center")
        self.guess_center_check.setChecked(True)
        self.guess_center_check.toggled.connect(self._on_guess_center_toggled)
        form.addRow(self.guess_center_check)

        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setRange(-90.0, 90.0)
        self.lat_spin.setDecimals(6)
        self.lat_spin.setValue(float(self.settings.value("last_lat", 48.7768)))
        self.lat_spin.setEnabled(False)
        form.addRow("Latitude:", self.lat_spin)

        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setRange(-180.0, 180.0)
        self.lon_spin.setDecimals(6)
        self.lon_spin.setValue(float(self.settings.value("last_lon", -121.8145)))
        self.lon_spin.setEnabled(False)
        form.addRow("Longitude:", self.lon_spin)

        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.0001, 1.0)
        self.radius_spin.setDecimals(4)
        self.radius_spin.setSingleStep(0.001)
        self.radius_spin.setValue(0.05)
        form.addRow("Search Radius (deg):", self.radius_spin)

        self.query_type_combo = QComboBox()
        for qt in QueryType:
            self.query_type_combo.addItem(qt.value.capitalize(), qt)
        self.query_type_combo.setCurrentIndex(0)
        form.addRow("Query Type:", self.query_type_combo)

        self.clean_check = QCheckBox()
        self.clean_check.setChecked(True)
        form.addRow("Clean Data:", self.clean_check)

        return group

    def _on_guess_center_toggled(self, checked: bool) -> None:
        self.lat_spin.setEnabled(not checked)
        self.lon_spin.setEnabled(not checked)

    def _build_options_group(self) -> QGroupBox:
        group = QGroupBox("Display Options")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(2, 2, 2, 2)

        self.options_tabs = QTabWidget()
        outer.addWidget(self.options_tabs)

        # --- Basic tab ---
        basic = QWidget()
        form = QFormLayout(basic)
        form.setContentsMargins(4, 4, 4, 4)

        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(1.5, 10.0)
        self.zoom_spin.setDecimals(1)
        self.zoom_spin.setSingleStep(0.5)
        self.zoom_spin.setValue(2.0)
        self.zoom_spin.setSuffix("x")
        form.addRow("Coarse Zoom:", self.zoom_spin)

        self.fine_zoom_spin = QDoubleSpinBox()
        self.fine_zoom_spin.setRange(1.1, 3.0)
        self.fine_zoom_spin.setDecimals(1)
        self.fine_zoom_spin.setSingleStep(0.1)
        self.fine_zoom_spin.setValue(1.3)
        self.fine_zoom_spin.setSuffix("x")
        form.addRow("Fine Zoom:", self.fine_zoom_spin)

        self.display_unit_combo = QComboBox()
        self.display_unit_combo.addItem("Feet", ("ft (international)", M2FT))
        self.display_unit_combo.addItem("Feet (US Survey)", ("ft (US survey)", M2FT_SURVEY))
        self.display_unit_combo.addItem("Meters", ("m", 1.0))
        self.display_unit_combo.currentIndexChanged.connect(self._on_display_unit_changed)
        form.addRow("Display In:", self.display_unit_combo)

        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 20)
        self.point_size_spin.setValue(15)
        self.point_size_spin.valueChanged.connect(self._on_point_size_changed)
        form.addRow("Point Size:", self.point_size_spin)

        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(1000, 500000)
        self.max_points_spin.setSingleStep(10000)
        self.max_points_spin.setValue(50000)
        self.max_points_spin.valueChanged.connect(self._on_max_points_changed)
        form.addRow("Max Points:", self.max_points_spin)

        self.bg_combo = QComboBox()
        self.bg_combo.addItem("Black", "black")
        self.bg_combo.addItem("Gray", "gray")
        self.bg_combo.addItem("White", "white")
        self.bg_combo.currentIndexChanged.connect(self._on_bg_changed)
        form.addRow("Background:", self.bg_combo)

        self.unit_combo = QComboBox()
        self.unit_combo.addItem("Auto", None)
        self.unit_combo.addItem("Feet", "ft")
        self.unit_combo.addItem("Meters", "m")
        form.addRow("Force Unit:", self.unit_combo)

        self.options_tabs.addTab(basic, "Basic")

        # --- Advanced tab (classification colors, populated after data loads) ---
        self._adv_widget = QWidget()
        self._class_color_combos: dict[int, QComboBox] = {}
        self._class_visible_checks: dict[int, QCheckBox] = {}
        self.class_color_layout = QFormLayout(self._adv_widget)
        self.class_color_layout.setContentsMargins(4, 4, 4, 4)
        self._adv_placeholder = QLabel("Load data to see classifications")
        self._adv_placeholder.setStyleSheet("color: #888;")
        self.class_color_layout.addRow(self._adv_placeholder)
        self.options_tabs.addTab(self._adv_widget, "Classifications")

        self.class_color_group = QWidget()
        self.class_color_group.setVisible(False)

        # Keep color_combo reference for _get_options (default colorscale)
        self.color_combo = None

        return group

    def _populate_class_colors(self, classifications: list[int]) -> None:
        """Build per-classification colormap dropdowns + visibility checkboxes."""
        while self.class_color_layout.rowCount() > 0:
            self.class_color_layout.removeRow(0)
        self._class_color_combos.clear()
        self._class_visible_checks.clear()

        # Default: show surface classes (Ground + Snow/Ice).
        # If none are present, show all so the view isn't empty.
        has_surface = bool(SURFACE_CLASSES & set(classifications))

        for cls in sorted(classifications):
            row = QHBoxLayout()

            check = QCheckBox()
            check.setChecked(cls in SURFACE_CLASSES if has_surface else True)
            check.toggled.connect(lambda checked, c=cls: self._on_class_visibility_toggled(c, checked))
            self._class_visible_checks[cls] = check
            row.addWidget(check)

            combo = QComboBox()
            for cs in ColorScale:
                combo.addItem(cs.value.capitalize(), cs)
            default = DEFAULT_CLASS_CMAPS.get(cls, DEFAULT_CMAP)
            combo.setCurrentText(default.value.capitalize())
            combo.currentIndexChanged.connect(self._on_live_option_changed)
            self._class_color_combos[cls] = combo
            row.addWidget(combo, stretch=1)

            name = ASPRS_CLASSIFICATION_NAMES.get(cls, f"Class {cls}")
            self.class_color_layout.addRow(f"{name}:", row)

    def _get_class_cmaps(self) -> dict[int, ColorScale]:
        """Read the current per-classification colormap selections."""
        return {cls: combo.currentData() for cls, combo in self._class_color_combos.items()}

    def _on_class_visibility_toggled(self, cls: int, visible: bool) -> None:
        self.embedded_search.set_class_visible(cls, visible)

    def _on_live_option_changed(self) -> None:
        """Re-render with updated colormaps when color combo changes."""
        if self.embedded_search.is_active:
            QTimer.singleShot(0, lambda: self.embedded_search.set_class_cmaps(self._get_class_cmaps()))

    def _on_display_unit_changed(self) -> None:
        if self.embedded_search.is_active:
            unit_label, scale = self.display_unit_combo.currentData()
            QTimer.singleShot(0, lambda: self.embedded_search.set_display_unit(unit_label, scale))

    def _on_point_size_changed(self, value: int) -> None:
        if self.embedded_search.is_active:
            QTimer.singleShot(0, lambda: self.embedded_search.set_point_size(value))

    def _on_max_points_changed(self, value: int) -> None:
        if self.embedded_search.is_active:
            QTimer.singleShot(0, lambda: self.embedded_search.set_max_points(value))

    def _set_camera_view(self, view: str) -> None:
        """Set camera to a preset view, keeping the current focal point."""
        fp = self.plotter.camera.focal_point
        if view == "top":
            self.plotter.camera_position = "xy"
            self.plotter.camera.focal_point = fp
        elif view == "iso":
            self.plotter.camera_position = "xz"
            self.plotter.camera.focal_point = fp
            self.plotter.camera.elevation = 25
            self.plotter.camera.azimuth = -60
        self.plotter.render()

    def _on_bg_changed(self) -> None:
        color = self.bg_combo.currentData()
        if self.embedded_search.is_active:
            QTimer.singleShot(0, lambda: self.embedded_search.set_background(color))
        else:
            QTimer.singleShot(0, lambda: self.plotter.set_background(color))

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
            colorscale=ColorScale.TURBO,
            filter_ground=False,
            query_radius_deg=self.radius_spin.value(),
            zoom_scale=1.0 / self.zoom_spin.value(),  # default for CLI; GUI passes directly
            force_unit=self.unit_combo.currentData(),
            dem_path=None,
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
        self._log(f"Running {query_type.value} query...\n")

        # Disconnect any stale worker signals
        if self._worker is not None:
            try:
                self._worker.disconnect()
            except (TypeError, RuntimeError):
                pass

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
        self._last_crs_name = crs_info.source if crs_info else ""

        # Populate classification color dropdowns from detected classes
        import numpy as np
        if cloud.classification is not None:
            unique_cls = sorted(int(c) for c in np.unique(cloud.classification))
        else:
            unique_cls = []
        self._populate_class_colors(unique_cls)

        options = self._get_options()
        unit_label, unit_scale = self.display_unit_combo.currentData()
        class_cmaps = self._get_class_cmaps()
        visible_classes = {
            cls for cls, check in self._class_visible_checks.items()
            if check.isChecked()
        }
        self.viewer_stack.setCurrentIndex(1)
        self.embedded_search.load(
            cloud, crs_info, options,
            display_unit=unit_label,
            unit_scale=unit_scale,
            class_cmaps=class_cmaps,
            visible_classes=visible_classes,
            max_points=self.max_points_spin.value(),
        )

        self._set_search_controls(True)
        self._update_mode_label()
        self._log(f"Loaded {cloud.n_points:,} points. Click to select.")

    def _on_query_finished(self, result: object) -> None:
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze")
        self._display_result(result)

    def _on_query_error(self, msg: str) -> None:
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze")
        self._log(f"ERROR: {msg}\n")
        logger.error("Query failed: %s", msg)

    def _on_screenshot(self) -> None:
        """Save the current 3D view as a PNG."""
        last_dir = self.settings.value("last_dir", os.path.expanduser("~"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", str(Path(last_dir) / "elidar_screenshot.png"),
            "PNG Images (*.png);;All Files (*)",
        )
        if path:
            self.plotter.screenshot(path)
            self._log(f"Screenshot saved: {path}")

    # --- Search controls ---

    def _set_search_controls(self, enabled: bool) -> None:
        self.find_peak_btn.setEnabled(enabled)
        self.zoom_btn.setEnabled(enabled)
        self.fine_zoom_btn.setEnabled(enabled)
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

        # Enable saddle when at full resolution (not subsampled) and enough points
        n = self.embedded_search.visible_point_count
        saddle_ok = not self.embedded_search.is_subsampled and n >= 10
        self.find_saddle_btn.setEnabled(saddle_ok)
        self.select_peaks_btn.setEnabled(saddle_ok)

    def _on_find_peak(self) -> None:
        self._last_select_type = "Peak"
        self.embedded_search.find_peak()

    def _on_find_saddle(self) -> None:
        self._last_select_type = "Saddle"
        if self._manual_peak_a is not None and self._manual_peak_b is not None:
            self.embedded_search.find_saddle(self._manual_peak_a, self._manual_peak_b)
        else:
            self.embedded_search.find_saddle()
        # Reset peak selection mode
        self._manual_peak_a = None
        self._manual_peak_b = None
        self._peak_select_mode = False
        self.select_peaks_btn.setChecked(False)

    def _on_select_peaks_toggled(self, checked: bool) -> None:
        self._peak_select_mode = checked
        if checked:
            self._manual_peak_a = None
            self._manual_peak_b = None
            self._log("Peak selection mode: click Peak A, then Peak B, then Find Saddle")
        else:
            self._manual_peak_a = None
            self._manual_peak_b = None

    def _on_zoom(self) -> None:
        scale = 1.0 / self.zoom_spin.value()
        self.embedded_search.zoom(scale)
        self._update_mode_label()

    def _on_fine_zoom(self) -> None:
        scale = 1.0 / self.fine_zoom_spin.value()
        self.embedded_search.zoom(scale)
        self._update_mode_label()

    def _on_done(self) -> None:
        result = self.embedded_search.done()
        if result is not None:
            self._set_search_controls(False)
            self._update_mode_label()
            self.exag_slider.setValue(10)  # Reset to 1.0x

    def _on_cancel(self) -> None:
        self.embedded_search.cancel()
        self._set_search_controls(False)
        self.find_saddle_btn.setEnabled(False)
        self.mode_label.setText("")
        self.viewer_stack.setCurrentIndex(0)
        self._log("Search cancelled.\n")

    def _on_exag_changed(self, value: int) -> None:
        factor = value / 10.0
        self.exag_value_label.setText(f"{factor:.1f}x")
        self.embedded_search.set_z_exaggeration(factor)

    def _on_stretch_changed(self, value: int) -> None:
        stretch = value / 10.0
        self.stretch_value_label.setText(f"{stretch:.1f}x")
        if self.embedded_search.is_active:
            QTimer.singleShot(0, lambda: self.embedded_search.set_color_stretch(stretch))

    # --- Search callbacks ---

    def _on_search_status(self, msg: str) -> None:
        if msg.startswith("Selected:") and self._peak_select_mode:
            idx = self.embedded_search._pick_idx
            if self._manual_peak_a is None:
                self._manual_peak_a = idx
                self._log("Peak A set. Now click Peak B.\n")
                return
            elif self._manual_peak_b is None:
                self._manual_peak_b = idx
                self._log("Peak B set. Click Find Saddle.\n")
                return
        elif msg.startswith("Selected:"):
            self._last_select_type = "Manual"
        self._log(msg + "\n")

    def _on_search_result(self, pick: PickResult) -> None:
        result = QueryResult(
            elevation_m=pick.z,
            elevation_ft=pick.z * M2FT,
            location_local=(pick.x_local, pick.y_local),
            location_deg=(pick.lat, pick.lon),
            query_type=QueryType.SEARCH,
            classification=pick.classification,
        )
        self._display_result(result)

    # --- Results ---

    def _log(self, msg: str) -> None:
        """Append a message to the info terminal."""
        self.results_text.append(msg)

    def _clear_result_fields(self) -> None:
        self.result_class_edit.clear()
        self.result_elev_edit.clear()
        self.result_lat_edit.clear()
        self.result_lon_edit.clear()
        self.result_code_edit.clear()
        self.result_units_edit.clear()
        self.result_crs_edit.clear()
        self.result_type_edit.clear()

    def _display_result(self, result: object) -> None:
        if result is None:
            self._log("No result (point not in data or cancelled)")
            return

        self._last_result = result

        # Populate result fields
        if result.classification is not None:
            cls_name = ASPRS_CLASSIFICATION_NAMES.get(result.classification, f"Class {result.classification}")
            self.result_class_edit.setText(cls_name)
        else:
            self.result_class_edit.setText("N/A")

        unit_label, unit_scale = self.display_unit_combo.currentData()
        self.result_elev_edit.setText(f"{result.elevation_m * unit_scale:,.2f}")
        self.result_lat_edit.setText(f"{result.location_deg[0]:.6f}")
        self.result_lon_edit.setText(f"{result.location_deg[1]:.6f}")

        # Second row
        self.result_code_edit.setText(
            str(result.classification) if result.classification is not None else "N/A"
        )
        self.result_units_edit.setText(unit_label)
        self.result_crs_edit.setText(self._last_crs_name)
        self.result_type_edit.setText(self._last_select_type or "Manual")

        # Log to terminal (always both units)
        self._log(
            f"[{result.query_type.value.upper()}] "
            f"{result.elevation_ft:,.2f} ft  ({result.elevation_m:,.2f} m)  "
            f"at ({result.location_deg[0]:.6f}, {result.location_deg[1]:.6f})"
        )
        if self._tools_tab is not None:
            self._tools_tab.set_last_result_ft(result.elevation_ft)

