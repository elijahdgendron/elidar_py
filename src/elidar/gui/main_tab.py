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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from elidar.core.types import ColorScale, QueryOptions, QueryType

logger = logging.getLogger(__name__)


class _QueryWorker(QThread):
    """Background thread for running LiDAR queries."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

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

            self.progress.emit("Loading and processing LiDAR data...")
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
    """Primary analysis tab with file selection, parameters, and results."""

    def __init__(self, settings: QSettings) -> None:
        super().__init__()
        self.settings = settings
        self._worker: _QueryWorker | None = None

        layout = QHBoxLayout(self)

        # Left panel: controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(self._build_file_group())
        left_layout.addWidget(self._build_query_group())
        left_layout.addWidget(self._build_options_group())
        left_layout.addWidget(self._build_action_group())
        left_layout.addStretch()

        # Right panel: results
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._build_results_group())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([400, 300])
        layout.addWidget(splitter)

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
        self.file_list.setMaximumHeight(100)
        layout.addWidget(self.file_list)

        # DEM (optional)
        dem_row = QHBoxLayout()
        dem_row.addWidget(QLabel("DEM (optional):"))
        self.dem_edit = QLineEdit()
        self.dem_edit.setPlaceholderText("Only needed if LAZ has no CRS")
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

    def _build_results_group(self) -> QGroupBox:
        group = QGroupBox("Results")
        layout = QVBoxLayout(group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self.results_text)

        # Prominence calculator
        prom_group = QGroupBox("Prominence Calculator")
        prom_layout = QFormLayout(prom_group)

        self.peak_elev_edit = QLineEdit()
        self.peak_elev_edit.setPlaceholderText("ft")
        prom_layout.addRow("Peak Elev (ft):", self.peak_elev_edit)

        self.saddle_elev_edit = QLineEdit()
        self.saddle_elev_edit.setPlaceholderText("ft")
        prom_layout.addRow("Saddle Elev (ft):", self.saddle_elev_edit)

        self.calc_prom_btn = QPushButton("Calculate")
        self.calc_prom_btn.clicked.connect(self._on_calc_prominence)
        prom_layout.addRow(self.calc_prom_btn)

        self.prom_label = QLabel("")
        self.prom_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        prom_layout.addRow("Prominence:", self.prom_label)

        layout.addWidget(prom_group)

        # Copy buttons
        btn_row = QHBoxLayout()
        copy_peak_btn = QPushButton("Copy Peak to Prominence")
        copy_peak_btn.clicked.connect(lambda: self._copy_to_prominence("peak"))
        btn_row.addWidget(copy_peak_btn)

        copy_saddle_btn = QPushButton("Copy Saddle to Prominence")
        copy_saddle_btn.clicked.connect(lambda: self._copy_to_prominence("saddle"))
        btn_row.addWidget(copy_saddle_btn)
        layout.addLayout(btn_row)

        return group

    # --- Slots ---

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

    def _on_analyze(self) -> None:
        paths = self._get_las_paths()
        if not paths:
            QMessageBox.warning(self, "No Files", "Add at least one LAS/LAZ file.")
            return

        lat = self.lat_spin.value()
        lon = self.lon_spin.value()
        query_type = self.query_type_combo.currentData()

        self.settings.setValue("last_lat", lat)
        self.settings.setValue("last_lon", lon)

        dem = self.dem_edit.text().strip() or None
        force_unit = self.unit_combo.currentData()

        options = QueryOptions(
            clean_data=self.clean_check.isChecked(),
            colorscale=self.color_combo.currentData(),
            filter_ground=self.ground_check.isChecked(),
            query_radius_deg=self.radius_spin.value(),
            zoom_scale=self.zoom_spin.value(),
            force_unit=force_unit,
            dem_path=dem,
        )

        if query_type == QueryType.SEARCH:
            self._run_search(lat, lon, paths, options)
        else:
            self._run_background_query(lat, lon, query_type, paths, options)

    def _run_search(
        self, lat: float, lon: float, paths: list[str], options: QueryOptions
    ) -> None:
        """Run search mode directly (it opens its own pyvista window)."""
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("Analyzing...")
        self.results_text.append("Running search mode...\n")

        try:
            from elidar.core.query import query_lidar

            result = query_lidar(
                query_lat=lat,
                query_lon=lon,
                query_type=QueryType.SEARCH,
                las_path=paths if len(paths) > 1 else paths[0],
                options=options,
            )
            self._display_result(result)
        except Exception as e:
            self._display_error(str(e))
        finally:
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText("Analyze")

    def _run_background_query(
        self,
        lat: float,
        lon: float,
        query_type: QueryType,
        paths: list[str],
        options: QueryOptions,
    ) -> None:
        """Run peak/point query in a background thread."""
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("Analyzing...")
        self.results_text.append(f"Running {query_type.value} query...\n")

        self._worker = _QueryWorker(lat, lon, query_type, paths, options)
        self._worker.finished.connect(self._on_query_finished)
        self._worker.error.connect(self._on_query_error)
        self._worker.start()

    def _on_query_finished(self, result: object) -> None:
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze")
        self._display_result(result)

    def _on_query_error(self, msg: str) -> None:
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze")
        self._display_error(msg)

    def _display_result(self, result: object) -> None:
        if result is None:
            self.results_text.append("No result (point not in data or cancelled)\n")
            return

        from elidar.core.types import M2FT

        self._last_result = result
        self.results_text.append(
            f"--- {result.query_type.value.upper()} ---\n"
            f"  Elevation: {result.elevation_ft:,.2f} ft  ({result.elevation_m:,.2f} m)\n"
            f"  Location:  {result.location_deg[0]:.6f}°N, {result.location_deg[1]:.6f}°W\n"
        )

    def _display_error(self, msg: str) -> None:
        self.results_text.append(f"ERROR: {msg}\n")
        logger.error("Query failed: %s", msg)

    def _on_calc_prominence(self) -> None:
        try:
            peak = float(self.peak_elev_edit.text())
            saddle = float(self.saddle_elev_edit.text())
        except ValueError:
            self.prom_label.setText("Enter valid numbers")
            return

        prom_ft = peak - saddle
        prom_m = prom_ft / 3.2808398950131233
        self.prom_label.setText(f"{prom_ft:,.2f} ft  ({prom_m:,.2f} m)")

    def _copy_to_prominence(self, target: str) -> None:
        if not hasattr(self, "_last_result") or self._last_result is None:
            return
        val = f"{self._last_result.elevation_ft:.2f}"
        if target == "peak":
            self.peak_elev_edit.setText(val)
        else:
            self.saddle_elev_edit.setText(val)
