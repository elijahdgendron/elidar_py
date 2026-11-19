"""Notes tab: scratch notes and prominence calculator."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class ToolsTab(QWidget):
    """Notes tab with prominence calculator and scratch notepad."""

    def __init__(self) -> None:
        super().__init__()
        self._unit_label: str = "ft (international)"
        self._unit_scale: float = 1.0  # multiply meters to get display unit
        self._last_elevation_display: float | None = None
        self._default_save_dir: str = os.path.expanduser("~")

        layout = QVBoxLayout(self)

        layout.addWidget(self._build_prominence_group())
        layout.addWidget(self._build_scratch_group(), stretch=1)

    def _build_prominence_group(self) -> QGroupBox:
        group = QGroupBox("Prominence Calculator")
        outer = QHBoxLayout(group)

        form = QFormLayout()

        self.peak_elev_edit = QLineEdit()
        self.peak_elev_edit.setPlaceholderText("ft")
        self._peak_row_label = QLabel("Peak (ft):")
        form.addRow(self._peak_row_label, self.peak_elev_edit)

        self.saddle_elev_edit = QLineEdit()
        self.saddle_elev_edit.setPlaceholderText("ft")
        self._saddle_row_label = QLabel("Saddle (ft):")
        form.addRow(self._saddle_row_label, self.saddle_elev_edit)

        self.prom_label = QLabel("")
        self.prom_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        form.addRow("Prominence:", self.prom_label)

        outer.addLayout(form)

        btn_col = QVBoxLayout()
        self.calc_prom_btn = QPushButton("Calculate")
        self.calc_prom_btn.clicked.connect(self._on_calc_prominence)
        btn_col.addWidget(self.calc_prom_btn)

        self.copy_peak_btn = QPushButton("Last Result → Peak")
        self.copy_peak_btn.clicked.connect(lambda: self._copy_to_prominence("peak"))
        btn_col.addWidget(self.copy_peak_btn)

        self.copy_saddle_btn = QPushButton("Last Result → Saddle")
        self.copy_saddle_btn.clicked.connect(lambda: self._copy_to_prominence("saddle"))
        btn_col.addWidget(self.copy_saddle_btn)

        btn_col.addStretch()
        outer.addLayout(btn_col)

        return group

    def _build_scratch_group(self) -> QGroupBox:
        group = QGroupBox("Scratch Notes")
        layout = QVBoxLayout(group)

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText(
            "Take notes here. Save to a text file when done."
        )
        layout.addWidget(self.notes_edit)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save Notes...")
        save_btn.clicked.connect(self._on_save_notes)
        btn_row.addWidget(save_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.notes_edit.clear)
        btn_row.addWidget(clear_btn)
        layout.addLayout(btn_row)

        return group

    # --- Prominence ---

    def set_display_unit(self, unit_label: str, unit_scale: float) -> None:
        """Update the display unit used by the prominence calculator.

        :param unit_label: Short unit string (e.g. "ft (international)", "m").
        :param unit_scale: Multiply meters by this to get the display unit.
        """
        self._unit_label = unit_label
        self._unit_scale = unit_scale
        short = self._short_unit()
        self._peak_row_label.setText(f"Peak ({short}):")
        self._saddle_row_label.setText(f"Saddle ({short}):")
        self.peak_elev_edit.setPlaceholderText(short)
        self.saddle_elev_edit.setPlaceholderText(short)

    def set_last_result(self, elevation_display: float) -> None:
        """Store the last query result elevation for copy buttons.

        :param elevation_display: Elevation in the current display unit.
        """
        self._last_elevation_display = elevation_display

    def _short_unit(self) -> str:
        """Return a compact unit abbreviation for labels."""
        label = self._unit_label.lower()
        if "survey" in label:
            return "ft-us"
        if "ft" in label or "feet" in label or "foot" in label:
            return "ft"
        return "m"

    def _on_calc_prominence(self) -> None:
        try:
            peak = float(self.peak_elev_edit.text())
            saddle = float(self.saddle_elev_edit.text())
        except ValueError:
            self.prom_label.setText("Enter valid numbers")
            return

        prom = peak - saddle
        short = self._short_unit()
        self.prom_label.setText(f"{prom:,.2f} {short}")

    def _copy_to_prominence(self, target: str) -> None:
        if self._last_elevation_display is None:
            return
        val = f"{self._last_elevation_display:.2f}"
        if target == "peak":
            self.peak_elev_edit.setText(val)
        else:
            self.saddle_elev_edit.setText(val)

    # --- Notes ---

    def append_note(self, text: str) -> None:
        """Append a line to the scratch notes.

        :param text: Text to append.
        """
        current = self.notes_edit.toPlainText()
        if current and not current.endswith("\n"):
            self.notes_edit.append("")
        self.notes_edit.append(text)

    def _on_save_notes(self) -> None:
        text = self.notes_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Empty", "No notes to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Notes", str(Path(self._default_save_dir) / "elidar_notes.txt"),
            "Text Files (*.txt);;All Files (*)",
        )
        if path:
            Path(path).write_text(text, encoding="utf-8")
            logger.info("Notes saved to %s", path)

    def has_unsaved_notes(self) -> bool:
        """Check if there are unsaved notes."""
        return bool(self.notes_edit.toPlainText().strip())
