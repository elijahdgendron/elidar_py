"""Tools tab: scratch notes and prominence calculator."""

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

from elidar.core.types import M2FT

logger = logging.getLogger(__name__)


class ToolsTab(QWidget):
    """Tools tab with prominence calculator and scratch notepad."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        layout.addWidget(self._build_prominence_group())
        layout.addWidget(self._build_scratch_group(), stretch=1)

    def _build_prominence_group(self) -> QGroupBox:
        group = QGroupBox("Prominence Calculator")
        outer = QHBoxLayout(group)

        form = QFormLayout()

        self.peak_elev_edit = QLineEdit()
        self.peak_elev_edit.setPlaceholderText("ft")
        form.addRow("Peak (ft):", self.peak_elev_edit)

        self.saddle_elev_edit = QLineEdit()
        self.saddle_elev_edit.setPlaceholderText("ft")
        form.addRow("Saddle (ft):", self.saddle_elev_edit)

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

    def set_last_result_ft(self, elevation_ft: float) -> None:
        """Store the last query result elevation for copy buttons.

        :param elevation_ft: Elevation in feet from the last query.
        """
        self._last_elevation_ft = elevation_ft

    def _on_calc_prominence(self) -> None:
        try:
            peak = float(self.peak_elev_edit.text())
            saddle = float(self.saddle_elev_edit.text())
        except ValueError:
            self.prom_label.setText("Enter valid numbers")
            return

        prom_ft = peak - saddle
        prom_m = prom_ft / M2FT
        self.prom_label.setText(f"{prom_ft:,.2f} ft  ({prom_m:,.2f} m)")

    def _copy_to_prominence(self, target: str) -> None:
        if not hasattr(self, "_last_elevation_ft"):
            return
        val = f"{self._last_elevation_ft:.2f}"
        if target == "peak":
            self.peak_elev_edit.setText(val)
        else:
            self.saddle_elev_edit.setText(val)

    # --- Notes ---

    def _on_save_notes(self) -> None:
        text = self.notes_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Empty", "No notes to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Notes", os.path.expanduser("~"),
            "Text Files (*.txt);;All Files (*)",
        )
        if path:
            Path(path).write_text(text, encoding="utf-8")
            logger.info("Notes saved to %s", path)

    def has_unsaved_notes(self) -> bool:
        """Check if there are unsaved notes."""
        return bool(self.notes_edit.toPlainText().strip())
