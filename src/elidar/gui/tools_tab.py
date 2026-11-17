"""Tools tab: scratch notes."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class ToolsTab(QWidget):
    """Tools tab with scratch notepad."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        layout.addWidget(self._build_scratch_group())

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
