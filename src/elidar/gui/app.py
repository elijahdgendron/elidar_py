"""ELiDAR main GUI application window."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from elidar.gui.main_tab import MainTab
from elidar.gui.tools_tab import ToolsTab

logger = logging.getLogger(__name__)


class ELiDARApp(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ELiDAR - Point Cloud Peak/Saddle Analysis and Visualization for Peakbaggers")
        self.setMinimumSize(950, 700)

        self.settings = QSettings("ELiDAR", "ELiDAR")
        self.resize(950, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.main_tab = MainTab(self.settings)
        self.tools_tab = ToolsTab()
        self.main_tab.set_tools_tab(self.tools_tab)

        self.tabs.addTab(self.main_tab, "Analysis")
        self.tabs.addTab(self.tools_tab, "Notes")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        super().closeEvent(event)

    def _restore_geometry(self) -> None:
        geom = self.settings.value("geometry")
        if geom:
            self.restoreGeometry(geom)
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)


def run_gui() -> int:
    """Launch the ELiDAR GUI application.

    :return: Exit code.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("ELiDAR")
    app.setOrganizationName("ELiDAR")

    window = ELiDARApp()
    window.show()

    return app.exec()
