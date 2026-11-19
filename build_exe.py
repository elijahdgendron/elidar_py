"""Build ELiDAR as a standalone Windows .exe using Nuitka.

Run on Windows with Python 3.10 or 3.11:

    pip install nuitka ordered-set
    python build_exe.py

Or install the build extras first:

    pip install -e ".[build]"
    python build_exe.py

Output: dist/ELiDAR.exe
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ENTRY_POINT = Path("src") / "elidar" / "cli.py"
OUTPUT_DIR = "dist"
OUTPUT_NAME = "ELiDAR.exe"

INCLUDE_PACKAGES = [
    "elidar",
    "pyvista",
    "pyvistaqt",
    "vtkmodules",
    "open3d",
    "laspy",
    "lazrs",
    "pyproj",
    "rasterio",
    "scipy",
    "numpy",
    "matplotlib",
]

INCLUDE_PACKAGE_DATA = [
    "elidar",
    "pyproj",
    "rasterio",
    "certifi",
]

EXCLUDE_IMPORTS = [
    "pytest",
    "ruff",
    "tkinter",
    "IPython",
]


def _check_prerequisites() -> bool:
    """Verify that required tools are available.

    :return: True if all prerequisites are met.
    """
    ok = True

    major, minor = sys.version_info[:2]
    print(f"Python {major}.{minor}.{sys.version_info[2]}")
    if not (major == 3 and minor in (10, 11)):
        print(
            f"  WARNING: Python 3.10 or 3.11 recommended for Open3D "
            f"compatibility (found {major}.{minor})"
        )

    if sys.platform != "win32":
        print("  WARNING: This script builds a Windows .exe. Run it on Windows.")
        print("           Nuitka does not support cross-compilation.")
        ok = False

    for mod_name in ("nuitka", "ordered_set"):
        try:
            importlib.import_module(mod_name)
            print(f"  {mod_name}: found")
        except ImportError:
            print(f"  {mod_name}: NOT FOUND — pip install {mod_name.replace('_', '-')}")
            ok = False

    if not ENTRY_POINT.exists():
        print(f"  Entry point not found: {ENTRY_POINT}")
        print("  Run this script from the project root directory.")
        ok = False

    return ok


def _build_command() -> list[str]:
    """Assemble the Nuitka command line.

    :return: Command as a list of strings.
    """
    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--standalone",
        "--onefile",
        "--enable-plugin=pyqt6",
    ]

    for pkg in INCLUDE_PACKAGES:
        cmd.append(f"--include-package={pkg}")

    for pkg in INCLUDE_PACKAGE_DATA:
        cmd.append(f"--include-package-data={pkg}")

    for mod in EXCLUDE_IMPORTS:
        cmd.append(f"--nofollow-import-to={mod}")

    cmd.extend([
        "--windows-console-mode=attach",
        f"--output-filename={OUTPUT_NAME}",
        f"--output-dir={OUTPUT_DIR}",
        str(ENTRY_POINT),
    ])

    return cmd


def main() -> int:
    """Run the Nuitka build.

    :return: Exit code.
    """
    print("=" * 60)
    print("ELiDAR — Nuitka .exe Build")
    print("=" * 60)
    print()

    print("Checking prerequisites...")
    if not _check_prerequisites():
        print("\nPrerequisites not met. Fix the issues above and retry.")
        return 1

    print()

    cmd = _build_command()
    print("Nuitka command:")
    print(f"  {' '.join(cmd)}")
    print()
    print("Starting build (this may take 10-30 minutes on first run)...")
    print("-" * 60)

    result = subprocess.run(cmd)

    print("-" * 60)

    if result.returncode != 0:
        print(f"\nBuild FAILED (exit code {result.returncode})")
        return result.returncode

    exe_path = Path(OUTPUT_DIR) / OUTPUT_NAME
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\nBuild SUCCEEDED")
        print(f"  Output: {exe_path}")
        print(f"  Size:   {size_mb:.1f} MB")
        print()
        print("Smoke test:")
        print(f"  {exe_path}              (launch GUI)")
        print(f"  {exe_path} --help       (show CLI help)")
    else:
        print(f"\nBuild completed but {exe_path} not found — check Nuitka output above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
