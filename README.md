# ELiDAR_py

Free, open-source tool for analyzing LiDAR point cloud data to find precise peak and saddle elevations and compute topographic prominence. Built for peakbaggers.

## Quick Start

```bash
cd elidar_py
source env/bin/activate   # Linux/Mac
elidar                    # Launch GUI
```

## Installation

```bash
cd elidar_py
python -m venv env
source env/bin/activate       # Linux/Mac
# env\Scripts\activate        # Windows
pip install -e ".[gui]"       # Full install with GUI
```

For CLI-only (no GUI):
```bash
pip install -e ".[viz]"       # Core + CLI + interactive search
```

## Data Sources

Download LAS/LAZ files from your state's public LiDAR portal:
- **Colorado**: https://coloradohazardmapping.com/lidarDownload
- **Washington**: https://lidarportal.dnr.wa.gov/
- **USGS 3DEP**: https://apps.nationalmap.gov/downloader/

You only need the LAZ/LAS file. CRS is read directly from the file.

## Documentation

See [FEATURES.md](FEATURES.md) for a full guide to the GUI, tools, display options, and saddle analysis workflow.

## Dependencies

- `laspy[lazrs]` -- LAS/LAZ file reading
- `pyproj` -- Coordinate reprojection
- `numpy`, `scipy` -- Array operations, Delaunay triangulation
- `open3d` -- Point cloud denoising
- `pyvista` + `pyvistaqt` -- 3D interactive visualization
- `PyQt6` -- Desktop GUI
- `rasterio` -- DEM reading (optional CRS fallback)
