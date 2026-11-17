# ELiDAR_py

Python tool for analyzing LiDAR data to find precise peak and saddle elevations and compute topographic prominence. Port of the original MATLAB ELiDAR application with key improvements.

## Key Improvements over MATLAB Version

- **No MATLAB license required** -- fully free and open-source
- **CRS read from LAZ files directly** -- no DEM file needed (was mandatory in MATLAB)
- **Multi-tile support** -- load multiple LAZ files at once, no separate combine step
- **Correct US survey foot handling** -- distinguishes US survey foot from international foot
- **Keyboard-driven search** -- Z/D/Escape instead of dialog boxes

## Quick Start (existing install)

```bash
cd elidar_py
source env/bin/activate
elidar                    # Launch GUI
```

## Installation (first time)

```bash
cd elidar_py
python -m venv env
source env/bin/activate       # Linux/Mac
# env\Scripts\activate        # Windows
pip install -e ".[gui]"       # Full install with GUI
```

For CLI-only (no GUI):
```bash
pip install -e "."            # Core + CLI
pip install -e ".[viz]"       # Core + CLI + interactive search mode
```

## Usage

### GUI (Recommended)

```bash
elidar          # Launch the desktop GUI
elidar gui      # Same thing
```

1. Click **Add Files** or **Add Folder** to load LAS/LAZ files
2. Enter the latitude and longitude of your query point
3. Select a query type:
   - **Search**: Interactive 3D scatter plot -- click a point, zoom in, repeat
   - **Peak**: Automatically finds the highest point near the query location
   - **Point**: Returns the closest LiDAR return to the query coordinates
4. Click **Analyze**
5. Use the **Prominence Calculator** to compute peak - saddle elevation

### Command Line

```bash
# Find the peak elevation near a point
elidar peak --lat 48.7768 --lon -121.8145 --las baker_summit.laz --radius 0.002

# Interactive 3D point selection
elidar search --lat 48.7768 --lon -121.8145 --las baker_summit.laz

# Multiple tiles at once
elidar peak --lat 38.828 --lon -104.906 --las tile1.laz tile2.laz

# Calculate prominence
elidar prominence 10725.0 10200.0

# Use a DEM for CRS fallback (rare)
elidar peak --lat 38.828 --lon -104.906 --las data.laz --dem dem.tif
```

### Search Mode Controls

When the 3D scatter plot opens:
- **Click** on a point to select it
- **Z** -- Zoom in around the selected point
- **D** or **Enter** -- Accept the current selection
- **Escape** -- Cancel

### Options

| Option | CLI Flag | Default | Description |
|--------|----------|---------|-------------|
| Clean Data | `--no-clean` | On | Statistical outlier removal and deduplication |
| Ground Only | `--ground-only` | Off | Filter to LiDAR classification 2 (ground) |
| Search Radius | `--radius` | 0.0008° | Initial query radius in degrees |
| Zoom Scale | `--zoom-scale` | 0.5 | Zoom reduction factor per iteration |
| Color Scale | `--colorscale` | jet | Plot coloring: jet, turbo, winter, greyscale, br |
| Force Unit | `--force-unit` | Auto | Override unit detection: ft or m |

## Data Sources

Download LiDAR data from your state's public data portal. Examples:
- **Colorado**: https://coloradohazardmapping.com/lidarDownload
- **Washington**: https://lidarportal.dnr.wa.gov/
- **USGS 3DEP**: https://apps.nationalmap.gov/downloader/

You typically need only the LAZ/LAS file. The DEM file is only needed as a fallback if the LAZ file lacks CRS metadata (uncommon with modern data).

## Dependencies

- `laspy[lazrs]` -- LAS/LAZ file reading
- `pyproj` -- Coordinate reprojection
- `numpy` -- Array operations
- `rasterio` -- DEM reading (optional CRS fallback)
- `open3d` -- Point cloud denoising
- `pyvista` + `pyvistaqt` -- 3D interactive visualization
- `PyQt6` -- Desktop GUI

## Project Structure

```
src/elidar/
  core/
    types.py      -- Data classes (QueryOptions, QueryResult, etc.)
    crs.py        -- CRS resolution (LAZ VLRs -> DEM fallback)
    cleaning.py   -- Statistical outlier removal + voxel downsampling
    query.py      -- Main query engine (peak, point, search modes)
  viz/
    scatter3d.py  -- Interactive 3D scatter with point picking + zoom
    colormaps.py  -- Color scale definitions
  gui/
    app.py        -- Main application window
    main_tab.py   -- Analysis tab
    tools_tab.py  -- Tools tab (notes)
  cli.py          -- Command-line interface
```
