# ELiDAR_py Features Guide

## Overview

ELiDAR_py is a desktop application for manually analyzing LiDAR point cloud data to determine precise peak and saddle (key col) elevations for topographic prominence calculation. It reads LAS/LAZ files directly, reprojects coordinates automatically, and provides an interactive 3D viewer for point selection.

---

## GUI Layout

```
+------------------+----------------------------------------------------------+
| Data Files       | [View: 3D|Top|Clear]  status text  [Util: Peak|Saddle..] |
| Query Options    |                                                          |
| Display Options  |              3D LiDAR Viewer                     Z  Color|
|   [Basic]        |              (pyvista embedded)                   Exag Focus|
|   [Classifications]                                                  |   |  |
|   [Colors]       |                                                   |   |  |
|                  +----------------------------------------------------------+
| [    Load    ]   | [Zoom] [Fine] [Done] [Reselect] [Cancel] [Screenshot]   |
|                  | Result: Class | Elev | Lat | Lon                         |
|                  |         Code | Units | CRS | Type                        |
|                  | Log                                                      |
+------------------+----------------------------------------------------------+
```

---

## Loading Data

1. Click **Add Files** or **Add Folder** to select LAS/LAZ files
2. Multiple files can be loaded simultaneously (each with its own CRS)
3. CRS is read directly from the LAZ file headers -- no DEM file needed
4. The **Use tile center** checkbox (default) automatically centers on the loaded data
5. **Search Radius** controls how far from the query center to include points
6. Click **Load** to process and display the point cloud

---

## 3D Viewer

### Navigation
- **Left drag**: Orbit around the terrain (turntable style, Z-axis locked)
- **Middle drag**: Pan
- **Right drag / Mouse wheel**: Zoom
- **[3D] button**: Reset to default isometric view
- **[Top] button**: Switch to top-down map view
- **[Clear] button**: Remove all markers, paths, and selections

### Point Selection
- **Click** on a point to select it
- The yellow status text shows classification, coordinates, and elevation
- Selected points persist through zoom operations
- **[Reselect (R)]**: Restore the last Find Peak / Auto Saddle selection after accidentally clicking away

### Zooming
- **[Zoom (Z)]**: Coarse zoom -- halves the view radius (default 2x)
- **[Fine (F)]**: Fine zoom -- small reduction (default 1.3x)
- Zoom always centers on your selected point
- Camera orientation is preserved across zooms
- Both zoom factors are adjustable in Display Options

### Accepting a Selection
- **[Done (D)]**: Accept the current point, reset to full view with annotation
- **[Cancel (Esc)]**: Discard and return to the placeholder

---

## Find Peak

Located in the **Util** toolbar group. Automatically selects the highest visible point in the current view.

- Uses only points from visible classifications
- Highlights with a magenta marker
- Does not auto-zoom -- click Zoom/Fine to drill in and verify
- The selection persists across zooms so you can inspect the point up close

---

## Saddle Analysis

### Auto Saddle
Finds the key col (saddle point) between the two most prominent peaks in the view.

**Algorithm:**
1. Builds a Delaunay triangulation of visible points
2. Identifies local maxima (points higher than all neighbors) as peak candidates
3. Selects the two highest peaks that are well-separated
4. Constructs a Maximum Spanning Tree (edges weighted by minimum endpoint elevation)
5. Traces the minimax path between the peaks through the MST
6. The lowest point on this path is the key col by definition

**What you see:**
- Green dots at the two reference peaks
- Cyan line showing the minimax path between them
- Magenta marker at the key col
- Log shows col elevation and both peak elevations

This algorithm is exact -- it uses actual LiDAR point elevations with no curve fitting or interpolation.

### Manual Saddle
Click **[Manual Saddle]** to enter two-click mode:
1. Click a point on/near Peak A
2. Click a point on/near Peak B
3. The saddle search runs automatically after the second click

You don't need to click the exact summits -- any points on opposite sides of the saddle converge to the same col. This is useful when auto-detection picks the wrong peaks.

---

## Display Options

### Basic Tab
| Option | Description |
|--------|-------------|
| Coarse Zoom | Zoom factor for the Zoom button (default 2.0x) |
| Fine Zoom | Zoom factor for the Fine button (default 1.3x) |
| Display In | Feet (international), Feet (US Survey), or Meters |
| Point Size | LiDAR point rendering size (1-20, default 15) |
| Max Points | Maximum points per classification to display (1k-500k, default 50k) |
| Background | Black (default), Gray, or White |
| Force Unit | Override CRS unit detection (Auto/Feet/Meters) |

### Classifications Tab
Shows each LiDAR classification found in the data with:
- **Checkbox**: Toggle visibility (instant, no re-render)
- **Colormap dropdown**: Per-class elevation coloring

Default visibility:
- Ground (class 2) and Snow/Ice (class 21) are shown
- Unclassified, noise, and other classes are hidden
- If no ground/ice exists, all classes are shown

Available colormaps: Jet, Turbo, Winter, Ice, Greyscale, BR (blue-red)

The **Ice** colormap (gray-to-ice-blue-to-white) is designed for snow/ice data and is visually distinct from Turbo's warm rainbow used for ground.

### Colors Tab
Configurable overlay colors:
| Element | Default | Description |
|---------|---------|-------------|
| Current Pick | Red | The pyvista click highlight |
| Selection | Magenta | The persistent selection marker |
| Peak Markers | Lime | Reference peaks for saddle analysis |
| Saddle Path | Cyan | The minimax path line |

---

## Sliders

### Z Exaggeration (right of viewer)
Multiplies the vertical axis (1x-10x) to make subtle terrain features visible. Updates in real-time. Resets to 1x when Done is clicked.

### Color Focus (right of viewer)
Allocates more of the color range to high elevations (1x-5x). At 1x the mapping is linear. At 3x, subtle differences near summits become clearly visible while lower elevations compress into fewer colors.

---

## Colorbars

When both Ground and Snow/Ice classifications are visible, two vertical colorbars appear side-by-side:
- Left bar: Ground colormap (e.g. Turbo) with "Elev (ft)" title
- Right bar: Snow/Ice colormap (e.g. Ice) with min/max labels

The colorbars share the same elevation range for consistent comparison.

---

## Result Fields

After selecting a point (manually, via Find Peak, or via saddle analysis):

| Field | Description |
|-------|-------------|
| Class | ASPRS classification name (Ground, Snow/Ice, etc.) |
| Elev | Elevation in the current display unit |
| Lat | Latitude in degrees |
| Lon | Longitude in degrees |
| Code | Numeric classification code (2, 21, etc.) |
| Units | Current display unit (ft (international), ft (US survey), m) |
| CRS | Source of coordinate reference system (laz_vlr, dem, merged) |
| Type | How the point was selected (Manual, Peak, Saddle) |

---

## Tools Tab

### Prominence Calculator
Enter peak and saddle elevations in feet to compute prominence:
- **Last Result -> Peak / Saddle**: Copy the most recent query result
- **Calculate**: Computes peak - saddle elevation

### Scratch Notes
Text editor for taking notes during analysis. Save to a .txt file.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Z | Coarse zoom |
| F | Fine zoom |
| D / Enter | Accept selection (Done) |
| R | Reselect last Find Peak / Saddle result |
| Escape | Cancel |

---

## Screenshot

Click **[Screenshot]** to save the current 3D view as a PNG file.

---

## Command Line

```bash
elidar                    # Launch GUI (default)
elidar gui                # Launch GUI (explicit)
elidar peak --lat 48.77 --lon -121.81 --las file.laz --radius 0.002
elidar search --lat 48.77 --lon -121.81 --las file.laz
elidar peak --las tile1.laz tile2.laz --lat 38.83 --lon -104.91
elidar prominence 10725.0 10200.0
```

---

## Technical Notes

### CRS Handling
CRS is read from the LAZ file's VLR headers (WKT or GeoKeyDirectoryTag). Supports LAS 1.0-1.4 formats. Falls back to a DEM file if the LAZ has no CRS metadata (rare with modern data).

### Unit Handling
Distinguishes US survey foot (1200/3937 m) from international foot (0.3048 m). The difference is ~0.02 ft at 10,000 ft -- small but systematic. All internal math is in meters; display conversion uses the correct factor for the selected unit.

### Point Cloud Cleaning
Statistical outlier removal (Open3D) followed by voxel deduplication. Both operate in local meters regardless of the file's native CRS. Controlled by the Clean Data checkbox.

### Multi-Tile Support
Multiple LAZ files can be loaded simultaneously. Each is read with its own CRS, reprojected to lat/lon independently, and Z normalized to meters before merging. No separate "combine" step needed.
