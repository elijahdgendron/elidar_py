"""Manual test for interactive search mode.

Run this script to open a pyvista 3D window with synthetic mountain data.
Use it to verify the point picking and zoom workflow:

  1. Click on a point in the 3D scatter plot
  2. Press Z to zoom in around that point
  3. Repeat until you have the exact point you want
  4. Press D or Enter to accept the selection
  5. Press Escape to cancel

Usage:
    python tests/manual_test_search.py
    python tests/manual_test_search.py --colorscale turbo
    python tests/manual_test_search.py --no-lla
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np


def create_synthetic_mountain() -> Path:
    """Create a synthetic LAS file that looks like a mountain with ridges."""
    center_lat = 38.828
    center_lon = -104.906
    n_points = 5000
    rng = np.random.RandomState(42)

    lat = center_lat + rng.uniform(-0.0006, 0.0006, n_points)
    lon = center_lon + rng.uniform(-0.0006, 0.0006, n_points)

    dist = np.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)
    z = 2400.0 + 80.0 * np.exp(-(dist / 0.0003) ** 2)

    # Add a ridge extending NE
    ridge_dist = np.abs(
        (lat - center_lat) * 0.7 - (lon - center_lon + 0.0001) * 0.7
    )
    z += 15.0 * np.exp(-(ridge_dist / 0.00005) ** 2) * np.exp(-(dist / 0.0004) ** 2)

    # Add terrain noise
    z += rng.normal(0, 0.5, n_points)

    # Place exact peak
    peak_idx = np.argmax(z)
    z[peak_idx] = 2482.5

    header = laspy.LasHeader(point_format=0, version="1.2")
    header.offsets = [np.min(lat), np.min(lon), np.min(z)]
    header.scales = [1e-7, 1e-7, 0.001]

    las = laspy.LasData(header)
    las.x = lat
    las.y = lon
    las.z = z

    tmp = tempfile.NamedTemporaryFile(suffix=".las", prefix="LLA_manual_test_", delete=False)
    las.write(tmp.name)
    tmp.close()

    print(f"Created synthetic mountain: {tmp.name}")
    print(f"  {n_points} points")
    print(f"  Peak: {z[peak_idx]:.1f} m at ({lat[peak_idx]:.6f}, {lon[peak_idx]:.6f})")
    print(f"  Elevation range: {z.min():.1f} - {z.max():.1f} m")
    return Path(tmp.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual test for interactive search mode")
    parser.add_argument(
        "--colorscale",
        choices=["jet", "turbo", "winter", "greyscale", "br"],
        default="turbo",
    )
    parser.add_argument("--no-lla", action="store_true", help="Show local coords instead of lat/lon")
    parser.add_argument("--las", type=str, default=None, help="Use a real LAS file instead of synthetic")
    parser.add_argument("--dem", type=str, default=None, help="DEM file for CRS (if using real data)")
    parser.add_argument("--lat", type=float, default=None, help="Query latitude (for real data)")
    parser.add_argument("--lon", type=float, default=None, help="Query longitude (for real data)")
    args = parser.parse_args()

    from elidar.core.query import query_lidar
    from elidar.core.types import ColorScale, QueryOptions, QueryType

    if args.las:
        las_path = Path(args.las)
        if args.lat is None or args.lon is None:
            print("ERROR: --lat and --lon required when using --las")
            return 1
        lat, lon = args.lat, args.lon
    else:
        las_path = create_synthetic_mountain()
        lat, lon = 38.828, -104.906

    options = QueryOptions(
        clean_data=False,
        colorscale=ColorScale(args.colorscale),
        display_lla=not args.no_lla,
        dem_path=args.dem,
    )

    print(f"\nRunning search at ({lat}, {lon})...")
    print("Controls: Click=pick, Z=zoom, D/Enter=done, Esc=cancel\n")

    result = query_lidar(
        query_lat=lat,
        query_lon=lon,
        query_type=QueryType.SEARCH,
        las_path=las_path,
        options=options,
    )

    if result is None:
        print("Search cancelled")
    else:
        print(f"\n--- SEARCH Result ---")
        print(f"  Elevation:  {result.elevation_ft:.4f} ft  ({result.elevation_m:.4f} m)")
        print(f"  Location:   {result.location_deg[0]:.6f}°N, {result.location_deg[1]:.6f}°W")

    # Clean up synthetic file
    if not args.las:
        las_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
