"""Command-line interface for ELiDAR."""

from __future__ import annotations

import argparse
import logging
import sys

from elidar.core.types import ColorScale, M2FT, QueryOptions, QueryType


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    :param argv: Command-line arguments (defaults to sys.argv[1:]).
    :return: Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="elidar",
        description="ELiDAR - LiDAR analysis for peak and saddle prominence",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Query type")

    _add_query_parser(subparsers, "peak", "Find the highest point near a location")
    _add_query_parser(subparsers, "point", "Find the closest LiDAR return to a location")
    _add_query_parser(subparsers, "search", "Interactive 3D point selection")

    prominence_parser = subparsers.add_parser(
        "prominence",
        help="Calculate prominence from peak and saddle elevations",
    )
    prominence_parser.add_argument("peak_ft", type=float, help="Peak elevation in feet")
    prominence_parser.add_argument("saddle_ft", type=float, help="Saddle elevation in feet")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "prominence":
        return _run_prominence(args)

    return _run_query(args)


def _add_query_parser(subparsers: argparse._SubParsersAction, name: str, help_text: str) -> None:
    """Add a query subcommand with common arguments."""
    p = subparsers.add_parser(name, help=help_text)
    p.add_argument("--lat", type=float, required=True, help="Latitude in degrees")
    p.add_argument("--lon", type=float, required=True, help="Longitude in degrees")
    p.add_argument("--las", type=str, required=True, nargs="+", help="Path(s) to LAS/LAZ file(s)")
    p.add_argument("--dem", type=str, default=None, help="Path to DEM file (optional fallback for CRS)")
    p.add_argument("--no-clean", action="store_true", help="Skip point cloud cleaning")
    p.add_argument("--ground-only", action="store_true", help="Filter to ground classification only")
    p.add_argument("--radius", type=float, default=0.0008, help="Search radius in degrees (default: 0.0008)")
    p.add_argument("--zoom-scale", type=float, default=0.5, help="Zoom factor per iteration (default: 0.5)")
    p.add_argument("--force-unit", choices=["ft", "m"], default=None, help="Override unit detection")
    p.add_argument(
        "--colorscale",
        choices=[c.value for c in ColorScale],
        default="jet",
        help="Color mapping for search plots (default: jet)",
    )
    p.add_argument("--center", action="store_true", help="Use dataset center as query point (dev)")


def _run_query(args: argparse.Namespace) -> int:
    """Execute a LiDAR query from CLI arguments.

    :param args: Parsed arguments.
    :return: Exit code.
    """
    from elidar.core.query import query_lidar

    options = QueryOptions(
        clean_data=not args.no_clean,
        colorscale=ColorScale(args.colorscale),
        filter_ground=args.ground_only,
        query_radius_deg=args.radius,
        zoom_scale=args.zoom_scale,
        force_unit=args.force_unit,
        display_lla=True,
        center=args.center,
        dem_path=args.dem,
    )

    query_type = QueryType(args.command)

    result = query_lidar(
        query_lat=args.lat,
        query_lon=args.lon,
        query_type=query_type,
        las_path=args.las,
        options=options,
    )

    if result is None:
        print("No result (query point not in data or cancelled)")
        return 1

    print(f"\n--- {result.query_type.value.upper()} Result ---")
    print(f"  Elevation:  {result.elevation_ft:.4f} ft  ({result.elevation_m:.4f} m)")
    print(f"  Location:   {result.location_deg[0]:.6f}°N, {result.location_deg[1]:.6f}°W")
    print(f"  Local CRS:  ({result.location_local[0]:.2f}, {result.location_local[1]:.2f})")
    return 0


def _run_prominence(args: argparse.Namespace) -> int:
    """Calculate and display prominence.

    :param args: Parsed arguments with peak_ft and saddle_ft.
    :return: Exit code.
    """
    prominence_ft = args.peak_ft - args.saddle_ft
    prominence_m = prominence_ft / M2FT

    print(f"\n--- Prominence ---")
    print(f"  Peak:        {args.peak_ft:.4f} ft")
    print(f"  Saddle:      {args.saddle_ft:.4f} ft")
    print(f"  Prominence:  {prominence_ft:.4f} ft  ({prominence_m:.4f} m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
