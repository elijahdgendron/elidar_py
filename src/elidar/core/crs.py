"""CRS (Coordinate Reference System) resolution from LAZ files and DEM fallback.

This is the key improvement over the MATLAB version: CRS is read directly from
the LAZ file's VLR headers, eliminating the mandatory DEM file requirement.

Resolution priority:
  1. User override (force_unit with explicit EPSG)
  2. LAZ file VLRs (GeoKeyDirectoryTag or WKT)
  3. DEM file fallback (rasterio)
  4. LLA_ filename convention (already in lat/lon/meters)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import laspy
import numpy as np
import pyproj

from elidar.core.types import CrsInfo

logger = logging.getLogger(__name__)

# GeoTIFF key IDs used in LAS GeoKeyDirectoryTag VLRs
_PROJECTED_CS_TYPE_GEO_KEY = 3072
_GEOGRAPHIC_TYPE_GEO_KEY = 2048
_VERTICAL_CS_TYPE_GEO_KEY = 4096

# LASF_Projection VLR identifiers
_GEO_KEY_DIRECTORY_TAG_RECORD_ID = 34735
_GEO_ASCII_PARAMS_RECORD_ID = 34737
_WKT_RECORD_ID = 2112
_LASF_PROJECTION_USER_ID = "LASF_Projection"


def resolve_crs(
    las_path: str | Path,
    dem_path: str | Path | None = None,
    force_unit: str | None = None,
) -> CrsInfo:
    """Resolve the coordinate reference system for a LAS/LAZ file.

    :param las_path: Path to the LAS/LAZ file.
    :param dem_path: Optional path to a DEM file (.tif/.img) as CRS fallback.
    :param force_unit: Override unit detection ("ft" or "m").
    :return: CrsInfo describing the resolved CRS.
    :raises ValueError: If no CRS can be determined.
    """
    las_path = Path(las_path)

    if _is_lla_file(las_path):
        logger.info("LLA_ prefix detected, assuming lat/lon/meters convention")
        return CrsInfo(
            crs=None,
            unit_name="metre",
            is_geographic=True,
            source="lla_convention",
        )

    crs_info = _try_crs_from_laz(las_path)
    if crs_info is None and dem_path is not None:
        crs_info = _try_crs_from_dem(dem_path)
    if crs_info is None:
        raise ValueError(
            f"Could not determine CRS for {las_path.name}. "
            "Provide a DEM file or specify force_unit."
        )

    if force_unit is not None:
        crs_info = _apply_unit_override(crs_info, force_unit)

    return crs_info


def _is_lla_file(las_path: Path) -> bool:
    """Check if the file uses the LLA_ naming convention from combineLAS."""
    return las_path.stem.startswith("LLA_") or las_path.stem.startswith("LLA")


def _try_crs_from_laz(las_path: Path) -> CrsInfo | None:
    """Attempt to read CRS from LAZ file VLR headers.

    :param las_path: Path to the LAS/LAZ file.
    :return: CrsInfo if CRS found, None otherwise.
    """
    try:
        with laspy.open(las_path) as reader:
            header = reader.header
            vlrs = header.vlrs

            crs = _try_wkt_vlr(vlrs)
            if crs is None:
                crs = _try_geokey_vlr(vlrs)

            if crs is not None:
                unit_name = _extract_unit_name(crs)
                logger.info("CRS from LAZ VLR: %s, unit: %s", crs.name, unit_name)
                return CrsInfo(
                    crs=crs,
                    unit_name=unit_name,
                    is_geographic=crs.is_geographic,
                    source="laz_vlr",
                )
    except Exception:
        logger.debug("Failed to read CRS from LAZ VLRs", exc_info=True)

    return None


def _try_wkt_vlr(vlrs: list) -> pyproj.CRS | None:
    """Try to extract CRS from a WKT VLR (LAS 1.4+)."""
    for vlr in vlrs:
        if (
            getattr(vlr, "user_id", "").strip() == _LASF_PROJECTION_USER_ID
            and vlr.record_id == _WKT_RECORD_ID
        ):
            wkt = vlr.record_data.decode("utf-8", errors="replace").rstrip("\x00")
            if wkt.strip():
                return pyproj.CRS.from_wkt(wkt)
    return None


def _try_geokey_vlr(vlrs: list) -> pyproj.CRS | None:
    """Try to extract CRS from GeoKeyDirectoryTag VLR."""
    for vlr in vlrs:
        if (
            getattr(vlr, "user_id", "").strip() == _LASF_PROJECTION_USER_ID
            and vlr.record_id == _GEO_KEY_DIRECTORY_TAG_RECORD_ID
        ):
            data = np.frombuffer(vlr.record_data, dtype=np.uint16)
            if len(data) < 4:
                continue

            n_keys = data[3]
            keys = {}
            for i in range(n_keys):
                offset = 4 + i * 4
                if offset + 3 >= len(data):
                    break
                key_id = data[offset]
                value = data[offset + 3]
                keys[key_id] = value

            epsg = keys.get(_PROJECTED_CS_TYPE_GEO_KEY)
            if epsg and epsg != 0 and epsg != 32767:
                try:
                    return pyproj.CRS.from_epsg(epsg)
                except pyproj.exceptions.CRSError:
                    logger.debug("Invalid projected EPSG code: %d", epsg)

            epsg = keys.get(_GEOGRAPHIC_TYPE_GEO_KEY)
            if epsg and epsg != 0 and epsg != 32767:
                try:
                    return pyproj.CRS.from_epsg(epsg)
                except pyproj.exceptions.CRSError:
                    logger.debug("Invalid geographic EPSG code: %d", epsg)

    return None


def _try_crs_from_dem(dem_path: str | Path) -> CrsInfo | None:
    """Fallback: read CRS from a DEM file using rasterio.

    :param dem_path: Path to DEM file (.tif/.img).
    :return: CrsInfo if CRS found, None otherwise.
    """
    try:
        import rasterio

        dem_path = Path(dem_path)
        if not dem_path.exists():
            logger.warning("DEM file does not exist: %s", dem_path)
            return None

        with rasterio.open(dem_path) as dataset:
            if dataset.crs is None:
                return None
            crs = pyproj.CRS(dataset.crs)
            unit_name = _extract_unit_name(crs)
            logger.info("CRS from DEM: %s, unit: %s", crs.name, unit_name)
            return CrsInfo(
                crs=crs,
                unit_name=unit_name,
                is_geographic=crs.is_geographic,
                source="dem",
            )
    except ImportError:
        logger.warning("rasterio not installed, cannot read DEM CRS")
        return None
    except Exception:
        logger.debug("Failed to read CRS from DEM", exc_info=True)
        return None


def _extract_unit_name(crs: pyproj.CRS) -> str:
    """Extract the linear unit name from a CRS.

    :param crs: pyproj CRS object.
    :return: Unit name string.
    """
    if crs.is_projected:
        axis_info = crs.axis_info
        if axis_info:
            return axis_info[0].unit_name
    if crs.is_geographic:
        return "degree"
    return "unknown"


def _apply_unit_override(crs_info: CrsInfo, force_unit: str) -> CrsInfo:
    """Override the detected unit in a CrsInfo.

    :param crs_info: Original CRS info.
    :param force_unit: "ft" or "m".
    :return: New CrsInfo with overridden unit.
    """
    if force_unit.lower() in ("ft", "foot", "feet"):
        unit_name = "foot"
    elif force_unit.lower() in ("m", "metre", "meter"):
        unit_name = "metre"
    else:
        logger.warning("Unknown force_unit '%s', ignoring", force_unit)
        return crs_info

    return CrsInfo(
        crs=crs_info.crs,
        unit_name=unit_name,
        is_geographic=crs_info.is_geographic,
        source="override",
    )


def make_transformer(crs_info: CrsInfo) -> pyproj.Transformer | None:
    """Create a pyproj Transformer from the CRS to WGS84 geographic.

    :param crs_info: Resolved CRS information.
    :return: Transformer for native CRS -> (lon, lat), or None if already geographic.
    """
    if crs_info.is_geographic or crs_info.crs is None:
        return None

    return pyproj.Transformer.from_crs(
        crs_info.crs,
        pyproj.CRS("EPSG:4326"),
        always_xy=True,
    )
