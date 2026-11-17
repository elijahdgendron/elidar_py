"""Custom colormap definitions for ELiDAR scatter plots.

Ports the MATLAB colormaps:
  - 'BR': blue (low) -> red (high)
  - 'Greyscale': black (low) -> white (high)
  - 'jet', 'turbo', 'winter': standard matplotlib colormaps
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap, Colormap
import matplotlib.pyplot as plt

from elidar.core.types import ColorScale

_BR_CMAP = LinearSegmentedColormap.from_list(
    "elidar_br",
    [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
)

_GREYSCALE_CMAP = LinearSegmentedColormap.from_list(
    "elidar_greyscale",
    [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
)

_ICE_CMAP = LinearSegmentedColormap.from_list(
    "elidar_ice",
    [
        (0.05, 0.05, 0.30),  # deep navy
        (0.15, 0.35, 0.65),  # steel blue
        (0.40, 0.70, 0.90),  # ice blue
        (0.75, 0.90, 0.97),  # frost
        (0.95, 0.98, 1.00),  # near-white
    ],
)

_COLORMAP_REGISTRY: dict[ColorScale, str | Colormap] = {
    ColorScale.JET: "jet",
    ColorScale.TURBO: "turbo",
    ColorScale.WINTER: "winter",
    ColorScale.ICE: _ICE_CMAP,
    ColorScale.BR: _BR_CMAP,
    ColorScale.GREYSCALE: _GREYSCALE_CMAP,
}


def get_colormap(scale: ColorScale) -> str | Colormap:
    """Get the matplotlib colormap for a ColorScale enum value.

    :param scale: The desired color scale.
    :return: A matplotlib colormap name or Colormap object usable by pyvista.
    """
    return _COLORMAP_REGISTRY[scale]
