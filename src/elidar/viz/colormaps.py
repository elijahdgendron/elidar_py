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

_COLORMAP_REGISTRY: dict[ColorScale, str | Colormap] = {
    ColorScale.JET: "jet",
    ColorScale.TURBO: "turbo",
    ColorScale.WINTER: "winter",
    ColorScale.BR: _BR_CMAP,
    ColorScale.GREYSCALE: _GREYSCALE_CMAP,
}


def get_colormap(scale: ColorScale) -> str | Colormap:
    """Get the matplotlib colormap for a ColorScale enum value.

    :param scale: The desired color scale.
    :return: A matplotlib colormap name or Colormap object usable by pyvista.
    """
    return _COLORMAP_REGISTRY[scale]
