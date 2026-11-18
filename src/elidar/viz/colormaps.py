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
        (0.45, 0.45, 0.48),  # medium gray
        (0.30, 0.50, 0.70),  # steel blue
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


def stretch_colormap(cmap: str | Colormap, stretch: float) -> Colormap:
    """Resample a colormap with a power curve for non-linear color mapping.

    Stretch > 1 gives more color differentiation at high values (peaks).
    Stretch = 1 is linear (no change).
    Stretch < 1 gives more differentiation at low values.

    :param cmap: Matplotlib colormap name or object.
    :param stretch: Power curve exponent (applied as ``t^(1/stretch)``).
    :return: Resampled Colormap.
    """
    import numpy as np

    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

    n = 256
    t = np.linspace(0, 1, n)
    t_stretched = t ** (1.0 / stretch)
    colors = cmap(t_stretched)
    return LinearSegmentedColormap.from_list(f"{cmap.name}_s{stretch:.1f}", colors, N=n)
