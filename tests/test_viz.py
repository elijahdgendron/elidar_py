"""Tests for visualization utilities (non-interactive parts)."""

import numpy as np

from elidar.core.types import ColorScale, PointCloudData
from elidar.viz.colormaps import get_colormap
from elidar.viz.scatter3d import PickResult, _find_picked_index, _PickState, _UserAction


class TestColormaps:
    def test_all_colormaps_resolve(self) -> None:
        for cs in ColorScale:
            cmap = get_colormap(cs)
            assert cmap is not None

    def test_standard_maps_are_strings(self) -> None:
        assert get_colormap(ColorScale.JET) == "jet"
        assert get_colormap(ColorScale.TURBO) == "turbo"
        assert get_colormap(ColorScale.WINTER) == "winter"

    def test_custom_maps_are_colormaps(self) -> None:
        from matplotlib.colors import Colormap

        assert isinstance(get_colormap(ColorScale.BR), Colormap)
        assert isinstance(get_colormap(ColorScale.GREYSCALE), Colormap)


class TestPickState:
    def test_initial_state(self) -> None:
        state = _PickState()
        assert state.action == _UserAction.NONE
        assert state.picked_point is None

    def test_reset(self) -> None:
        state = _PickState()
        state.action = _UserAction.ZOOM
        state.picked_point = np.array([1.0, 2.0, 3.0])
        state.reset_for_iteration()
        assert state.action == _UserAction.NONE
        assert state.picked_point is None


class TestFindPickedIndex:
    def _make_cloud(self) -> PointCloudData:
        return PointCloudData(
            x=np.array([38.0, 39.0, 40.0]),
            y=np.array([-105.0, -106.0, -107.0]),
            z=np.array([1000.0, 2000.0, 3000.0]),
            lat=np.array([38.0, 39.0, 40.0]),
            lon=np.array([-105.0, -106.0, -107.0]),
        )

    def test_find_exact_match(self) -> None:
        cloud = self._make_cloud()
        picked = np.array([39.0, -106.0, 2000.0])
        idx = _find_picked_index(cloud, picked)
        assert idx == 1

    def test_find_last_point(self) -> None:
        cloud = self._make_cloud()
        picked = np.array([40.0, -107.0, 3000.0])
        idx = _find_picked_index(cloud, picked)
        assert idx == 2

    def test_find_nearest_with_noise(self) -> None:
        cloud = self._make_cloud()
        picked = np.array([38.001, -105.001, 1000.1])
        idx = _find_picked_index(cloud, picked)
        assert idx == 0


class TestPickResult:
    def test_creation(self) -> None:
        pr = PickResult(
            z=8122.6,
            x_local=3169300.0,
            y_local=1362700.0,
            lat=38.8278,
            lon=-104.9056,
            index=42,
        )
        assert pr.z == 8122.6
        assert pr.lat == 38.8278
        assert pr.index == 42
