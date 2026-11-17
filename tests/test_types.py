"""Tests for core type definitions."""

import numpy as np

from elidar.core.types import (
    CrsInfo,
    M2FT,
    PointCloudData,
    QueryOptions,
    QueryResult,
    QueryType,
    US_SURVEY_FOOT_TO_M,
    INTL_FOOT_TO_M,
)


class TestCrsInfo:
    def test_metre_unit(self) -> None:
        info = CrsInfo(crs=None, unit_name="metre", is_geographic=False, source="laz_vlr")
        assert not info.is_foot
        assert info.unit_to_meters == 1.0

    def test_foot_unit(self) -> None:
        info = CrsInfo(crs=None, unit_name="foot", is_geographic=False, source="laz_vlr")
        assert info.is_foot
        assert info.unit_to_meters == INTL_FOOT_TO_M

    def test_us_survey_foot(self) -> None:
        info = CrsInfo(crs=None, unit_name="US survey foot", is_geographic=False, source="laz_vlr")
        assert info.is_foot
        assert info.unit_to_meters == US_SURVEY_FOOT_TO_M
        assert info.unit_to_meters != INTL_FOOT_TO_M

    def test_us_survey_foot_elevation_difference(self) -> None:
        """At 14,000 ft, US survey foot vs intl foot differ by ~0.03 ft."""
        elev_ft = 14000.0
        m_survey = elev_ft * US_SURVEY_FOOT_TO_M
        m_intl = elev_ft * INTL_FOOT_TO_M
        diff_ft = (m_survey - m_intl) * M2FT
        assert 0.02 < abs(diff_ft) < 0.04


class TestPointCloudData:
    def test_subset(self) -> None:
        cloud = PointCloudData(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([4.0, 5.0, 6.0]),
            z=np.array([7.0, 8.0, 9.0]),
            lat=np.array([10.0, 11.0, 12.0]),
            lon=np.array([13.0, 14.0, 15.0]),
            classification=np.array([1, 2, 2], dtype=np.uint8),
        )
        mask = np.array([False, True, True])
        sub = cloud.subset(mask)

        assert sub.n_points == 2
        np.testing.assert_array_equal(sub.x, [2.0, 3.0])
        np.testing.assert_array_equal(sub.z, [8.0, 9.0])
        np.testing.assert_array_equal(sub.classification, [2, 2])

    def test_subset_no_lat_lon(self) -> None:
        cloud = PointCloudData(
            x=np.array([1.0, 2.0]),
            y=np.array([3.0, 4.0]),
            z=np.array([5.0, 6.0]),
        )
        mask = np.array([True, False])
        sub = cloud.subset(mask)
        assert sub.n_points == 1
        assert len(sub.lat) == 0

    def test_n_points(self) -> None:
        cloud = PointCloudData(
            x=np.zeros(100),
            y=np.zeros(100),
            z=np.zeros(100),
        )
        assert cloud.n_points == 100


class TestQueryOptions:
    def test_defaults(self) -> None:
        opts = QueryOptions()
        assert opts.clean_data is True
        assert opts.filter_ground is False
        assert opts.query_radius_deg == 0.0008
        assert opts.zoom_scale == 0.5
        assert opts.dem_path is None
        assert opts.force_unit is None
