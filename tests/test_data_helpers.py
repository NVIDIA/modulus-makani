# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime as dt
import time
import unittest

import numpy as np

from makani.utils.dataloaders.data_helpers import (
    parse_data_normalization_mode,
    get_lat_lon_grid,
    get_data_normalization,
    get_time_diff_stds,
    get_climatology,
    get_timestamp,
    as_utc,
    get_date_from_string,
    get_date_from_timestamp,
    get_seconds_from_times,
    get_timedelta_from_timestamp,
    get_date_ranges,
)

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import set_seed, get_default_parameters, compare_arrays, local_timezone as _local_timezone

UTC = dt.timezone.utc


# ===========================================================================
# 1. get_lat_lon_grid
# ===========================================================================
class TestGetLatLonGrid(unittest.TestCase):
    """
    Contracts:
      - latitude  = linspace(90, -90, nlat, endpoint=True)   → includes both poles
      - longitude = linspace(0, 360, nlon, endpoint=False)   → excludes 360 (wraps)
    """

    def test_lat_shape(self):
        lat, _ = get_lat_lon_grid((36, 72))
        self.assertEqual(lat.shape, (36,))

    def test_lon_shape(self):
        _, lon = get_lat_lon_grid((36, 72))
        self.assertEqual(lon.shape, (72,))

    def test_lat_starts_at_north_pole(self):
        lat, _ = get_lat_lon_grid((36, 72))
        self.assertAlmostEqual(lat[0], 90.0)

    def test_lat_ends_at_south_pole(self):
        """endpoint=True means -90 is included."""
        lat, _ = get_lat_lon_grid((36, 72))
        self.assertAlmostEqual(lat[-1], -90.0)

    def test_lat_monotonically_decreasing(self):
        """Latitude goes north-to-south."""
        lat, _ = get_lat_lon_grid((36, 72))
        self.assertTrue(np.all(np.diff(lat) < 0))

    def test_lon_starts_at_zero(self):
        _, lon = get_lat_lon_grid((36, 72))
        self.assertAlmostEqual(lon[0], 0.0)

    def test_lon_does_not_reach_360(self):
        """endpoint=False means 360 is excluded."""
        _, lon = get_lat_lon_grid((36, 72))
        self.assertLess(lon[-1], 360.0)

    def test_lon_monotonically_increasing(self):
        _, lon = get_lat_lon_grid((36, 72))
        self.assertTrue(np.all(np.diff(lon) > 0))

    def test_single_point(self):
        """Edge case: 1×1 grid returns scalar arrays."""
        lat, lon = get_lat_lon_grid((1, 1))
        self.assertEqual(lat.shape, (1,))
        self.assertEqual(lon.shape, (1,))
        self.assertAlmostEqual(lat[0], 90.0)
        self.assertAlmostEqual(lon[0], 0.0)


# ===========================================================================
# 2. get_data_normalization
# ===========================================================================
class TestGetDataNormalization(unittest.TestCase):

    def setUp(self):
        self.params = get_default_parameters()  # normalization = "none"

    def test_normalization_none_string_returns_none(self):
        """'none' does not match any branch → (None, None)."""
        bias, scale = get_data_normalization(self.params)
        self.assertIsNone(bias)
        self.assertIsNone(scale)

    def test_no_normalization_attr_returns_none(self):
        """Without a normalization attribute at all → (None, None)."""
        params = get_default_parameters()
        del params.normalization  # remove the attribute
        bias, scale = get_data_normalization(params)
        self.assertIsNone(bias)
        self.assertIsNone(scale)

    def test_minmax_missing_file_raises(self):
        """minmax mode raises when min_path does not exist."""
        self.params.normalization = "minmax"
        self.params.min_path = "/nonexistent/mins.npy"
        self.params.max_path = "/nonexistent/maxs.npy"
        with self.assertRaises(FileNotFoundError):
            get_data_normalization(self.params)

    def test_zscore_missing_file_raises(self):
        """zscore mode raises when global_means_path does not exist."""
        self.params.normalization = "zscore"
        self.params.global_means_path = "/nonexistent/means.npy"
        self.params.global_stds_path = "/nonexistent/stds.npy"
        with self.assertRaises(FileNotFoundError):
            get_data_normalization(self.params)

    def test_synthetic_data_needs_no_stats_files(self):
        """Synthetic data has no statistics, and the .npy files it would load do not exist."""
        self.params.enable_synthetic_data = True
        self.params.normalization = "zscore"
        self.params.global_means_path = "/nonexistent/means.npy"
        self.params.global_stds_path = "/nonexistent/stds.npy"

        bias, scale = get_data_normalization(self.params)

        num_channels = len(self.params.channel_names)
        self.assertEqual(bias.shape, (1, num_channels, 1, 1))
        self.assertEqual(scale.shape, (1, num_channels, 1, 1))
        np.testing.assert_array_equal(bias, np.zeros_like(bias))
        np.testing.assert_array_equal(scale, np.ones_like(scale))

    def test_synthetic_time_diff_stds_need_no_files(self):
        """The same holds for the time-difference stats a temp_diff_normalization loss wants."""
        self.params.enable_synthetic_data = True
        self.params.time_diff_stds_path = "/nonexistent/time_diff_stds.npy"

        time_diff_stds = get_time_diff_stds(self.params)

        num_channels = len(self.params.channel_names)
        self.assertEqual(time_diff_stds.shape, (1, num_channels, 1, 1))
        np.testing.assert_array_equal(time_diff_stds, np.ones_like(time_diff_stds))

        # the loss indexes the flattened array with out_channels, so it has to be long enough
        self.assertGreater(len(time_diff_stds.flatten()), max(self.params.out_channels))


# ===========================================================================
# 3. get_climatology (synthetic-data path only — no file I/O)
# ===========================================================================
class TestGetClimatology(unittest.TestCase):

    def setUp(self):
        set_seed(333)
        self.params = get_default_parameters()
        self.params.enable_synthetic_data = True
        self.params.N_out_channels = 5
        self.params.img_crop_shape_x = 32
        self.params.img_crop_shape_y = 64
        self.params.subsampling_factor = 1

    def test_shape(self):
        """Synthetic climatology has shape (1, C, H, W)."""
        clim = get_climatology(self.params)
        self.assertEqual(
            clim.shape,
            (1, self.params.N_out_channels, self.params.img_crop_shape_x, self.params.img_crop_shape_y),
        )

    def test_all_zeros(self):
        """Synthetic climatology is all zeros."""
        clim = get_climatology(self.params)
        self.assertTrue(np.all(clim == 0.0))

    def test_dtype_float32(self):
        clim = get_climatology(self.params)
        self.assertEqual(clim.dtype, np.float32)

    def test_subsampling_halves_spatial_dims(self):
        """subsampling_factor=2 halves both spatial dimensions."""
        self.params.subsampling_factor = 2
        clim = get_climatology(self.params)
        self.assertEqual(clim.shape[-2], self.params.img_crop_shape_x // 2)
        self.assertEqual(clim.shape[-1], self.params.img_crop_shape_y // 2)

    def test_subsampling_preserves_zeros(self):
        """Subsampled synthetic climatology is still all zeros."""
        self.params.subsampling_factor = 2
        clim = get_climatology(self.params)
        self.assertTrue(np.all(clim == 0.0))


# ===========================================================================
# 4. get_timestamp
# ===========================================================================
class TestGetTimestamp(unittest.TestCase):
    """
    get_timestamp(year, hour) → Jan 1 <year> 00:00 UTC + <hour> hours.
    """

    def test_hour_zero_is_jan_1_midnight(self):
        ts = get_timestamp(2020, 0)
        self.assertEqual(ts, dt.datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC))

    def test_hour_24_is_jan_2(self):
        ts = get_timestamp(2020, 24)
        self.assertEqual(ts, dt.datetime(2020, 1, 2, 0, 0, 0, tzinfo=UTC))

    def test_hour_offset(self):
        ts = get_timestamp(2020, 6)
        self.assertEqual(ts, dt.datetime(2020, 1, 1, 6, 0, 0, tzinfo=UTC))

    def test_is_utc(self):
        ts = get_timestamp(2020, 0)
        self.assertEqual(ts.tzinfo, UTC)

    def test_year_boundary(self):
        """hour = 8760 (365 days) should land at the start of the next year."""
        ts = get_timestamp(2019, 365 * 24)
        self.assertEqual(ts.year, 2020)
        self.assertEqual(ts.month, 1)
        self.assertEqual(ts.day, 1)

    def test_different_years(self):
        ts_a = get_timestamp(2019, 0)
        ts_b = get_timestamp(2020, 0)
        self.assertLess(ts_a, ts_b)


# ===========================================================================
# 5. get_date_from_string
# ===========================================================================
class TestAsUtc(unittest.TestCase):

    def test_naive_is_labelled_not_converted(self):
        self.assertEqual(as_utc(dt.datetime(2020, 6, 15, 12)), dt.datetime(2020, 6, 15, 12, tzinfo=UTC))

    def test_aware_is_converted(self):
        plus_five = dt.timezone(dt.timedelta(hours=5))
        result = as_utc(dt.datetime(2020, 6, 15, 17, tzinfo=plus_five))
        self.assertEqual(result.tzinfo, UTC)
        self.assertEqual(result, dt.datetime(2020, 6, 15, 12, tzinfo=UTC))

    @unittest.skipUnless(hasattr(time, "tzset"), "TZ manipulation is POSIX only")
    def test_does_not_depend_on_the_machine_timezone(self):
        expected = dt.datetime(2020, 6, 15, 12, tzinfo=UTC).timestamp()
        for zone in ("UTC", "Europe/Berlin", "America/Los_Angeles"):
            with _local_timezone(zone):
                with self.subTest(zone=zone):
                    self.assertEqual(as_utc(dt.datetime(2020, 6, 15, 12)).timestamp(), expected)


class TestGetSecondsFromTimes(unittest.TestCase):
    """Shared by the multifiles loader and SampleSource to turn discovered times into floats."""

    def test_datetimes_become_epoch_seconds(self):
        plus_five = dt.timezone(dt.timedelta(hours=5))
        times = [
            dt.datetime(2020, 6, 15, 12, tzinfo=UTC),
            dt.datetime(2020, 6, 15, 17, tzinfo=plus_five),
            dt.datetime(2020, 6, 15, 12),
        ]
        expected = dt.datetime(2020, 6, 15, 12, tzinfo=UTC).timestamp()
        # times are compared exactly and shape aware: a relative tolerance on epoch
        # seconds would accept shifts of hours, and broadcasting a wrong shape
        self.assertTrue(
            compare_arrays(
                "epoch seconds",
                get_seconds_from_times(times),
                np.full(3, expected),
                atol=0.0,
                rtol=0.0,
                shape_check=True,
            )
        )

    def test_timedeltas_become_offsets(self):
        times = [dt.timedelta(hours=6), dt.timedelta(days=1)]
        self.assertTrue(
            compare_arrays(
                "offset seconds",
                get_seconds_from_times(times),
                np.array([6 * 3600.0, 86400.0]),
                atol=0.0,
                rtol=0.0,
                shape_check=True,
            )
        )

    @unittest.skipUnless(hasattr(time, "tzset"), "TZ manipulation is POSIX only")
    def test_naive_times_do_not_depend_on_the_machine_timezone(self):
        expected = dt.datetime(2020, 6, 15, 12, tzinfo=UTC).timestamp()
        for zone in ("UTC", "Europe/Berlin", "America/Los_Angeles"):
            with _local_timezone(zone):
                with self.subTest(zone=zone):
                    seconds = get_seconds_from_times([dt.datetime(2020, 6, 15, 12)])
                    self.assertTrue(
                        compare_arrays(
                            "naive epoch seconds", seconds, np.array([expected]), atol=0.0, rtol=0.0, shape_check=True
                        )
                    )


class TestGetDateFromString(unittest.TestCase):

    def test_utc_aware_string_preserved(self):
        """An already-UTC ISO string should parse to the same moment."""
        ts = get_date_from_string("2020-06-15T12:00:00+00:00")
        self.assertEqual(ts, dt.datetime(2020, 6, 15, 12, 0, 0, tzinfo=UTC))

    def test_offset_aware_string_converted_to_utc(self):
        """A +05:00 string should be converted: 17:00+05 → 12:00 UTC."""
        ts = get_date_from_string("2020-06-15T17:00:00+05:00")
        self.assertEqual(ts.tzinfo, UTC)
        self.assertEqual(ts.hour, 12)
        self.assertEqual(ts.date(), dt.date(2020, 6, 15))

    def test_naive_string_is_taken_to_be_utc(self):
        """A naive ISO string is labelled UTC, not converted from local time.

        Asserting only that the result is timezone aware is not enough: the
        moment itself has to be unchanged, which is what distinguishes
        labelling from converting.
        """
        ts = get_date_from_string("2020-06-15T12:00:00")

        self.assertEqual(ts, dt.datetime(2020, 6, 15, 12, 0, 0, tzinfo=UTC))

    @unittest.skipUnless(hasattr(time, "tzset"), "TZ manipulation is POSIX only")
    def test_parsing_does_not_depend_on_the_machine_timezone(self):
        """The same string must give the same moment on any machine.

        This is the reason the helper exists. ``astimezone`` on a naive datetime
        does not raise, it assumes the *local* zone, so a date written without a
        designator would land an hour out on a CET laptop and be correct on a
        UTC cluster -- a discrepancy that only shows up when comparing runs
        across systems.
        """
        cases = {
            "2020-06-15T12:00:00+00:00": dt.datetime(2020, 6, 15, 12, tzinfo=UTC),
            "2020-06-15T17:00:00+05:00": dt.datetime(2020, 6, 15, 12, tzinfo=UTC),
            "2020-06-15T12:00:00": dt.datetime(2020, 6, 15, 12, tzinfo=UTC),
            "2020-06-15": dt.datetime(2020, 6, 15, 0, tzinfo=UTC),
        }

        for zone in ("UTC", "Europe/Berlin", "America/Los_Angeles"):
            with _local_timezone(zone):
                for isostring, expected in cases.items():
                    with self.subTest(timezone=zone, isostring=isostring):
                        self.assertEqual(get_date_from_string(isostring), expected)

    def test_result_is_datetime(self):
        ts = get_date_from_string("2020-01-01")
        self.assertIsInstance(ts, dt.datetime)


# ===========================================================================
# 6. get_date_from_timestamp
# ===========================================================================
class TestGetDateFromTimestamp(unittest.TestCase):

    def test_unix_epoch(self):
        """Timestamp 0 is 1970-01-01 00:00:00 UTC."""
        ts = get_date_from_timestamp(0)
        self.assertEqual(ts, dt.datetime(1970, 1, 1, 0, 0, 0, tzinfo=UTC))

    def test_one_hour(self):
        ts = get_date_from_timestamp(3600)
        self.assertEqual(ts, dt.datetime(1970, 1, 1, 1, 0, 0, tzinfo=UTC))

    def test_one_day(self):
        ts = get_date_from_timestamp(86400)
        self.assertEqual(ts, dt.datetime(1970, 1, 2, 0, 0, 0, tzinfo=UTC))

    def test_is_utc(self):
        ts = get_date_from_timestamp(0)
        self.assertEqual(ts.tzinfo, UTC)


# ===========================================================================
# 7. get_timedelta_from_timestamp
# ===========================================================================
class TestGetTimedeltaFromTimestamp(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(get_timedelta_from_timestamp(0), dt.timedelta(0))

    def test_one_hour(self):
        self.assertEqual(get_timedelta_from_timestamp(3600), dt.timedelta(hours=1))

    def test_one_day(self):
        self.assertEqual(get_timedelta_from_timestamp(86400), dt.timedelta(days=1))

    def test_fractional_seconds(self):
        self.assertEqual(
            get_timedelta_from_timestamp(90),
            dt.timedelta(minutes=1, seconds=30),
        )


# ===========================================================================
# 8. get_date_ranges
# ===========================================================================
class TestGetDateRanges(unittest.TestCase):

    def _date(self, y=2020, m=1, d=1, h=12):
        return dt.datetime(y, m, d, h, 0, 0, tzinfo=UTC)

    def test_single_date_produces_one_range(self):
        ranges = get_date_ranges([self._date()], lookback_hours=6, lookahead_hours=24)
        self.assertEqual(len(ranges), 1)

    def test_multiple_dates_produce_matching_count(self):
        dates = [self._date(d=1), self._date(d=2), self._date(d=3)]
        ranges = get_date_ranges(dates, lookback_hours=6, lookahead_hours=6)
        self.assertEqual(len(ranges), 3)

    def test_range_start_is_date_minus_lookback(self):
        date = self._date()
        ranges = get_date_ranges([date], lookback_hours=6, lookahead_hours=0)
        start, _ = ranges[0]
        self.assertEqual(start, date - dt.timedelta(hours=6))

    def test_range_end_is_date_plus_lookahead(self):
        date = self._date()
        ranges = get_date_ranges([date], lookback_hours=0, lookahead_hours=24)
        _, end = ranges[0]
        self.assertEqual(end, date + dt.timedelta(hours=24))

    def test_zero_offsets_give_point_range(self):
        date = self._date()
        ranges = get_date_ranges([date], lookback_hours=0, lookahead_hours=0)
        start, end = ranges[0]
        self.assertEqual(start, date)
        self.assertEqual(end, date)

    def test_start_before_end_when_offsets_nonzero(self):
        date = self._date()
        ranges = get_date_ranges([date], lookback_hours=6, lookahead_hours=24)
        start, end = ranges[0]
        self.assertLess(start, end)

    def test_each_range_is_independent(self):
        """Ranges for different dates should not overlap when dates are far apart."""
        d1 = self._date(d=1)
        d2 = self._date(d=10)
        ranges = get_date_ranges([d1, d2], lookback_hours=6, lookahead_hours=6)
        _, end1 = ranges[0]
        start2, _ = ranges[1]
        self.assertLess(end1, start2)


class TestParseDataNormalizationMode(unittest.TestCase):
    def test_default_is_offline_over_space(self):
        self.assertEqual(parse_data_normalization_mode(None), ("offline", "FHW", False))

    def test_mode_without_axes(self):
        self.assertEqual(parse_data_normalization_mode("online"), ("online", "FHW", False))

    def test_axes(self):
        self.assertEqual(parse_data_normalization_mode("offline-HW"), ("offline", "HW", False))

    def test_batch_axis_is_split_off(self):
        # the B used to be left in the axis names, which DALI does not know
        self.assertEqual(parse_data_normalization_mode("offline-FHWB"), ("offline", "FHW", True))

    def test_unknown_mode_is_rejected(self):
        # anything but offline would otherwise be treated as online normalization
        for mode in ["offine", "offine-FHW", ""]:
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    parse_data_normalization_mode(mode)


if __name__ == "__main__":
    unittest.main()
