# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys
import unittest
import datetime as dt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.utils.inference.helpers import (
    split_list,
    SortedIndexSampler,
    SimpleIndexSampler,
    translate_date_sampler_to_timedelta_sampler,
    compute_inference_range,
)


class TestSplitList(unittest.TestCase):
    """
    split_list distributes a list across N chunks; the inferencer uses it to
    assign sample indices to batch-parallel ranks. Bugs cause uneven workload
    or duplicated/missing samples across ranks.
    """

    def test_even_split(self):
        self.assertEqual(split_list([0, 1, 2, 3, 4, 5], 2), [[0, 1, 2], [3, 4, 5]])

    def test_uneven_split_preserves_partition(self):
        # 7 items into 3 chunks — chunk sizes are implementation-defined, but
        # the partition must (a) cover every input element exactly once and
        # (b) produce exactly nchunks chunks.
        chunks = split_list(list(range(7)), 3)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(sorted(sum(chunks, [])), list(range(7)))

    def test_single_chunk(self):
        self.assertEqual(split_list([1, 2, 3], 1), [[1, 2, 3]])

    def test_empty_list(self):
        # Distributing an empty list across N chunks should yield N empty chunks.
        chunks = split_list([], 3)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(c == [] for c in chunks))

    def test_split_preserves_order(self):
        # Each chunk's elements must remain in the original input order.
        chunks = split_list([10, 20, 30, 40, 50], 2)
        flat = sum(chunks, [])
        self.assertEqual(flat, [10, 20, 30, 40, 50])


class TestSortedIndexSampler(unittest.TestCase):
    """
    SortedIndexSampler is the schedule generator for autoregressive rollouts.
    Bugs cause misaligned forecast↔IC mapping, off-by-one rollout step counts,
    or silently dropped batches near the dataset boundary — failures that
    produce wrong skill numbers without crashing.
    """

    def test_basic_schedule(self):
        # ICs at [0, 100, 200], batch_size=2 ⇒ batched([0,100], [200]).
        # rollout_steps=3 ⇒ 4 timesteps per batch (init + 3 targets).
        # Stride is rollout_dt=6 between successive timesteps.
        sampler = SortedIndexSampler(
            indices=[0, 100, 200],
            maxind=10000,
            batch_size=2,
            rollout_steps=3,
            rollout_dt=6,
        )
        self.assertEqual(
            list(sampler),
            [
                [0, 100],
                [6, 106],
                [12, 112],
                [18, 118],  # batch 1, 4 timesteps
                [200],
                [206],
                [212],
                [218],  # batch 2, 4 timesteps
            ],
        )
        self.assertEqual(len(sampler), 8)

    def test_drops_batch_when_any_member_overflows(self):
        # Pin the actual (slightly surprising) behavior: with incomplete_rollouts=False,
        # the ENTIRE batch is dropped if ANY member's full rollout would exceed maxind.
        # Here batch=[0, 90], rollout_steps=3, dt=6 → step 3 produces shift=[18, 108];
        # 108 ≥ 100 → drop the whole batch (including the safe step-0/step-1/step-2 rows).
        sampler = SortedIndexSampler(
            indices=[0, 90],
            maxind=100,
            batch_size=2,
            rollout_steps=3,
            rollout_dt=6,
        )
        self.assertEqual(list(sampler), [])

    def test_keeps_partial_rollout_when_flagged(self):
        # incomplete_rollouts=True keeps the rollout steps that DO fit before the
        # overflow. For batch=[0, 90], dt=6, maxind=100:
        #   s=0: [0, 90]   max=90  ✓ keep
        #   s=1: [6, 96]   max=96  ✓ keep
        #   s=2: [12, 102] max=102 ✗ break, do not append this row
        # Result: 2 batches kept.
        sampler = SortedIndexSampler(
            indices=[0, 90],
            maxind=100,
            batch_size=2,
            rollout_steps=3,
            rollout_dt=6,
            incomplete_rollouts=True,
        )
        self.assertEqual(list(sampler), [[0, 90], [6, 96]])

    def test_zero_rollout_steps(self):
        # rollout_steps=0 ⇒ only the IC, no targets.
        sampler = SortedIndexSampler(
            indices=[0, 5, 10],
            maxind=100,
            batch_size=2,
            rollout_steps=0,
            rollout_dt=6,
        )
        self.assertEqual(list(sampler), [[0, 5], [10]])

    def test_batch_size_larger_than_indices_is_capped(self):
        # batch_size=10 with only 3 indices ⇒ effective batch size capped at 3,
        # producing one batch followed by rollout steps in lockstep.
        sampler = SortedIndexSampler(
            indices=[0, 5, 10],
            maxind=1000,
            batch_size=10,
            rollout_steps=2,
            rollout_dt=6,
        )
        self.assertEqual(list(sampler), [[0, 5, 10], [6, 11, 16], [12, 17, 22]])

    def test_empty_indices(self):
        sampler = SortedIndexSampler(
            indices=[],
            maxind=100,
            batch_size=2,
            rollout_steps=3,
            rollout_dt=6,
        )
        self.assertEqual(list(sampler), [])
        self.assertEqual(len(sampler), 0)

    def test_one_batch_drop_does_not_affect_other_batches(self):
        # Two batches: [0, 50] safely fits, [200, 250] does not (with maxind=210
        # batch member 250 already overflows even at step 0). Confirm only the
        # offending batch is dropped, not the whole schedule.
        sampler = SortedIndexSampler(
            indices=[0, 50, 200, 250],
            maxind=210,
            batch_size=2,
            rollout_steps=1,
            rollout_dt=6,
        )
        # batch 1: [0, 50] → s=0 [0,50] max=50 < 210, s=1 [6,56] max=56 < 210 → keep
        # batch 2: [200, 250] → s=0 [200,250] max=250 ≥ 210 → drop
        self.assertEqual(list(sampler), [[0, 50], [6, 56]])


class TestSimpleIndexSampler(unittest.TestCase):
    """SimpleIndexSampler is a thin Sampler over a fixed list of batches."""

    def test_iterates_provided_lists(self):
        sampler = SimpleIndexSampler([[1, 2], [3, 4, 5]])
        self.assertEqual(list(sampler), [[1, 2], [3, 4, 5]])
        self.assertEqual(len(sampler), 2)

    def test_empty(self):
        sampler = SimpleIndexSampler([])
        self.assertEqual(list(sampler), [])
        self.assertEqual(len(sampler), 0)


class TestTranslateDateSamplerToTimedeltaSampler(unittest.TestCase):
    """
    Translates a sampler that addresses a *date-indexed* dataset into one that
    addresses a *timedelta-indexed* dataset (e.g. mask or climatology that's
    keyed by hours-from-Jan-1 of an arbitrary year). The crucial property is
    that the timedelta is computed from Jan 1 of *each timestamp's own year*,
    so multi-year inputs are handled correctly — bugs here cause masks or
    climatologies to be applied to the wrong target.
    """

    @staticmethod
    def _make_fake_datasets(dates):
        """Build a mock date-indexed dataset and a mock timedelta-indexed
        dataset that records each queried timedelta in insertion order."""

        class FakeDateDataset:
            def __init__(self, ds):
                self._dates = ds

            def get_time_at_index(self, idx):
                return self._dates[idx]

        class FakeTimedeltaDataset:
            # covers every time of a year, so that only the recorded lookups matter
            end_date = dt.timedelta(days=366)
            dhours = 1

            def __init__(self):
                self._lookup = {}
                self._next = 0

            def get_index_at_time(self, td):
                if td not in self._lookup:
                    self._lookup[td] = self._next
                    self._next += 1
                return self._lookup[td]

        return FakeDateDataset(dates), FakeTimedeltaDataset()

    def test_single_year_basic(self):
        # Two dates in the same year: deltas from that year's Jan 1.
        dates = [
            dt.datetime(2017, 1, 2, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2017, 3, 15, 6, 0, 0, tzinfo=dt.timezone.utc),
        ]
        date_ds, td_ds = self._make_fake_datasets(dates)
        src = SimpleIndexSampler([[0, 1]])

        out = list(translate_date_sampler_to_timedelta_sampler(src, date_ds, td_ds))

        # Jan 2 12:00 = 1 day 12 h after Jan 1 00:00
        td0 = dt.timedelta(days=1, hours=12)
        # Mar 15 06:00 of 2017 = 31 (Jan) + 28 (Feb, non-leap) + 14 (to Mar 15) days, plus 6h
        td1 = dt.timedelta(days=31 + 28 + 14, hours=6)

        self.assertEqual(out, [[td_ds._lookup[td0], td_ds._lookup[td1]]])

    def test_multi_year_uses_each_years_jan_1(self):
        # Two dates in DIFFERENT years. The function must compute each delta
        # against Jan 1 of its OWN year, not against a single shared reference.
        # This is the property that catches "I refactored to use one base
        # epoch" regressions.
        dates = [
            dt.datetime(2017, 1, 2, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2018, 7, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
        ]
        date_ds, td_ds = self._make_fake_datasets(dates)
        src = SimpleIndexSampler([[0, 1]])

        out = list(translate_date_sampler_to_timedelta_sampler(src, date_ds, td_ds))

        td0 = dt.timedelta(days=1, hours=12)  # Jan 2 12:00 of 2017
        td1 = dt.timedelta(days=31 + 28 + 31 + 30 + 31 + 30)  # Jul 1 of 2018 (non-leap)
        self.assertEqual(td1, dt.timedelta(days=181))  # sanity

        self.assertEqual(out, [[td_ds._lookup[td0], td_ds._lookup[td1]]])

    def test_leap_year_jan_1_is_correct_reference(self):
        # 2020 is a leap year. Mar 1 of 2020 = 31 (Jan) + 29 (Feb leap) = 60 days.
        # If the implementation accidentally used a fixed non-leap reference
        # year (e.g. always 2017), this would be off by one day.
        dates = [dt.datetime(2020, 3, 1, 0, 0, 0, tzinfo=dt.timezone.utc)]
        date_ds, td_ds = self._make_fake_datasets(dates)
        src = SimpleIndexSampler([[0]])

        out = list(translate_date_sampler_to_timedelta_sampler(src, date_ds, td_ds))

        expected_td = dt.timedelta(days=60)
        self.assertEqual(out, [[td_ds._lookup[expected_td]]])

    def test_multiple_batches(self):
        # Two batches in the source sampler — verify they map independently.
        dates = [
            dt.datetime(2017, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2017, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2017, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc),
        ]
        date_ds, td_ds = self._make_fake_datasets(dates)
        src = SimpleIndexSampler([[0, 1], [2]])

        out = list(translate_date_sampler_to_timedelta_sampler(src, date_ds, td_ds))

        td0 = dt.timedelta(0)
        td1 = dt.timedelta(hours=12)
        td2 = dt.timedelta(days=1)
        self.assertEqual(
            out,
            [
                [td_ds._lookup[td0], td_ds._lookup[td1]],
                [td_ds._lookup[td2]],
            ],
        )

    def test_index_offset(self):
        # with history, the time looked up is that of the latest step of the window, index_offset steps on
        dates = [dt.datetime(2017, 1, 1, 6 * h, 0, 0, tzinfo=dt.timezone.utc) for h in range(4)]
        date_ds, td_ds = self._make_fake_datasets(dates)
        src = SimpleIndexSampler([[0, 1]])

        out = list(translate_date_sampler_to_timedelta_sampler(src, date_ds, td_ds, index_offset=2))

        self.assertEqual(out, [[td_ds._lookup[dt.timedelta(hours=12)], td_ds._lookup[dt.timedelta(hours=18)]]])

    def test_last_step_of_leap_year(self):
        # a 6 hourly climatology of 366 days ends at day 365, 18:00. With hourly data, Dec 31 of a leap year
        # runs to 23:00, past that sample, and has to map to it like any other time within its step
        class Climatology6h:
            dhours = 6
            start_date = dt.timedelta(0)
            end_date = dt.timedelta(days=365, hours=18)

            def get_index_at_time(self, td):
                if td < self.start_date or td > self.end_date:
                    return None
                return int(td.total_seconds() // (6 * 3600))

        dates = [dt.datetime(2020, 12, 31, h, 0, 0, tzinfo=dt.timezone.utc) for h in [12, 17, 18, 19, 23]]
        date_ds, _ = self._make_fake_datasets(dates)
        src = SimpleIndexSampler([[0, 1, 2, 3, 4]])

        out = list(translate_date_sampler_to_timedelta_sampler(src, date_ds, Climatology6h()))

        self.assertEqual(out, [[1462, 1462, 1463, 1463, 1463]])

    def test_time_outside_relative_dataset_raises(self):
        # a relative dataset returns None for a time it does not cover, which must not reach its loader
        class ShortTimedeltaDataset:
            start_date = dt.timedelta(0)
            end_date = dt.timedelta(days=1)
            dhours = 1

            def get_index_at_time(self, td):
                return None if td > self.end_date else int(td.total_seconds() // 3600)

        dates = [
            dt.datetime(2017, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2017, 1, 3, 0, 0, 0, tzinfo=dt.timezone.utc),
        ]
        date_ds, _ = self._make_fake_datasets(dates)
        src = SimpleIndexSampler([[0, 1]])

        with self.assertRaisesRegex(ValueError, "outside the range"):
            translate_date_sampler_to_timedelta_sampler(src, date_ds, ShortTimedeltaDataset())


class TestComputeInferenceRange(unittest.TestCase):
    """
    compute_inference_range resolves user-supplied (start_date, end_date, date_step)
    into the index-and-stride form the inferencer needs for its sampler. It also
    validates the range — bugs here lead to silent off-by-ones, accepting empty
    ranges, or letting an inverted range through.

    The dataset interface required is small (``get_index_at_time``,
    ``get_time_at_index``, ``dhours``, ``start_date``, ``end_date``, ``__len__``),
    so we stub it with a simple class.
    """

    @staticmethod
    def _make_dataset(start, count, dhours):
        """Create a fake date-indexed dataset with `count` regularly-spaced samples."""
        dates = [start + dt.timedelta(hours=h * dhours) for h in range(count)]

        class FakeDataset:
            def __init__(self):
                self._dates = dates
                self.dhours = dhours
                self.start_date = dates[0]
                self.end_date = dates[-1]

            def __len__(self):
                return len(self._dates)

            def get_index_at_time(self, t):
                try:
                    return self._dates.index(t)
                except ValueError:
                    return None

            def get_time_at_index(self, idx):
                return self._dates[idx]

        return FakeDataset(), dates

    def test_both_dates_none_uses_full_range(self):
        ds, dates = self._make_dataset(
            start=dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc),
            count=10,
            dhours=6,
        )
        start_idx, end_idx, step, sd, ed = compute_inference_range(ds, None, None, date_step=6)
        self.assertEqual(start_idx, 0)
        self.assertEqual(end_idx, 9)  # len-1
        self.assertEqual(step, 1)  # 6 // 6
        self.assertEqual(sd, dates[0])
        self.assertEqual(ed, dates[-1])

    def test_explicit_dates_resolve_to_indices(self):
        ds, dates = self._make_dataset(
            start=dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc),
            count=20,
            dhours=6,
        )
        start_idx, end_idx, step, sd, ed = compute_inference_range(
            ds,
            start_date=dates[3],
            end_date=dates[15],
            date_step=12,
        )
        self.assertEqual(start_idx, 3)
        self.assertEqual(end_idx, 15)
        self.assertEqual(step, 2)  # 12 // 6
        self.assertEqual(sd, dates[3])
        self.assertEqual(ed, dates[15])

    def test_only_start_given(self):
        ds, dates = self._make_dataset(
            start=dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc),
            count=10,
            dhours=6,
        )
        start_idx, end_idx, _, _, _ = compute_inference_range(ds, dates[2], None, date_step=6)
        self.assertEqual(start_idx, 2)
        self.assertEqual(end_idx, 9)

    def test_only_end_given(self):
        ds, dates = self._make_dataset(
            start=dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc),
            count=10,
            dhours=6,
        )
        start_idx, end_idx, _, _, _ = compute_inference_range(ds, None, dates[5], date_step=6)
        self.assertEqual(start_idx, 0)
        self.assertEqual(end_idx, 5)

    def test_inverted_range_raises_value_error(self):
        ds, dates = self._make_dataset(
            start=dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc),
            count=10,
            dhours=6,
        )
        with self.assertRaises(ValueError):
            compute_inference_range(ds, start_date=dates[7], end_date=dates[3], date_step=6)

    def test_equal_range_raises_value_error(self):
        # ``end_index <= start_index`` raises — equal indices count as empty range
        ds, dates = self._make_dataset(
            start=dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc),
            count=10,
            dhours=6,
        )
        with self.assertRaises(ValueError):
            compute_inference_range(ds, start_date=dates[4], end_date=dates[4], date_step=6)

    def test_step_smaller_than_dhours_raises_value_error(self):
        ds, _ = self._make_dataset(
            start=dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc),
            count=10,
            dhours=6,
        )
        with self.assertRaises(ValueError):
            compute_inference_range(ds, None, None, date_step=3)  # 3 < 6

    def test_step_equal_to_dhours_is_allowed(self):
        # Boundary of the validation: date_step == dhours → step == 1
        ds, _ = self._make_dataset(
            start=dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc),
            count=10,
            dhours=6,
        )
        _, _, step, _, _ = compute_inference_range(ds, None, None, date_step=6)
        self.assertEqual(step, 1)

    def test_step_not_divisible_by_dhours_floors_silently(self):
        # Pinned behavior: date_step // dhours floors the result. With dhours=6
        # and date_step=7, step=1 (i.e. user requested ~7h cadence but got 6h).
        # If a future change rounds up or raises, this test will surface it.
        ds, _ = self._make_dataset(
            start=dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc),
            count=10,
            dhours=6,
        )
        _, _, step, _, _ = compute_inference_range(ds, None, None, date_step=7)
        self.assertEqual(step, 1)

        _, _, step2, _, _ = compute_inference_range(ds, None, None, date_step=13)
        self.assertEqual(step2, 2)  # 13 // 6 = 2 (i.e. ~12h actual cadence)

    def test_start_date_out_of_range_raises_index_error(self):
        ds, _ = self._make_dataset(
            start=dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc),
            count=10,
            dhours=6,
        )
        # A date that doesn't exist in the dataset (off-grid by 1 hour)
        bad = dt.datetime(2017, 1, 1, 1, tzinfo=dt.timezone.utc)
        with self.assertRaises(IndexError):
            compute_inference_range(ds, start_date=bad, end_date=None, date_step=6)

    def test_end_date_out_of_range_raises_index_error(self):
        ds, _ = self._make_dataset(
            start=dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc),
            count=10,
            dhours=6,
        )
        # Date past the dataset's last sample
        bad = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
        with self.assertRaises(IndexError):
            compute_inference_range(ds, start_date=None, end_date=bad, date_step=6)

    def test_resolved_dates_are_canonical_dataset_timestamps(self):
        # The resolved dates should be the dataset's actual timestamps at the
        # resolved indices — i.e. snapped to dataset cadence — not the user's
        # input dates verbatim. (Not strongly testable here because the fake
        # dataset only allows exact-match lookups; this test pins the contract.)
        ds, dates = self._make_dataset(
            start=dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc),
            count=10,
            dhours=6,
        )
        _, _, _, sd, ed = compute_inference_range(ds, dates[2], dates[7], date_step=6)
        # Returned datetimes are obtained via get_time_at_index, not just echoed
        self.assertEqual(sd, dates[2])
        self.assertEqual(ed, dates[7])
        # Identity check: ``ds.get_time_at_index(start_idx) is sd`` (same object)
        self.assertIs(sd, ds.get_time_at_index(2))


if __name__ == "__main__":
    unittest.main()
