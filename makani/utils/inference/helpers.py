# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from more_itertools import batched, divide
from typing import Optional, List, Iterator, Tuple
import datetime as dt

import torch.utils.data as tud


def split_list(lst: List[int], nchunks: int) -> List[List[int]]:
    return [list(x) for x in list(divide(nchunks, lst))]


class SortedIndexSampler(tud.Sampler[List[int]]):
    def __init__(
        self,
        indices: List[int],
        maxind: int,
        batch_size: int,
        rollout_steps: int,
        rollout_dt: int,
        incomplete_rollouts: Optional[bool] = False,
    ) -> None:

        # Cap the batch size to the input length, but never below 1: ``batched(..., 0)``
        # raises in more_itertools, so empty ``indices`` would otherwise crash here.
        # With max(1, ...) and empty indices, batched([], 1) is an empty iterator and
        # the loop below produces an empty schedule — the natural "no work" outcome.
        batch_size = max(1, min(len(indices), batch_size))
        batches = map(list, batched(indices, batch_size))
        self.indices = []
        for batch in batches:
            rollout = []
            append = True
            for s in range(0, rollout_steps + 1):
                shift = [b + rollout_dt * s for b in batch]
                if max(shift) >= maxind:
                    append = False
                    break

                rollout.append(shift)

            if append or incomplete_rollouts:
                self.indices += rollout

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.indices:
            yield batch


class SimpleIndexSampler(tud.Sampler[List[int]]):
    def __init__(self, indices: List[List[int]]) -> None:
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.indices:
            yield batch


def compute_inference_range(
    dataset,
    start_date: Optional[dt.datetime] = None,
    end_date: Optional[dt.datetime] = None,
    date_step: int = 1,
) -> Tuple[int, int, int, dt.datetime, dt.datetime]:
    """
    Resolve a (start, end, step) tuple against a date-indexed dataset for inference.

    Given an optional ``start_date`` and ``end_date`` (already parsed to ``datetime``),
    look them up in the dataset to obtain integer indices and validate the resulting
    range. ``date_step`` is given in hours and converted to a sample-index stride via
    floor-division by ``dataset.dhours``.

    The dataset is expected to expose:
        - ``get_index_at_time(date) -> Optional[int]`` (None if out of range)
        - ``get_time_at_index(idx) -> datetime``
        - ``dhours: int``
        - ``start_date``, ``end_date`` (datetimes; only used in error messages)
        - ``__len__``

    Returns ``(start_index, end_index, step, resolved_start_date, resolved_end_date)``,
    where the resolved dates are the dataset's actual timestamps at the resolved
    indices (i.e. canonical, snapped to dataset cadence).

    Raises:
        IndexError: if either supplied date falls outside the dataset's coverage.
        ValueError: if ``end_index <= start_index`` (range is empty or inverted).
        ValueError: if ``date_step < dataset.dhours``.
    """
    if start_date is not None:
        start_index = dataset.get_index_at_time(start_date)
        if start_index is None:
            raise IndexError(
                f"Error, start date {start_date} is outside the dataset range of "
                f"{dataset.start_date} to {dataset.end_date}"
            )
    else:
        start_index = 0

    if end_date is not None:
        end_index = dataset.get_index_at_time(end_date)
        if end_index is None:
            raise IndexError(
                f"Error, end date {end_date} is outside the dataset range of "
                f"{dataset.start_date} to {dataset.end_date}"
            )
    else:
        end_index = len(dataset) - 1

    if end_index <= start_index:
        raise ValueError(f"Error, start date {start_date} has to be strictly smaller than end date {end_date}")

    if date_step < dataset.dhours:
        raise ValueError(f"date_step {date_step} is smaller than the dataset dhours {dataset.dhours}")

    # Floor-division: a date_step that's not an exact multiple of dhours rounds down.
    # E.g. dhours=6, date_step=7 → step=1 (one-sample stride). Pinned by tests.
    step = date_step // dataset.dhours

    resolved_start_date = dataset.get_time_at_index(start_index)
    resolved_end_date = dataset.get_time_at_index(end_index)

    return start_index, end_index, step, resolved_start_date, resolved_end_date


def translate_date_sampler_to_timedelta_sampler(sampler, date_dataset, timedelta_dataset, index_offset=0):
    # index_offset shifts each sample index to the step whose time is looked up, e.g. n_history * dt for the
    # latest step of a window, which is the step the predictions are verified against
    indexlist = []

    # like every other sample, the last one of the relative dataset stands for the times up to the next step, e.g.
    # the 18:00 sample of a 6 hourly climatology for 18:00 to 23:59 of the last day. get_index_at_time refuses
    # times past the last sample, which hourly data reaches on Dec 31 of a leap year
    end_of_last_step = timedelta_dataset.end_date + dt.timedelta(hours=timedelta_dataset.dhours)

    def lookup(tdelta):
        if timedelta_dataset.end_date < tdelta < end_of_last_step:
            tdelta = timedelta_dataset.end_date
        return timedelta_dataset.get_index_at_time(tdelta)

    iterator = iter(sampler)
    for indices in iterator:
        tstamps = [date_dataset.get_time_at_index(idx + index_offset) for idx in indices]
        timedeltas = [
            t - dt.datetime(year=t.year, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc)
            for t in tstamps
        ]
        tindices = [lookup(t) for t in timedeltas]

        # a time outside the relative dataset would reach its loader as index None, fail here instead
        for idx, t, tdelta, tidx in zip(indices, tstamps, timedeltas, tindices):
            if tidx is None:
                raise ValueError(
                    f"Error, sample {idx} at {t} maps to {tdelta} into the year, which is outside the range "
                    f"{timedelta_dataset.start_date} to {timedelta_dataset.end_date} of the relative dataset"
                )

        indexlist.append(tindices)

    return SimpleIndexSampler(indexlist)
