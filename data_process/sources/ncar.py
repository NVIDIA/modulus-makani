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

"""NSF NCAR ERA5 (RDA d633000) source, read from the public ``nsf-ncar-era5`` S3 bucket.

The NCAR grid is already the makani grid, so no regridding or latitude flipping
takes place; a mismatch against the metadata grid is an error rather than
something this source tries to fix.
"""

from typing import Dict, List, Optional
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import io
import threading
import os
import numpy as np
import h5py as h5
import warnings

from makani.utils.dataloaders.ncar_helpers import (
    NCAR_ERA5_BUCKET,
    accumulation_key,
    analysis_pl_key,
    analysis_sfc_key,
    build_ncar_channel_groups,
    invariant_key,
    invariant_variables,
    resolve_accumulation_segments,
    to_ncar_hours,
)
from data_process.sources import InvariantSource, Sample, Source, Unit


class NcarStore(object):
    """Reader for NCAR ERA5 netCDF4 objects on S3, with optional local caching.

    The objects are chunked such that a single chunk holds one timestep of all
    pressure levels, so h5py byte range reads over S3 subset in time for free
    while subsetting in level is not possible. Open file handles are kept in a
    small LRU cache because a rank walks days in order and therefore revisits
    the same monthly surface file many times.

    Parameters
    ----------
    bucket : str
        Name of the S3 bucket to read from.
    cache_dir : str, optional
        If given, objects are downloaded here on first use and reopened from
        local disk afterwards. Downloads are staged through a temporary file and
        renamed into place, so concurrent ranks cannot observe a partial file.
    max_open : int, optional
        Number of file handles to keep open. Wants to comfortably exceed the
        number of monthly surface and accumulation files a day touches, since
        those are revisited on every day of the month; pressure level files are
        released explicitly and do not compete for slots.
    prefetch_workers : int, optional
        Number of background threads fetching upcoming objects. Zero disables
        prefetching and every object is read lazily on demand.

        Concurrency has to happen at this level rather than around the h5py
        reads: h5py serializes every HDF5 call behind a global lock, and because
        the file object is a Python fsspec stream the network waits happen
        inside that lock too, so threading the reads themselves buys nothing.
        Fetching whole objects in the background instead gives genuinely
        concurrent S3 streams and overlaps transfer with decompression.

        Each in flight object is held whole, in memory when streaming, so peak
        memory grows by roughly ``prefetch_workers`` times the object size, and
        pressure level files are on the order of a gigabyte. Size this against
        the ranks you place per node. With ``cache_dir`` set the prefetch lands
        on disk instead and costs no memory.
    """

    def __init__(
        self,
        bucket: str,
        cache_dir: Optional[str] = None,
        max_open: Optional[int] = 16,
        prefetch_workers: Optional[int] = 0,
    ):
        import s3fs

        self.bucket = bucket
        self.cache_dir = cache_dir
        self.max_open = max_open
        self.prefetch_workers = prefetch_workers
        self.fs = s3fs.S3FileSystem(anon=True)
        self._handles = OrderedDict()
        self._coords = {}
        self._pool = ThreadPoolExecutor(prefetch_workers) if prefetch_workers > 0 else None
        self._pending = OrderedDict()
        self._plan = []
        self._plan_position = 0

    def _fetch(self, key: str) -> str:
        """Download ``key`` into the cache directory if not already present, return its local path."""
        path = os.path.join(self.cache_dir, key)
        if os.path.isfile(path):
            return path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # unique per rank and per prefetch thread, so two fetches can never
        # stage through the same temporary path
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.part"
        self.fs.get(f"{self.bucket}/{key}", tmp)
        os.replace(tmp, path)
        return path

    def set_read_plan(self, keys: List[str]):
        """Declare the objects this rank will read, in order, and start filling the pipe.

        The plan is what makes prefetching useful: the store can keep exactly
        ``prefetch_workers`` fetches in flight at all times, topping the queue up
        as each object is consumed, rather than draining and refilling in bursts.
        """
        self._plan = list(keys)
        self._plan_position = 0
        self._top_up()

    def _top_up(self):
        """Queue background fetches until ``prefetch_workers`` objects are in flight."""
        if self._pool is None:
            return
        while len(self._pending) < self.prefetch_workers and self._plan_position < len(self._plan):
            key = self._plan[self._plan_position]
            self._plan_position += 1
            if key in self._pending or key in self._handles:
                continue
            if self.cache_dir is not None:
                self._pending[key] = self._pool.submit(self._fetch, key)
            else:
                self._pending[key] = self._pool.submit(self.fs.cat_file, f"{self.bucket}/{key}")

    def open(self, key: str) -> h5.File:
        """Return an open :class:`h5py.File` for ``key``, reusing a cached handle if possible."""
        if key in self._handles:
            self._handles.move_to_end(key)
            return self._handles[key][0]

        while len(self._handles) >= self.max_open:
            stale_key, (stale_file, stale_raw) = self._handles.popitem(last=False)
            stale_file.close()
            if stale_raw is not None:
                stale_raw.close()
            for cached in [k for k in self._coords if k[0] == stale_key]:
                del self._coords[cached]

        pending = self._pending.pop(key, None)
        if pending is not None:
            # blocks only if the background fetch has not finished yet
            fetched = pending.result()
            self._top_up()
            if self.cache_dir is not None:
                handle, raw = h5.File(fetched, "r"), None
            else:
                raw = io.BytesIO(fetched)
                handle = h5.File(raw, "r")
        elif self.cache_dir is not None:
            handle, raw = h5.File(self._fetch(key), "r"), None
        else:
            raw = self.fs.open(f"{self.bucket}/{key}", "rb")
            handle = h5.File(raw, "r")

        self._handles[key] = (handle, raw)
        return handle

    def release(self, key: str):
        """Close ``key`` and drop its cached coordinates.

        Called once a file is known to be finished with. This matters for the
        pressure level files: they are consumed entirely within a single day, and
        under prefetching they are resident in memory whole, so holding them in
        the LRU until something evicts them would pin gigabytes per rank for no
        benefit.
        """
        entry = self._handles.pop(key, None)
        if entry is not None:
            handle, raw = entry
            handle.close()
            if raw is not None:
                raw.close()
        for cached in [k for k in self._coords if k[0] == key]:
            del self._coords[cached]

    def coord(self, key: str, name: str) -> np.ndarray:
        """Return a one dimensional coordinate of ``key``, cached in memory.

        Coordinates are tiny but are consulted once per sample per variable to
        locate a timestep, and over S3 each of those lookups is a round trip.
        Entries are dropped when the owning handle is evicted, so a reopened
        file is always read afresh.
        """
        if (key, name) not in self._coords:
            self._coords[(key, name)] = self.open(key)[name][:]
        return self._coords[(key, name)]

    def close(self):
        """Close all open handles and drop anything still being prefetched."""
        for pending in self._pending.values():
            pending.cancel()
        self._pending.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for handle, raw in self._handles.values():
            handle.close()
            if raw is not None:
                raw.close()
        self._handles.clear()
        self._coords.clear()


def _mask_fill(data: np.ndarray, dset: h5.Dataset) -> np.ndarray:
    """Replace the netCDF ``_FillValue`` with NaN, in place.

    Fields that are only defined over part of the globe, most notably sea
    surface temperature, carry a large sentinel over the undefined region. The
    makani dataloaders understand NaN, so translate it here.

    ``data`` must be a freshly read array that the caller owns, since it is
    modified in place: the pressure level blocks are 150 MB apiece and neither a
    copy nor the float64 promotion that ``np.where(..., np.nan, ...)`` would
    introduce is worth paying for on this path.
    """
    fill = dset.attrs.get("_FillValue", None)
    if fill is None:
        return data
    mask = data == np.asarray(fill).astype(data.dtype).item()
    if mask.any():
        data[mask] = np.nan
    return data


def _coord_index(values: np.ndarray, wanted: int, name: str) -> int:
    """Return the position of ``wanted`` (hours since 1900-01-01) in a time coordinate."""
    matches = np.nonzero(values == wanted)[0]
    if matches.size == 0:
        raise IndexError(f"Time {wanted} not found in coordinate '{name}'.")
    return int(matches[0])


def _check_grid(handle: h5.File, lat: List[float], lon: List[float]):
    """Verify that the source grid matches the grid declared in the metadata."""
    file_lat, file_lon = handle["latitude"][:], handle["longitude"][:]
    if file_lat.shape[0] != len(lat) or file_lon.shape[0] != len(lon):
        raise ValueError(
            f"Grid mismatch: NCAR ERA5 is {file_lat.shape[0]}x{file_lon.shape[0]} but the metadata declares "
            f"{len(lat)}x{len(lon)}. This converter does not regrid; use a metadata file on the native "
            f"0.25 degree grid."
        )
    if not (np.allclose(file_lat, lat) and np.allclose(file_lon, lon)):
        raise ValueError(
            "Grid mismatch: NCAR ERA5 coordinates differ from the metadata coordinates. Expected latitude "
            "descending from 90 to -90 and longitude ascending from 0 to 359.75."
        )


def _keys_for_day(groups, day, day_times, window_hours) -> List[str]:
    """List every object needed to fill one day, in the order the fill loop reads them."""
    keys = []
    for group in groups:
        if group.kind == "pl":
            keys.append(analysis_pl_key(group.variables[0], day))
        elif group.kind == "sfc":
            keys.append(analysis_sfc_key(group.variables[0], day))
        else:
            for _, valid_time in day_times:
                for init_time, _, _ in resolve_accumulation_segments(valid_time, window_hours):
                    keys += [accumulation_key(variable, init_time) for variable in group.variables]
    # preserve order while dropping repeats, since accumulation runs are shared between samples
    return list(dict.fromkeys(keys))


def _fill_pressure_levels(store, group, out, entry_key, day, day_times, lat, lon, grid_checked):
    """Fill all channels of one pressure level variable for a single day."""
    key = analysis_pl_key(group.variables[0], day)
    handle = store.open(key)
    if key not in grid_checked:
        _check_grid(handle, lat, lon)
        grid_checked.add(key)

    dset = handle[group.variables[0].h5_name]
    levels = list(store.coord(key, "level").astype(int))
    level_positions = [levels.index(level) for level in group.levels]
    times = store.coord(key, "time")

    for sample_index, valid_time in day_times:
        # one read pulls the whole chunk, which holds every level of this timestep
        block = _mask_fill(dset[_coord_index(times, to_ncar_hours(valid_time), "time")], dset)
        for cidx, position in zip(group.channel_indices, level_positions):
            out[entry_key][sample_index, cidx, ...] = block[position]

    # a pressure level file covers exactly one day, so it is finished with here
    store.release(key)


def _fill_surface(store, group, out, entry_key, day, day_times, lat, lon, grid_checked):
    """Fill one surface analysis channel for a single day."""
    variable = group.variables[0]
    key = analysis_sfc_key(variable, day)
    handle = store.open(key)
    if key not in grid_checked:
        _check_grid(handle, lat, lon)
        grid_checked.add(key)

    dset = handle[variable.h5_name]
    times = store.coord(key, "time")
    cidx = group.channel_indices[0]
    for sample_index, valid_time in day_times:
        index = _coord_index(times, to_ncar_hours(valid_time), "time")
        out[entry_key][sample_index, cidx, ...] = _mask_fill(dset[index], dset)


def _fill_accumulated(store, group, out, entry_key, day, day_times, window_hours, lat, lon, grid_checked):
    """Fill one accumulated channel for a single day.

    NCAR stores the accumulated fields already de-accumulated, one value per
    forecast hour, so the value over the window is the sum of the hours it
    spans, taken across every run the window touches. Each read returns all 12
    forecast hours of a run, and consecutive samples usually share a run, so the
    runs are memoized for the duration of the day.
    """
    cidx = group.channel_indices[0]
    runs: Dict[tuple, np.ndarray] = {}

    def read_run(variable, init_time):
        key = accumulation_key(variable, init_time)
        if (key, variable.h5_name) not in runs:
            handle = store.open(key)
            if key not in grid_checked:
                _check_grid(handle, lat, lon)
                grid_checked.add(key)
            dset = handle[variable.h5_name]
            inits = store.coord(key, "forecast_initial_time")
            index = _coord_index(inits, to_ncar_hours(init_time), "forecast_initial_time")
            runs[(key, variable.h5_name)] = _mask_fill(dset[index], dset)
        return runs[(key, variable.h5_name)]

    for sample_index, valid_time in day_times:
        total = np.zeros((len(lat), len(lon)), dtype=np.float32)
        for init_time, hour_start, hour_end in resolve_accumulation_segments(valid_time, window_hours):
            for variable in group.variables:
                run = read_run(variable, init_time)
                # forecast hour h holds the accumulation over the hour ending at h
                total += run[hour_start:hour_end].sum(axis=0)

        out[entry_key][sample_index, cidx, ...] = total


class NcarSource(Source):
    """Read ERA5 from the NSF NCAR bucket on S3.

    Work is split over whole days rather than individual timestamps, because the
    pressure level files are chunked as one chunk per timestep across all levels
    and splitting a day across ranks would make several ranks fetch and
    decompress the same chunk. Days are handed out in contiguous runs, so that a
    rank keeps hitting the same monthly surface file.

    Parameters
    ----------
    metadata : Dict
        Dataset metadata, see :class:`data_process.sources.Source`.
    comm_rank : int
        MPI rank, used to restrict informational output to rank 0.
    bucket : str
        Name of the S3 bucket holding the NCAR ERA5 data.
    cache_dir : str, optional
        Directory used to cache the raw NCAR files. Without it nothing is written
        to local disk, but an interrupted run has to refetch everything. Note that
        the raw files are considerably larger than the converted output.
    accumulation_hours : int, optional
        Length in hours of the window used for accumulated channels such as tp.
        Defaults to dhours. Windows longer than a forecast run are stitched
        together from consecutive runs.
    prefetch_workers : int, optional
        Background threads per rank fetching upcoming objects, overlapping
        transfer with decompression and giving each rank several concurrent S3
        streams. Zero, the default, reads lazily on demand. This only helps if
        the reads are latency bound rather than capped by aggregate bandwidth;
        when the link is already saturated it adds memory pressure for nothing.
        While streaming, peak memory grows by roughly this many object sizes,
        and pressure level objects are on the order of a gigabyte, so weigh it
        against the ranks placed per node.
    skip_missing_channels : bool
        Setting this flag to True will skip channels without an NCAR counterpart instead of failing.
    impute_missing_timestamps : bool
        Setting this flag to True will write NaN and mark ``valid_data`` for
        data that is not on the bucket, instead of failing. This works per
        channel group and day: if any object or timestep a group needs for a
        day is missing, all of that group's channels are imputed for the whole
        day. Typical case is converting the current year, where the forecast
        streams behind ``tp`` lag the analysis streams by about a month.
    """

    def __init__(
        self,
        metadata: Dict,
        comm_rank: int,
        bucket: Optional[str] = NCAR_ERA5_BUCKET,
        cache_dir: Optional[str] = None,
        accumulation_hours: Optional[int] = None,
        prefetch_workers: Optional[int] = 0,
        skip_missing_channels: Optional[bool] = False,
        impute_missing_timestamps: Optional[bool] = False,
    ):
        super().__init__(metadata, comm_rank)
        self.impute_missing_timestamps = impute_missing_timestamps

        # group channels by the source file that provides them
        self.groups = build_ncar_channel_groups(self.channel_names, skip_missing_channels=skip_missing_channels)
        self.accumulation_hours = self.dhours if accumulation_hours is None else accumulation_hours
        if comm_rank == 0:
            covered = sum(len(g.channel_indices) for g in self.groups)
            if covered != len(self.channel_names):
                warnings.warn(f"Skipping {len(self.channel_names) - covered} channels without an NCAR counterpart.")
            if any(g.kind == "accum" for g in self.groups):
                print(f"Accumulated channels use a {self.accumulation_hours}h window ending at the sample time.")

        self.store = NcarStore(bucket, cache_dir=cache_dir, prefetch_workers=prefetch_workers)
        self.grid_checked = set()

    def skipped_channel_indices(self) -> List[int]:
        covered = {cidx for group in self.groups for cidx in group.channel_indices}
        return [cidx for cidx in range(len(self.channel_names)) if cidx not in covered]

    def split_units(self, samples: List[Sample]) -> List[Unit]:
        """Bucket the samples by calendar day."""
        days = OrderedDict()
        for sample in samples:
            days.setdefault(sample[1].date(), []).append(sample)
        return list(days.values())

    def begin_year(self, units: List[Unit]):
        # Declare the objects to prefetch, so the store can keep the queue topped
        # up across day boundaries rather than in bursts. Only the pressure level
        # files qualify: they are read in full within one day, which is exactly
        # what a whole-object fetch suits, and they dominate the byte volume. The
        # surface and accumulation files are read in slivers spread over a month
        # and are better left on the lazy block-cached path, which also keeps
        # them from sitting in memory whole for weeks of simulated time.
        pressure_level_groups = [group for group in self.groups if group.kind == "pl"]
        self.store.set_read_plan(
            [
                key
                for day_times in units
                for key in _keys_for_day(
                    pressure_level_groups, day_times[0][1].date(), day_times, self.accumulation_hours
                )
            ]
        )

    def fill(self, out: h5.File, entry_key: str, units: List[Unit]):
        for day_times in units:
            day = day_times[0][1].date()
            for group in self.groups:
                try:
                    self._fill_group(out, entry_key, group, day, day_times)
                except (FileNotFoundError, IndexError) as error:
                    # a missing object surfaces as FileNotFoundError, a missing
                    # timestep within an object as IndexError from _coord_index
                    if not self.impute_missing_timestamps:
                        raise
                    self._impute(out, entry_key, group, day, day_times, error)

    def _fill_group(self, out, entry_key, group, day, day_times):
        if group.kind == "pl":
            _fill_pressure_levels(
                self.store, group, out, entry_key, day, day_times, self.lat, self.lon, self.grid_checked
            )
        elif group.kind == "sfc":
            _fill_surface(self.store, group, out, entry_key, day, day_times, self.lat, self.lon, self.grid_checked)
        else:
            _fill_accumulated(
                self.store,
                group,
                out,
                entry_key,
                day,
                day_times,
                self.accumulation_hours,
                self.lat,
                self.lon,
                self.grid_checked,
            )

    def _impute(self, out, entry_key, group, day, day_times, error):
        """Write NaN for all channels of ``group`` on ``day`` and mark them invalid."""
        if group.kind == "pl":
            # the fill bailed out before releasing the day's pressure level file
            self.store.release(analysis_pl_key(group.variables[0], day))
        names = [self.channel_names[cidx] for cidx in group.channel_indices]
        print(f"Imputing {day} for {', '.join(names)}: {error}")
        # one explicit array per write, rather than a scalar h5py would expand row by row
        nan = np.full((len(self.lat), len(self.lon)), np.nan, dtype=np.float32)
        for sample_index, _ in day_times:
            for cidx in group.channel_indices:
                out[entry_key][sample_index, cidx, ...] = nan
                out["valid_data"][sample_index, cidx] = 0

    def close(self):
        self.store.close()


class NcarInvariantSource(InvariantSource):
    """Read the time invariant ERA5 fields from the NSF NCAR bucket on S3.

    Each field is a single small object holding one timestep, so every object is
    simply opened and read whole.

    Parameters
    ----------
    metadata : Dict
        Invariant metadata, see :class:`data_process.sources.InvariantSource`.
    bucket : str
        Name of the S3 bucket holding the NCAR ERA5 data.
    skip_missing_channels : bool
        Setting this flag to True will skip channels without an NCAR counterpart
        instead of failing. A field with a counterpart that is absent from the
        bucket always fails.
    """

    def __init__(
        self,
        metadata: Dict,
        bucket: Optional[str] = NCAR_ERA5_BUCKET,
        skip_missing_channels: Optional[bool] = False,
    ):
        super().__init__(metadata)

        self.missing = [name for name in self.channel_names if name not in invariant_variables]
        if self.missing:
            if not skip_missing_channels:
                raise ValueError(
                    f"Channels {self.missing} have no NCAR invariant counterpart. Known names: {list(invariant_variables)}"
                )
            warnings.warn(f"Skipping channels without an NCAR invariant counterpart: {self.missing}")

        self.store = NcarStore(bucket)

    def skipped_channel_indices(self) -> List[int]:
        return [self.channel_names.index(name) for name in self.missing]

    def fill(self, out: h5.File, entry_key: str):
        for cidx, name in enumerate(self.channel_names):
            if name in self.missing:
                continue
            variable = invariant_variables[name]
            key = invariant_key(variable)
            handle = self.store.open(key)
            _check_grid(handle, self.lat, self.lon)
            dset = handle[variable.h5_name]
            # the single timestep of the stream
            out[entry_key][cidx, ...] = _mask_fill(dset[0], dset)
            self.store.release(key)

    def summary(self) -> Optional[str]:
        return f"Skipped channels: {self.missing}" if self.missing else None

    def close(self):
        self.store.close()
