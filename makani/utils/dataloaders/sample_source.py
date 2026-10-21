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

"""Turns a dataset into training samples.

Given an index, produce one sample, the input and target windows with their
dynamic features and timestamps, see :meth:`SampleSource.__call__`: which
samples exist, which one an index refers to, how the epoch is shuffled and
sharded, and how a window of timesteps is assembled. Where the bytes come from
is a :mod:`~makani.utils.dataloaders.backends` concern; this module never opens
a file.

It is written as a DALI external source callback -- called with a ``sample_info``
and raising ``StopIteration`` at the end of an epoch -- because that is what
consumes it today. Nothing else about it is DALI specific.

Shuffling, and why the seed matters
-----------------------------------
The permutation for an epoch is derived from ``base_seed + cycle_epoch_idx``
alone, never from generator state carried between calls. Every worker therefore
computes the same permutation independently, and the source can be restarted
mid-epoch from nothing but its counters -- which is what makes the DALI
checkpointing in :mod:`~makani.utils.dataloaders.dali_dataloader` work. A seed
that differed between workers would silently give each of them a different
epoch.
"""

import logging
import math
import time
from bisect import bisect_right
from itertools import accumulate
from typing import Optional

import numpy as np
import torch

from ..grid_types import DEFAULT_GRID_TYPE
from .backends import get_backend
from .data_helpers import get_date_from_string, get_date_from_timestamp, get_date_ranges, get_seconds_from_times


class SampleSource(object):
    """Assembles samples from a dataset, one window at a time.

    Parameters
    ----------
    location : str or list of str
        Where the dataset lives. The layout is detected unless ``backend`` names
        it.
    max_samples : int, optional
        Cap on how many of the available samples to use.
    samples_per_epoch : int, optional
        Samples that constitute an epoch, when it should differ from the dataset
        size.
    train : bool
        Whether to shuffle. Evaluation walks the dataset in order.
    batch_size, dt, dhours, n_history, n_future : int
        Sample geometry: batch size, stride between timesteps in samples, hours
        between samples, and how many steps of past and future a window carries.
    in_channels, out_channels : sequence of int
        Channel indices for the input and target windows, in the order the model
        expects them.
    crop_size, crop_anchor : sequence
        Region of the grid to use.
    subsampling_factor : int
        Take every n-th grid point.
    num_shards, shard_id : int
        Data parallel decomposition: each shard walks its own slice of the
        shuffled epoch.
    io_grid, io_rank : sequence of int
        Spatial decomposition, as ``[channel, lat, lon]``. Channel parallelism is
        not supported, so the first entry has to be 1.
    device_id : int
        GPU this source feeds. Recorded for the caller; nothing here runs on it.
    truncate_old : bool
        When ``max_samples`` caps the dataset, take the newest samples rather than
        the oldest.
    enable_logging : bool
        Whether to report what was found. Set on the ranks that log to screen, so
        one process describes the dataset instead of all of them.
    zenith_angle : bool
        Also return the cosine of the solar zenith angle for every timestep of
        the windows, computed on this rank's coordinates.
    lat_lon : tuple, optional
        Coordinates to use instead of those found in the files.
    dataset_name, timestamp_name : str
        Names of the field array and the time coordinate within a file, where the
        layout has them.
    channel_names : list of str, optional
        Required by layouts that address variables by name rather than by index,
        such as WeatherBench2.
    enable_odirect, odirect_alignment, enable_s3 : bool, int, bool
        HDF5 access modes, passed through to the backend and ignored by the
        others. See :class:`.backends.MakaniHDF5Backend`.
    seed : int
        Base of the shuffling seed. Has to be identical across workers.
    is_parallel : bool
        Whether this instance will be pickled into worker processes, which
        decides when the read buffers are allocated.
    timestamp_boundary_list : list of str
        ISO timestamps at which the data has a discontinuity, e.g. the start of a
        new analysis epoch. Windows spanning one are dropped.
    backend : str, optional
        Force a backend rather than detecting the layout.
    """

    def __init__(
        self,
        location,
        max_samples,
        samples_per_epoch,
        train,
        batch_size,
        dt,
        dhours,
        n_history,
        n_future,
        in_channels,
        out_channels,
        crop_size,
        crop_anchor,
        subsampling_factor=1,
        grid_type=DEFAULT_GRID_TYPE,
        num_shards=1,
        shard_id=0,
        io_grid=[1, 1, 1],
        io_rank=[0, 0, 0],
        device_id=0,
        truncate_old=True,
        enable_logging=True,
        zenith_angle=True,
        lat_lon=None,
        dataset_name="fields",
        timestamp_name="timestamp",
        channel_names=None,
        enable_odirect=False,
        odirect_alignment=0,
        enable_s3=False,
        seed=333,
        is_parallel=True,
        timestamp_boundary_list=[],
        backend: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.location = location
        self.max_samples = max_samples
        self.n_samples_per_epoch = samples_per_epoch
        self.truncate_old = truncate_old
        self.train = train
        self.dt = dt
        self.dhours = dhours
        self.n_history = n_history
        self.n_future = n_future
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_in_channels = len(in_channels)
        self.n_out_channels = len(out_channels)
        self.subsampling_factor = subsampling_factor
        self.base_seed = seed
        self.num_shards = num_shards
        self.device_id = device_id
        self.shard_id = shard_id
        self.is_parallel = is_parallel
        self.zenith_angle = zenith_angle
        self.channel_names = channel_names
        self.num_retries = 5

        # channels are read in ascending order because that is the only order
        # storage can serve efficiently, then permuted back into the requested one
        self.in_channels_sorted = np.sort(self.in_channels)
        self.in_channels_unsort = np.argsort(np.argsort(self.in_channels))
        self.in_channels_is_sorted = bool(np.all(self.in_channels_sorted == self.in_channels))
        self.out_channels_sorted = np.sort(self.out_channels)
        self.out_channels_unsort = np.argsort(np.argsort(self.out_channels))
        self.out_channels_is_sorted = bool(np.all(self.out_channels_sorted == self.out_channels))

        if io_grid[0] != 1:
            raise ValueError(f"channel parallelism is not supported, expected io_grid[0] == 1 but got {io_grid[0]}")

        self.backend = get_backend(
            location,
            backend=backend,
            dataset_name=dataset_name,
            timestamp_name=timestamp_name,
            dhours=dhours,
            channel_names=channel_names,
            crop_anchor=list(crop_anchor),
            crop_size=list(crop_size),
            io_grid=io_grid[1:],
            io_rank=io_rank[1:],
            subsampling_factor=subsampling_factor,
            grid_type=grid_type,
            lat_lon=lat_lon,
            enable_odirect=enable_odirect,
            odirect_alignment=odirect_alignment,
            enable_s3=enable_s3,
        )
        metadata = self.backend.discover(enable_logging=enable_logging)

        # everything below reads the dataset through the metadata, never the files
        self.years = metadata.labels
        self.n_years = len(metadata.files)
        self.n_samples_year = metadata.samples_per_file
        self.timestamps = metadata.timestamps
        self.total_channels = metadata.total_channels
        self.img_shape = metadata.grid.shape

        # the geometry of what this rank reads, settled by the backend
        self.crop_size = self.backend.crop_size
        self.crop_anchor = self.backend.crop_anchor
        self.read_shape = self.backend.read_shape
        self.read_anchor = self.backend.read_anchor
        self.return_shape = self.backend.chunk.shape
        self.grid = self.backend.chunk
        self.lat_lon_local = (self.backend.chunk.lat.tolist(), self.backend.chunk.lon.tolist())
        self.img_shape_resampled = (
            math.ceil(self.img_shape[0] / self.subsampling_factor),
            math.ceil(self.img_shape[1] / self.subsampling_factor),
        )

        # local coordinate mesh, for the solar zenith angle
        self.lon_grid_local, self.lat_grid_local = np.meshgrid(self.backend.chunk.lon, self.backend.chunk.lat)

        self._initialize_dataset_properties(enable_logging, timestamp_boundary_list)

        self.shuffle = bool(train)
        self.date_fn = np.vectorize(get_date_from_timestamp)

    # ---- index arithmetic --------------------------------------------------

    def _generate_indexlist(self, timestamp_boundary_list):
        """Select the sample indices a window can legitimately start at."""
        self.indices_full = np.arange(self.samples_start, self.samples_end)

        dt_total = self.dhours * self.dt
        if timestamp_boundary_list:
            boundaries = [get_date_from_string(timestamp) for timestamp in timestamp_boundary_list]

            # a window overlapping a discontinuity mixes two regimes, so the
            # exclusion reaches back a whole window and forward the history
            exclusions = get_date_ranges(
                boundaries,
                lookback_hours=dt_total * (self.n_future + 1),
                lookahead_hours=dt_total * self.n_history,
            )

            allowed = np.vectorize(lambda date: not any(start <= date < end for start, end in exclusions))
            self.indices_select = self.indices_full[allowed(np.asarray(self.timestamps)[self.indices_full])]
        else:
            self.indices_select = self.indices_full.copy()

        # a window is read from one file, so it has to sit inside one: an index
        # whose window would cross a file boundary is dropped rather than clamped,
        # since clamping would silently return duplicate samples
        file_offsets = np.asarray(self.file_offsets)
        samples_per_file = np.asarray(self.n_samples_year)
        file_idx = np.searchsorted(file_offsets, self.indices_select, side="right") - 1
        local_idx = self.indices_select - file_offsets[file_idx]
        lengths = samples_per_file[file_idx]
        fits = (local_idx >= self.dt * self.n_history) & (local_idx + self.dt * (self.n_future + 1) <= lengths - 1)
        self.indices_select = self.indices_select[fits]

    def _initialize_dataset_properties(self, enable_logging, timestamp_boundary_list):
        self.file_offsets = list(accumulate(self.n_samples_year, lambda a, b: a + b))[:-1]
        self.file_offsets.insert(0, 0)
        # kept under the old name because the DALI loader and the tests read it
        self.year_offsets = self.file_offsets
        self.n_samples_available = sum(self.n_samples_year)

        requested = (
            self.n_samples_available if self.max_samples is None else min(self.n_samples_available, self.max_samples)
        )

        # a window needs history before it and future after it, so the usable
        # range is shorter than the dataset at both ends
        if self.truncate_old:
            self.samples_start = max(
                self.dt * self.n_history,
                self.n_samples_available - requested - self.dt * (self.n_future + 1) - 1,
            )
        else:
            self.samples_start = self.dt * self.n_history
        self.samples_end = min(self.samples_start + requested, self.n_samples_available) - self.dt * (self.n_future + 1)

        self._generate_indexlist(timestamp_boundary_list)

        smallest, largest = self.indices_select.min(), self.indices_select.max()
        if (smallest < self.dt * self.n_history) or (
            largest >= (self.n_samples_available - self.dt * (self.n_future + 1))
        ):
            raise IndexError(
                f"Sample index {smallest} or {largest} is out of bounds "
                f"[{self.dt * self.n_history}, {self.n_samples_available - self.dt * (self.n_future + 1)}). "
                "Please check your index list."
            )

        self.n_samples_total = self.indices_select.shape[0]
        self.n_samples_shard = self.n_samples_total // self.num_shards

        self.num_steps_per_cycle = self.n_samples_shard // self.batch_size
        if self.n_samples_per_epoch is None:
            self.n_samples_per_epoch = self.n_samples_total
        self.num_steps_per_epoch = self.n_samples_per_epoch // (self.batch_size * self.num_shards)
        self.num_samples_per_cycle_shard = self.num_steps_per_cycle * self.batch_size
        self.num_samples_per_epoch_shard = self.num_steps_per_epoch * self.batch_size

        if enable_logging:
            self._log_summary()

        self.last_cycle_epoch = None
        self.index_permutation = None

        if not self.is_parallel:
            self._init_buffers()

    def _log_summary(self):
        logging.info(
            "Average number of samples per file: {:.1f}".format(float(self.n_samples_total) / float(self.n_years))
        )
        logging.info(
            "Found data at path {}. Number of examples: {} (distributed over {} files). "
            "Full image Shape: {} x {} x {}. Read Shape: {} x {} x {}".format(
                self.location,
                self.n_samples_available,
                self.n_years,
                self.img_shape[0],
                self.img_shape[1],
                self.total_channels,
                self.read_shape[0],
                self.read_shape[1],
                self.n_in_channels,
            )
        )
        logging.info(
            "Using {} from the total number of available samples with {} samples per epoch "
            "(corresponds to {} steps for {} shards with local batch size {})".format(
                self.n_samples_total,
                self.n_samples_per_epoch,
                self.num_steps_per_epoch,
                self.num_shards,
                self.batch_size,
            )
        )
        logging.info(f"Date range for data set: {self.timestamps[0]} to {self.timestamps[-1]}.")
        logging.info("Delta t: {} hours".format(self.dhours * self.dt))
        logging.info(
            "Including {} hours of past history in training at a frequency of {} hours".format(
                self.dhours * self.dt * (self.n_history + 1), self.dhours * self.dt
            )
        )
        logging.info(
            "Including {} hours of future targets in training at a frequency of {} hours".format(
                self.dhours * self.dt * (self.n_future + 1), self.dhours * self.dt
            )
        )

    def _get_local_year_index_from_global_index(self, sample_idx):
        file_idx = bisect_right(self.file_offsets, sample_idx) - 1
        return sample_idx - self.file_offsets[file_idx], file_idx

    # ---- buffers and windows -----------------------------------------------

    def _init_buffers(self):
        self.inp_buff = np.zeros((self.n_history + 1, self.n_in_channels, *self.return_shape), dtype=np.float32)
        self.tar_buff = np.zeros((self.n_future + 1, self.n_out_channels, *self.return_shape), dtype=np.float32)

    def _reorder_channels(self, inp, tar):
        inp = inp[:, self.in_channels_unsort, ...].copy() if not self.in_channels_is_sorted else inp.copy()
        tar = tar[:, self.out_channels_unsort, ...].copy() if not self.out_channels_is_sorted else tar.copy()
        return inp, tar

    def _open_with_retries(self, file_idx):
        """Open a file, tolerating the transient failures a busy filesystem gives."""
        for _ in range(self.num_retries):
            try:
                self.backend.open(file_idx)
                return
            except Exception as err:
                print(f"Cannot get handle for file {file_idx}. Reason {err}, retrying.", flush=True)
                time.sleep(5)
        raise OSError(f"Unable to retrieve handle for file {file_idx} after {self.num_retries} attempts, aborting.")

    def _read_window(self, file_idx, local_idx):
        input_slice = slice(local_idx - self.dt * self.n_history, local_idx + 1, self.dt)
        target_slice = slice(local_idx + self.dt, local_idx + self.dt * (self.n_future + 1) + 1, self.dt)

        self.backend.read(file_idx, input_slice, self.in_channels_sorted, out=self.inp_buff)
        self.backend.read(file_idx, target_slice, self.out_channels_sorted, out=self.tar_buff)

        return self._reorder_channels(self.inp_buff, self.tar_buff)

    def output_layout(self):
        """DALI layout of each output of :meth:`__call__`, in the order they are returned.

        The pipeline takes the number of outputs from this as well, so that the two
        cannot drift apart: fields are ``FCHW``, timestamps ``F``.
        """
        layout = ["FCHW", "FCHW"]
        if self.zenith_angle:
            layout += ["FCHW", "FCHW"]
        layout += ["F", "F"]
        return layout

    def _compute_timestamps(self, local_idx, file_idx):
        # the times the backend discovered, not ones derived from the year label:
        # a file need not start on January 1st, a partial year being the obvious case
        global_idx = self.file_offsets[file_idx] + local_idx
        inp_time = self.timestamps[global_idx - self.dt * self.n_history : global_idx + 1 : self.dt]
        tar_time = self.timestamps[global_idx + self.dt : global_idx + self.dt * (self.n_future + 1) + 1 : self.dt]
        return get_seconds_from_times(inp_time), get_seconds_from_times(tar_time)

    def _compute_zenith_angle(self, inp_times, tar_times):
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push("SampleSource:_compute_zenith_angle")

        from makani.third_party.climt.zenith_angle_v2 import cos_zenith_angle

        cos_zenith_inp = np.expand_dims(
            cos_zenith_angle(self.date_fn(inp_times), self.lon_grid_local, self.lat_grid_local).astype(np.float32),
            axis=1,
        )
        cos_zenith_tar = np.expand_dims(
            cos_zenith_angle(self.date_fn(tar_times), self.lon_grid_local, self.lat_grid_local).astype(np.float32),
            axis=1,
        )

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
        return cos_zenith_inp, cos_zenith_tar

    # ---- lifecycle ---------------------------------------------------------

    def __getstate__(self):
        # the backend drops its own handles; nothing else here resists pickling
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

        if self.is_parallel:
            self._init_buffers()

        # open every file up front so a read never stalls mid training. Starting
        # at an offset per shard staggers concurrent ranks across files, which
        # keeps them from colliding on the same one at worker startup.
        start = self.shard_id % self.n_years
        for offset in range(self.n_years):
            self._open_with_retries((start + offset) % self.n_years)

    def __len__(self):
        return self.n_samples_shard

    # ---- the callback ------------------------------------------------------

    def __call__(self, sample_info):
        """Return the sample DALI asks for, or end the epoch.

        The index is derived from the counters DALI supplies rather than from any
        state kept here, which is what lets a restarted pipeline land on the same
        sample; see the note on shuffling in the module docstring.

        Returns
        -------
        tuple of numpy.ndarray
            The outputs of one sample, positional because DALI takes them so, in
            the order of :meth:`output_layout`: the input window
            ``(n_history + 1, C, H, W)`` and the target window
            ``(n_future + 1, C, H, W)``; with ``zenith_angle``, the cosine of the
            solar zenith angle over each window, ``(T, 1, H, W)``; and the
            timestamps of each window in seconds since the epoch,
            ``(n_history + 1,)`` and ``(n_future + 1,)``, float64. The DALI
            loader names them as the keys of a batch, see
            :mod:`makani.utils.dataloaders.batch`.

        Raises
        ------
        StopIteration
            At the end of the epoch.
        """
        global_sample_idx = sample_info.idx_in_epoch + sample_info.epoch_idx * self.num_samples_per_epoch_shard
        cycle_sample_idx = global_sample_idx % self.num_samples_per_cycle_shard
        cycle_epoch_idx = global_sample_idx // self.num_samples_per_cycle_shard

        if sample_info.iteration >= self.num_steps_per_epoch:
            raise StopIteration

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push("SampleSource:__call__")

        if cycle_epoch_idx != self.last_cycle_epoch:
            self.last_cycle_epoch = cycle_epoch_idx

            # seeded only by the cycle index, so every worker agrees and a
            # restart reproduces the ordering
            rng = np.random.default_rng(seed=self.base_seed + cycle_epoch_idx)
            if self.shuffle:
                self.index_permutation = rng.permutation(self.indices_select)
            else:
                self.index_permutation = self.indices_select.copy()

            start = self.n_samples_shard * self.shard_id
            self.index_permutation = self.index_permutation[start : start + self.n_samples_shard]

        sample_idx = self.index_permutation[cycle_sample_idx]
        local_idx, file_idx = self._get_local_year_index_from_global_index(sample_idx)

        self._open_with_retries(file_idx)

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push("SampleSource:read")
        inp, tar = self._read_window(file_idx, local_idx)
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

        inp_time, tar_time = self._compute_timestamps(local_idx, file_idx)

        result = (inp, tar)
        if self.zenith_angle:
            result = result + self._compute_zenith_angle(inp_time, tar_time)
        result = result + (inp_time, tar_time)

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

        return result
