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

"""A torch dataset over a makani dataset.

The map-style counterpart to the DALI pipeline: indexable, so a sample can be
asked for by index or by time, which is what inference and the climatology and
mask lookups need. Where the bytes come from is a
:mod:`~makani.utils.dataloaders.backends` concern; this module assembles windows
from them, normalizes, and converts to the model grid.

Unlike the training source, a window here may span files -- an inference run
walks the dataset in order and does not care where one file ends -- so each
timestep is resolved to its own file and read separately.
"""

import logging
from typing import Optional, List, Tuple, Union
from itertools import accumulate
import operator
from bisect import bisect_right
import math
import datetime as dt

import torch
import numpy as np
from torch.utils.data import Dataset

# for data normalization
from makani.utils.dataloaders.data_helpers import (
    get_date_from_timestamp,
    get_seconds_from_times,
    get_timedelta_from_timestamp,
)

# storage backends
from makani.utils.dataloaders.backends import get_backend
from makani.utils.dataloaders.data_shapes import DataShapes
from makani.utils.dataloaders.batch import make_batch

# for grid conversion
from makani.utils.grid_types import DEFAULT_GRID_TYPE
from makani.utils.grids import GridConverter


class MultifilesDataset(Dataset):
    def __init__(
        self,
        location: Union[str, List[str]],
        dt: int,
        in_channels: List[int],
        out_channels: List[int],
        n_history: Optional[int] = 0,
        n_future: Optional[int] = 0,
        add_zenith: Optional[bool] = False,
        data_grid_type: Optional[str] = "equiangular",
        model_grid_type: Optional[str] = "equiangular",
        bias: Optional[np.array] = None,
        scale: Optional[np.array] = None,
        relative_timestamp: Optional[bool] = False,
        return_target: Optional[bool] = True,
        dataset_name: Optional[str] = "fields",
        timestamp_name: Optional[str] = "timestamp",
        channel_names: Optional[List[str]] = None,
        latitude_name: Optional[str] = "lat",
        longitude_name: Optional[str] = "lon",
        enable_s3: Optional[bool] = False,
        enable_odirect: Optional[bool] = False,
        odirect_alignment: Optional[int] = 0,
        crop_size: Optional[Tuple[int, int]] = (None, None),
        crop_anchor: Optional[Tuple[int, int]] = (0, 0),
        subsampling_factor: Optional[int] = 1,
        grid_type: Optional[str] = DEFAULT_GRID_TYPE,
        io_grid: Optional[List[int]] = [1, 1, 1],
        io_rank: Optional[List[int]] = [0, 0, 0],
        enable_logging: Optional[bool] = True,
        backend: Optional[str] = None,
        **kwargs,
    ):

        self.location = location
        self.dt = dt
        self.n_history = n_history
        self.n_future = n_future
        self.in_channels = np.array(in_channels)
        self.out_channels = np.array(out_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)
        self.add_zenith = add_zenith
        self.relative_timestamp = relative_timestamp
        self.return_target = return_target
        self.dataset_name = dataset_name
        self.timestamp_name = timestamp_name
        self.channel_names = channel_names
        self.latitude_name = latitude_name
        self.longitude_name = longitude_name
        self.enable_s3 = enable_s3
        self.enable_odirect = enable_odirect
        self.odirect_alignment = odirect_alignment
        self.backend_name = backend

        # channels are read in ascending order because that is the only order
        # storage can serve efficiently, then permuted back into the requested one
        self.in_channels_sorted = np.sort(self.in_channels)
        self.in_channels_unsort = np.argsort(np.argsort(self.in_channels))
        self.in_channels_is_sorted = bool(np.all(self.in_channels_sorted == self.in_channels))
        self.out_channels_sorted = np.sort(self.out_channels)
        self.out_channels_unsort = np.argsort(np.argsort(self.out_channels))
        self.out_channels_is_sorted = bool(np.all(self.out_channels_sorted == self.out_channels))

        # multifiles dataloader doesn't support channel parallelism yet
        # set the read slices
        if io_grid[0] != 1:
            raise ValueError(f"channel parallelism is not supported, expected io_grid[0] == 1 but got {io_grid[0]}")
        self.io_grid = io_grid[1:]
        self.io_rank = io_rank[1:]

        # crop info
        self.crop_size = crop_size
        self.crop_anchor = crop_anchor
        self.subsampling_factor = subsampling_factor
        self.grid_type = grid_type

        # datetime logic
        if self.relative_timestamp:
            self.date_fn = np.vectorize(get_timedelta_from_timestamp)
        else:
            self.date_fn = np.vectorize(get_date_from_timestamp)

        # get more info
        self._get_files_stats(enable_logging)

        # for normalization load the statistics
        self.normalize = True

        if bias is not None:
            self.in_bias = bias[:, self.in_channels]
            self.out_bias = bias[:, self.out_channels]
        else:
            self.in_bias = np.zeros((1, len(self.in_channels), 1, 1))
            self.out_bias = np.zeros((1, len(self.out_channels), 1, 1))

        if scale is not None:
            self.in_scale = scale[:, self.in_channels]
            self.out_scale = scale[:, self.out_channels]
        else:
            self.in_scale = np.ones((1, len(self.in_channels), 1, 1))
            self.out_scale = np.ones((1, len(self.out_channels), 1, 1))

        # local coordinate mesh, for the solar zenith angle
        self.lon_grid_local, self.lat_grid_local = np.meshgrid(self.backend.chunk.lon, self.backend.chunk.lat)

        # grid types
        self.grid_converter = GridConverter(
            data_grid_type,
            model_grid_type,
            torch.deg2rad(torch.tensor(self.lat_lon_local[0])).to(torch.float32),
            torch.deg2rad(torch.tensor(self.lat_lon_local[1])).to(torch.float32),
        )

        # the geometry the run needs, in the one shape every loader reports it
        self.data_shapes = DataShapes.from_loader(self)

    def _get_files_stats(self, enable_logging):
        """Ask the backend what is there, and settle what this rank reads."""

        # arbitrary file names, ordered by time rather than by name: an inference
        # dataset is not necessarily a file per year
        self.backend = get_backend(
            self.location,
            backend=self.backend_name,
            dataset_name=self.dataset_name,
            timestamp_name=self.timestamp_name,
            channel_names=self.channel_names,
            latitude_name=self.latitude_name,
            longitude_name=self.longitude_name,
            file_pattern="*",
            relative_timestamp=self.relative_timestamp,
            crop_anchor=list(self.crop_anchor),
            crop_size=list(self.crop_size),
            io_grid=self.io_grid,
            io_rank=self.io_rank,
            subsampling_factor=self.subsampling_factor,
            grid_type=self.grid_type,
            enable_s3=self.enable_s3,
            enable_odirect=self.enable_odirect,
            odirect_alignment=self.odirect_alignment,
        )
        metadata = self.backend.discover(enable_logging=enable_logging)

        self.files_paths = metadata.files
        self.n_samples_file = metadata.samples_per_file

        # the backend reports times as datetimes (or timedeltas), which is what
        # the by-time lookups compare against; a sample carries the same
        # times as float seconds, which is what a tensor can hold
        self.datestamps = metadata.timestamps
        self.timestamps = get_seconds_from_times(self.datestamps)
        self.img_shape = metadata.grid.shape
        self.total_channels = metadata.total_channels
        self.lat_lon = (metadata.grid.lat.tolist(), metadata.grid.lon.tolist())

        # the geometry of what this rank reads, settled by the backend
        self.crop_size = tuple(self.backend.crop_size)
        self.read_anchor = tuple(self.backend.read_anchor)
        self.read_shape = tuple(self.backend.read_shape)
        self.return_shape = self.backend.chunk.shape
        self.lat_lon_local = (self.backend.chunk.lat.tolist(), self.backend.chunk.lon.tolist())

        # the cadence has to be constant: a sample index means a time only if the
        # step between samples is the same everywhere
        steps = {int(delta.total_seconds() // 3600) for delta in np.diff(self.datestamps).tolist()}
        if len(steps) > 1:
            raise RuntimeError(
                "The time difference between steps is not constant, provide a dataset where this is the case"
            )
        self.dhours = steps.pop()

        if not self.relative_timestamp:
            self.years = sorted({date.year for date in self.datestamps.tolist()})
            self.n_years = len(self.years)

        self.start_date = self.datestamps[0]
        self.end_date = self.datestamps[-1]

        # sample indexing
        self.file_offsets = list(accumulate(self.n_samples_file, operator.add))[:-1]
        self.file_offsets.insert(0, 0)
        self.n_samples_available = sum(self.n_samples_file)
        self.n_samples_total = self.n_samples_available

        if enable_logging:
            self._log_summary()

        # set properties for compatibility
        self.img_shape_x = self.img_shape[0]
        self.img_shape_y = self.img_shape[1]

        self.img_crop_shape_x = self.crop_size[0]
        self.img_crop_shape_y = self.crop_size[1]
        self.img_crop_offset_x = self.crop_anchor[0]
        self.img_crop_offset_y = self.crop_anchor[1]

        self.img_local_shape_x = self.read_shape[0]
        self.img_local_shape_y = self.read_shape[1]
        self.img_local_offset_x = self.read_anchor[0]
        self.img_local_offset_y = self.read_anchor[1]

        # resampling stuff
        self.img_shape_resampled = (
            math.ceil(self.img_shape[0] / self.subsampling_factor),
            math.ceil(self.img_shape[1] / self.subsampling_factor),
        )
        self.img_local_shape_x_resampled = self.return_shape[0]
        self.img_local_shape_y_resampled = self.return_shape[1]
        self.img_shape_x_resampled = self.img_shape_resampled[0]
        self.img_shape_y_resampled = self.img_shape_resampled[1]

    def _log_summary(self):
        logging.info(
            "Average number of samples per file: {:.1f}".format(
                float(self.n_samples_total) / float(len(self.files_paths))
            )
        )
        logging.info(
            "Found data at path {}. Number of examples: {}. Full image Shape: {} x {} x {}. "
            "Read Shape: {} x {} x {}".format(
                self.location,
                self.n_samples_available,
                self.img_shape[0],
                self.img_shape[1],
                self.total_channels,
                self.read_shape[0],
                self.read_shape[1],
                self.n_in_channels,
            )
        )
        logging.info(
            f"Dataset covers a timespan from {self.start_date} to {self.end_date} "
            f"with a resolution of {self.dhours} hour(s)."
        )
        logging.info(f"Using a step size of {self.dhours*self.dt} hour(s) for inference.")
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

    def _compute_timestamps(self, global_idx, offset_start, offset_end):
        # the same samples _get_data reads: dt apart, not adjacent
        return self.timestamps[global_idx + self.dt * offset_start : global_idx + self.dt * offset_end : self.dt]

    def _compute_zenith_angle(self, times):
        # import
        from makani.third_party.climt.zenith_angle_v2 import cos_zenith_angle

        # convert to datetimes:
        times_dt = self.date_fn(times)

        # compute the corresponding zenith angles
        cos_zenith = np.expand_dims(
            cos_zenith_angle(times_dt, self.lon_grid_local, self.lat_grid_local).astype(np.float32), axis=1
        )

        return cos_zenith

    def _get_indices(self, global_idx):
        file_idx = bisect_right(self.file_offsets, global_idx) - 1
        local_idx = global_idx - self.file_offsets[file_idx]
        return file_idx, local_idx

    def _get_data(self, global_idx, offset_start, offset_end, target=False):
        # a window may span files, so every timestep is resolved to its own file
        # and read on its own; the backend already knows which region to read
        channels = self.out_channels_sorted if target else self.in_channels_sorted

        data_list = []
        for offset_idx in range(offset_start, offset_end):
            file_idx, local_idx = self._get_indices(global_idx + self.dt * offset_idx)
            data_list.append(self.backend.read(file_idx, slice(local_idx, local_idx + 1), channels))

        data = np.concatenate(data_list, axis=0)

        if target:
            if not self.out_channels_is_sorted:
                data = data[:, self.out_channels_unsort, ...]
        elif not self.in_channels_is_sorted:
            data = data[:, self.in_channels_unsort, ...]

        if self.normalize:
            if target:
                data = (data - self.out_bias) / self.out_scale
            else:
                data = (data - self.in_bias) / self.in_scale

        return data

    def __len__(self):
        toff = 1 if self.return_target else 0
        return self.n_samples_total - self.dt * (self.n_history + self.n_future + toff)

    def get_sample_at_index(self, global_idx, return_target=True):
        """The sample starting at ``global_idx``, as a dict, see :mod:`makani.utils.dataloaders.batch`."""

        def as_field(arr):
            return self.grid_converter(torch.as_tensor(arr, dtype=torch.float32))

        def as_time(arr):
            return torch.as_tensor(arr, dtype=torch.float64)

        inp = self._get_data(global_idx, 0, self.n_history + 1, target=False)
        inp_time = self._compute_timestamps(global_idx, 0, self.n_history + 1)
        inp_zen = self._compute_zenith_angle(inp_time) if self.add_zenith else None

        tar = tar_time = tar_zen = None
        if return_target:
            tar = self._get_data(global_idx, self.n_history + 1, self.n_history + self.n_future + 2, target=True)
            tar_time = self._compute_timestamps(global_idx, self.n_history + 1, self.n_history + self.n_future + 2)
            tar_zen = self._compute_zenith_angle(tar_time) if self.add_zenith else None

        return make_batch(
            inp=as_field(inp),
            inp_timestamp=as_time(inp_time),
            tar=None if tar is None else as_field(tar),
            tar_timestamp=None if tar_time is None else as_time(tar_time),
            inp_zenith=None if inp_zen is None else as_field(inp_zen),
            tar_zenith=None if tar_zen is None else as_field(tar_zen),
        )

    # this is just for the torch dataloader
    def __getitem__(self, global_idx):

        result = self.get_sample_at_index(global_idx, return_target=self.return_target)

        return result

    def get_index_at_time(self, tstamp):
        # return the sample which is equal or smaller than timestamp:
        if self.relative_timestamp:
            if not isinstance(tstamp, dt.timedelta):
                tstamp = get_timedelta_from_timestamp(tstamp)
        else:
            if not isinstance(tstamp, dt.datetime):
                tstamp = get_date_from_timestamp(tstamp)

        if (tstamp < self.start_date) or (tstamp > self.end_date):
            return None

        # this returns the position in the sorted list. We need to find it in the original list then
        gidx = bisect_right(self.datestamps, tstamp) - 1

        return gidx

    def get_time_at_index(self, global_idx):
        return self.datestamps[global_idx]

    def get_sample_at_time(self, timestamp):
        global_idx = self.get_index_at_time(timestamp)
        if global_idx is None:
            raise IndexError(f"Time stamp {timestamp} is out of range of the dataset.")
        return self.get_sample_at_index(global_idx, return_target=self.return_target)

    def get_output_normalization(self):
        return self.out_bias, self.out_scale

    def get_input_normalization(self):
        return self.in_bias, self.in_scale
