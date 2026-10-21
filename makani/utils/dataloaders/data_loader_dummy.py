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

import logging
import datetime
from bisect import bisect_right
from typing import Optional, List, Tuple
import glob
import torch
import numpy as np
import h5py
import math

# distributed stuff
from makani.utils import comm

# we need this
from torch_harmonics.distributed import compute_split_shapes

# for grid conversion
from makani.utils.dataloaders.data_shapes import DataShapes
from makani.utils.dataloaders.batch import make_batch
from makani.utils.grids import GridConverter

# data helpers
from .data_helpers import get_lat_lon_grid, get_date_from_string, get_date_from_timestamp


class DummyLoader(object):
    """Synthetic dataloader for model-perf benchmarking.

    Unlike conventional dataloaders, ``DummyLoader`` pre-allocates its output
    tensors on the device passed via the ``device`` kwarg and returns the same
    tensors on every iteration. This is intentional — for benchmarking model
    throughput, it avoids the per-batch CPU→device transfer cost that would
    otherwise dominate the timing. Pass ``device=torch.device("cpu")`` for
    testing or for a more conventional dataloader-like behavior.

    For multi-GPU runs the caller is responsible for passing the rank-specific
    device (e.g. ``cuda:<local_rank>``); the loader does not infer it.
    """

    def __init__(
        self,
        location: str,
        batch_size: int,
        dt: int,
        dhours: int,
        in_channels: List[int],
        out_channels: List[int],
        img_shape: Optional[Tuple[int, int]] = None,
        max_samples: Optional[int] = None,
        n_samples_per_epoch: Optional[int] = None,
        n_history: Optional[int] = 0,
        n_future: Optional[int] = 0,
        add_zenith: Optional[bool] = False,
        latitudes: Optional[np.array] = None,
        longitudes: Optional[np.array] = None,
        data_grid_type: Optional[str] = "equiangular",
        model_grid_type: Optional[str] = "equiangular",
        return_target: Optional[bool] = True,
        dataset_name: Optional[str] = "fields",
        crop_size: Optional[Tuple[int, int]] = (None, None),
        crop_anchor: Optional[Tuple[int, int]] = (0, 0),
        subsampling_factor: Optional[int] = 1,
        io_grid: Optional[List[int]] = [1, 1, 1],
        io_rank: Optional[List[int]] = [0, 0, 0],
        device: Optional[torch.device] = torch.device("cpu"),
        enable_logging: Optional[bool] = True,
        **kwargs,
    ):

        self.location = location
        self.dt = dt
        self.dhours = dhours
        self.max_samples = max_samples
        self.n_samples_per_epoch = n_samples_per_epoch
        self.batch_size = batch_size
        self.n_history = n_history
        self.n_future = n_future
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_in_channels = len(in_channels)
        self.n_out_channels = len(out_channels)
        self.img_shape = img_shape
        self.device = device
        self.subsampling_factor = subsampling_factor
        self.return_target = return_target
        self.io_grid = io_grid[1:]
        self.io_rank = io_rank[1:]
        if (latitudes is not None) and (longitudes is not None):
            self.lat_lon = (latitudes, longitudes)
        else:
            self.lat_lon = None

        # get cropping:
        self.crop_size = crop_size
        self.crop_anchor = crop_anchor

        self._get_files_stats()

        # get local lat lon (overrides the read_anchor-based slice from _get_files_stats)
        self.lat_lon_local = (
            self.lat_lon[0][self.crop_anchor[0] : self.crop_anchor[0] + self.crop_size[0]],
            self.lat_lon[1][self.crop_anchor[1] : self.crop_anchor[1] + self.crop_size[1]],
        )

        # zenith angle yes or no?
        self.add_zenith = add_zenith
        if self.add_zenith:
            # input and target zenith span their own windows, n_history+1 and n_future+1 steps
            self.zen_inp_dummy = torch.zeros(
                (self.batch_size, self.n_history + 1, 1, self.return_shape[0], self.return_shape[1]),
                dtype=torch.float32,
                device=self.device,
            )
            if self.return_target:
                self.zen_tar_dummy = torch.zeros(
                    (self.batch_size, self.n_future + 1, 1, self.return_shape[0], self.return_shape[1]),
                    dtype=torch.float32,
                    device=self.device,
                )

        # grid types (lat_lon_local may be a list if auto-created; coerce to tensor)
        self.grid_converter = GridConverter(
            data_grid_type,
            model_grid_type,
            torch.deg2rad(torch.as_tensor(self.lat_lon_local[0])).to(torch.float32),
            torch.deg2rad(torch.as_tensor(self.lat_lon_local[1])).to(torch.float32),
        )

        # the geometry the run needs, in the one shape every loader reports it
        self.data_shapes = DataShapes.from_loader(self)

    def _get_files_stats(self):

        if self.img_shape is None:
            self.files_paths = glob.glob(self.location + "/*.h5")

            if not self.files_paths:
                raise RuntimeError(
                    "You have to specify img_shape if you do not provide a data path from which shapes can be deferred."
                )

            self.files_paths.sort()

            n_samples_total = 0
            for fname in self.files_paths:
                with h5py.File(fname, "r") as _f:
                    n_samples_total += _f["fields"].shape[0]
                    self.img_shape = _f["fields"].shape[2:4]

            if self.n_samples_per_epoch is None:
                self.n_samples_per_epoch = n_samples_total

            if self.max_samples is None:
                self.max_samples = n_samples_total

            # the user can regulate the number of samples using either variables
            self.n_samples_per_epoch = min(self.n_samples_per_epoch, self.max_samples)

        else:
            if (self.n_samples_per_epoch is not None) and (self.max_samples is None):
                self.max_samples = self.n_samples_per_epoch
            elif (self.n_samples_per_epoch is None) and (self.max_samples is not None):
                self.n_samples_per_epoch = self.max_samples

        # perform a sanity check here
        if (self.n_samples_per_epoch == 0) or (self.n_samples_per_epoch is None):
            raise RuntimeError("You have noit specified a valid number of samples per epoch.")

        # determine local read size:
        # sanitize the crops first
        if self.crop_size[0] is None:
            self.crop_size_x = self.img_shape[0]
        else:
            self.crop_size_x = self.crop_size[0]
        if self.crop_size[1] is None:
            self.crop_size_y = self.img_shape[1]
        else:
            self.crop_size_y = self.crop_size[1]
        self.crop_size = (self.crop_size_x, self.crop_size_y)

        if self.crop_anchor[0] + self.crop_size[0] > self.img_shape[0]:
            raise ValueError(
                f"crop in dimension 0 (anchor {self.crop_anchor[0]} + shape {self.crop_size[0]}) exceeds image shape {self.img_shape[0]}"
            )
        if self.crop_anchor[1] + self.crop_size[1] > self.img_shape[1]:
            raise ValueError(
                f"crop in dimension 1 (anchor {self.crop_anchor[1]} + shape {self.crop_size[1]}) exceeds image shape {self.img_shape[1]}"
            )

        # for x
        split_shapes_x = compute_split_shapes(self.crop_size[0], self.io_grid[0])
        read_shape_x = split_shapes_x[self.io_rank[0]]
        read_anchor_x = sum(split_shapes_x[: self.io_rank[0]])

        # for y
        split_shapes_y = compute_split_shapes(self.crop_size[1], self.io_grid[1])
        read_shape_y = split_shapes_y[self.io_rank[1]]
        read_anchor_y = sum(split_shapes_y[: self.io_rank[1]])

        # store exposed variables
        self.read_anchor = (read_anchor_x, read_anchor_y)
        self.read_shape = (read_shape_x, read_shape_y)
        self.return_shape = (
            math.ceil(self.read_shape[0] / self.subsampling_factor),
            math.ceil(self.read_shape[1] / self.subsampling_factor),
        )

        self.img_shape_resampled = (
            math.ceil(self.img_shape[0] / self.subsampling_factor),
            math.ceil(self.img_shape[1] / self.subsampling_factor),
        )

        # auto-create lat/lon if the caller didn't supply them
        # (matches the pattern used by SampleSource)
        if self.lat_lon is None:
            latitude, longitude = get_lat_lon_grid(self.img_shape)
            self.lat_lon = (latitude.tolist(), longitude.tolist())

        # lat lon coords
        self.lat_lon_local = (
            self.lat_lon[0][self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0]],
            self.lat_lon[1][self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1]],
        )

        # sharding
        self.n_samples_total = self.n_samples_per_epoch
        self.n_samples_shard = self.n_samples_total // comm.get_size("data")

        # channels
        self.n_in_channels_local = self.n_in_channels
        self.n_out_channels_local = self.n_out_channels

        logging.info(
            f"Number of examples: {self.n_samples_per_epoch}. Image Shape: {self.img_shape[0]} x {self.img_shape[1]} x {self.n_in_channels_local}"
        )
        # the window counts its latest step too, as in the other loaders
        logging.info(
            f"Including {self.dhours * self.dt * (self.n_history + 1)} hours of past history in training at a "
            f"frequency of {self.dhours * self.dt} hours"
        )
        logging.info("WARNING: using dummy data")

        # create tensors for dummy data on the device passed at construction time
        self.inp = torch.zeros(
            (self.batch_size, self.n_history + 1, self.n_in_channels, self.return_shape[0], self.return_shape[1]),
            dtype=torch.float32,
            device=self.device,
        )
        self.tar = torch.zeros(
            (self.batch_size, self.n_future + 1, self.n_out_channels_local, self.return_shape[0], self.return_shape[1]),
            dtype=torch.float32,
            device=self.device,
        )

        # initialize output
        self.inp.uniform_()
        self.tar.uniform_()

        # the values are as meaningless as the payload; they are there because every batch has them
        self.inp_time = torch.zeros((self.batch_size, self.n_history + 1), dtype=torch.float64, device=self.device)
        self.tar_time = torch.ones((self.batch_size, self.n_future + 1), dtype=torch.float64, device=self.device)

        self.in_bias = np.zeros((1, self.n_in_channels, 1, 1)).astype(np.float32)
        self.in_scale = np.ones((1, self.n_in_channels, 1, 1)).astype(np.float32)
        self.out_bias = np.zeros((1, self.n_out_channels_local, 1, 1)).astype(np.float32)
        self.out_scale = np.ones((1, self.n_out_channels_local, 1, 1)).astype(np.float32)

    def get_input_normalization(self):
        return self.in_bias, self.in_scale

    def get_output_normalization(self):
        return self.out_bias, self.out_scale

    def __len__(self):
        return self.n_samples_shard

    def __iter__(self):
        self.sample_idx = 0
        return self

    def __next__(self):
        if self.sample_idx < self.n_samples_shard:
            self.sample_idx += 1

            return make_batch(
                inp=self.inp,
                inp_timestamp=self.inp_time,
                tar=self.tar if self.return_target else None,
                tar_timestamp=self.tar_time if self.return_target else None,
                inp_zenith=self.zen_inp_dummy if self.add_zenith else None,
                tar_zenith=self.zen_tar_dummy if (self.add_zenith and self.return_target) else None,
            )
        else:
            raise StopIteration()


class DummyInferenceDataset(DummyLoader, torch.utils.data.Dataset):
    """Map-style synthetic dataset for the inference pipeline.

    ``DummyLoader`` is an iterator that hands back the same pre-allocated batch forever, which
    suits a training benchmark. The inferencer cannot use it: it wraps its dataset in a
    ``torch.utils.data.DataLoader`` driven by a ``SortedIndexSampler``, so it needs a map-style
    dataset addressable by index, with timestamps that advance -- the rollout schedule is built
    from index arithmetic and the output written by ``RolloutBuffer`` is labelled by time.

    This subclass reuses ``DummyLoader``'s geometry (sharding, cropping, resampling, grid
    conversion) and replaces the iteration protocol with indexed access. Two deliberate
    differences from the parent:

    * Samples are allocated on the **host**, not the device. The inferencer builds its loader
      with ``num_workers`` and ``pin_memory=True``; CUDA tensors cannot cross a worker process
      boundary or be pinned. It also makes the measurement more faithful, since real inference
      pays the H2D copy.
    * The payload is allocated once and returned for every index, as in the parent. The values
      are meaningless either way, and re-randomizing per sample would put host-side RNG work
      into a measurement aimed at the model and the output path.

    Timestamps are synthesized from ``start_date`` at a spacing of ``dhours``, so
    ``get_date_from_timestamp`` and the rollout buffer's time coordinates behave as they would
    on a real dataset.
    """

    def __init__(self, *args, start_date: Optional[str] = "2020-01-01T00:00:00Z", **kwargs):
        # the parent allocates its batch on this device; force the host, and keep the batch
        # dimension minimal since indexed access never uses the parent's buffers
        kwargs["device"] = torch.device("cpu")
        kwargs["batch_size"] = 1

        super().__init__(*args, **kwargs)

        # timestamps as float seconds, matching what MultifilesDataset stores
        start = get_date_from_string(start_date)
        self.datestamps = [start + datetime.timedelta(hours=self.dhours * idx) for idx in range(self.n_samples_total)]
        self.timestamps = np.asarray([stamp.timestamp() for stamp in self.datestamps], dtype=np.float64)
        self.start_date = self.datestamps[0]
        self.end_date = self.datestamps[-1]

        # per-sample payload: the parent's buffers carry a batch dimension the sampler supplies
        sample_shape = (self.n_history + 1, self.n_in_channels, self.return_shape[0], self.return_shape[1])
        self.sample_inp = torch.zeros(sample_shape, dtype=torch.float32).uniform_()

        target_shape = (self.n_future + 1, self.n_out_channels, self.return_shape[0], self.return_shape[1])
        self.sample_tar = torch.zeros(target_shape, dtype=torch.float32).uniform_()

        if self.add_zenith:
            self.sample_zen = torch.zeros(
                (self.n_history + 1, 1, self.return_shape[0], self.return_shape[1]), dtype=torch.float32
            )
            self.sample_zen_tar = torch.zeros(
                (self.n_future + 1, 1, self.return_shape[0], self.return_shape[1]), dtype=torch.float32
            )

    def __len__(self):
        # mirrors MultifilesDataset: the last samples cannot start a full history/future window
        toff = 1 if self.return_target else 0
        return self.n_samples_total - self.dt * (self.n_history + self.n_future + toff)

    def _timestamps_at(self, global_idx, offset_start, offset_end):
        return self.timestamps[global_idx + self.dt * offset_start : global_idx + self.dt * offset_end : self.dt]

    def get_sample_at_index(self, global_idx, return_target=True):
        inp_time = self._timestamps_at(global_idx, 0, self.n_history + 1)
        tar_time = None
        if return_target:
            tar_time = self._timestamps_at(global_idx, self.n_history + 1, self.n_history + self.n_future + 2)

        return make_batch(
            inp=self.sample_inp,
            inp_timestamp=torch.as_tensor(inp_time, dtype=torch.float64),
            tar=self.sample_tar if return_target else None,
            tar_timestamp=torch.as_tensor(tar_time, dtype=torch.float64) if return_target else None,
            inp_zenith=self.sample_zen if self.add_zenith else None,
            tar_zenith=self.sample_zen_tar if (self.add_zenith and return_target) else None,
        )

    def __getitem__(self, global_idx):
        return self.get_sample_at_index(global_idx, return_target=self.return_target)

    def get_index_at_time(self, tstamp):
        if not isinstance(tstamp, datetime.datetime):
            tstamp = get_date_from_timestamp(tstamp)

        if (tstamp < self.start_date) or (tstamp > self.end_date):
            return None

        return bisect_right(self.datestamps, tstamp) - 1

    def get_time_at_index(self, global_idx):
        return self.datestamps[global_idx]

    def get_sample_at_time(self, timestamp):
        global_idx = self.get_index_at_time(timestamp)
        if global_idx is None:
            raise IndexError(f"Time stamp {timestamp} is out of range of the dataset.")

        return self.get_sample_at_index(global_idx, return_target=self.return_target)
