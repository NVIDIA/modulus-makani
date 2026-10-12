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

from packaging import version
import os
import contextlib
import time
import re
import json
import datetime as dt
import random
from typing import List, Optional, Sequence

import numpy as np
import h5py as h5
import zarr

import torch

from makani.utils.YParams import ParamsBase

H5_PATH = "fields"
NUM_CHANNELS = 5
IMG_SIZE_H = 64
IMG_SIZE_W = 128
CHANNEL_NAMES = ["u10m", "t2m", "u500", "z500", "t500"]

# Dataset layout used by init_hdf5_dataset
NUM_SAMPLES_PER_YEAR = 365
TRAIN_YEARS = [2017, 2018]
TEST_YEARS = [2019]
DHOURS = (365 * 24) // NUM_SAMPLES_PER_YEAR  # 24


def set_seed(seed=333):
    """Seed torch, torch.cuda, and numpy for reproducible tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return


def disable_tf32():
    # the api for this was changed lately in pytorch
    if torch.cuda.is_available():
        if version.parse(torch.__version__) >= version.parse("2.9.0"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            torch.backends.cudnn.rnn.fp32_precision = "ieee"
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    return


@contextlib.contextmanager
def local_timezone(name):
    """Run the enclosed block as if the machine were in another timezone."""
    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def get_default_parameters():

    # instantiate parameters
    params = ParamsBase()

    # dataset related
    params.dt = 1
    params.n_history = 0
    params.n_future = 0
    params.normalization = "none"
    params.data_grid_type = "equiangular"
    params.model_grid_type = "equiangular"
    params.sht_grid_type = "legendre-gauss"

    params.resuming = False
    params.amp_mode = "none"
    params.jit_mode = "none"
    params.disable_ddp = False
    params.checkpointing_level = 0
    params.enable_synthetic_data = False
    params.split_data_channels = False

    # dataloader related
    params.in_channels = list(range(NUM_CHANNELS))
    params.out_channels = list(range(NUM_CHANNELS))
    params.channel_names = [CHANNEL_NAMES[i] for i in range(NUM_CHANNELS)]

    # number of channels
    params.N_in_channels = len(params.in_channels)
    params.N_out_channels = len(params.out_channels)

    params.batch_size = 1
    params.valid_autoreg_steps = 0
    params.num_data_workers = 1
    params.multifiles = True
    params.io_grid = [1, 1, 1]
    params.io_rank = [0, 0, 0]

    # extra channels
    params.add_grid = False
    params.add_zenith = False
    params.invariants = None

    # logging stuff, needed for higher level tests
    params.log_to_screen = False
    params.log_to_wandb = False

    # preprocessor fields — required by Preprocessor2D; tests that need different
    # values override these in their own setUp after calling get_default_parameters()
    params.img_shape_x = IMG_SIZE_H
    params.img_shape_y = IMG_SIZE_W
    params.img_shape_x_resampled = IMG_SIZE_H
    params.img_shape_y_resampled = IMG_SIZE_W
    params.history_normalization_mode = "none"
    params.history_normalization_decay = 0.5
    params.normalize_residual = False

    return params


def init_dataset_metadata(
    path: str,
    dataset_name: str = "testing",
    h5_path: str = H5_PATH,
    dhours: int = DHOURS,
    channel_names: Optional[List[str]] = None,
    lat: Optional[List[float]] = None,
    lon: Optional[List[float]] = None,
    grid_type: str = "equiangular",
    attrs: Optional[dict] = None,
    analysis_epoch_start_dates: Optional[List[str]] = None,
) -> str:
    """Write a dataset metadata JSON to <path>/data.json and return the file path.

    This is the canonical way to create a metadata fixture for tests.  It is
    also called internally by ``init_hdf5_dataset`` so that both produce identical
    JSON layouts (including the ``attrs`` key required by
    ``parse_dataset_metadata``).
    """
    if channel_names is None:
        channel_names = [f"chan_{i}" for i in range(NUM_CHANNELS)]
    if lat is None:
        lat = np.linspace(90, -90, IMG_SIZE_H, endpoint=True).tolist()
    if lon is None:
        lon = np.linspace(0, 360, IMG_SIZE_W, endpoint=False).tolist()
    if attrs is None:
        attrs = {"description": "A synthetic test dataset."}

    metadata = dict(
        dataset_name=dataset_name,
        h5_path=h5_path,
        dims=["time", "channel", "lat", "lon"],
        dhours=dhours,
        attrs=attrs,
        coords=dict(
            grid_type=grid_type,
            lat=lat,
            lon=lon,
            channel=channel_names,
        ),
    )
    if analysis_epoch_start_dates is not None:
        metadata["analysis_epoch_start_dates"] = analysis_epoch_start_dates

    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "data.json")
    with open(file_path, "w") as f:
        json.dump(metadata, f)
    return file_path


def init_hdf5_dataset(
    path: str,
    num_samples_per_year: Optional[int] = 365,
    num_channels: Optional[int] = NUM_CHANNELS,
    img_size_h: Optional[int] = IMG_SIZE_H,
    img_size_w: Optional[int] = IMG_SIZE_W,
    nan_fraction: Optional[float] = 0.0,
    annotate: Optional[bool] = True,
    create_concat: Optional[bool] = False,
):

    test_path = os.path.join(path, "test")
    os.makedirs(test_path)

    train_path = os.path.join(path, "train")
    os.makedirs(train_path)

    stats_path = os.path.join(path, "stats")
    os.makedirs(stats_path)

    metadata_path = os.path.join(path, "metadata")
    os.makedirs(metadata_path)

    # rng:
    rng = np.random.default_rng(seed=333)

    # create lon lat grid
    longitude = np.linspace(0, 360, img_size_w, endpoint=False)
    latitude = np.linspace(90, -90, img_size_h, endpoint=True)

    # channels names
    channel_names = [f"chan_{idx}" for idx in range(num_channels)]
    chanlen = max([len(x) for x in channel_names])

    # set dhours:
    hours_per_year = 365 * 24
    dhours = hours_per_year // num_samples_per_year

    # storage for the concatenated training file (populated when create_concat=True)
    concat_data_list = []
    concat_ts_list = []

    # create training files
    num_train = 0
    for y in [2017, 2018]:
        data_path = os.path.join(train_path, f"{y}.h5")
        with h5.File(data_path, "w") as hf:
            hf.create_dataset(H5_PATH, shape=(num_samples_per_year, num_channels, img_size_h, img_size_w), dtype="f4")

            num_dof = num_samples_per_year * num_channels * img_size_h * img_size_w
            data = rng.random((num_dof,), dtype=np.float32)

            # add NaNs
            if nan_fraction > 0.0:
                indices = np.arange(num_samples_per_year * num_channels * img_size_h * img_size_w, dtype=np.int32)
                nan_count = int(nan_fraction * num_dof)
                rng.shuffle(indices)
                nan_indices = indices[0:nan_count]
                data[nan_indices] = np.nan

            # reshape to correct shape
            data = data.reshape(num_samples_per_year, num_channels, img_size_h, img_size_w)

            # store in file
            hf[H5_PATH][...] = data[...]

            # compute timestamps for this year (used for both annotation and concat)
            year_start = dt.datetime(year=y, month=1, day=1, hour=0, tzinfo=dt.timezone.utc).timestamp()
            timestamps = year_start + np.arange(0, hours_per_year * 3600, dhours * 3600, dtype=np.float64)

            # annotations
            if annotate:
                hf.create_dataset("timestamp", data=timestamps)
                hf.create_dataset("channel", len(channel_names), dtype=h5.string_dtype(length=chanlen))
                hf["channel"][...] = channel_names
                hf.create_dataset("lat", data=latitude)
                hf.create_dataset("lon", data=longitude)
                # create scales
                hf["timestamp"].make_scale("timestamp")
                hf["channel"].make_scale("channel")
                hf["lat"].make_scale("lat")
                hf["lon"].make_scale("lon")
                # attach scales
                hf[H5_PATH].dims[0].attach_scale(hf["timestamp"])
                hf[H5_PATH].dims[1].attach_scale(hf["channel"])
                hf[H5_PATH].dims[2].attach_scale(hf["lat"])
                hf[H5_PATH].dims[3].attach_scale(hf["lon"])

            # collect for the concatenated file
            if create_concat:
                concat_data_list.append(data.copy())
                concat_ts_list.append(timestamps)

        num_train += num_samples_per_year

    # create validation files
    num_test = 0
    for y in [2019]:
        data_path = os.path.join(test_path, f"{y}.h5")
        with h5.File(data_path, "w") as hf:
            hf.create_dataset(H5_PATH, shape=(num_samples_per_year, num_channels, img_size_h, img_size_w), dtype="f4")
            hf[H5_PATH][...] = rng.random(
                (num_samples_per_year, num_channels, img_size_h, img_size_w), dtype=np.float32
            )

            # annotations
            if annotate:
                # create datasets
                year_start = dt.datetime(year=y, month=1, day=1, hour=0, tzinfo=dt.timezone.utc).timestamp()
                timestamps = year_start + np.arange(0, hours_per_year * 3600, dhours * 3600, dtype=np.float64)
                hf.create_dataset("timestamp", data=timestamps)
                hf.create_dataset("channel", len(channel_names), dtype=h5.string_dtype(length=chanlen))
                hf["channel"][...] = channel_names
                hf.create_dataset("lat", data=latitude)
                hf.create_dataset("lon", data=longitude)
                # create scales
                hf["timestamp"].make_scale("timestamp")
                hf["channel"].make_scale("channel")
                hf["lat"].make_scale("lat")
                hf["lon"].make_scale("lon")
                # attach scales
                hf[H5_PATH].dims[0].attach_scale(hf["timestamp"])
                hf[H5_PATH].dims[1].attach_scale(hf["channel"])
                hf[H5_PATH].dims[2].attach_scale(hf["lat"])
                hf[H5_PATH].dims[3].attach_scale(hf["lon"])

        num_test += num_samples_per_year

    # create stats files
    np.save(os.path.join(stats_path, "mins.npy"), np.zeros((1, num_channels, 1, 1), dtype=np.float64))

    np.save(os.path.join(stats_path, "maxs.npy"), np.ones((1, num_channels, 1, 1), dtype=np.float64))

    np.save(
        os.path.join(stats_path, "time_means.npy"),
        np.zeros((1, num_channels, img_size_h, img_size_w), dtype=np.float64),
    )

    np.save(os.path.join(stats_path, "global_means.npy"), np.zeros((1, num_channels, 1, 1), dtype=np.float64))

    np.save(os.path.join(stats_path, "global_stds.npy"), np.ones((1, num_channels, 1, 1), dtype=np.float64))

    np.save(os.path.join(stats_path, "time_diff_means.npy"), np.zeros((1, num_channels, 1, 1), dtype=np.float64))

    np.save(os.path.join(stats_path, "time_diff_stds.npy"), np.ones((1, num_channels, 1, 1), dtype=np.float64))

    # create concatenated training file (all years in one HDF5)
    if create_concat:
        concat_train_path = os.path.join(path, "train_concat.h5")
        concat_data = np.concatenate(concat_data_list, axis=0)
        concat_ts = np.concatenate(concat_ts_list)
        with h5.File(concat_train_path, "w") as hf:
            hf.create_dataset(H5_PATH, data=concat_data)
            hf.create_dataset("timestamp", data=concat_ts)
            hf.create_dataset("channel", len(channel_names), dtype=h5.string_dtype(length=chanlen))
            hf["channel"][...] = channel_names
            hf.create_dataset("lat", data=latitude)
            hf.create_dataset("lon", data=longitude)
            hf["timestamp"].make_scale("timestamp")
            hf["channel"].make_scale("channel")
            hf["lat"].make_scale("lat")
            hf["lon"].make_scale("lon")
            hf[H5_PATH].dims[0].attach_scale(hf["timestamp"])
            hf[H5_PATH].dims[1].attach_scale(hf["channel"])
            hf[H5_PATH].dims[2].attach_scale(hf["lat"])
            hf[H5_PATH].dims[3].attach_scale(hf["lon"])
    else:
        concat_train_path = None

    # create metadata file:
    init_dataset_metadata(
        path=metadata_path,
        dataset_name="testing",
        h5_path=H5_PATH,
        dhours=dhours,
        channel_names=channel_names,
        lat=latitude.tolist(),
        lon=longitude.tolist(),
    )

    return train_path, num_train, test_path, num_test, stats_path, metadata_path, concat_train_path


def init_zarr_dataset(
    path: str,
    num_samples_per_year: Optional[int] = 365,
    num_channels: Optional[int] = NUM_CHANNELS,
    img_size_h: Optional[int] = IMG_SIZE_H,
    img_size_w: Optional[int] = IMG_SIZE_W,
    nan_fraction: Optional[float] = 0.0,
    annotate: Optional[bool] = True,
    consolidate: Optional[bool] = True,
):
    """Create a zarr-backed dummy dataset with the same layout as init_hdf5_dataset.

    Files are written as ``<year>.zarr`` directories (matching the glob pattern
    the dataloader uses).  When ``annotate=True`` a ``"time"`` coordinate array
    is written at the group level so that ``_zarr_read_timestamps`` picks it up
    via the primary lookup path.  ``consolidate=True`` calls
    ``zarr.consolidate_metadata`` so the ``open_consolidated`` fast path is
    exercised in the dataloader.

    Returns the same tuple as ``init_hdf5_dataset`` (with ``concat_train_path=None``
    since the concat path is HDF5-only).
    """
    test_path = os.path.join(path, "test_zarr")
    os.makedirs(test_path)

    train_path = os.path.join(path, "train_zarr")
    os.makedirs(train_path)

    stats_path = os.path.join(path, "stats_zarr")
    os.makedirs(stats_path)

    metadata_path = os.path.join(path, "metadata_zarr")
    os.makedirs(metadata_path)

    rng = np.random.default_rng(seed=333)

    longitude = np.linspace(0, 360, img_size_w, endpoint=False)
    latitude = np.linspace(90, -90, img_size_h, endpoint=True)
    channel_names = [f"chan_{idx}" for idx in range(num_channels)]

    hours_per_year = 365 * 24
    dhours = hours_per_year // num_samples_per_year

    def _write_year(store_path, year, data):
        g = zarr.open_group(store_path, mode="w")
        g.create_array(H5_PATH, data=data, chunks=(1, num_channels, img_size_h, img_size_w))
        if annotate:
            year_start = dt.datetime(year=year, month=1, day=1, hour=0, tzinfo=dt.timezone.utc).timestamp()
            timestamps = year_start + np.arange(0, hours_per_year * 3600, dhours * 3600, dtype=np.float64)
            g.create_array("time", data=timestamps)
            g.create_array("channel", data=np.array(channel_names))
            g.create_array("lat", data=latitude)
            g.create_array("lon", data=longitude)
        if consolidate:
            zarr.consolidate_metadata(store_path)

    num_train = 0
    for y in [2017, 2018]:
        num_dof = num_samples_per_year * num_channels * img_size_h * img_size_w
        data = rng.random((num_dof,), dtype=np.float32).reshape(
            num_samples_per_year, num_channels, img_size_h, img_size_w
        )
        if nan_fraction > 0.0:
            n_nan = int(nan_fraction * num_dof)
            idx = rng.choice(num_dof, size=n_nan, replace=False)
            data.flat[idx] = np.nan
        _write_year(os.path.join(train_path, f"{y}.zarr"), y, data)
        num_train += num_samples_per_year

    num_test = 0
    for y in [2019]:
        data = rng.random((num_samples_per_year, num_channels, img_size_h, img_size_w), dtype=np.float32)
        _write_year(os.path.join(test_path, f"{y}.zarr"), y, data)
        num_test += num_samples_per_year

    # stats files are format-agnostic — reuse the same .npy layout
    np.save(os.path.join(stats_path, "mins.npy"), np.zeros((1, num_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "maxs.npy"), np.ones((1, num_channels, 1, 1), dtype=np.float64))
    np.save(
        os.path.join(stats_path, "time_means.npy"),
        np.zeros((1, num_channels, img_size_h, img_size_w), dtype=np.float64),
    )
    np.save(os.path.join(stats_path, "global_means.npy"), np.zeros((1, num_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "global_stds.npy"), np.ones((1, num_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "time_diff_means.npy"), np.zeros((1, num_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "time_diff_stds.npy"), np.ones((1, num_channels, 1, 1), dtype=np.float64))

    init_dataset_metadata(
        path=metadata_path,
        dataset_name="testing_zarr",
        h5_path=H5_PATH,
        dhours=dhours,
        channel_names=channel_names,
        lat=latitude.tolist(),
        lon=longitude.tolist(),
    )

    return train_path, num_train, test_path, num_test, stats_path, metadata_path, None


def init_wb2_zarr_dataset(
    path: str,
    channel_names: Optional[List[str]] = None,
    num_samples_per_year: Optional[int] = 365,
    img_size_h: Optional[int] = IMG_SIZE_H,
    img_size_w: Optional[int] = IMG_SIZE_W,
    annotate: Optional[bool] = True,
    consolidate: Optional[bool] = True,
):
    """Create a WB2-layout zarr dataset for testing.

    Mirrors the structure of real WeatherBench2 zarr stores:
      - one zarr array per variable (no single stacked ``fields`` array)
      - surface variables: shape ``(time, latitude, longitude)``
      - atmospheric variables: shape ``(time, level, latitude, longitude)``
      - ``time`` coordinate stored as ``datetime64[ns]``
      - ``latitude``, ``longitude``, ``level`` coordinates at group level

    ``channel_names`` defaults to ``CHANNEL_NAMES`` which contains both surface
    (u10m, t2m) and atmospheric (u500, z500, t500) variables so both code paths
    are exercised.

    Returns the same tuple as ``init_hdf5_dataset``/``init_zarr_dataset``
    (with ``concat_path=None``).
    """
    from makani.utils.dataloaders.wb2_helpers import atmospheric_wb2_name, surface_wb2_name

    if channel_names is None:
        channel_names = list(CHANNEL_NAMES)

    test_path = os.path.join(path, "test_zarr_wb2")
    os.makedirs(test_path)
    train_path = os.path.join(path, "train_zarr_wb2")
    os.makedirs(train_path)
    stats_path = os.path.join(path, "stats_zarr_wb2")
    os.makedirs(stats_path)
    metadata_path = os.path.join(path, "metadata_zarr_wb2")
    os.makedirs(metadata_path)

    rng = np.random.default_rng(seed=333)
    longitude = np.linspace(0, 360, img_size_w, endpoint=False).astype(np.float32)
    latitude = np.linspace(90, -90, img_size_h, endpoint=True).astype(np.float32)

    hours_per_year = 365 * 24
    dhours = hours_per_year // num_samples_per_year

    # parse channel_names to determine which WB2 variables are needed and at what levels
    atm_vars = {}  # wb2_name -> set of required pressure levels
    surf_vars = set()

    for ch_name in channel_names:
        m = re.search(r"[0-9]{1,4}$", ch_name)
        if m is not None and ch_name != "d2":
            pressure = int(m.group())
            prefix = ch_name[: m.start()]
            wb2_name = atmospheric_wb2_name(prefix)
            atm_vars.setdefault(wb2_name, set()).add(pressure)
        else:
            surf_vars.add(surface_wb2_name(ch_name))

    # unified sorted level array covering all atmospheric variables in the fixture
    all_levels = sorted(set().union(*atm_vars.values())) if atm_vars else []
    levels = np.array(all_levels, dtype=np.int64)
    n_levels = len(levels)

    def _write_year(store_path, year):
        g = zarr.open_group(store_path, mode="w")

        if annotate:
            year_start = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
            times = np.array(
                [
                    np.datetime64(int((year_start + dt.timedelta(hours=i * dhours)).timestamp() * 1e9), "ns")
                    for i in range(num_samples_per_year)
                ]
            )
            g.create_array("time", data=times)
            g.create_array("latitude", data=latitude)
            g.create_array("longitude", data=longitude)
            if n_levels > 0:
                g.create_array("level", data=levels)

        for wb2_name in surf_vars:
            data = rng.random((num_samples_per_year, img_size_h, img_size_w), dtype=np.float32)
            g.create_array(wb2_name, data=data, chunks=(1, img_size_h, img_size_w))

        for wb2_name in atm_vars:
            data = rng.random((num_samples_per_year, n_levels, img_size_h, img_size_w), dtype=np.float32)
            g.create_array(wb2_name, data=data, chunks=(1, n_levels, img_size_h, img_size_w))

        if consolidate:
            zarr.consolidate_metadata(store_path)

    num_train = 0
    for y in [2017, 2018]:
        _write_year(os.path.join(train_path, f"{y}.zarr"), y)
        num_train += num_samples_per_year

    num_test = 0
    for y in [2019]:
        _write_year(os.path.join(test_path, f"{y}.zarr"), y)
        num_test += num_samples_per_year

    n_channels = len(channel_names)
    np.save(os.path.join(stats_path, "mins.npy"), np.zeros((1, n_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "maxs.npy"), np.ones((1, n_channels, 1, 1), dtype=np.float64))
    np.save(
        os.path.join(stats_path, "time_means.npy"), np.zeros((1, n_channels, img_size_h, img_size_w), dtype=np.float64)
    )
    np.save(os.path.join(stats_path, "global_means.npy"), np.zeros((1, n_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "global_stds.npy"), np.ones((1, n_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "time_diff_means.npy"), np.zeros((1, n_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "time_diff_stds.npy"), np.ones((1, n_channels, 1, 1), dtype=np.float64))

    init_dataset_metadata(
        path=metadata_path,
        dataset_name="testing_wb2",
        h5_path=H5_PATH,
        dhours=dhours,
        channel_names=channel_names,
        lat=latitude.tolist(),
        lon=longitude.tolist(),
    )

    return train_path, num_train, test_path, num_test, stats_path, metadata_path, None


def _is_inexact_tensor(tensor):
    # floating point or complex
    return tensor.is_floating_point() or tensor.is_complex()


def compare_tensors(msg, tensor1, tensor2, atol=1e-8, rtol=1e-5, verbose=False, shape_check=False):
    # shape_check: require identical shapes, since torch.allclose broadcasts and
    # would otherwise let a tensor of the wrong shape pass on matching values

    # some None checks
    if tensor1 is None and tensor2 is None:
        allclose = True
    elif tensor1 is None and tensor2 is not None:
        allclose = False
        if verbose:
            print("tensor1 is None and tensor2 is not None")
    elif tensor1 is not None and tensor2 is None:
        allclose = False
        if verbose:
            print("tensor1 is not None and tensor2 is None")
    elif shape_check and tensor1.shape != tensor2.shape:
        allclose = False
        if verbose:
            print(f"Shape mismatch on {msg}: {tuple(tensor1.shape)} vs {tuple(tensor2.shape)}")
    elif not (_is_inexact_tensor(tensor1) or _is_inexact_tensor(tensor2)):
        # integers of any width (or bools): exact, and no difference statistics, since
        # bools cannot be subtracted, integer tensors have no mean, and unsigned
        # differences wrap around
        tensor1, tensor2 = torch.broadcast_tensors(tensor1, tensor2)
        mismatch = (tensor1 != tensor2).flatten()
        allclose = not bool(mismatch.any())
        if not allclose and verbose:
            first = int(mismatch.nonzero()[0])
            print(
                f"{int(mismatch.sum())} mismatching elements on {msg}, first at flat index {first}: {tensor1.flatten()[first]} and {tensor2.flatten()[first]}"
            )
    else:
        diff = torch.abs(tensor1 - tensor2)
        abs_diff = torch.mean(diff, dim=0)
        rel_diff = torch.mean(diff / torch.clamp(torch.abs(tensor2), min=1e-6), dim=0)
        allclose = torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)
        if not allclose and verbose:
            print(
                f"Absolute difference on {msg}: min = {abs_diff.min()}, mean = {abs_diff.mean()}, max = {abs_diff.max()}"
            )
            print(
                f"Relative difference on {msg}: min = {rel_diff.min()}, mean = {rel_diff.mean()}, max = {rel_diff.max()}"
            )
            print(
                f"Element values with max difference on {msg}: {tensor1.flatten()[diff.argmax()]} and {tensor2.flatten()[diff.argmax()]}"
            )
            # find violating entry
            worst_diff = torch.argmax(diff - (atol + rtol * torch.abs(tensor2)))
            diff_bad = diff.flatten()[worst_diff].item()
            tensor2_abs_bad = torch.abs(tensor2).flatten()[worst_diff].item()
            print(
                f"Worst allclose condition violation: {diff_bad} <= {atol} + {rtol} * {tensor2_abs_bad} = {atol + rtol * tensor2_abs_bad}"
            )

    return allclose


def _is_inexact(array):
    # floating point or complex
    return np.issubdtype(np.asarray(array).dtype, np.inexact)


def compare_arrays(msg, array1, array2, atol=1e-8, rtol=1e-5, verbose=False, shape_check=False):
    # shape_check: require identical shapes, since np.allclose broadcasts and
    # would otherwise let an array of the wrong shape pass on matching values
    # some None checks
    if array1 is None and array2 is None:
        allclose = True
    elif array1 is None and array2 is not None:
        allclose = False
        if verbose:
            print("array1 is None and array2 is not None")
    elif array1 is not None and array2 is None:
        allclose = False
        if verbose:
            print("array1 is not None and array2 is None")
    elif shape_check and np.shape(array1) != np.shape(array2):
        allclose = False
        if verbose:
            print(f"Shape mismatch on {msg}: {np.shape(array1)} vs {np.shape(array2)}")
    elif not (_is_inexact(array1) or _is_inexact(array2)):
        # integers of any width (or bools): exact, and no difference statistics, since
        # bools cannot be subtracted and unsigned differences wrap around
        array1, array2 = np.broadcast_arrays(array1, array2)
        mismatch = (array1 != array2).flatten()
        allclose = not mismatch.any()
        if not allclose and verbose:
            first = int(np.flatnonzero(mismatch)[0])
            print(
                f"{int(mismatch.sum())} mismatching elements on {msg}, first at flat index {first}: {array1.flatten()[first]} and {array2.flatten()[first]}"
            )
    else:
        # some sanitization
        if array1.ndim == 0:
            array1 = array1.reshape(1)
        if array2.ndim == 0:
            array2 = array2.reshape(1)
        # compute error
        diff = np.abs(array1 - array2)
        abs_diff = np.mean(diff, axis=0)
        rel_diff = np.mean(diff / np.clip(np.abs(array2), a_min=1e-6, a_max=None), axis=0)
        allclose = np.allclose(array1, array2, atol=atol, rtol=rtol)
        if not allclose and verbose:
            print(
                f"Absolute difference on {msg}: min = {abs_diff.min()}, mean = {abs_diff.mean()}, max = {abs_diff.max()}"
            )
            print(
                f"Relative difference on {msg}: min = {rel_diff.min()}, mean = {rel_diff.mean()}, max = {rel_diff.max()}"
            )
            print(
                f"Element values with max difference on {msg}: {array1.flatten()[diff.argmax()]} and {array2.flatten()[diff.argmax()]}"
            )
            # find violating entry
            worst_diff = np.argmax(diff - (atol + rtol * np.abs(array2)))
            diff_bad = diff.flatten()[worst_diff].item()
            array2_abs_bad = np.abs(array2).flatten()[worst_diff].item()
            print(
                f"Worst allclose condition violation: {diff_bad} <= {atol} + {rtol} * {array2_abs_bad} = {atol + rtol * array2_abs_bad}"
            )

    return allclose


# ---------------------------------------------------------------------------
# ICON fixture
#
# Scaled down from the EXCLAIM R02B10 DYAMOND output, keeping the properties the
# backend has to cope with and shrinking only the ones it does not care about:
#
#   * one variable per file, so a sample spans several of them
#   * cadences that differ per variable -- three hourly on pressure levels,
#     hourly at the surface -- so the sample axis is an intersection
#   * file spans that differ -- daily for the level variables, one file covering
#     the whole period for the surface one -- so unit boundaries do not line up
#   * two variables called ``u``, one on pressure levels and one on altitude
#     levels, which is what forces resolution by level coordinate
#   * the cell axis chunked into 21 pieces, as the real files are
#
# What is smaller: 320 cells rather than 83,886,080 (both are 20 * 4 * 4**b, the
# icosahedral count, here with b=1 instead of b=10), four pressure levels rather
# than 37, and two days rather than a simulation.
# ---------------------------------------------------------------------------

ICON_N_CELLS = 320
ICON_PLEV_HPA = (1000, 850, 500, 250)
ICON_UUID = "e6f7602e-2027-11ef-ab56-6d1c8c7a4d8d"
ICON_EPOCH = "minutes since 2020-1-1 00:00:00"
ICON_FILL = np.float32(-9.99e-08)
ICON_CHANNEL_NAMES = ["u500", "u850", "t500", "t850", "t2m"]
#: cells written as _FillValue, which a reader has to turn into NaN
ICON_FILL_CELLS = (7,)


def _icosahedral_ordering(lon_rad, lat_rad):
    """Order cells the way ICON does: by which root region they fall in.

    ICON's cell index follows the icosahedron, so cells that are close on the
    sphere are close in the file -- coarsely, and with boundaries at the vertex
    latitudes of +-arctan(1/2) = +-26.57 degrees. Reproducing that is what makes
    the run coalescing in the backend meaningful to test: a lat/lon block should
    land in a handful of long index ranges rather than scattered singletons.
    """
    # the twelve icosahedron vertices, ordered as a sweep from one pole to the
    # other rather than by construction: regions next to each other in the index
    # then lie next to each other on the sphere, which is the property being
    # imitated. Listing the poles first would make cell 0 and cell n/12
    # antipodal, and destroy exactly the locality this exists to provide.
    ring = np.arctan(0.5)
    vertex_lat = [np.pi / 2] + [ring] * 5 + [-ring] * 5 + [-np.pi / 2]
    vertex_lon = (
        [0.0]
        + [index * 2 * np.pi / 5 for index in range(5)]
        + [(index + 0.5) * 2 * np.pi / 5 for index in range(5)]
        + [0.0]
    )

    vertex_lat = np.asarray(vertex_lat)
    vertex_lon = np.asarray(vertex_lon)

    # nearest vertex by great circle distance
    cosine = np.sin(lat_rad[:, None]) * np.sin(vertex_lat[None, :]) + np.cos(lat_rad[:, None]) * np.cos(
        vertex_lat[None, :]
    ) * np.cos(lon_rad[:, None] - vertex_lon[None, :])
    region = np.argmax(cosine, axis=1)

    return np.lexsort((lon_rad, lat_rad, region))


def icon_grid_coordinates(n_cells: int = ICON_N_CELLS):
    """Cell centres for the fixture, in radians, ordered like ICON's."""
    # a Fibonacci sphere: near uniform, and deterministic
    indices = np.arange(n_cells, dtype=np.float64)
    lat = np.arcsin(1.0 - 2.0 * (indices + 0.5) / n_cells)
    lon = np.mod(indices * np.pi * (3.0 - np.sqrt(5.0)), 2.0 * np.pi) - np.pi

    order = _icosahedral_ordering(lon, lat)
    return lon[order], lat[order]


def icon_expected_value(variable: str, time_index: int, level_index: int, cell: np.ndarray):
    """The value the fixture stores, so a test can assert an exact read."""
    base = float(sum(ord(character) for character in variable))
    return np.float32(base + 1000.0 * time_index + 10.0 * level_index) + np.asarray(cell, dtype=np.float32) / 1000.0


def icon_expected_field(variable: str, time_index: int, level_index: int, n_cells: int = ICON_N_CELLS):
    """The whole field a reader should return, fills included as NaN."""
    field = icon_expected_value(variable, time_index, level_index, np.arange(n_cells))
    field[list(ICON_FILL_CELLS)] = np.nan
    return field


def _write_icon_file(path, variable, units, times, level_name, level_values, level_units, n_cells, fill_cells=()):
    """One ICON style netCDF4 file: a single variable of (time, level, ncells)."""
    n_chunks = 21  # as in the real files, so the cell axis is chunked the same way
    chunk = int(np.ceil(n_cells / n_chunks))

    with h5.File(path, "w") as handle:
        handle.attrs["Conventions"] = np.bytes_("CF-1.6")
        handle.attrs["uuidOfHGrid"] = np.bytes_(ICON_UUID)
        handle.attrs["grid_file_uri"] = np.bytes_("http://example.invalid/icon_grid_test.nc")

        time_dset = handle.create_dataset("time", data=np.asarray(times, dtype=np.float64))
        time_dset.attrs["units"] = np.bytes_(ICON_EPOCH)

        level_dset = handle.create_dataset(level_name, data=np.asarray(level_values, dtype=np.float64))
        level_dset.attrs["units"] = np.bytes_(level_units)

        cells = np.arange(n_cells)
        data = np.empty((len(times), len(level_values), n_cells), dtype=np.float32)
        for time_index in range(len(times)):
            for level_index in range(len(level_values)):
                data[time_index, level_index] = icon_expected_value(variable, time_index, level_index, cells)
        for cell in fill_cells:
            data[:, :, cell] = ICON_FILL

        field = handle.create_dataset(variable, data=data, chunks=(1, 1, chunk))
        field.attrs["units"] = np.bytes_(units)
        field.attrs["_FillValue"] = ICON_FILL
        field.attrs["missing_value"] = ICON_FILL
        field.attrs["CDI_grid_type"] = np.bytes_("unstructured")
        field.attrs["number_of_grid_in_reference"] = np.int32(1)


def init_icon_dataset(
    path: str,
    n_cells: Optional[int] = ICON_N_CELLS,
    n_days: Optional[int] = 2,
    fill_cells: Optional[Sequence[int]] = ICON_FILL_CELLS,
):
    """Create a scaled down ICON dataset and its grid file.

    Returns ``(data_path, grid_path, n_samples, channel_names)``, where
    ``n_samples`` counts the times every required variable has -- the three
    hourly axis, since that is the coarsest.
    """
    data_path = os.path.join(path, "icon")
    grid_path = os.path.join(path, "icon_grid_test.nc")
    os.makedirs(data_path, exist_ok=True)

    lon_rad, lat_rad = icon_grid_coordinates(n_cells)
    with h5.File(grid_path, "w") as handle:
        handle.attrs["uuidOfHGrid"] = np.bytes_(ICON_UUID)
        handle.attrs["grid_root"] = np.int32(2)
        handle.attrs["grid_level"] = np.int32(1)
        for name, values in (("clon", lon_rad), ("clat", lat_rad)):
            dset = handle.create_dataset(name, data=values)
            dset.attrs["units"] = np.bytes_("radian")

    day_zero = dt.datetime(2021, 6, 1, tzinfo=dt.timezone.utc)
    epoch = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    minutes = lambda when: (when - epoch).total_seconds() / 60.0  # noqa: E731

    pressures = np.asarray(ICON_PLEV_HPA, dtype=np.float64) * 100.0  # the files store Pa

    for day in range(n_days):
        start = day_zero + dt.timedelta(days=day)
        stamp = start.strftime("%Y%m%dT%H%M%SZ")

        # three hourly on pressure levels, one file per day
        three_hourly = [minutes(start + dt.timedelta(hours=3 * step)) for step in range(8)]
        for variable, units in (("temp", "K"), ("u", "m s-1")):
            _write_icon_file(
                os.path.join(data_path, f"{variable}_dyamond_atm_1_3_{stamp}.nc"),
                variable,
                units,
                three_hourly,
                "plev",
                pressures,
                "Pa",
                n_cells,
                fill_cells,
            )

        # the same name on altitude levels, hourly: what forces resolution by
        # level coordinate rather than by name
        hourly = [minutes(start + dt.timedelta(hours=step)) for step in range(24)]
        _write_icon_file(
            os.path.join(data_path, f"u_dyamond_atm_4_{stamp}.nc"),
            "u",
            "m s-1",
            hourly,
            "alt",
            np.asarray([10.0, 100.0]),
            "m",
            n_cells,
            fill_cells,
        )

    # hourly at the surface, and one file for the whole period rather than one
    # per day, so its unit boundaries do not line up with the others
    all_hours = [minutes(day_zero + dt.timedelta(hours=step)) for step in range(24 * n_days)]
    _write_icon_file(
        os.path.join(data_path, "t_2m_dyamond_atm_3_202106.nc"),
        "t_2m",
        "K",
        all_hours,
        "height_2",
        np.asarray([2.0]),
        "m",
        n_cells,
        fill_cells,
    )

    return data_path, grid_path, 8 * n_days, list(ICON_CHANNEL_NAMES)
