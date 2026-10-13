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

import importlib.util
import os
import sys
import glob
import copy
import math
import tempfile
import datetime as dt
from typing import Optional
from parameterized import parameterized, parameterized_class

import unittest
from unittest import mock
import torch
import numpy as np
import h5py as h5
import zarr

from makani.utils.dataloader import get_dataloader
from makani.utils.dataloaders.batch import (
    INPUT,
    TARGET,
    INPUT_DYNAMIC_FEATURES,
    TARGET_DYNAMIC_FEATURES,
    INPUT_TIMESTAMP,
    TARGET_TIMESTAMP,
    ZENITH,
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import (
    disable_tf32,
    get_default_parameters,
    init_hdf5_dataset,
    init_zarr_dataset,
    init_wb2_zarr_dataset,
    compare_tensors,
    compare_arrays,
)
from .testutils import H5_PATH, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W, CHANNEL_NAMES

_TRAINING_KEYS = {INPUT, TARGET, INPUT_DYNAMIC_FEATURES, TARGET_DYNAMIC_FEATURES, INPUT_TIMESTAMP, TARGET_TIMESTAMP}
_INFERENCE_KEYS = {INPUT, INPUT_DYNAMIC_FEATURES, INPUT_TIMESTAMP}

_multifiles_params = [True]
_have_dali = importlib.util.find_spec("nvidia.dali") is not None

if _have_dali:
    _multifiles_params.append(False)

_dali_devices = [("cpu",)]
if torch.cuda.is_available():
    _dali_devices.append(("gpu",))


def get_sample(path: str, idx):
    files = sorted(glob.glob(os.path.join(path, "*.h5")))
    h5file = h5.File(files[0], "r")
    num_samples_per_file = h5file[H5_PATH].shape[0]
    h5file.close()
    file_id = idx // num_samples_per_file
    file_index = idx % num_samples_per_file

    with h5.File(files[file_id], "r") as f:
        data = f[H5_PATH][file_index, ...]

    return data


def get_zarr_sample(path: str, idx):
    files = sorted(glob.glob(os.path.join(path, "*.zarr")))
    first = zarr.open_group(files[0], mode="r")
    num_samples_per_file = first[H5_PATH].shape[0]
    file_id = idx // num_samples_per_file
    file_index = idx % num_samples_per_file
    g = zarr.open_group(files[file_id], mode="r")
    return np.array(g[H5_PATH][file_index, ...])


def get_wb2_zarr_sample(path: str, idx: int, channel_names=None):
    """Read one sample from a WB2-layout zarr store and return it as (C, H, W)."""
    from makani.utils.dataloaders.wb2_helpers import build_wb2_channel_map

    if channel_names is None:
        channel_names = list(CHANNEL_NAMES)

    files = sorted(glob.glob(os.path.join(path, "*.zarr")))
    first = zarr.open_group(files[0], mode="r")
    level_values = np.asarray(first["level"]) if "level" in first else None
    channel_map = build_wb2_channel_map(channel_names, level_values)
    probe_name = channel_map[0][0]
    num_samples_per_file = first[probe_name].shape[0]

    file_id = idx // num_samples_per_file
    file_index = idx % num_samples_per_file
    g = zarr.open_group(files[file_id], mode="r")

    h, w = first[probe_name].shape[-2], first[probe_name].shape[-1]
    result = np.zeros((len(channel_names), h, w), dtype=np.float32)
    for ch_idx, (zarr_name, level_idx) in enumerate(channel_map):
        if level_idx is None:
            result[ch_idx] = g[zarr_name][file_index]
        else:
            result[ch_idx] = g[zarr_name][file_index, level_idx]
    return result


def init_dataset_params(
    train_path: str,
    valid_path: str,
    stats_path: str,
    batch_size: int,
    n_history: int,
    n_future: int,
    normalization: str,
    num_data_workers: int,
):

    # instantiate params base
    params = get_default_parameters()

    # init paths
    params.train_data_path = train_path
    params.valid_data_path = valid_path
    params.min_path = os.path.join(stats_path, "mins.npy")
    params.max_path = os.path.join(stats_path, "maxs.npy")
    params.time_means_path = os.path.join(stats_path, "time_means.npy")
    params.global_means_path = os.path.join(stats_path, "global_means.npy")
    params.global_stds_path = os.path.join(stats_path, "global_stds.npy")
    params.time_diff_means_path = os.path.join(stats_path, "time_diff_means.npy")
    params.time_diff_stds_path = os.path.join(stats_path, "time_diff_stds.npy")

    # general parameters
    params.dhours = 24
    params.h5_path = H5_PATH
    params.n_history = n_history
    params.n_future = n_future
    params.batch_size = batch_size
    params.normalization = normalization

    # performance parameters
    params.num_data_workers = num_data_workers
    params.enable_odirect = False
    params.odirect_alignment = 0
    params.enable_s3 = False

    return params


class DataLoaderBase:
    """Shared tests for all dataloader backends (HDF5, zarr-makani, zarr-wb2, …).

    Not a TestCase subclass so pytest does not collect it directly.
    Concrete subclasses must inherit from both this class and
    ``unittest.TestCase``, implement ``setUpClass`` / ``tearDownClass``, and
    implement ``_get_sample``.
    """

    def _get_dataloader(self, *args, **kwargs):
        """
        get_dataloader that shuts a DALI pipeline down when the test ends. Left to the
        garbage collector, its native threads would outlive the test and run into later ones.
        """
        loader, dataset, sampler = get_dataloader(*args, **kwargs)
        if hasattr(loader, "close"):
            self.addCleanup(loader.close)
        return loader, dataset, sampler

    def _assert_plain_batch(self, batch):
        """A batch with a target and nothing optional, such as the zenith angle."""
        self.assertEqual(set(batch), _TRAINING_KEYS)
        self.assertEqual(batch[INPUT_DYNAMIC_FEATURES], {})
        self.assertEqual(batch[TARGET_DYNAMIC_FEATURES], {})

    def _get_sample(self, path, idx):
        raise NotImplementedError

    def setUp(self):
        disable_tf32()

        self.params = init_dataset_params(
            self.train_path,
            self.valid_path,
            self.stats_path,
            batch_size=2,
            n_history=0,
            n_future=0,
            normalization="zscore",
            num_data_workers=1,
        )

        self.params.multifiles = True
        self.params.num_train = self.num_train
        self.params.num_valid = self.num_valid

        # this is also correct for most cases:
        self.params.io_grid = [1, 1, 1]
        self.params.io_rank = [0, 0, 0]

        self.num_steps = 5

    def _test_shapes_and_sample_counts(self, multifiles):

        # set mutltifiles
        self.params.multifiles = multifiles

        # create dataloaders
        valid_loader, valid_dataset, _ = self._get_dataloader(
            self.params, self.params.valid_data_path, mode="eval", device=self.device
        )

        # do tests
        num_valid_steps = self.params.num_valid // self.params.batch_size
        for idt, token in enumerate(valid_loader):
            self._assert_plain_batch(token)
            inp, tar = token[INPUT], token[TARGET]

            self.assertEqual(tuple(inp.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))
            self.assertEqual(tuple(tar.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))

        self.assertEqual((idt + 1), num_valid_steps)

    def _test_content(self, multifiles):

        # set mutltifiles
        self.params.multifiles = multifiles

        # create dataloaders
        valid_loader, valid_dataset, _ = self._get_dataloader(
            self.params, self.params.valid_data_path, mode="eval", device=self.device
        )

        # do tests
        for idt, token in enumerate(valid_loader):
            # get loader samples
            inp, tar = token[INPUT], token[TARGET]

            # get test samples
            off = self.params.batch_size * idt
            inp_res = []
            tar_res = []
            for b in range(self.params.batch_size):
                inp_res.append(self._get_sample(self.params.valid_data_path, off + b))
                tar_res.append(self._get_sample(self.params.valid_data_path, off + b + 1))
            test_inp = np.squeeze(np.stack(inp_res, axis=0))
            test_tar = np.squeeze(np.stack(tar_res, axis=0))

            inp = np.squeeze(inp.cpu().numpy())
            tar = np.squeeze(tar.cpu().numpy())

            self.assertTrue(compare_arrays("test_content inp", inp, test_inp))
            self.assertTrue(compare_arrays("test_content tar", tar, test_tar))

            if idt > self.num_steps:
                break

    def _test_content_normalization_zscore(self, multifiles):
        """With non-trivial means/stds, zscore output equals (raw - mean) / std."""
        self.params.multifiles = multifiles

        # non-trivial stats: per-channel means/stds distinct from (0, 1)
        rng = np.random.default_rng(seed=1234)
        means = rng.normal(loc=0.5, scale=1.0, size=(1, NUM_CHANNELS, 1, 1)).astype(np.float64)
        stds = rng.uniform(low=0.5, high=2.0, size=(1, NUM_CHANNELS, 1, 1)).astype(np.float64)

        with tempfile.TemporaryDirectory() as tmp_stats:
            np.save(os.path.join(tmp_stats, "global_means.npy"), means)
            np.save(os.path.join(tmp_stats, "global_stds.npy"), stds)
            # ancillary stats files: copy defaults from the shared stats dir
            for fname in ("mins.npy", "maxs.npy", "time_means.npy", "time_diff_means.npy", "time_diff_stds.npy"):
                src = os.path.join(self.stats_path, fname)
                np.save(os.path.join(tmp_stats, fname), np.load(src))

            params = copy.deepcopy(self.params)
            params.global_means_path = os.path.join(tmp_stats, "global_means.npy")
            params.global_stds_path = os.path.join(tmp_stats, "global_stds.npy")
            params.min_path = os.path.join(tmp_stats, "mins.npy")
            params.max_path = os.path.join(tmp_stats, "maxs.npy")
            params.time_means_path = os.path.join(tmp_stats, "time_means.npy")
            params.time_diff_means_path = os.path.join(tmp_stats, "time_diff_means.npy")
            params.time_diff_stds_path = os.path.join(tmp_stats, "time_diff_stds.npy")

            valid_loader, valid_dataset, _ = self._get_dataloader(
                params, params.valid_data_path, mode="eval", device=self.device
            )

            # the same (B, C, 1, 1) stats broadcast over (B, C, H, W) raw samples
            means_b = means.astype(np.float32)
            stds_b = stds.astype(np.float32)

            for idt, token in enumerate(valid_loader):
                inp, tar = token[INPUT], token[TARGET]

                off = params.batch_size * idt
                inp_raw = np.stack(
                    [self._get_sample(params.valid_data_path, off + b) for b in range(params.batch_size)],
                    axis=0,
                ).astype(np.float32)
                tar_raw = np.stack(
                    [self._get_sample(params.valid_data_path, off + b + 1) for b in range(params.batch_size)],
                    axis=0,
                ).astype(np.float32)

                # analytical zscore: broadcast (1, C, 1, 1) stats against (B, C, H, W) samples
                expected_inp = (inp_raw - means_b) / stds_b
                expected_tar = (tar_raw - means_b) / stds_b

                inp_np = np.squeeze(inp.cpu().numpy())
                tar_np = np.squeeze(tar.cpu().numpy())

                self.assertTrue(
                    compare_arrays("zscore normalized inp", inp_np, np.squeeze(expected_inp), atol=1e-5, rtol=1e-4)
                )
                self.assertTrue(
                    compare_arrays("zscore normalized tar", tar_np, np.squeeze(expected_tar), atol=1e-5, rtol=1e-4)
                )

                if idt > self.num_steps:
                    break

    def _test_channel_ordering(self, multifiles):

        # set mutltifiles
        self.params.multifiles = multifiles

        # create dataloaders
        valid_loader, valid_dataset, _ = self._get_dataloader(
            self.params, self.params.valid_data_path, mode="eval", device=self.device
        )

        # create flipped dataloader
        params = copy.deepcopy(self.params)
        params.in_channels = params.in_channels[::-1]
        params.out_channels = params.out_channels[::-1]
        valid_loader_flip, valid_dataset_flip, _ = self._get_dataloader(
            params, self.params.valid_data_path, mode="eval", device=self.device
        )

        for idt, (token, token_flip) in enumerate(zip(valid_loader, valid_loader_flip)):
            inp, tar = token[INPUT], token[TARGET]
            inp_flip, tar_flip = token_flip[INPUT], token_flip[TARGET]

            self.assertFalse(compare_tensors("inp vs flipped inp", inp, inp_flip))
            inp_flip_flip = torch.flip(inp_flip, dims=(2,))
            self.assertTrue(compare_tensors("inp vs double-flipped inp", inp, inp_flip_flip))

            self.assertFalse(compare_tensors("tar vs flipped tar", tar, tar_flip))
            tar_flip_flip = torch.flip(tar_flip, dims=(2,))
            self.assertTrue(compare_tensors("tar vs double-flipped tar", tar, tar_flip_flip))

    def _test_history(self, multifiles):

        # set mutltifiles
        self.params.multifiles = multifiles

        # set history:
        self.params.n_history = 3

        # create dataloaders
        valid_loader, valid_dataset, _ = self._get_dataloader(
            self.params, self.params.valid_data_path, mode="eval", device=self.device
        )

        # do tests
        for idt, token in enumerate(valid_loader):
            self._assert_plain_batch(token)
            inp, tar = token[INPUT], token[TARGET]

            self.assertEqual(
                tuple(inp.shape),
                (self.params.batch_size, self.params.n_history + 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W),
            )
            self.assertEqual(tuple(tar.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))

            if idt > self.num_steps:
                break

    def _test_future(self, multifiles):

        # set mutltifiles
        self.params.multifiles = multifiles

        # set future:
        self.params.n_future = 3

        # create dataloaders
        train_loader, train_dataset, _ = self._get_dataloader(
            self.params, self.params.train_data_path, mode="train", device=self.device
        )

        # do tests
        for idt, token in enumerate(train_loader):
            self._assert_plain_batch(token)
            inp, tar = token[INPUT], token[TARGET]

            self.assertEqual(tuple(inp.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))
            self.assertEqual(
                tuple(tar.shape),
                (self.params.batch_size, self.params.n_future + 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W),
            )

            if idt > self.num_steps:
                break

    def _test_autoreg(self, multifiles):

        # set mutltifiles
        self.params.multifiles = multifiles

        # set autoreg
        self.params.valid_autoreg_steps = 3

        # create dataloaders
        valid_loader, valid_dataset, _ = self._get_dataloader(
            self.params, self.params.valid_data_path, mode="eval", device=self.device
        )

        # do tests
        for idt, token in enumerate(valid_loader):
            self._assert_plain_batch(token)
            inp, tar = token[INPUT], token[TARGET]

            self.assertEqual(tuple(inp.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))
            self.assertEqual(
                tuple(tar.shape),
                (self.params.batch_size, self.params.valid_autoreg_steps + 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W),
            )

            if idt > self.num_steps:
                break

    def _test_distributed(self, multifiles):

        # set multifiles
        self.params.multifiles = multifiles

        # set IO grid
        self.params.io_grid = [1, 2, 1]
        self.params.io_rank = [0, 1, 0]

        valid_loader, valid_dataset, _ = self._get_dataloader(
            self.params, self.params.valid_data_path, mode="eval", device=self.device
        )

        off_x = valid_dataset.img_local_offset_x
        off_y = valid_dataset.img_local_offset_y
        range_x = valid_dataset.img_local_shape_x
        range_y = valid_dataset.img_local_shape_y

        # do tests
        num_steps = 3
        for idt, token in enumerate(valid_loader):
            # get loader samples
            inp, tar = token[INPUT], token[TARGET]

            self.assertEqual(tuple(inp.shape), (self.params.batch_size, 1, NUM_CHANNELS, range_x, range_y))
            self.assertEqual(tuple(tar.shape), (self.params.batch_size, 1, NUM_CHANNELS, range_x, range_y))

            # get test samples
            off = self.params.batch_size * idt
            inp_res = []
            tar_res = []
            for b in range(self.params.batch_size):
                tmp = self._get_sample(self.params.valid_data_path, off + b)
                inp_res.append(tmp[:, off_x : off_x + range_x, off_y : off_y + range_y])
                tmp = self._get_sample(self.params.valid_data_path, off + b + 1)
                tar_res.append(tmp[:, off_x : off_x + range_x, off_y : off_y + range_y])

            # stack
            test_inp = np.squeeze(np.stack(inp_res, axis=0))
            test_tar = np.squeeze(np.stack(tar_res, axis=0))

            inp = np.squeeze(inp.cpu().numpy())
            tar = np.squeeze(tar.cpu().numpy())

            self.assertTrue(compare_arrays("test_distributed inp", inp, test_inp))
            self.assertTrue(compare_arrays("test_distributed tar", tar, test_tar))

            if idt > self.num_steps:
                break

    def _test_distributed_subsampling(self, multifiles):

        # set multifiles
        self.params.multifiles = multifiles

        # set subsampling factor
        subsample = 2
        self.params.subsampling_factor = subsample

        # set IO grid
        self.params.io_grid = [1, 2, 1]
        self.params.io_rank = [0, 1, 0]

        valid_loader, valid_dataset, _ = self._get_dataloader(
            self.params, self.params.valid_data_path, mode="eval", device=self.device
        )

        off_x = valid_dataset.img_local_offset_x
        off_y = valid_dataset.img_local_offset_y
        range_x = valid_dataset.img_local_shape_x
        range_y = valid_dataset.img_local_shape_y

        # do tests
        num_steps = 3
        for idt, token in enumerate(valid_loader):
            # get loader samples
            inp, tar = token[INPUT], token[TARGET]

            self.assertEqual(
                tuple(inp.shape),
                (
                    self.params.batch_size,
                    1,
                    NUM_CHANNELS,
                    math.ceil(range_x / subsample),
                    math.ceil(range_y / subsample),
                ),
            )
            self.assertEqual(
                tuple(tar.shape),
                (
                    self.params.batch_size,
                    1,
                    NUM_CHANNELS,
                    math.ceil(range_x / subsample),
                    math.ceil(range_y / subsample),
                ),
            )

            # get test samples
            off = self.params.batch_size * idt
            inp_res = []
            tar_res = []
            for b in range(self.params.batch_size):
                tmp = self._get_sample(self.params.valid_data_path, off + b)
                inp_res.append(tmp[:, off_x : off_x + range_x : subsample, off_y : off_y + range_y : subsample])
                tmp = self._get_sample(self.params.valid_data_path, off + b + 1)
                tar_res.append(tmp[:, off_x : off_x + range_x : subsample, off_y : off_y + range_y : subsample])

            # stack
            test_inp = np.squeeze(np.stack(inp_res, axis=0))
            test_tar = np.squeeze(np.stack(tar_res, axis=0))

            inp = np.squeeze(inp.cpu().numpy())
            tar = np.squeeze(tar.cpu().numpy())

            self.assertTrue(compare_arrays("test_distributed_subsampling inp", inp, test_inp))
            self.assertTrue(compare_arrays("test_distributed_subsampling tar", tar, test_tar))

            if idt > self.num_steps:
                break

    def _test_dali_temporal_window_across_year_boundary_no_duplicates(self):
        """
        DALI path, eval mode on the 2-year training set with n_future > 0.

        Invariants on a full non-shuffled epoch:
          1) every loaded sample is distinct, and
          2) the loaded set equals direct HDF5 reads of the indices the
             loader claims to visit (loader.extsource.indices_select).

        (1) catches the original failure mode: a per-year clamp in
        SampleSource.__call__ that silently maps indices near a year boundary
        to the same frame, producing duplicates.  (2) additionally catches
        a regression where the clamp returned a fixed-but-different frame
        (no duplicates but every sample wrong) and other indexing slips
        that uniqueness alone would miss.  The test stats fixture is
        zero-mean / unit-std (testutils.py:311-315) so zscore is a
        pass-through and the loader's float32 output equals on-disk bytes.
        """
        params = copy.deepcopy(self.params)
        params.multifiles = False
        params.valid_autoreg_steps = 3  # n_future=3 in eval mode
        params.batch_size = 1

        loader, _, _ = self._get_dataloader(
            params,
            params.train_data_path,
            mode="eval",
            device=self.device,
            dali_device=self.dali_device,
        )

        fingerprints = []
        for token in loader:
            inp = token[INPUT]
            for b in range(inp.shape[0]):
                fingerprints.append(inp[b].cpu().numpy().tobytes())

        unique = len(set(fingerprints))
        total = len(fingerprints)
        self.assertEqual(
            unique,
            total,
            f"{total - unique} duplicate samples in a non-shuffled DALI epoch " f"(total={total}, unique={unique}).",
        )

        expected = {
            self._get_sample(params.train_data_path, int(idx)).tobytes() for idx in loader.extsource.indices_select
        }
        self.assertEqual(
            set(fingerprints),
            expected,
            "loaded samples do not match direct HDF5 reads of indices_select",
        )

    def _test_dali_parallel_workers_full_epoch_coverage(self, num_workers):
        """
        DALI path, eval mode, parameterized on num_data_workers.

        num_data_workers is plumbed through to both py_num_workers and the
        external_source's prefetch_queue_depth (dali_dataloader.py:41,70).
        Under parallel execution DALI runs a SharedBatchDispatcher in a
        thread per worker that serialises the callback's returned numpy
        arrays into shared memory.  If SampleSource.__call__ ever returned a
        reference to a reused per-worker buffer instead of a fresh copy,
        the main worker thread would invoke the next sample-fetch before
        the dispatcher had finished reading the previous buffer, producing
        either duplicate or torn samples.  SampleSource._reorder_channels
        sidesteps this by always returning a copy; this test locks that
        invariant in.

        Baseline: raw HDF5 reads via get_sample(), since the test stats
        fixture is zero-mean / unit-std (testutils.py:311-315) so the
        zscore normalisation is a pass-through and the loader's float32
        output matches the on-disk bytes.  This catches a wider class of
        bugs than a loader-vs-loader comparison would (e.g. wrong-index
        reads that would alias themselves).

        For each worker count, a full non-shuffled eval epoch must
          1) be duplicate-free,
          2) yield exactly the loader's reported step count,
          3) yield the same SET of samples as direct HDF5 reads of the
             indices the loader claims to visit.
        """
        params = copy.deepcopy(self.params)
        params.multifiles = False
        params.batch_size = 1
        # cap dataset to keep the multi-config sweep cheap; truncate_old
        # picks the trailing window, so a small cap still exercises a full
        # epoch through the parallel ES path
        params.n_eval_samples = 40
        params.num_data_workers = num_workers

        loader, _, _ = self._get_dataloader(
            params,
            params.valid_data_path,
            mode="eval",
            device=self.device,
            dali_device=self.dali_device,
        )

        fps = []
        for token in loader:
            inp = token[INPUT]
            for b in range(inp.shape[0]):
                fps.append(inp[b].cpu().numpy().tobytes())

        self.assertEqual(
            len(set(fps)),
            len(fps),
            f"{len(fps) - len(set(fps))} duplicates with num_data_workers={num_workers} "
            f"(total={len(fps)}, unique={len(set(fps))})",
        )
        self.assertEqual(len(fps), len(loader))

        # ground truth: read the same indices straight from the HDF5 file
        expected = {
            self._get_sample(params.valid_data_path, int(idx)).tobytes() for idx in loader.extsource.indices_select
        }
        self.assertEqual(
            set(fps),
            expected,
            f"loader samples at num_data_workers={num_workers} do not match " f"direct HDF5 reads",
        )

    def _test_dali_multi_epoch_reset_state(self):
        """
        DALI path, train mode on the 2-year training set.

        Drives the pipeline through two full back-to-back epochs via
        auto_reset, then a third epoch after an explicit reset_pipeline().
        Checks
          - each epoch fully iterates without raising,
          - each epoch yields distinct samples (no intra-epoch duplicates),
          - the two shuffled epochs use different permutations (seed is
            base_seed + cycle_epoch_idx; if the DALI/ES epoch counter got
            stuck the two epochs would match).
        """
        params = copy.deepcopy(self.params)
        params.multifiles = False
        params.batch_size = 1
        params.n_train_samples = 20  # keep the test fast
        params.n_future = 0

        loader, _, _ = self._get_dataloader(
            params,
            params.train_data_path,
            mode="train",
            device=self.device,
            dali_device=self.dali_device,
        )

        def _collect_epoch():
            fps = []
            for token in loader:
                inp = token[INPUT]
                for b in range(inp.shape[0]):
                    fps.append(inp[b].cpu().numpy().tobytes())
            return fps

        # auto_reset handles the first-to-second transition
        fps_ep0 = _collect_epoch()
        fps_ep1 = _collect_epoch()

        # explicit reset_pipeline() should also produce a valid epoch
        loader.reset_pipeline()
        fps_ep2 = _collect_epoch()

        for ep_idx, fps in enumerate((fps_ep0, fps_ep1, fps_ep2)):
            with self.subTest(desc=f"epoch {ep_idx} non-empty"):
                self.assertGreater(len(fps), 0)
            with self.subTest(desc=f"no intra-epoch duplicates in epoch {ep_idx}"):
                self.assertEqual(len(fps), len(set(fps)))

        # auto_reset must advance the shuffle seed: two consecutive epochs
        # cannot produce identical orderings.
        with self.subTest(desc="auto_reset advances shuffle state"):
            self.assertNotEqual(fps_ep0, fps_ep1)

    def _test_dali_checkpoint_restore_survives_rebuild(self):
        """
        DALI path, train mode: a restore can be performed at all on this backend.

        Restoring rebuilds the pipeline inside a live process, so the python worker
        pool is spawned a second time and every worker re-opens its file handles
        through this backend's reader (``_get_year_h5`` / ``_zarr_open`` / the WB2
        variant). That is the only part of the restore which is backend-specific;
        the resume logic itself works off DALI's counters and is backend-independent,
        so it is covered once, in ``_test_dali_checkpoint_mid_epoch_resume``. This
        test therefore only asserts that the rebuild survives and keeps delivering
        data, not which samples come out.
        """
        params = copy.deepcopy(self.params)
        params.multifiles = False
        params.batch_size = 1
        params.n_train_samples = 8  # keep the test fast
        params.n_future = 0

        loader, _, _ = self._get_dataloader(
            params,
            params.train_data_path,
            mode="train",
            device=self.device,
            dali_device=self.dali_device,
        )

        before = next(iter(loader))[INPUT]

        # round-trip the state through the loader, which tears the pipeline down and
        # builds a new one with a fresh worker pool
        self.assertTrue(loader.load_state_dict(loader.state_dict()))

        after = next(iter(loader))[INPUT]

        self.assertEqual(tuple(after.shape), tuple(before.shape))

    def _test_dali_checkpoint_mid_epoch_resume(self):
        """
        DALI path, train mode: state_dict/load_state_dict resume the sample sequence.

        Captures the loader state in the middle of a (shuffled) epoch, records
        the samples that follow, and checks that a freshly built loader which
        restores that state produces exactly those samples. A third loader
        without a restore must NOT match, which pins down that the comparison
        is actually sensitive to the stored position rather than trivially
        satisfied by both loaders starting from the same seed.
        """
        params = copy.deepcopy(self.params)
        params.multifiles = False
        params.batch_size = 1
        params.n_train_samples = 20  # keep the test fast
        params.n_future = 0

        def _make_loader():
            loader, _, _ = self._get_dataloader(
                params,
                params.train_data_path,
                mode="train",
                device=self.device,
                dali_device=self.dali_device,
            )
            return loader

        def _collect(iterator, num_steps):
            fps = []
            for _ in range(num_steps):
                inp = next(iterator)[INPUT]
                for b in range(inp.shape[0]):
                    fps.append(inp[b].cpu().numpy().tobytes())
            return fps

        num_consumed = 3
        num_compared = 3

        # consume a part of the epoch, then checkpoint mid-epoch
        loader = _make_loader()
        iterator = iter(loader)
        _collect(iterator, num_consumed)
        state_dict = loader.state_dict()

        # the continuation of the interrupted epoch is the ground truth
        reference = _collect(iterator, num_compared)

        # a restored loader has to continue where the state was captured
        restored_loader = _make_loader()
        self.assertTrue(restored_loader.load_state_dict(state_dict))
        resumed = _collect(iter(restored_loader), num_compared)

        with self.subTest(desc="restored loader resumes the sample sequence"):
            self.assertEqual(resumed, reference)

        # without a restore the loader starts at the beginning of the epoch instead
        fresh = _collect(iter(_make_loader()), num_compared)

        with self.subTest(desc="loader without restore does not resume"):
            self.assertNotEqual(fresh, reference)


@parameterized_class(("dali_device",), _dali_devices)
class TestHDF5DataLoader(DataLoaderBase, unittest.TestCase):
    """DataLoaderBase exercised against HDF5-backed files.

    Both ``multifiles=True`` (MultifilesDataset) and ``multifiles=False``
    (DALI) paths are tested.  Additionally tests the MultifilesDataset-specific
    date/index retrieval API which is not available on the DALI path.
    """

    @classmethod
    def setUpClass(cls, path: Optional[str] = "/tmp"):
        cls.device = torch.device("cpu")
        cls.tmpdir = tempfile.TemporaryDirectory(dir=path)
        tmp_path = cls.tmpdir.name
        cls.train_path, cls.num_train, cls.valid_path, cls.num_valid, cls.stats_path, cls.metadata_path, _ = (
            init_hdf5_dataset(tmp_path)
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _get_sample(self, path, idx):
        return get_sample(path, idx)

    # multifiles-parameterized tests: both True (MultifilesDataset) and False (DALI)
    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_shapes_and_sample_counts(self, multifiles):
        self._test_shapes_and_sample_counts(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_content(self, multifiles):
        self._test_content(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_content_normalization_zscore(self, multifiles):
        self._test_content_normalization_zscore(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_channel_ordering(self, multifiles):
        self._test_channel_ordering(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_history(self, multifiles):
        self._test_history(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_future(self, multifiles):
        self._test_future(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_autoreg(self, multifiles):
        self._test_autoreg(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_distributed(self, multifiles):
        self._test_distributed(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_distributed_subsampling(self, multifiles):
        self._test_distributed_subsampling(multifiles)

    # DALI-specific mechanics tests — defined here (not in DataLoaderBase) so
    # @parameterized_class strips them from the unparameterized base class.
    @unittest.skipUnless(_have_dali, "nvidia.dali is not installed")
    def test_dali_temporal_window_across_year_boundary_no_duplicates(self):
        self._test_dali_temporal_window_across_year_boundary_no_duplicates()

    @parameterized.expand([(1,), (2,), (4,)], skip_on_empty=False)
    @unittest.skipUnless(_have_dali, "nvidia.dali is not installed")
    def test_dali_parallel_workers_full_epoch_coverage(self, num_workers):
        self._test_dali_parallel_workers_full_epoch_coverage(num_workers)

    @unittest.skipUnless(_have_dali, "nvidia.dali is not installed")
    def test_dali_multi_epoch_reset_state(self):
        self._test_dali_multi_epoch_reset_state()

    @unittest.skipUnless(_have_dali, "nvidia.dali is not installed")
    def test_dali_checkpoint_mid_epoch_resume(self):
        self._test_dali_checkpoint_mid_epoch_resume()

    @unittest.skipUnless(_have_dali, "nvidia.dali is not installed")
    def test_dali_batch_carries_named_timestamps(self):
        # timestamps used to be positional, and a training loader that returned them put
        # them where the trainers expect the zenith angle
        self.params.multifiles = False
        self.params.n_future = 1
        train_loader, _, _ = self._get_dataloader(
            self.params, self.params.train_data_path, mode="train", device=self.device, dali_device=self.dali_device
        )
        batch = next(iter(train_loader))
        self._assert_plain_batch(batch)
        self.assertEqual(tuple(batch[INPUT_TIMESTAMP].shape), (self.params.batch_size, 1))
        self.assertEqual(tuple(batch[TARGET_TIMESTAMP].shape), (self.params.batch_size, 2))
        self.assertEqual(batch[INPUT_TIMESTAMP].dtype, torch.float64)

    @unittest.skipUnless(_have_dali, "nvidia.dali is not installed")
    def test_dali_close_releases_the_pipeline(self):
        self.params.multifiles = False
        train_loader, _, _ = self._get_dataloader(
            self.params, self.params.train_data_path, mode="train", device=self.device, dali_device=self.dali_device
        )
        next(iter(train_loader))
        train_loader.close()
        self.assertIsNone(train_loader.pipeline)
        self.assertIsNone(train_loader.iterator)
        # a second close, e.g. the registered test cleanup, must be harmless
        train_loader.close()

    # HDF5-only: MultifilesDataset date/index retrieval API
    def test_date_retrieval(self):
        self.params.multifiles = True
        train_loader, _, _ = self._get_dataloader(
            self.params, self.params.train_data_path, mode="train", device=self.device
        )
        dhours = 24
        time1 = train_loader.dataset.get_time_at_index(10)
        time1_comp = dt.datetime(
            year=2017, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc
        ) + dt.timedelta(hours=dhours * 10)
        self.assertEqual(time1, time1_comp)
        time2 = train_loader.dataset.get_time_at_index(365)
        time2_comp = dt.datetime(year=2018, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc)
        self.assertEqual(time2, time2_comp)

    def test_index_retrieval(self):
        self.params.multifiles = True
        train_loader, _, _ = self._get_dataloader(
            self.params, self.params.train_data_path, mode="train", device=self.device
        )
        dhours = 24
        tstamp = dt.datetime(
            year=2017, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc
        ) + dt.timedelta(hours=dhours * 10)
        self.assertEqual(train_loader.dataset.get_index_at_time(tstamp), 10)
        tstamp = dt.datetime(year=2018, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc)
        self.assertEqual(train_loader.dataset.get_index_at_time(tstamp), 365)


@parameterized_class(("dali_device",), _dali_devices)
@unittest.skipUnless(_have_dali, "nvidia.dali is not installed — zarr tests require DALI")
class TestZarrDataLoader(DataLoaderBase, unittest.TestCase):
    """DataLoaderBase exercised against zarr-backed files (makani flat format).

    Both ``multifiles=True`` (MultifilesDataset, now zarr-aware) and
    ``multifiles=False`` (DALI) paths are tested, mirroring TestHDF5DataLoader.
    MultifilesDataset-specific date/index retrieval is also exercised.

    ``zarr_format="wb2"`` testing lives in ``TestZarrWB2DataLoader``.
    """

    @classmethod
    def setUpClass(cls, path: Optional[str] = "/tmp"):
        cls.device = torch.device("cpu")
        cls.tmpdir = tempfile.TemporaryDirectory(dir=path)
        tmp_path = cls.tmpdir.name
        cls.train_path, cls.num_train, cls.valid_path, cls.num_valid, cls.stats_path, cls.metadata_path, _ = (
            init_zarr_dataset(tmp_path)
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _get_sample(self, path, idx):
        return get_zarr_sample(path, idx)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_shapes_and_sample_counts(self, multifiles):
        self._test_shapes_and_sample_counts(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_content(self, multifiles):
        self._test_content(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_content_normalization_zscore(self, multifiles):
        self._test_content_normalization_zscore(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_channel_ordering(self, multifiles):
        self._test_channel_ordering(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_history(self, multifiles):
        self._test_history(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_future(self, multifiles):
        self._test_future(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_autoreg(self, multifiles):
        self._test_autoreg(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_distributed(self, multifiles):
        self._test_distributed(multifiles)

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_distributed_subsampling(self, multifiles):
        self._test_distributed_subsampling(multifiles)

    # the full mid-epoch resume is exercised on the HDF5 backend; here we only check
    # that the pipeline rebuild a restore performs works with the zarr reader
    def test_dali_checkpoint_restore_survives_rebuild(self):
        self._test_dali_checkpoint_restore_survives_rebuild()

    def test_date_retrieval(self):
        self.params.multifiles = True
        train_loader, _, _ = self._get_dataloader(
            self.params, self.params.train_data_path, mode="train", device=self.device
        )
        dhours = 24
        time1 = train_loader.dataset.get_time_at_index(10)
        time1_comp = dt.datetime(
            year=2017, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc
        ) + dt.timedelta(hours=dhours * 10)
        self.assertEqual(time1, time1_comp)
        time2 = train_loader.dataset.get_time_at_index(365)
        time2_comp = dt.datetime(year=2018, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc)
        self.assertEqual(time2, time2_comp)

    def test_index_retrieval(self):
        self.params.multifiles = True
        train_loader, _, _ = self._get_dataloader(
            self.params, self.params.train_data_path, mode="train", device=self.device
        )
        dhours = 24
        tstamp = dt.datetime(
            year=2017, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc
        ) + dt.timedelta(hours=dhours * 10)
        self.assertEqual(train_loader.dataset.get_index_at_time(tstamp), 10)
        tstamp = dt.datetime(year=2018, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc)
        self.assertEqual(train_loader.dataset.get_index_at_time(tstamp), 365)


@parameterized_class(("dali_device",), _dali_devices)
@unittest.skipUnless(_have_dali, "nvidia.dali is not installed — zarr WB2 tests require DALI")
class TestZarrWB2DataLoader(DataLoaderBase, unittest.TestCase):
    """DataLoaderBase exercised against WB2-layout zarr files, DALI path only.

    The fixture (``init_wb2_zarr_dataset``) creates one zarr array per ERA5
    variable (surface: ``(time, lat, lon)``, atmospheric: ``(time, level, lat, lon)``)
    mirroring the real WeatherBench2 store layout.  The ES helper auto-detects
    the WB2 format because ``dataset_name="fields"`` is absent from the group,
    then builds the variable→level conversion table from ``params.channel_names``
    (set to ``CHANNEL_NAMES`` by ``get_default_parameters()``).
    """

    @classmethod
    def setUpClass(cls, path: Optional[str] = "/tmp"):
        cls.device = torch.device("cpu")
        cls.tmpdir = tempfile.TemporaryDirectory(dir=path)
        tmp_path = cls.tmpdir.name
        cls.train_path, cls.num_train, cls.valid_path, cls.num_valid, cls.stats_path, cls.metadata_path, _ = (
            init_wb2_zarr_dataset(tmp_path)
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _get_sample(self, path, idx):
        return get_wb2_zarr_sample(path, idx)

    # DALI-only (multifiles=False) — one test per behaviour, no parameterization
    def test_shapes_and_sample_counts(self):
        self._test_shapes_and_sample_counts(False)

    def test_content(self):
        self._test_content(False)

    def test_content_normalization_zscore(self):
        self._test_content_normalization_zscore(False)

    def test_channel_ordering(self):
        self._test_channel_ordering(False)

    def test_history(self):
        self._test_history(False)

    def test_future(self):
        self._test_future(False)

    # the full mid-epoch resume is exercised on the HDF5 backend; here we only check
    # that the pipeline rebuild a restore performs works with the WB2 zarr reader
    def test_dali_checkpoint_restore_survives_rebuild(self):
        self._test_dali_checkpoint_restore_survives_rebuild()

    def test_autoreg(self):
        self._test_autoreg(False)

    def test_distributed(self):
        self._test_distributed(False)

    def test_distributed_subsampling(self):
        self._test_distributed_subsampling(False)


class TestDummyLoader(unittest.TestCase):
    """Smoke tests for the synthetic DummyLoader. CPU-friendly — no DALI, no HDF5 fixture."""

    def setUp(self):
        disable_tf32()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.img_shape = (32, 64)
        self.in_channels = list(range(5))
        self.out_channels = list(range(5))

    def _make_loader(self, **overrides):
        from makani.utils.dataloaders.data_loader_dummy import DummyLoader

        kwargs = dict(
            location="/nonexistent",
            device=self.device,
            batch_size=self.batch_size,
            dt=1,
            dhours=6,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            img_shape=self.img_shape,
            n_samples_per_epoch=4,
        )
        kwargs.update(overrides)
        return DummyLoader(**kwargs)

    def test_basic_iteration_shapes_and_dtype(self):
        loader = self._make_loader()
        self.assertEqual(len(loader), 4)
        batches = list(loader)
        self.assertEqual(len(batches), 4)
        for batch in batches:
            inp, tar = batch[INPUT], batch[TARGET]
            self.assertEqual(tuple(inp.shape), (self.batch_size, 1, len(self.in_channels), *self.img_shape))
            self.assertEqual(tuple(tar.shape), (self.batch_size, 1, len(self.out_channels), *self.img_shape))
            self.assertEqual(inp.dtype, torch.float32)
            self.assertEqual(tar.dtype, torch.float32)

    def test_history_and_future_dimensions(self):
        loader = self._make_loader(n_history=2, n_future=3)
        batch = next(iter(loader))
        inp, tar = batch[INPUT], batch[TARGET]
        self.assertEqual(inp.shape[1], 3)  # n_history + 1
        self.assertEqual(tar.shape[1], 4)  # n_future  + 1

    def test_add_zenith_adds_zenith_features(self):
        loader = self._make_loader(add_zenith=True)
        batch = next(iter(loader))
        self.assertEqual(set(batch[INPUT_DYNAMIC_FEATURES]), {ZENITH})
        self.assertEqual(set(batch[TARGET_DYNAMIC_FEATURES]), {ZENITH})
        inp_zen = batch[INPUT_DYNAMIC_FEATURES][ZENITH]
        self.assertEqual(tuple(inp_zen.shape), (self.batch_size, 1, 1, *self.img_shape))

    def test_target_zenith_spans_the_target_window(self):
        # the target zenith used to be the input zenith tensor, n_history+1 steps long
        loader = self._make_loader(add_zenith=True, n_history=1, n_future=2)
        batch = next(iter(loader))
        inp_zen = batch[INPUT_DYNAMIC_FEATURES][ZENITH]
        tar_zen = batch[TARGET_DYNAMIC_FEATURES][ZENITH]
        self.assertEqual(tuple(inp_zen.shape), (self.batch_size, 2, 1, *self.img_shape))
        self.assertEqual(tuple(tar_zen.shape), (self.batch_size, 3, 1, *self.img_shape))

    def test_batch_carries_timestamps(self):
        loader = self._make_loader()
        batch = next(iter(loader))
        self.assertEqual(set(batch), _TRAINING_KEYS)
        inp_time = batch[INPUT_TIMESTAMP]
        self.assertEqual(tuple(inp_time.shape), (self.batch_size, 1))
        self.assertEqual(inp_time.dtype, torch.float64)

    def test_return_target_false_yields_input_only(self):
        loader = self._make_loader(return_target=False)
        batch = next(iter(loader))
        self.assertEqual(set(batch), _INFERENCE_KEYS)

    def test_subsampling_factor_reduces_spatial_shape(self):
        loader = self._make_loader(subsampling_factor=2)
        inp = next(iter(loader))[INPUT]
        self.assertEqual(tuple(inp.shape[-2:]), (self.img_shape[0] // 2, self.img_shape[1] // 2))

    def test_normalization_helpers_return_neutral_stats(self):
        loader = self._make_loader()
        in_bias, in_scale = loader.get_input_normalization()
        out_bias, out_scale = loader.get_output_normalization()
        self.assertEqual(in_bias.shape, (1, len(self.in_channels), 1, 1))
        self.assertTrue((in_bias == 0).all())
        self.assertTrue((in_scale == 1).all())
        self.assertTrue((out_bias == 0).all())
        self.assertTrue((out_scale == 1).all())


def _without_dali(name, *args, **kwargs):
    """``importlib.util.find_spec`` as on a system without DALI."""
    if name == "nvidia.dali":
        return None
    return _find_spec(name, *args, **kwargs)


_find_spec = importlib.util.find_spec


class TestGetDataloaderWithoutDali(unittest.TestCase):
    """Only the DALI path needs DALI; the synthetic loader used to be refused without it."""

    def _params(self):
        params = get_default_parameters()
        params.multifiles = False
        params.dhours = 6
        params.batch_size = 2
        params.n_train_samples = 4
        return params

    def test_synthetic_loader_does_not_need_dali(self):
        from makani.utils.dataloaders.data_loader_dummy import DummyLoader

        params = self._params()
        params.enable_synthetic_data = True
        with mock.patch("importlib.util.find_spec", side_effect=_without_dali):
            loader, _, _ = get_dataloader(params, "/nonexistent", device=torch.device("cpu"), mode="train")
        self.assertIsInstance(loader, DummyLoader)

    def test_dali_loader_still_reports_missing_dali(self):
        params = self._params()
        params.enable_synthetic_data = False
        with mock.patch("importlib.util.find_spec", side_effect=_without_dali):
            with self.assertRaisesRegex(ImportError, "nvidia-dali"):
                get_dataloader(params, "/nonexistent", device=torch.device("cpu"), mode="train")


class TestMultifilesDataset(unittest.TestCase):
    """MultifilesDataset built directly, without DALI or a GPU.

    The parameterized suites above compare it against the DALI path on the same
    fixture, which is the broad check. What is pinned here is what only this
    dataset does: reading a target window, striding by ``dt``, surviving the
    pickling a torch worker pool performs, and reporting the grid the files
    actually carry.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.train_path, *_ = init_hdf5_dataset(cls.tmpdir.name)

        # channel c holds the constant value c + 1, so which channels came back
        # can be read off the values
        # one file: the dataset requires a constant cadence, and a second file
        # starting at its own new year would leave a 347 day gap after twenty
        # daily samples
        cls.distinct_path = os.path.join(cls.tmpdir.name, "distinct")
        os.makedirs(cls.distinct_path, exist_ok=True)
        base = dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc)
        n_samples = 20
        timestamps = np.array(
            [(base + dt.timedelta(hours=24 * idx)).timestamp() for idx in range(n_samples)], dtype=np.float64
        )
        data = np.zeros((n_samples, NUM_CHANNELS, 8, 16), dtype=np.float32)
        for channel in range(NUM_CHANNELS):
            data[:, channel] = float(channel + 1)
        with h5.File(os.path.join(cls.distinct_path, "2017.h5"), "w") as handle:
            dset = handle.create_dataset(H5_PATH, data=data)
            scale = handle.create_dataset("timestamp", data=timestamps)
            scale.make_scale("timestamp")
            dset.dims[0].attach_scale(scale)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _make(self, location=None, **overrides):
        from makani.utils.dataloaders.data_loader_multifiles import MultifilesDataset

        kwargs = dict(
            location=location or self.train_path,
            dt=1,
            in_channels=list(range(NUM_CHANNELS)),
            out_channels=list(range(NUM_CHANNELS)),
            n_history=0,
            n_future=0,
            enable_logging=False,
        )
        kwargs.update(overrides)
        return MultifilesDataset(**kwargs)

    def test_target_uses_the_output_channels(self):
        """The target window is the output channels, not the input ones.

        These were read with the input selection and then normalized with the
        output statistics, so a run whose outputs differ from its inputs
        trained against the wrong fields. Nothing catches that downstream: the
        shapes agree whenever the two selections are the same length.
        """
        dataset = self._make(location=self.distinct_path, in_channels=[0, 1], out_channels=[3, 4], n_future=0)

        sample = dataset[0]
        inp, tar = sample[INPUT], sample[TARGET]

        # channel c holds c + 1
        for name, values, constant in (
            ("inp c0", inp[:, 0], 1.0),
            ("inp c1", inp[:, 1], 2.0),
            ("tar c0", tar[:, 0], 4.0),
            ("tar c1", tar[:, 1], 5.0),
        ):
            with self.subTest(field=name):
                got = values.numpy()
                self.assertTrue(compare_arrays(name, got, np.full_like(got, constant)))

    def test_target_channels_are_reordered_like_the_inputs(self):
        # an unsorted request has to come back in the order it was asked for,
        # on the target side as well
        dataset = self._make(location=self.distinct_path, in_channels=[1, 0], out_channels=[4, 2])

        sample = dataset[0]
        inp, tar = sample[INPUT], sample[TARGET]

        for name, values, constant in (
            ("inp c0", inp[:, 0], 2.0),
            ("inp c1", inp[:, 1], 1.0),
            ("tar c0", tar[:, 0], 5.0),
            ("tar c1", tar[:, 1], 3.0),
        ):
            with self.subTest(field=name):
                got = values.numpy()
                self.assertTrue(compare_arrays(name, got, np.full_like(got, constant)))

    def test_timestamps_stride_by_dt(self):
        """The times returned belong to the samples returned.

        The window steps ``dt`` samples at a time, so its timestamps do too.
        Slicing adjacent ones instead gives times that drift from the data as
        soon as dt > 1, which reaches the model through the zenith angle.
        """
        dataset = self._make(dt=2, n_history=1, add_zenith=False)

        inp_time = dataset[0][INPUT_TIMESTAMP]

        dhours = dataset.dhours
        self.assertEqual(float(inp_time[1] - inp_time[0]), 2 * dhours * 3600.0)

    def test_timestamps_match_the_samples_they_came_from(self):
        # the stronger statement: the returned times are the dataset's own
        # times at the indices the data was read from
        dataset = self._make(dt=3, n_history=1, add_zenith=False)

        inp_time = dataset[5][INPUT_TIMESTAMP]

        expected = np.asarray([dataset.timestamps[5], dataset.timestamps[5 + 3]])
        self.assertTrue(compare_arrays("input timestamps", inp_time.numpy(), expected))

    def test_target_times_stride_by_dt(self):
        dataset = self._make(dt=2, n_future=1, add_zenith=False)

        tar_time = dataset[0][TARGET_TIMESTAMP]

        self.assertEqual(float(tar_time[1] - tar_time[0]), 2 * dataset.dhours * 3600.0)

    def test_survives_pickling(self):
        # a torch DataLoader with workers pickles the dataset; the backend has
        # to drop its handles and reopen them on the other side
        import pickle

        dataset = self._make()
        before = dataset[3]

        revived = pickle.loads(pickle.dumps(dataset))
        after = revived[3]

        self.assertEqual(set(after), set(before))
        for key in (INPUT, TARGET, INPUT_TIMESTAMP, TARGET_TIMESTAMP):
            with self.subTest(output=key):
                self.assertTrue(compare_tensors(f"unpickled {key}", after[key], before[key]))

    def test_reports_the_grid_the_files_carry(self):
        # the fixture writes its coordinates, so they should be what comes back
        dataset = self._make()

        with h5.File(sorted(glob.glob(os.path.join(self.train_path, "*.h5")))[0], "r") as handle:
            latitude = handle["lat"][...]

        self.assertTrue(compare_arrays("latitudes", np.asarray(dataset.lat_lon[0]), latitude, rtol=1e-6))

    def test_lookup_by_time_round_trips(self):
        # inference addresses samples by timestamp, so the two directions have
        # to agree
        dataset = self._make()

        for index in (0, 17, 400):
            with self.subTest(index=index):
                self.assertEqual(dataset.get_index_at_time(dataset.get_time_at_index(index)), index)


class TestMultifilesWb2(unittest.TestCase):
    """The WeatherBench2 layout, read through the torch dataset.

    Reachable only since the dataset moved onto the backends and started
    passing ``channel_names`` -- which the layout needs, since it addresses
    variables by name rather than by index. Inference uses this dataset, so
    this is what makes a WB2 store usable there.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.train_path, *_ = init_wb2_zarr_dataset(cls.tmpdir.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _make(self, **overrides):
        from makani.utils.dataloaders.data_loader_multifiles import MultifilesDataset

        kwargs = dict(
            location=self.train_path,
            dt=1,
            in_channels=list(range(NUM_CHANNELS)),
            out_channels=list(range(NUM_CHANNELS)),
            n_history=0,
            n_future=0,
            channel_names=list(CHANNEL_NAMES),
            enable_logging=False,
        )
        kwargs.update(overrides)
        return MultifilesDataset(**kwargs)

    def test_reads_the_same_values_as_the_store(self):
        dataset = self._make()

        inp = dataset[2][INPUT]
        expected = get_wb2_zarr_sample(self.train_path, 2)

        self.assertTrue(compare_arrays("wb2 through multifiles", inp[0].numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_without_channel_names_it_says_so(self):
        # the layout cannot be read by index, and the message should name what
        # is missing rather than what was not found
        with self.assertRaises(ValueError) as ctx:
            self._make(channel_names=None)
        self.assertIn("channel_names", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
