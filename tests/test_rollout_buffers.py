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

import unittest
from parameterized import parameterized, parameterized_class
import tempfile
import os
import sys
import numpy as np
import torch
import h5py as h5
from typing import Optional

import torch_harmonics as th

from makani.utils.inference.rollout_buffer import (
    MeanStdBuffer,
    RolloutBuffer,
    TemporalAverageBuffer,
    SpectrumAverageBuffer,
    ZonalSpectrumAverageBuffer,
)
from makani.utils.dataloaders.data_helpers import get_lat_lon_grid

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import (
    disable_tf32,
    set_seed,
    init_hdf5_dataset,
    get_default_parameters,
    compare_arrays,
    compare_tensors,
    H5_PATH,
    IMG_SIZE_H,
    IMG_SIZE_W,
)

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


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
    """Initialize dataset parameters using the same approach as test_dataloader.py"""

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


@parameterized_class(("device",), _devices)
class TestTemporalAverageBufferIntegration(unittest.TestCase):
    """
    End-to-end test for ``TemporalAverageBuffer``: feeds real HDF5 data through
    the buffer (update + finalize), writes the output, reads it back, and
    compares against numpy-computed mean/std. Complements
    ``TestTemporalAverageBufferUnit`` (in-memory, no I/O) below.

    Named ``Integration`` rather than ``...Buffers`` (plural) — the prior name
    misleadingly suggested coverage of all rollout-buffer classes when in fact
    only ``TemporalAverageBuffer`` is exercised here. The actual ``RolloutBuffer``
    data-dump class has its own ``TestRolloutBuffer`` (singular) elsewhere.
    """

    @classmethod
    def setUpClass(cls, path: Optional[str] = "/tmp"):
        """Set up test environment using class-level setup like test_dataloader.py"""

        # Set random seed for reproducibility
        set_seed(333)

        # create temporary directory
        cls.tmpdir = tempfile.TemporaryDirectory(dir=path)
        tmp_path = cls.tmpdir.name

        # init datasets and stats using the same approach as test_dataloader.py
        cls.train_path, cls.num_train, cls.valid_path, cls.num_valid, cls.stats_path, cls.metadata_path, _ = (
            init_hdf5_dataset(tmp_path)
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after test using class-level teardown"""
        cls.tmpdir.cleanup()

    def setUp(self):
        """Set up test parameters for each test method"""

        disable_tf32()

        # Initialize parameters using the same approach as test_dataloader.py
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

        # Set rollout parameters
        self.rollout_dt = 6  # 6 hours
        self.ensemble_size = 4

        # Channel configuration - use the same channels as testutils
        self.channel_names = ["u10m", "t2m", "u500", "z500", "t500"]
        self.output_channels = ["u10m", "t2m", "u500"]  # Only track some channels
        self.num_channels = len(self.channel_names)
        self.num_output_channels = len(self.output_channels)

        # Set image dimensions - use the same as testutils
        self.img_shape = (IMG_SIZE_H, IMG_SIZE_W)  # (lat, lon)
        self.local_shape = (IMG_SIZE_H, IMG_SIZE_W)  # Same as img_shape for non-distributed test
        self.local_offset = (0, 0)

        # Create lat/lon grid
        self.latitude, self.longitude = get_lat_lon_grid(self.img_shape)
        self.lat_lon = (self.latitude.tolist(), self.longitude.tolist())

    @parameterized.expand(
        [
            (1, 1, False),
            (2, 1, False),
            (4, 1, False),
            (1, 2, False),
            (2, 2, False),
            (4, 2, False),
            (1, 1, True),
            (2, 1, True),
            (4, 1, True),
            (1, 2, True),
            (2, 2, True),
            (4, 2, True),
        ],
        skip_on_empty=True,
    )
    def test_temporal_averaging_buffer(self, batch_size, num_rollout_steps, scale_bias, verbose=False):
        """
        Test TemporalAverageBuffer by feeding data one tensor at a time and comparing
        with manual mean and variance calculations
        """
        # Create output file path
        output_file = os.path.join(self.tmpdir.name, "temporal_average_output.h5")

        if not scale_bias:
            scale = None
            bias = None
        else:
            scale = torch.ones((self.num_channels,), dtype=torch.float32)
            bias = torch.zeros((self.num_channels,), dtype=torch.float32)

        # Initialize TemporalAverageBuffer
        buffer = TemporalAverageBuffer(
            num_rollout_steps=num_rollout_steps,
            rollout_dt=6,
            img_shape=self.img_shape,
            local_shape=self.local_shape,
            local_offset=self.local_offset,
            channel_names=self.channel_names,
            lat_lon=self.lat_lon,
            device=self.device,
            output_channels=self.output_channels,
            output_file=output_file,
            scale=scale,
            bias=bias,
        )

        # Load test data from the dummy dataset
        test_file = os.path.join(self.valid_path, "2019.h5")
        with h5.File(test_file, "r") as hf:
            # Load all data from the test file
            data = hf[H5_PATH][:]  # Shape: (num_samples, num_channels, lat, lon)

        # Convert to torch tensor
        data_tensor = torch.from_numpy(data).to(self.device)

        # Get channel indices for output channels
        output_channel_indices = [self.channel_names.index(ch) for ch in self.output_channels]

        # Prepare manual calculation arrays
        manual_data = []  # Will store all data for manual calculation

        # Feed data one tensor at a time to the buffer
        num_samples = (data_tensor.shape[0] // batch_size // num_rollout_steps) * num_rollout_steps * batch_size

        for idt, step in enumerate(range(0, num_samples, batch_size)):
            # Extract single sample and reshape for buffer
            # Buffer expects: (batch_size, num_channels, lat, lon)
            sample = data_tensor[step : step + batch_size, ...]  # Add batch and ensemble dimension

            # Update buffer
            idte = idt % num_rollout_steps
            buffer.update(sample, idte)

            # Store for manual calculation
            manual_data.append(sample[:, output_channel_indices, :, :].cpu().numpy())

        # Finalize buffer
        buffer.finalize()

        # Manual calculation
        manual_data = np.stack(manual_data, axis=0)
        manual_data = manual_data.reshape(-1, num_rollout_steps, batch_size, self.num_output_channels, *self.img_shape)
        manual_data = np.transpose(manual_data, axes=(0, 2, 1, 3, 4, 5)).reshape(
            -1, num_rollout_steps, self.num_output_channels, *self.img_shape
        )

        # Calculate manual mean and std for output channels only
        manual_mean = np.mean(manual_data, axis=0)
        manual_std = np.std(manual_data, axis=0, ddof=1)  # ddof=1 for sample std

        # Read results from HDF5 file
        with h5.File(output_file, "r") as hf:
            buffer_mean = hf["mean"][:]  # Shape: (num_rollout_steps, num_channels, lat, lon)
            buffer_std = hf["std"][:]  # Shape: (num_rollout_steps, num_channels, lat, lon)
            lead_time = hf["lead_time"][:]
            channels = [x.decode() for x in hf["channel"][:]]
            lats = hf["lat"][:]
            lons = hf["lon"][:]

        # Verify file structure
        with self.subTest(desc="buffer shapes"):
            self.assertEqual(buffer_mean.shape, (num_rollout_steps, len(self.output_channels), *self.img_shape))
            self.assertEqual(buffer_std.shape, (num_rollout_steps, len(self.output_channels), *self.img_shape))
            self.assertEqual(len(lead_time), num_rollout_steps)
            self.assertEqual(len(channels), len(self.output_channels))
            self.assertEqual(len(lats), self.img_shape[0])
            self.assertEqual(len(lons), self.img_shape[1])

        # Verify channel names
        with self.subTest(desc="channel names"):
            self.assertEqual(channels, self.output_channels)

        # Verify lead times
        expected_lead_times = np.arange(
            self.rollout_dt, (num_rollout_steps + 1) * self.rollout_dt, self.rollout_dt, dtype=np.float64
        )
        with self.subTest(desc="lead times"):
            self.assertTrue(
                compare_arrays("lead times", lead_time, expected_lead_times, atol=0.0, rtol=1e-6, verbose=verbose)
            )

        # Verify lat/lon coordinates
        with self.subTest(desc="latitudes"):
            self.assertTrue(compare_arrays("latitudes", lats, self.latitude, atol=0.0, rtol=1e-6, verbose=verbose))
        with self.subTest(desc="longitudes"):
            self.assertTrue(compare_arrays("longitudes", lons, self.longitude, atol=0.0, rtol=1e-6, verbose=verbose))

        # Compare with buffer output
        with self.subTest(desc="mean"):
            self.assertTrue(compare_arrays("mean", buffer_mean, manual_mean, atol=0.0, rtol=1e-5, verbose=verbose))
        with self.subTest(desc="std"):
            self.assertTrue(compare_arrays("std", buffer_std, manual_std, atol=0.0, rtol=1e-5, verbose=verbose))


class TestMeanStdBuffer(unittest.TestCase):
    """
    Tests the Welford online accumulator that all averaging buffers inherit from.
    Bugs in this math produce silent skill-metric regressions, so we check it
    against torch's offline mean/m2 on the concatenated batch.

    ``MeanStdBuffer`` is abstract (inherits the ``DataBuffer`` ABC), so we use
    a thin concrete subclass that exposes the protected math methods through
    ``update``.
    """

    @staticmethod
    def _make_buffer(num_rollout_steps, num_channels, variable_shape, scale=None, bias=None):
        class _Concrete(MeanStdBuffer):
            def update(self, data, idt):
                mean, m2, count = self._compute_stats(data, dim=0)
                self._welford_combine(mean, m2, count, idt)

            def finalize(self):
                pass

        channel_names = [f"ch_{i}" for i in range(num_channels)]
        return _Concrete(
            num_rollout_steps=num_rollout_steps,
            rollout_dt=1,
            variable_shape=variable_shape,
            channel_names=channel_names,
            device="cpu",
            scale=scale,
            bias=bias,
            output_channels=channel_names,  # output all channels
            output_file=None,
        )

    def test_compute_stats_matches_torch(self, verbose=False):
        buf = self._make_buffer(num_rollout_steps=1, num_channels=2, variable_shape=(4, 4))
        data = torch.randn(10, 2, 4, 4)
        mean, m2, count = buf._compute_stats(data, dim=0)
        expected_mean = data.mean(dim=0)
        expected_m2 = (data - expected_mean).pow(2).sum(dim=0)

        with self.subTest(desc="mean"):
            self.assertTrue(compare_tensors("compute_stats mean", mean, expected_mean, atol=1e-6, verbose=verbose))
        with self.subTest(desc="m2"):
            self.assertTrue(compare_tensors("compute_stats m2", m2, expected_m2, atol=1e-4, verbose=verbose))
        with self.subTest(desc="count"):
            self.assertEqual(count.item(), 10)

    def test_single_batch_welford(self, verbose=False):
        # Welford from one batch should reproduce the batch's mean and m2 directly.
        buf = self._make_buffer(num_rollout_steps=1, num_channels=3, variable_shape=(4, 4))
        torch.manual_seed(0)
        data = torch.randn(8, 3, 4, 4)
        buf.update(data, idt=0)

        expected_mean = data.mean(dim=0)
        expected_m2 = (data - expected_mean).pow(2).sum(dim=0)

        with self.subTest(desc="mean"):
            self.assertTrue(
                compare_tensors("running_mean", buf.running_mean[0], expected_mean, atol=1e-5, verbose=verbose)
            )
        with self.subTest(desc="m2"):
            self.assertTrue(compare_tensors("running_var", buf.running_var[0], expected_m2, atol=1e-4, verbose=verbose))
        with self.subTest(desc="count"):
            self.assertEqual(buf.num_samples_tracked[0].item(), 8)

    def test_two_batch_welford_matches_concatenated(self, verbose=False):
        # Online Welford on two unequal batches must match offline mean/m2 on
        # their concatenation. This is the core invariant.
        buf = self._make_buffer(num_rollout_steps=1, num_channels=2, variable_shape=(3, 5))
        torch.manual_seed(7)
        data1 = torch.randn(5, 2, 3, 5)
        data2 = torch.randn(7, 2, 3, 5)
        buf.update(data1, idt=0)
        buf.update(data2, idt=0)

        joined = torch.cat([data1, data2], dim=0)
        expected_mean = joined.mean(dim=0)
        expected_m2 = (joined - expected_mean).pow(2).sum(dim=0)

        with self.subTest(desc="mean"):
            self.assertTrue(
                compare_tensors("two-batch mean", buf.running_mean[0], expected_mean, atol=1e-5, verbose=verbose)
            )
        with self.subTest(desc="m2"):
            self.assertTrue(
                compare_tensors("two-batch m2", buf.running_var[0], expected_m2, atol=1e-3, verbose=verbose)
            )
        with self.subTest(desc="count"):
            self.assertEqual(buf.num_samples_tracked[0].item(), 12)

    def test_three_batch_welford(self, verbose=False):
        # More batches to confirm the recurrence holds beyond the easy 2-batch case.
        buf = self._make_buffer(num_rollout_steps=1, num_channels=1, variable_shape=(3, 3))
        torch.manual_seed(42)
        batches = [torch.randn(b, 1, 3, 3) for b in [3, 5, 7]]
        for b in batches:
            buf.update(b, idt=0)

        joined = torch.cat(batches, dim=0)
        expected_mean = joined.mean(dim=0)
        expected_m2 = (joined - expected_mean).pow(2).sum(dim=0)

        with self.subTest(desc="mean"):
            self.assertTrue(
                compare_tensors("three-batch mean", buf.running_mean[0], expected_mean, atol=1e-5, verbose=verbose)
            )
        with self.subTest(desc="m2"):
            self.assertTrue(
                compare_tensors("three-batch m2", buf.running_var[0], expected_m2, atol=1e-3, verbose=verbose)
            )
        with self.subTest(desc="count"):
            self.assertEqual(buf.num_samples_tracked[0].item(), 15)

    def test_rollout_steps_track_independently(self, verbose=False):
        # Each rollout step has its own running stats — updating one must
        # leave the others zero.
        buf = self._make_buffer(num_rollout_steps=3, num_channels=2, variable_shape=(4, 4))
        buf.update(torch.randn(4, 2, 4, 4), idt=1)

        with self.subTest(desc="counts"):
            self.assertEqual(buf.num_samples_tracked[0].item(), 0)
            self.assertEqual(buf.num_samples_tracked[1].item(), 4)
            self.assertEqual(buf.num_samples_tracked[2].item(), 0)
        with self.subTest(desc="mean step 0 untouched"):
            self.assertTrue(
                compare_tensors(
                    "step 0 mean", buf.running_mean[0], torch.zeros_like(buf.running_mean[0]), verbose=verbose
                )
            )
        with self.subTest(desc="mean step 2 untouched"):
            self.assertTrue(
                compare_tensors(
                    "step 2 mean", buf.running_mean[2], torch.zeros_like(buf.running_mean[2]), verbose=verbose
                )
            )

    def test_zero_buffers_resets_all_state(self, verbose=False):
        buf = self._make_buffer(num_rollout_steps=2, num_channels=2, variable_shape=(4, 4))
        buf.update(torch.randn(5, 2, 4, 4), idt=0)
        buf.update(torch.randn(5, 2, 4, 4), idt=1)
        self.assertGreater(buf.num_samples_tracked.sum().item(), 0)

        buf.zero_buffers()

        with self.subTest(desc="counts"):
            self.assertEqual(buf.num_samples_tracked.sum().item(), 0)
        with self.subTest(desc="running_mean reset"):
            self.assertTrue(
                compare_tensors("running_mean", buf.running_mean, torch.zeros_like(buf.running_mean), verbose=verbose)
            )
        with self.subTest(desc="running_var reset"):
            self.assertTrue(
                compare_tensors("running_var", buf.running_var, torch.zeros_like(buf.running_var), verbose=verbose)
            )


class TestTemporalAverageBufferUnit(unittest.TestCase):
    """
    In-memory unit tests for TemporalAverageBuffer that don't touch disk and
    don't depend on the dataset fixture. Complements ``TestRolloutBuffers``,
    which is the integration-style test that loads real HDF5 data and writes
    to an output file.
    """

    @staticmethod
    def _make_buffer(num_rollout_steps, channel_names, output_channels, img_shape, scale=None, bias=None):
        H, W = img_shape
        return TemporalAverageBuffer(
            num_rollout_steps=num_rollout_steps,
            rollout_dt=1,
            img_shape=(H, W),
            local_shape=(H, W),
            local_offset=(0, 0),
            channel_names=channel_names,
            lat_lon=(list(range(H)), list(range(W))),
            device="cpu",
            scale=scale,
            bias=bias,
            output_channels=output_channels,
            output_file=None,
        )

    def test_update_subset_of_channels(self, verbose=False):
        # output_channels selects a subset; only those channels are accumulated.
        # 'a' (index 0) and 'c' (index 2) but not 'b' (index 1).
        channel_names = ["a", "b", "c"]
        output_channels = ["a", "c"]
        buf = self._make_buffer(
            num_rollout_steps=2,
            channel_names=channel_names,
            output_channels=output_channels,
            img_shape=(4, 8),
        )

        torch.manual_seed(1)
        data1 = torch.randn(3, 3, 4, 8)
        data2 = torch.randn(2, 3, 4, 8)
        buf.update(data1, idt=0)
        buf.update(data2, idt=0)

        joined = torch.cat([data1, data2], dim=0)
        sub = joined[:, [0, 2], :, :]
        expected_mean = sub.mean(dim=0)
        expected_m2 = (sub - expected_mean).pow(2).sum(dim=0)

        with self.subTest(desc="mean"):
            self.assertTrue(
                compare_tensors("subset mean", buf.running_mean[0], expected_mean, atol=1e-5, verbose=verbose)
            )
        with self.subTest(desc="m2"):
            self.assertTrue(compare_tensors("subset m2", buf.running_var[0], expected_m2, atol=1e-3, verbose=verbose))
        with self.subTest(desc="count"):
            self.assertEqual(buf.num_samples_tracked[0].item(), 5)

    def test_scale_and_bias_applied_before_averaging(self, verbose=False):
        # scale and bias are applied per-channel before the Welford update.
        # With constant data == 1, the per-channel mean must be scale*1 + bias.
        channel_names = ["a", "b"]
        output_channels = ["a", "b"]
        scale = torch.tensor([2.0, 0.5])
        bias = torch.tensor([1.0, -1.0])
        buf = self._make_buffer(
            num_rollout_steps=1,
            channel_names=channel_names,
            output_channels=output_channels,
            img_shape=(2, 2),
            scale=scale,
            bias=bias,
        )

        data = torch.ones(4, 2, 2, 2)
        buf.update(data, idt=0)

        # mean per channel = scale*1 + bias = [3.0, -0.5], spatially constant.
        with self.subTest(desc="ch_a mean = scale*1 + bias"):
            self.assertTrue(
                compare_tensors("ch_a", buf.running_mean[0, 0], 3.0 * torch.ones(2, 2), atol=1e-6, verbose=verbose)
            )
        with self.subTest(desc="ch_b mean = scale*1 + bias"):
            self.assertTrue(
                compare_tensors("ch_b", buf.running_mean[0, 1], -0.5 * torch.ones(2, 2), atol=1e-6, verbose=verbose)
            )
        with self.subTest(desc="m2 zero for constant input"):
            # constant input → zero variance → m2 = 0
            self.assertTrue(
                compare_tensors("zero m2", buf.running_var[0], torch.zeros_like(buf.running_var[0]), verbose=verbose)
            )

    def test_rollout_steps_independent(self, verbose=False):
        # Updates to different rollout steps must not contaminate each other.
        channel_names = ["a"]
        buf = self._make_buffer(
            num_rollout_steps=3,
            channel_names=channel_names,
            output_channels=channel_names,
            img_shape=(2, 2),
        )

        # Step 0: known constant 1.0
        buf.update(torch.ones(4, 1, 2, 2), idt=0)
        # Step 2: known constant 5.0
        buf.update(5.0 * torch.ones(4, 1, 2, 2), idt=2)

        with self.subTest(desc="step 0"):
            self.assertTrue(
                compare_tensors("step 0", buf.running_mean[0], torch.ones_like(buf.running_mean[0]), verbose=verbose)
            )
        with self.subTest(desc="step 1 untouched"):
            self.assertTrue(
                compare_tensors("step 1", buf.running_mean[1], torch.zeros_like(buf.running_mean[1]), verbose=verbose)
            )
        with self.subTest(desc="step 2"):
            self.assertTrue(
                compare_tensors(
                    "step 2", buf.running_mean[2], 5.0 * torch.ones_like(buf.running_mean[2]), verbose=verbose
                )
            )
        with self.subTest(desc="step 1 count"):
            self.assertEqual(buf.num_samples_tracked[1].item(), 0)

    def test_finalize_no_output_file_runs_without_error(self, verbose=False):
        # finalize() with no output_file should compute std and return cleanly.
        # std = sqrt(running_var / (n - 1)) — sample-corrected.
        channel_names = ["a"]
        buf = self._make_buffer(
            num_rollout_steps=1,
            channel_names=channel_names,
            output_channels=channel_names,
            img_shape=(2, 2),
        )

        # Two-sample batch with values [-1, +1] → mean=0, m2 = 2 (two unit deltas).
        data = torch.tensor([[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]]).reshape(2, 1, 2, 2)
        buf.update(data, idt=0)

        with self.subTest(desc="post-update mean"):
            self.assertTrue(compare_tensors("mean", buf.running_mean[0], torch.zeros(2, 2), atol=1e-6, verbose=verbose))
        with self.subTest(desc="post-update m2"):
            self.assertTrue(
                compare_tensors("m2", buf.running_var[0], 2.0 * torch.ones(2, 2), atol=1e-6, verbose=verbose)
            )

        # finalize is in-place math + (no-op write) — should not raise
        buf.finalize()


def _spectrum_buffer_inputs(fields, channel_names, output_channels, scale, bias, generator):
    """
    Build the (data, targ) pair a spectrum buffer is updated with from the fields it should see.

    ``fields`` has shape (batch, 1 + ensemble, len(output_channels), nlat, nlon) and holds the target
    followed by the ensemble members, in the units after the buffer applies ``scale * x + bias``. The
    inputs are those fields mapped back through the inverse transform; the channels that are not
    recorded carry noise, so that recording the wrong channel shows.
    """
    B, E1, _, H, W = fields.shape
    full = torch.randn(B, E1, len(channel_names), H, W, generator=generator, dtype=fields.dtype)
    for k, cname in enumerate(output_channels):
        c = channel_names.index(cname)
        full[:, :, c] = (fields[:, :, k] - bias[c]) / scale[c]
    return full[:, 1:], full[:, :1]


def _welford_reference(samples):
    """Mean and sum of squared deviations over the sample dimension, the state the Welford buffers hold."""
    samples = torch.cat(samples, dim=0)
    mean = samples.mean(dim=0)
    m2 = (samples - mean).square().sum(dim=0)
    return mean, m2, samples.shape[0]


class TestSpectrumAverageBufferUnit(unittest.TestCase):
    """
    SpectrumAverageBuffer accumulates the angular power spectrum of the target and of every ensemble
    member per rollout step. The fields are synthesized from known spherical harmonic coefficients on a
    Legendre-Gauss grid, where the SHT of a band-limited field is exact, so the expected spectrum

        P_l = |c_l0|^2 + 2 sum_{m>0} |c_lm|^2

    follows from the coefficients without a forward transform. With orthonormal harmonics this is the
    spectrum whose sum over l is the squared L2 norm of the field on the sphere (Parseval).
    """

    nlat, nlon = 16, 32
    grid = "legendre-gauss"

    def setUp(self):
        self.isht = th.InverseRealSHT(self.nlat, self.nlon, grid=self.grid)
        self.lmax, self.mmax = self.isht.lmax, self.isht.mmax
        self.generator = torch.Generator().manual_seed(4321)

    def _make_buffer(
        self, num_rollout_steps, ensemble_size, channel_names, output_channels, scale=None, bias=None, output_file=None
    ):
        return SpectrumAverageBuffer(
            num_rollout_steps=num_rollout_steps,
            rollout_dt=6,
            img_shape=(self.nlat, self.nlon),
            ensemble_size=ensemble_size,
            grid_type=self.grid,
            channel_names=channel_names,
            device="cpu",
            scale=scale,
            bias=bias,
            output_channels=output_channels,
            output_file=output_file,
        )

    def _random_coeffs(self, *shape):
        coeffs = torch.complex(
            torch.randn(*shape, self.lmax, self.mmax, generator=self.generator),
            torch.randn(*shape, self.lmax, self.mmax, generator=self.generator),
        )
        # a real field has no m > l modes and a real m = 0 mode
        l = torch.arange(self.lmax).reshape(-1, 1)
        m = torch.arange(self.mmax).reshape(1, -1)
        coeffs = torch.where(m <= l, coeffs, torch.zeros_like(coeffs))
        coeffs[..., 0] = torch.complex(coeffs[..., 0].real, torch.zeros_like(coeffs[..., 0].real))
        return coeffs

    @staticmethod
    def _power_spectrum(coeffs):
        # the orders m > 0 stand for m and -m of the real field
        power = coeffs.abs().square()
        power[..., 1:] *= 2.0
        # (batch, 1 + ensemble, channel, l) -> (batch, channel, 1 + ensemble, l), the buffer layout
        return power.sum(dim=-1).permute(0, 2, 1, 3)

    def _update_sequence(self, buf, batch_sizes, ensemble_size, channel_names, output_channels, scale, bias):
        """Update the buffer with random band-limited fields and return the expected spectra per rollout step."""
        expected = {}
        for idt, B in batch_sizes:
            coeffs = self._random_coeffs(B, 1 + ensemble_size, len(output_channels))
            fields = self.isht(coeffs)
            data, targ = _spectrum_buffer_inputs(fields, channel_names, output_channels, scale, bias, self.generator)
            buf.update(data, targ, idt)
            expected.setdefault(idt, []).append(self._power_spectrum(coeffs))
        return {idt: _welford_reference(spectra) for idt, spectra in expected.items()}

    def test_single_harmonic(self, verbose=False):
        # one degree with m > 0 and one with m = 0, in the target and the single ensemble member
        coeffs = torch.zeros(1, 2, 1, self.lmax, self.mmax, dtype=torch.complex64)
        coeffs[..., 5, 3] = 1.0
        coeffs[..., 2, 0] = 0.5
        fields = self.isht(coeffs)

        buf = self._make_buffer(1, 1, ["a"], ["a"])
        buf.update(fields[:, 1:], fields[:, :1], 0)

        expected = torch.zeros(1, 2, self.lmax)
        expected[..., 5] = 2.0
        expected[..., 2] = 0.25
        with self.subTest(desc="spectrum"):
            self.assertTrue(
                compare_tensors("spectrum", buf.running_mean[0], expected, atol=1e-5, rtol=1e-5, verbose=verbose)
            )

        # Parseval with the Legendre-Gauss quadrature, independent of the SHT: the latitude weights are
        # symmetric, so the order of the latitudes does not matter
        _, weights = np.polynomial.legendre.leggauss(self.nlat)
        weights = torch.from_numpy(weights).to(torch.float64) * 2.0 * np.pi / self.nlon
        norm2 = torch.einsum("...hw,h->...", fields.to(torch.float64).square(), weights)
        with self.subTest(desc="Parseval"):
            self.assertTrue(
                compare_tensors(
                    "Parseval",
                    buf.running_mean[0].sum(dim=-1).to(torch.float64),
                    norm2.reshape(1, 2),
                    atol=1e-5,
                    rtol=1e-5,
                    verbose=verbose,
                )
            )

    def test_statistics_with_scale_bias_and_channel_subset(self, verbose=False):
        # a reordered channel subset, so that a wrong channel mask or an ignored bias changes the spectra
        channel_names = ["a", "b", "c"]
        output_channels = ["c", "a"]
        scale = torch.tensor([2.0, 0.5, 4.0])
        bias = torch.tensor([1.0, -3.0, 0.25])
        ensemble_size = 2
        buf = self._make_buffer(
            2,
            ensemble_size,
            channel_names,
            output_channels,
            scale=scale.reshape(1, -1, 1, 1),
            bias=bias.reshape(1, -1, 1, 1),
        )

        # two batches of different size for the first rollout step, one for the second
        expected = self._update_sequence(
            buf, [(0, 3), (0, 2), (1, 4)], ensemble_size, channel_names, output_channels, scale, bias
        )

        for idt, (mean, m2, count) in expected.items():
            with self.subTest(desc=f"mean step {idt}"):
                self.assertTrue(
                    compare_tensors(
                        f"mean step {idt}", buf.running_mean[idt], mean, atol=1e-4, rtol=1e-4, verbose=verbose
                    )
                )
            with self.subTest(desc=f"m2 step {idt}"):
                self.assertTrue(
                    compare_tensors(f"m2 step {idt}", buf.running_var[idt], m2, atol=1e-3, rtol=1e-4, verbose=verbose)
                )
            with self.subTest(desc=f"count step {idt}"):
                self.assertEqual(buf.num_samples_tracked[idt].item(), count)

    def test_finalize_writes_mean_and_std(self, verbose=False):
        channel_names = ["a", "b"]
        output_channels = ["b", "a"]
        scale = torch.ones(2)
        bias = torch.zeros(2)
        ensemble_size = 2
        num_rollout_steps = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "spectrum.h5")
            buf = self._make_buffer(
                num_rollout_steps, ensemble_size, channel_names, output_channels, output_file=output_file
            )
            expected = self._update_sequence(
                buf, [(0, 3), (1, 2), (1, 2)], ensemble_size, channel_names, output_channels, scale, bias
            )
            buf.finalize()

            with h5.File(output_file, "r") as f:
                mean = torch.from_numpy(f["mean"][...])
                std = torch.from_numpy(f["std"][...])
                lead_time = f["lead_time"][...]
                channels = [c.decode() if isinstance(c, bytes) else str(c) for c in f["channel"][...]]

        with self.subTest(desc="shape"):
            self.assertEqual(tuple(mean.shape), (num_rollout_steps, 2, ensemble_size + 1, self.lmax))
        for idt, (ref_mean, ref_m2, count) in expected.items():
            ref_std = torch.sqrt(ref_m2 / (count - 1))
            with self.subTest(desc=f"mean step {idt}"):
                self.assertTrue(
                    compare_tensors(
                        f"mean step {idt}", mean[idt], ref_mean.double(), atol=1e-4, rtol=1e-4, verbose=verbose
                    )
                )
            with self.subTest(desc=f"std step {idt}"):
                self.assertTrue(
                    compare_tensors(
                        f"std step {idt}", std[idt], ref_std.double(), atol=1e-4, rtol=1e-4, verbose=verbose
                    )
                )
        with self.subTest(desc="lead time"):
            self.assertTrue(np.array_equal(lead_time, np.array([6.0, 12.0])))
        with self.subTest(desc="channels"):
            self.assertEqual(channels, output_channels)

    def test_finalize_without_output_file(self, verbose=False):
        buf = self._make_buffer(1, 1, ["a"], ["a"])
        expected = self._update_sequence(buf, [(0, 2)], 1, ["a"], ["a"], torch.ones(1), torch.zeros(1))
        buf.finalize()

        # nothing to write, and the accumulated statistics stay as they are
        mean, _, _ = expected[0]
        self.assertTrue(compare_tensors("mean", buf.running_mean[0], mean, atol=1e-4, rtol=1e-4, verbose=verbose))


class TestZonalSpectrumAverageBufferUnit(unittest.TestCase):
    """
    ZonalSpectrumAverageBuffer accumulates, per latitude, the one-sided power spectrum of the field
    along the longitudes, weighted by the cosine of the latitude. With the Fourier coefficients
    a_m = 1/N sum_k f_k exp(-2 pi i m k / N), the one-sided spectrum counts every order twice except
    the mean and, for an even number of longitudes N, the Nyquist order m = N / 2, which is its own
    conjugate. That makes the spectrum sum to the mean of f^2 over the longitudes (Parseval).
    """

    def setUp(self):
        self.generator = torch.Generator().manual_seed(1234)

    @staticmethod
    def _lat_lon(nlat, nlon):
        lats = np.linspace(75.0, -75.0, nlat).tolist()
        lons = np.linspace(0.0, 360.0, nlon, endpoint=False).tolist()
        return lats, lons

    def _make_buffer(
        self, nlat, nlon, num_rollout_steps, ensemble_size, channel_names, output_channels, scale=None, bias=None
    ):
        return ZonalSpectrumAverageBuffer(
            num_rollout_steps=num_rollout_steps,
            rollout_dt=6,
            img_shape=(nlat, nlon),
            ensemble_size=ensemble_size,
            channel_names=channel_names,
            lat_lon=self._lat_lon(nlat, nlon),
            device="cpu",
            scale=scale,
            bias=bias,
            output_channels=output_channels,
            output_file=None,
        )

    @staticmethod
    def _zonal_power(fields, lats):
        nlon = fields.shape[-1]
        coeffs = torch.fft.rfft(fields.to(torch.float64), dim=-1) / nlon
        power = coeffs.abs().square()
        power[..., 1:] *= 2.0
        if nlon % 2 == 0:
            power[..., nlon // 2] /= 2.0
        weights = torch.cos(torch.deg2rad(torch.tensor(lats, dtype=torch.float64))).reshape(-1, 1)
        # (batch, 1 + ensemble, channel, lat, m) -> (batch, channel, 1 + ensemble, lat, m), the buffer layout
        return (weights * power).permute(0, 2, 1, 3, 4)

    def _cos_lat(self, nlat):
        lats, _ = self._lat_lon(nlat, 1)
        return torch.cos(torch.deg2rad(torch.tensor(lats, dtype=torch.float64)))

    def test_single_wave(self, verbose=False):
        nlat, nlon = 4, 32
        phi = 2.0 * np.pi * torch.arange(nlon, dtype=torch.float32) / nlon
        # mean 0.5 and a wave of order 3 with amplitude 1: a_0 = 0.5 and a_3 = a_-3 = 0.5
        field = (0.5 + torch.cos(3.0 * phi)).expand(1, 2, 1, nlat, nlon)

        buf = self._make_buffer(nlat, nlon, 1, 1, ["a"], ["a"])
        buf.update(field[:, 1:], field[:, :1], 0)

        expected = torch.zeros(1, 2, nlat, nlon // 2 + 1, dtype=torch.float64)
        expected[..., 0] = 0.25 * self._cos_lat(nlat).reshape(-1)
        expected[..., 3] = 0.5 * self._cos_lat(nlat).reshape(-1)
        self.assertTrue(
            compare_tensors("spectrum", buf.running_mean[0].double(), expected, atol=1e-6, rtol=1e-5, verbose=verbose)
        )

    def test_nyquist_order_counted_once(self, verbose=False):
        # (-1)^k is the Nyquist wave of an even number of longitudes: a_{N/2} = 1, and its power is
        # the mean of f^2 = 1, not 2
        nlat, nlon = 4, 32
        field = torch.ones(nlon)
        field[1::2] = -1.0
        field = field.expand(1, 2, 1, nlat, nlon)

        buf = self._make_buffer(nlat, nlon, 1, 1, ["a"], ["a"])
        buf.update(field[:, 1:], field[:, :1], 0)

        expected = self._cos_lat(nlat).reshape(1, 1, -1).expand(1, 2, nlat)
        self.assertTrue(
            compare_tensors(
                "Nyquist power", buf.running_mean[0, ..., nlon // 2].double(), expected, atol=1e-6, verbose=verbose
            )
        )

    def test_parseval(self, verbose=False):
        # the spectrum of a random field sums to the cosine-weighted mean of f^2 over the longitudes,
        # for an even and an odd number of longitudes
        for nlon in [32, 31]:
            nlat = 6
            fields = torch.randn(1, 2, 1, nlat, nlon, generator=self.generator)
            buf = self._make_buffer(nlat, nlon, 1, 1, ["a"], ["a"])
            buf.update(fields[:, 1:], fields[:, :1], 0)

            expected = self._cos_lat(nlat) * fields.to(torch.float64).square().mean(dim=-1)
            expected = expected.permute(0, 2, 1, 3)[0]
            with self.subTest(desc=f"nlon={nlon}"):
                self.assertTrue(
                    compare_tensors(
                        f"Parseval nlon={nlon}",
                        buf.running_mean[0].sum(dim=-1).double(),
                        expected,
                        atol=1e-6,
                        rtol=1e-5,
                        verbose=verbose,
                    )
                )

    def test_statistics_with_scale_bias_and_channel_subset(self, verbose=False):
        # an odd number of longitudes has no Nyquist order, so this test is independent of how it is counted
        nlat, nlon = 6, 31
        lats, _ = self._lat_lon(nlat, nlon)
        channel_names = ["a", "b", "c"]
        output_channels = ["b", "c"]
        scale = torch.tensor([2.0, 0.5, 4.0])
        bias = torch.tensor([1.0, -3.0, 0.25])
        ensemble_size = 2
        buf = self._make_buffer(
            nlat,
            nlon,
            2,
            ensemble_size,
            channel_names,
            output_channels,
            scale=scale.reshape(1, -1, 1, 1),
            bias=bias.reshape(1, -1, 1, 1),
        )

        expected = {}
        for idt, B in [(0, 3), (0, 2), (1, 4)]:
            fields = torch.randn(B, 1 + ensemble_size, len(output_channels), nlat, nlon, generator=self.generator)
            data, targ = _spectrum_buffer_inputs(fields, channel_names, output_channels, scale, bias, self.generator)
            buf.update(data, targ, idt)
            expected.setdefault(idt, []).append(self._zonal_power(fields, lats))

        for idt, spectra in expected.items():
            mean, m2, count = _welford_reference(spectra)
            with self.subTest(desc=f"mean step {idt}"):
                self.assertTrue(
                    compare_tensors(
                        f"mean step {idt}", buf.running_mean[idt].double(), mean, atol=1e-5, rtol=1e-4, verbose=verbose
                    )
                )
            with self.subTest(desc=f"m2 step {idt}"):
                self.assertTrue(
                    compare_tensors(
                        f"m2 step {idt}", buf.running_var[idt].double(), m2, atol=1e-5, rtol=1e-4, verbose=verbose
                    )
                )
            with self.subTest(desc=f"count step {idt}"):
                self.assertEqual(buf.num_samples_tracked[idt].item(), count)


class TestRolloutBuffer(unittest.TestCase):
    """
    In-memory unit tests for ``RolloutBuffer`` — the data-dump buffer used to
    record forecast tensors during inference rollouts. None of these tests
    write to disk (``output_file=None``), so they exercise the pure logic of
    update/flush/zero in isolation:

      * update() routing into the flat (slot, ensemble, channel, lat, lon) buffer
      * buffer_offset advance by B slots per update() call
      * scale*x + bias projection
      * channel_mask channel selection
      * timestamp-only-at-idt=0
      * zero_buffers() data reset
      * auto-flush when a fresh update would overflow the buffer
      * output_memory_buffer_size defaults and clamping (step semantics)

    Runs on CPU end-to-end: ``_flush_buffer_to_disk``'s
    ``torch.cuda.synchronize`` is now device-guarded, so the auto-flush test
    no longer requires CUDA.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")

    @staticmethod
    def _make_buffer(
        *,
        num_samples=4,
        batch_size=2,
        num_rollout_steps=1,
        ensemble_size=1,
        img_shape=(2, 2),
        channel_names=("a",),
        output_channels=("a",),
        scale=None,
        bias=None,
        output_memory_buffer_size=None,
        device=None,
    ):
        if device is None:
            device = torch.device("cpu")
        H, W = img_shape
        return RolloutBuffer(
            num_samples=num_samples,
            batch_size=batch_size,
            num_rollout_steps=num_rollout_steps,
            rollout_dt=1,
            ensemble_size=ensemble_size,
            img_shape=(H, W),
            local_shape=(H, W),
            local_offset=(0, 0),
            channel_names=list(channel_names),
            lat_lon=(list(range(H)), list(range(W))),
            device=device,
            scale=scale,
            bias=bias,
            output_channels=list(output_channels),
            output_file=None,
            output_memory_buffer_size=output_memory_buffer_size,
        )

    def test_update_writes_to_correct_slot(self, verbose=False):
        # Channel mask should select 'a' (idx 0) and 'c' (idx 2), drop 'b' (idx 1).
        # The flat buffer has shape (num_steps, E, C', H, W). After one update at idt=0
        # with B=2, slots 0..1 hold the two ICs at idt=0; later slots are untouched.
        buf = self._make_buffer(
            num_samples=4,
            batch_size=2,
            num_rollout_steps=1,
            ensemble_size=2,
            img_shape=(4, 6),
            channel_names=("a", "b", "c"),
            output_channels=("a", "c"),
        )

        pred = torch.zeros(2, 2, 3, 4, 6)
        pred[..., 0, :, :] = 1.0  # 'a'
        pred[..., 1, :, :] = 2.0  # 'b' (NOT in output_channels, must be dropped)
        pred[..., 2, :, :] = 3.0  # 'c'
        tstamps = torch.tensor([100.0, 200.0])

        buf.update(pred, tstamps, idt=0)

        # buffer shape: (num_steps, E=2, C'=2, H=4, W=6); the two ICs at idt=0
        # occupy slots [0, 2).
        with self.subTest(desc="channel a values"):
            self.assertTrue(torch.all(buf.rollout_data[0:2, :, 0, :, :] == 1.0))
        with self.subTest(desc="channel c values"):
            self.assertTrue(torch.all(buf.rollout_data[0:2, :, 1, :, :] == 3.0))
        with self.subTest(desc="other slots untouched"):
            # remaining slots (those reserved for later leadtimes / batches) are still zero
            self.assertTrue(torch.all(buf.rollout_data[2:] == 0.0))

    def test_buffer_offset_advances_per_update(self, verbose=False):
        # With step-granular buffering, every update() consumes B slots regardless
        # of the leadtime — buffer_offset advances by batch_size on each call.
        # ``file_offset``, by contrast, advances only when a batch finishes its rollout.
        num_rollout_steps = 3
        batch_size = 2
        buf = self._make_buffer(
            num_samples=4,
            batch_size=batch_size,
            num_rollout_steps=num_rollout_steps,
        )
        file_offset_start = buf.file_offset

        pred = torch.zeros(batch_size, 1, 1, 2, 2)
        tstamps = torch.tensor([0.0, 1.0])

        with self.subTest(desc="initial offset"):
            self.assertEqual(buf.buffer_offset, 0)

        # Intermediate steps: buffer_offset advances by B, file_offset stays put.
        for idt in range(num_rollout_steps):
            buf.update(pred, tstamps, idt=idt)
            with self.subTest(desc=f"idt={idt} buffer_offset"):
                self.assertEqual(buf.buffer_offset, batch_size * (idt + 1))
            with self.subTest(desc=f"idt={idt} file_offset"):
                self.assertEqual(buf.file_offset, file_offset_start)

        # Final step (idt == num_rollout_steps): file_offset advances by batch_size,
        # buffer_offset advances one more time too.
        buf.update(pred, tstamps, idt=num_rollout_steps)
        with self.subTest(desc="post-final-step buffer_offset"):
            self.assertEqual(buf.buffer_offset, batch_size * (num_rollout_steps + 1))
        with self.subTest(desc="post-final-step file_offset"):
            self.assertEqual(buf.file_offset, file_offset_start + batch_size)

    def test_multi_step_rollout_fills_slots_in_order(self, verbose=False):
        # Each step writes a known constant. With B=1, the flat slot index equals idt:
        # slot[j*B + i] == (ic[i], idt_start + j), so for B=1 the j-th leadtime
        # of the single IC lands at slot j.
        num_rollout_steps = 3
        buf = self._make_buffer(
            num_samples=1,
            batch_size=1,
            num_rollout_steps=num_rollout_steps,
        )

        for idt in range(num_rollout_steps + 1):
            pred = torch.full((1, 1, 1, 2, 2), float(idt))
            buf.update(pred, torch.tensor([42.0]), idt=idt)

        for idt in range(num_rollout_steps + 1):
            with self.subTest(desc=f"slot {idt}"):
                slot = buf.rollout_data[idt]
                self.assertTrue(torch.all(slot == float(idt)))

    def test_scale_and_bias_applied(self, verbose=False):
        # With constant input ones, output per channel should equal scale*1 + bias.
        scale = torch.tensor([2.0, 3.0])
        bias = torch.tensor([1.0, -1.0])
        buf = self._make_buffer(
            num_samples=1,
            batch_size=1,
            num_rollout_steps=0,
            ensemble_size=1,
            channel_names=("a", "b"),
            output_channels=("a", "b"),
            scale=scale,
            bias=bias,
        )

        pred = torch.ones(1, 1, 2, 2, 2)
        buf.update(pred, torch.tensor([0.0]), idt=0)

        with self.subTest(desc="ch a = 2*1 + 1"):
            self.assertTrue(torch.all(buf.rollout_data[0, :, 0, :, :] == 3.0))
        with self.subTest(desc="ch b = 3*1 - 1"):
            self.assertTrue(torch.all(buf.rollout_data[0, :, 1, :, :] == 2.0))

    def test_default_scale_and_bias_are_identity(self, verbose=False):
        # When scale and bias are None, DataBuffer fills them with ones/zeros so
        # the projection is the identity. Verify input passes through unchanged.
        buf = self._make_buffer(num_samples=1, batch_size=1, num_rollout_steps=0)

        pred = torch.full((1, 1, 1, 2, 2), 7.5)
        buf.update(pred, torch.tensor([0.0]), idt=0)

        self.assertTrue(torch.all(buf.rollout_data[0] == 7.5))

    def test_timestamps_recorded_only_at_idt_zero(self, verbose=False):
        # Timestamps for a given IC should be captured at idt=0 and NOT
        # overwritten by subsequent rollout steps that pass different values.
        buf = self._make_buffer(
            num_samples=1,
            batch_size=1,
            num_rollout_steps=2,
        )

        pred = torch.zeros(1, 1, 1, 2, 2)

        buf.update(pred, torch.tensor([100.0]), idt=0)
        with self.subTest(desc="recorded at idt=0"):
            self.assertEqual(buf.timestamp_data[0].item(), 100.0)

        # Pass a different tstamp at later steps — it should be ignored.
        buf.update(pred, torch.tensor([999.0]), idt=1)
        buf.update(pred, torch.tensor([777.0]), idt=2)
        with self.subTest(desc="not overwritten by idt>0"):
            self.assertEqual(buf.timestamp_data[0].item(), 100.0)

    def test_zero_buffers_resets_all_state(self, verbose=False):
        # zero_buffers() is fully self-contained: zeros data tensors AND
        # resets buffer_offset, so it's safe to call mid-rollout (puts the
        # buffer back into a fresh post-construction state). Mirrors
        # MeanStdBuffer.zero_buffers's contract for the offset analog.
        buf = self._make_buffer(num_samples=2, batch_size=1, num_rollout_steps=0)

        pred = torch.full((1, 1, 1, 2, 2), 5.0)
        buf.update(pred, torch.tensor([100.0]), idt=0)  # finishes rollout (R=0)

        with self.subTest(desc="pre-zero state"):
            self.assertEqual(buf.buffer_offset, 1)
            self.assertNotEqual(buf.rollout_data.abs().sum().item(), 0.0)
            self.assertNotEqual(buf.timestamp_data.abs().sum().item(), 0.0)

        buf.zero_buffers()

        with self.subTest(desc="data zeroed"):
            self.assertEqual(buf.rollout_data.abs().sum().item(), 0.0)
            self.assertEqual(buf.timestamp_data.abs().sum().item(), 0.0)
        with self.subTest(desc="offset reset"):
            self.assertEqual(buf.buffer_offset, 0)

    def test_auto_flush_on_buffer_overflow(self, verbose=False):
        # Buffer holds 2 steps (output_memory_buffer_size=2), batch_size=2,
        # rollout_steps=0 (R+1=1 step per IC). The first batch fills the
        # buffer; the second batch's idt=0 must trigger _flush_buffer_to_disk
        # before writing, leaving the buffer holding only the second batch's data.
        # _flush_buffer_to_disk's cuda.synchronize is device-guarded so this runs on CPU.
        buf = self._make_buffer(
            num_samples=4,
            batch_size=2,
            num_rollout_steps=0,
            output_memory_buffer_size=2,
            device=self.device,
        )

        pred1 = torch.full((2, 1, 1, 2, 2), 1.0, device=self.device)
        buf.update(pred1, torch.tensor([1.0, 2.0]), idt=0)
        with self.subTest(desc="buffer full after first batch"):
            self.assertEqual(buf.buffer_offset, 2)

        # Second batch at idt=0: pre-flush condition (offset+B > capacity) holds.
        # The implementation auto-flushes, then writes the new batch into a
        # freshly-zeroed buffer.
        pred2 = torch.full((2, 1, 1, 2, 2), 9.0, device=self.device)
        buf.update(pred2, torch.tensor([3.0, 4.0]), idt=0)

        with self.subTest(desc="buffer holds second batch only"):
            self.assertEqual(buf.buffer_offset, 2)
            self.assertTrue(torch.all(buf.rollout_data[0:2] == 9.0))

    def test_auto_flush_mid_rollout(self, verbose=False):
        # Step-granular buffering: with R+1=3 steps per IC and buffer_size=2 (below R+1),
        # the buffer cannot hold even one full rollout. The flush must happen mid-rollout
        # and the carry-over chunk must point at the in-flight batch's next leadtime.
        num_rollout_steps = 2  # R+1 = 3
        buf = self._make_buffer(
            num_samples=1,
            batch_size=1,
            num_rollout_steps=num_rollout_steps,
            output_memory_buffer_size=2,
            device=self.device,
        )

        # idt=0 and idt=1 fit into the buffer; idt=2 triggers a flush.
        for idt in range(num_rollout_steps + 1):
            pred = torch.full((1, 1, 1, 2, 2), float(idt), device=self.device)
            buf.update(pred, torch.tensor([42.0]), idt=idt)
            with self.subTest(desc=f"after idt={idt}"):
                if idt < 2:
                    # still filling the buffer
                    self.assertEqual(buf.buffer_offset, idt + 1)
                else:
                    # idt=2 triggered a flush before the write, then wrote slot 0.
                    self.assertEqual(buf.buffer_offset, 1)
                    # carry-over chunk pinpoints the in-flight batch at idt=2.
                    self.assertEqual(len(buf.chunks), 1)
                    self.assertEqual(buf.chunks[-1]["idt_start"], 2)
                    self.assertEqual(buf.chunks[-1]["idt_count"], 1)
                    self.assertEqual(buf.chunks[-1]["batch_size"], 1)
                    # post-flush slot 0 holds the idt=2 data
                    self.assertTrue(torch.all(buf.rollout_data[0] == 2.0))

    def test_output_memory_buffer_size_defaults_to_full_run(self, verbose=False):
        # output_memory_buffer_size=None means "buffer the entire run in memory",
        # which is num_samples * (R + 1) steps under the step-granular semantics.
        # _make_buffer's defaults give num_samples=10, num_rollout_steps=1 → 10 * 2 = 20.
        buf = self._make_buffer(
            num_samples=10,
            batch_size=2,
            output_memory_buffer_size=None,
        )
        self.assertEqual(buf.num_buffered_samples, 20)

    def test_output_memory_buffer_size_clamping(self, verbose=False):
        # The buffer size is clamped to [batch_size, num_samples * (R + 1)] inclusive.
        # Below the floor: clamped up to batch_size (otherwise update() with a
        # full batch would never fit).
        with self.subTest(desc="clamps below batch_size"):
            buf = self._make_buffer(
                num_samples=10,
                batch_size=4,
                num_rollout_steps=1,
                output_memory_buffer_size=2,  # below floor (batch_size=4)
            )
            self.assertEqual(buf.num_buffered_samples, 4)

        # Above the ceiling: clamped down to num_samples * (R + 1) (no point
        # in over-allocating beyond the full run).
        with self.subTest(desc="clamps above num_samples * (R + 1)"):
            buf = self._make_buffer(
                num_samples=10,
                batch_size=2,
                num_rollout_steps=1,
                output_memory_buffer_size=100,  # above ceiling (10 * 2 = 20)
            )
            self.assertEqual(buf.num_buffered_samples, 20)


class TestRolloutBufferIO(unittest.TestCase):
    """
    End-to-end round-trip tests for ``RolloutBuffer``'s HDF5 output path.

    These tests drive only the public API: construct with ``output_file``,
    call ``update()`` per (batch, idt), then ``finalize()``. They verify
    correctness by **reading the resulting HDF5 file** rather than by
    inspecting the buffer's in-memory state. This keeps the tests valid
    under a future streaming write-path (e.g. GDS-backed) that may bypass
    the in-memory buffer entirely.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_buffer(
        self,
        *,
        output_file,
        num_samples=4,
        batch_size=2,
        num_rollout_steps=1,
        rollout_dt=6,
        ensemble_size=1,
        img_shape=(2, 4),
        channel_names=("a", "b"),
        output_channels=None,
        scale=None,
        bias=None,
        output_memory_buffer_size=None,
        lat_lon=None,
    ):
        if output_channels is None:
            output_channels = list(channel_names)
        if lat_lon is None:
            H, W = img_shape
            # synthetic lat/lon so we can verify file content
            lat_lon = (
                [float(90 - 180.0 * i / max(H - 1, 1)) for i in range(H)],
                [float(360.0 * j / W) for j in range(W)],
            )
        H, W = img_shape
        return RolloutBuffer(
            num_samples=num_samples,
            batch_size=batch_size,
            num_rollout_steps=num_rollout_steps,
            rollout_dt=rollout_dt,
            ensemble_size=ensemble_size,
            img_shape=(H, W),
            local_shape=(H, W),
            local_offset=(0, 0),
            channel_names=list(channel_names),
            lat_lon=lat_lon,
            device=self.device,
            scale=scale,
            bias=bias,
            output_channels=list(output_channels),
            output_file=output_file,
            output_memory_buffer_size=output_memory_buffer_size,
        )

    def _drive_full_rollout(self, buf, *, ic_data_per_batch, tstamps_per_batch):
        """Feed multiple batches through update() with the canonical step ordering.

        ``ic_data_per_batch[b]`` shape: (batch_size, num_rollout_steps+1,
        ensemble_size, num_channels, H, W). For each batch and each timestep,
        we extract the corresponding step slice and call update().
        """
        R_plus_1 = ic_data_per_batch[0].shape[1]
        for batch_idx, (batch, tstamps) in enumerate(zip(ic_data_per_batch, tstamps_per_batch)):
            for idt in range(R_plus_1):
                buf.update(batch[:, idt], tstamps, idt=idt)

    def test_dataset_shape_and_dtype(self, verbose=False):
        # Verify the output HDF5 ``fields`` dataset has the expected
        # (num_samples, R+1, ensemble_size, num_channels, H, W) shape and
        # float32 dtype. This is the contract every consumer relies on.
        out_path = os.path.join(self.tmpdir, "structure.h5")
        num_samples = 4
        num_rollout_steps = 2
        ensemble_size = 3
        img_shape = (4, 6)
        output_channels = ["a", "c"]

        buf = self._make_buffer(
            output_file=out_path,
            num_samples=num_samples,
            batch_size=2,
            num_rollout_steps=num_rollout_steps,
            ensemble_size=ensemble_size,
            img_shape=img_shape,
            channel_names=("a", "b", "c"),
            output_channels=output_channels,
        )

        # Feed enough zero data to fill all ICs through full rollouts.
        zero_batches = [
            torch.zeros(2, num_rollout_steps + 1, ensemble_size, 3, *img_shape)
            for _ in range(2)  # 2 batches × 2 ICs each = 4 ICs total
        ]
        tstamps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        self._drive_full_rollout(buf, ic_data_per_batch=zero_batches, tstamps_per_batch=tstamps)
        buf.finalize()

        with h5.File(out_path, "r") as f:
            with self.subTest(desc="dataset shape"):
                self.assertEqual(
                    f["fields"].shape,
                    (num_samples, num_rollout_steps + 1, ensemble_size, len(output_channels), *img_shape),
                )
            with self.subTest(desc="dataset dtype"):
                self.assertEqual(f["fields"].dtype, np.float32)
            with self.subTest(desc="dim labels"):
                self.assertEqual(f["fields"].dims[0].label, "Timestamp in UTC time zone")
                self.assertEqual(f["fields"].dims[1].label, "Lead time relative to timestamp")
                self.assertEqual(f["fields"].dims[2].label, "Ensemble index")
                self.assertEqual(f["fields"].dims[3].label, "Channel name")
                self.assertEqual(f["fields"].dims[4].label, "Latitude in degrees")
                self.assertEqual(f["fields"].dims[5].label, "Longitude in degrees")

    def test_round_trip_single_flush(self, verbose=False):
        # Whole run fits in a single in-memory buffer (no auto-flush).
        # Per-IC data is a unique constant so we can verify routing into
        # (sample, step, ensemble, channel, lat, lon) slots after readback.
        out_path = os.path.join(self.tmpdir, "single_flush.h5")
        num_samples = 4
        R = 2
        E = 1
        H, W = 2, 4

        buf = self._make_buffer(
            output_file=out_path,
            num_samples=num_samples,
            batch_size=2,
            num_rollout_steps=R,
            ensemble_size=E,
            img_shape=(H, W),
            channel_names=("a",),
        )

        # batch 0: ICs 0, 1; batch 1: ICs 2, 3
        # data[ic, idt, e, c, h, w] = ic * 10 + idt
        batches = []
        for b in range(2):
            arr = torch.zeros(2, R + 1, E, 1, H, W)
            for i in range(2):
                ic = b * 2 + i
                for idt in range(R + 1):
                    arr[i, idt, ...] = float(ic * 10 + idt)
            batches.append(arr)
        tstamps = [torch.tensor([100.0, 200.0]), torch.tensor([300.0, 400.0])]

        self._drive_full_rollout(buf, ic_data_per_batch=batches, tstamps_per_batch=tstamps)
        buf.finalize()

        with h5.File(out_path, "r") as f:
            data = f["fields"][...]
            ts = f["timestamp"][...]

        for ic in range(num_samples):
            for idt in range(R + 1):
                with self.subTest(desc=f"ic={ic} idt={idt}"):
                    self.assertTrue(
                        np.all(data[ic, idt] == float(ic * 10 + idt)),
                        msg=f"slot (ic={ic}, idt={idt}) has wrong content",
                    )
        with self.subTest(desc="timestamps"):
            self.assertTrue(np.array_equal(ts, np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)))

    def test_round_trip_with_auto_flush(self, verbose=False):
        # ``output_memory_buffer_size`` < ``num_samples`` forces at least one
        # mid-run flush. Verify the full file still contains every IC at the
        # right (sample, step, ...) coordinates — i.e. the auto-flush bookkeeping
        # for ``file_offset`` is correct.
        out_path = os.path.join(self.tmpdir, "auto_flush.h5")
        num_samples = 6
        R = 1
        H, W = 2, 3

        buf = self._make_buffer(
            output_file=out_path,
            num_samples=num_samples,
            batch_size=2,
            num_rollout_steps=R,
            ensemble_size=1,
            img_shape=(H, W),
            channel_names=("a",),
            output_memory_buffer_size=2,  # forces auto-flush every batch
        )

        # 3 batches × 2 ICs each = 6 ICs. Same encoding as single-flush case.
        batches = []
        for b in range(3):
            arr = torch.zeros(2, R + 1, 1, 1, H, W)
            for i in range(2):
                ic = b * 2 + i
                for idt in range(R + 1):
                    arr[i, idt, ...] = float(ic * 10 + idt)
            batches.append(arr)
        tstamps = [torch.tensor([float(b * 2), float(b * 2 + 1)]) for b in range(3)]

        self._drive_full_rollout(buf, ic_data_per_batch=batches, tstamps_per_batch=tstamps)
        buf.finalize()

        with h5.File(out_path, "r") as f:
            data = f["fields"][...]
            ts = f["timestamp"][...]

        with self.subTest(desc="all ICs present"):
            for ic in range(num_samples):
                for idt in range(R + 1):
                    self.assertTrue(
                        np.all(data[ic, idt] == float(ic * 10 + idt)),
                        msg=f"after auto-flush: slot (ic={ic}, idt={idt}) corrupt",
                    )
        with self.subTest(desc="timestamps in correct order across flushes"):
            self.assertTrue(np.array_equal(ts, np.arange(num_samples, dtype=np.float64)))

    def test_round_trip_with_mid_rollout_flush(self, verbose=False):
        # Step-granular buffering: ``output_memory_buffer_size`` smaller than R+1
        # forces flushes mid-rollout (a single trajectory does not fit in the buffer).
        # The carry-over chunk must correctly continue the in-flight batch at the
        # next leadtime so the round-trip file matches the non-flushed case exactly.
        # Mirrors the production motivation: long rollouts that overflow CPU memory.
        out_path = os.path.join(self.tmpdir, "mid_rollout_flush.h5")
        num_samples = 4
        R = 6  # R+1 = 7 steps per IC
        H, W = 2, 3
        buffer_steps = 3  # below R+1 — forces multiple mid-rollout flushes

        buf = self._make_buffer(
            output_file=out_path,
            num_samples=num_samples,
            batch_size=2,
            num_rollout_steps=R,
            ensemble_size=1,
            img_shape=(H, W),
            channel_names=("a",),
            output_memory_buffer_size=buffer_steps,
        )

        # 2 batches × 2 ICs each = 4 ICs. data[ic, idt] = ic * 100 + idt.
        batches = []
        for b in range(2):
            arr = torch.zeros(2, R + 1, 1, 1, H, W)
            for i in range(2):
                ic = b * 2 + i
                for idt in range(R + 1):
                    arr[i, idt, ...] = float(ic * 100 + idt)
            batches.append(arr)
        tstamps = [torch.tensor([float(b * 2), float(b * 2 + 1)]) for b in range(2)]

        self._drive_full_rollout(buf, ic_data_per_batch=batches, tstamps_per_batch=tstamps)
        buf.finalize()

        with h5.File(out_path, "r") as f:
            data = f["fields"][...]
            ts = f["timestamp"][...]

        with self.subTest(desc="all (ic, idt) slots correct after mid-rollout flushes"):
            for ic in range(num_samples):
                for idt in range(R + 1):
                    self.assertTrue(
                        np.all(data[ic, idt] == float(ic * 100 + idt)),
                        msg=f"slot (ic={ic}, idt={idt}) corrupt after mid-rollout flush",
                    )
        with self.subTest(desc="timestamps survive mid-rollout flushes"):
            self.assertTrue(np.array_equal(ts, np.arange(num_samples, dtype=np.float64)))

    def test_dim_scales_attached_and_correct(self, verbose=False):
        # The fields dataset has 5 named dim scales: timestamp, lead_time,
        # channel, lat, lon. Verify each resolves and contains the expected
        # values (channels and lat/lon set unconditionally; timestamp populated
        # via update() calls; lead_time computed from rollout_dt).
        out_path = os.path.join(self.tmpdir, "scales.h5")
        H, W = 3, 5
        rollout_dt = 6
        R = 2
        channel_names = ("a", "b")
        output_channels = ["a", "b"]
        lat = [60.0, 0.0, -60.0]
        lon = [0.0, 72.0, 144.0, 216.0, 288.0]

        buf = self._make_buffer(
            output_file=out_path,
            num_samples=2,
            batch_size=2,
            num_rollout_steps=R,
            rollout_dt=rollout_dt,
            ensemble_size=1,
            img_shape=(H, W),
            channel_names=channel_names,
            output_channels=output_channels,
            lat_lon=(lat, lon),
        )

        zero_batch = torch.zeros(2, R + 1, 1, 2, H, W)
        self._drive_full_rollout(
            buf,
            ic_data_per_batch=[zero_batch],
            tstamps_per_batch=[torch.tensor([1234.0, 5678.0])],
        )
        buf.finalize()

        with h5.File(out_path, "r") as f:
            dset = f["fields"]
            with self.subTest(desc="timestamp scale on dim 0"):
                self.assertTrue(np.array_equal(dset.dims[0]["timestamp"][...], [1234.0, 5678.0]))
            with self.subTest(desc="lead_time scale on dim 1"):
                expected_lt = np.arange(0, (R + 1) * rollout_dt, rollout_dt, dtype=np.float64)
                self.assertTrue(np.array_equal(dset.dims[1]["lead_time"][...], expected_lt))
            with self.subTest(desc="channel scale on dim 3"):
                names = [c.decode() for c in dset.dims[3]["channel"][...].tolist()]
                self.assertEqual(names, output_channels)
            with self.subTest(desc="lat scale on dim 4"):
                self.assertTrue(np.allclose(dset.dims[4]["lat"][...], np.array(lat, dtype=np.float32)))
            with self.subTest(desc="lon scale on dim 5"):
                self.assertTrue(np.allclose(dset.dims[5]["lon"][...], np.array(lon, dtype=np.float32)))

    def test_scale_and_bias_propagated_to_file(self, verbose=False):
        # Constant input (ones) with non-trivial scale + bias should land in
        # the file as scale*1 + bias per channel.
        out_path = os.path.join(self.tmpdir, "scale_bias.h5")
        scale = torch.tensor([2.0, 0.5])
        bias = torch.tensor([1.0, -1.0])

        buf = self._make_buffer(
            output_file=out_path,
            num_samples=2,
            batch_size=2,
            num_rollout_steps=0,
            ensemble_size=1,
            img_shape=(2, 2),
            channel_names=("a", "b"),
            output_channels=["a", "b"],
            scale=scale,
            bias=bias,
        )

        ones = torch.ones(2, 1, 1, 2, 2, 2)  # (batch, R+1, E, C, H, W)
        self._drive_full_rollout(buf, ic_data_per_batch=[ones], tstamps_per_batch=[torch.tensor([0.0, 1.0])])
        buf.finalize()

        with h5.File(out_path, "r") as f:
            data = f["fields"][...]  # (2, 1, 1, 2, 2, 2)

        with self.subTest(desc="channel a = 2*1 + 1 = 3"):
            self.assertTrue(np.all(data[:, :, :, 0, :, :] == 3.0))
        with self.subTest(desc="channel b = 0.5*1 + (-1) = -0.5"):
            self.assertTrue(np.all(data[:, :, :, 1, :, :] == -0.5))

    def test_channel_mask_subset_written(self, verbose=False):
        # full input has 3 channels (a, b, c); output_channels = (a, c).
        # The file should have only 2 channels and they should be a/c values.
        out_path = os.path.join(self.tmpdir, "mask.h5")

        buf = self._make_buffer(
            output_file=out_path,
            num_samples=2,
            batch_size=2,
            num_rollout_steps=0,
            ensemble_size=1,
            img_shape=(2, 2),
            channel_names=("a", "b", "c"),
            output_channels=["a", "c"],
        )

        # full pred has 3 channels; values 100/200/300 per channel
        pred = torch.zeros(2, 1, 1, 3, 2, 2)
        pred[..., 0, :, :] = 100.0  # a → kept
        pred[..., 1, :, :] = 200.0  # b → dropped
        pred[..., 2, :, :] = 300.0  # c → kept

        self._drive_full_rollout(buf, ic_data_per_batch=[pred], tstamps_per_batch=[torch.tensor([0.0, 1.0])])
        buf.finalize()

        with h5.File(out_path, "r") as f:
            data = f["fields"][...]
            channel_names = [c.decode() for c in f["channel"][...].tolist()]

        with self.subTest(desc="only 2 channels in file"):
            self.assertEqual(data.shape[3], 2)
            self.assertEqual(channel_names, ["a", "c"])
        with self.subTest(desc="channel 'a' kept its value"):
            self.assertTrue(np.all(data[:, :, :, 0, :, :] == 100.0))
        with self.subTest(desc="channel 'c' kept its value"):
            self.assertTrue(np.all(data[:, :, :, 1, :, :] == 300.0))

    def test_finalize_idempotent(self, verbose=False):
        # finalize() must be safe to call more than once. The recent fix
        # (file_handle = None after close) makes the second call a no-op
        # rather than an h5py error.
        out_path = os.path.join(self.tmpdir, "idempotent.h5")
        buf = self._make_buffer(output_file=out_path)

        zero = torch.zeros(2, 2, 1, 2, 2, 4)
        self._drive_full_rollout(buf, ic_data_per_batch=[zero], tstamps_per_batch=[torch.tensor([0.0, 1.0])])

        buf.finalize()
        with self.subTest(desc="file_handle nulled after first finalize"):
            self.assertIsNone(buf.file_handle)

        # Second finalize should not raise.
        try:
            buf.finalize()
        except Exception as e:
            self.fail(f"second finalize() raised: {e!r}")

        # File on disk is unaffected and still readable.
        with h5.File(out_path, "r") as f:
            self.assertEqual(f["fields"].shape, (4, 2, 1, 2, 2, 4))


class TestRolloutBufferStreaming(unittest.TestCase):
    """
    Tests for ``RolloutBuffer``'s streaming mode and ``buffer_device``
    parameter.

    Streaming mode skips the in-memory buffer entirely and writes directly to
    HDF5 on every ``update()``. ``buffer_device`` controls where the
    in-memory buffer (when present) lives — defaulting to CPU but allowing a
    GPU-resident buffer to avoid GPU→CPU traffic on every update.

    These tests reuse the public-API-only style of ``TestRolloutBufferIO``:
    construction → ``update()`` → ``finalize()`` → readback. We assert on
    file content (the contract) rather than internal state where possible.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_buffer(
        self,
        *,
        output_file,
        num_samples=4,
        batch_size=2,
        num_rollout_steps=1,
        rollout_dt=6,
        ensemble_size=1,
        img_shape=(2, 4),
        channel_names=("a", "b"),
        output_channels=None,
        scale=None,
        bias=None,
        output_memory_buffer_size=None,
        streaming_mode=False,
        buffer_device=torch.device("cpu"),
        enable_odirect=False,
        odirect_alignment=0,
    ):
        if output_channels is None:
            output_channels = list(channel_names)
        H, W = img_shape
        lat_lon = (
            [float(90 - 180.0 * i / max(H - 1, 1)) for i in range(H)],
            [float(360.0 * j / W) for j in range(W)],
        )
        return RolloutBuffer(
            num_samples=num_samples,
            batch_size=batch_size,
            num_rollout_steps=num_rollout_steps,
            rollout_dt=rollout_dt,
            ensemble_size=ensemble_size,
            img_shape=(H, W),
            local_shape=(H, W),
            local_offset=(0, 0),
            channel_names=list(channel_names),
            lat_lon=lat_lon,
            device=self.device,
            scale=scale,
            bias=bias,
            output_channels=list(output_channels),
            output_file=output_file,
            output_memory_buffer_size=output_memory_buffer_size,
            streaming_mode=streaming_mode,
            buffer_device=buffer_device,
            enable_odirect=enable_odirect,
            odirect_alignment=odirect_alignment,
        )

    def _drive_full_rollout(self, buf, *, ic_data_per_batch, tstamps_per_batch):
        R_plus_1 = ic_data_per_batch[0].shape[1]
        for batch, tstamps in zip(ic_data_per_batch, tstamps_per_batch):
            for idt in range(R_plus_1):
                buf.update(batch[:, idt], tstamps, idt=idt)

    def test_streaming_without_output_file_raises(self, verbose=False):
        # Streaming mode has no in-memory fallback — every update writes
        # directly to disk, so output_file is mandatory.
        with self.assertRaises(ValueError):
            self._make_buffer(output_file=None, streaming_mode=True)

    def test_enable_odirect_without_output_file_raises(self, verbose=False):
        # O_DIRECT writes need a real file — same contract as streaming mode.
        with self.assertRaises(ValueError):
            self._make_buffer(output_file=None, enable_odirect=True)

    def test_enable_odirect_round_trip(self, verbose=False):
        # When the HDF5 build includes the direct VFD, enable_odirect=True must:
        #   (1) construct without error,
        #   (2) open the file with driver="direct" (no host page cache),
        #   (3) produce a file readable with the default driver — same bytes as
        #       a control run with enable_odirect=False.
        # Skip-if-unavailable: not every HDF5 build ships the direct VFD.
        ctrl_path = os.path.join(self.tmpdir, "odirect_ctrl.h5")
        odirect_path = os.path.join(self.tmpdir, "odirect.h5")

        # build identical inputs
        H, W = 2, 4
        num_samples = 4
        R = 1
        batches = []
        for b in range(2):
            arr = torch.zeros(2, R + 1, 1, 2, H, W)
            for i in range(2):
                ic = b * 2 + i
                for idt in range(R + 1):
                    arr[i, idt, ...] = float(ic * 10 + idt)
            batches.append(arr)
        tstamps = [torch.tensor([float(b * 2), float(b * 2 + 1)]) for b in range(2)]

        # control
        buf_ctrl = self._make_buffer(
            output_file=ctrl_path,
            num_samples=num_samples,
            batch_size=2,
            num_rollout_steps=R,
            ensemble_size=1,
            img_shape=(H, W),
        )
        self._drive_full_rollout(buf_ctrl, ic_data_per_batch=batches, tstamps_per_batch=tstamps)
        buf_ctrl.finalize()

        # O_DIRECT path — skip if the local HDF5 build lacks the direct VFD
        try:
            buf_od = self._make_buffer(
                output_file=odirect_path,
                num_samples=num_samples,
                batch_size=2,
                num_rollout_steps=R,
                ensemble_size=1,
                img_shape=(H, W),
                enable_odirect=True,
                odirect_alignment=4096,
            )
        except (ValueError, OSError, RuntimeError) as e:
            self.skipTest(f"HDF5 build does not support the direct VFD: {e}")

        self._drive_full_rollout(buf_od, ic_data_per_batch=batches, tstamps_per_batch=tstamps)
        buf_od.finalize()

        with h5.File(ctrl_path, "r") as f_c, h5.File(odirect_path, "r") as f_o:
            with self.subTest(desc="fields match control"):
                self.assertTrue(np.array_equal(f_c["fields"][...], f_o["fields"][...]))
            with self.subTest(desc="timestamps match control"):
                self.assertTrue(np.array_equal(f_c["timestamp"][...], f_o["timestamp"][...]))

    def test_output_memory_buffer_size_zero_implies_streaming(self, verbose=False):
        # Legacy shorthand from tkurth/gds-support: passing
        # ``output_memory_buffer_size=0`` (or any non-positive value) enables
        # streaming mode without the caller needing to set streaming_mode=True.
        # Inference dispatcher scripts pass ``--output_memory_buffer_size=0`` to
        # disable buffering; this test pins that contract.
        out_path = os.path.join(self.tmpdir, "stream_via_zero.h5")
        buf = self._make_buffer(output_file=out_path, output_memory_buffer_size=0)
        try:
            with self.subTest(desc="streaming_mode auto-enabled"):
                self.assertTrue(buf.streaming_mode)
            with self.subTest(desc="no in-memory rollout_data allocated"):
                self.assertIsNone(buf.rollout_data)
            with self.subTest(desc="no in-memory timestamp_data allocated"):
                self.assertIsNone(buf.timestamp_data)
        finally:
            zero = torch.zeros(2, 2, 1, 2, 2, 4)
            self._drive_full_rollout(buf, ic_data_per_batch=[zero], tstamps_per_batch=[torch.tensor([0.0, 1.0])])
            buf.finalize()

    def test_output_memory_buffer_size_zero_without_file_raises(self, verbose=False):
        # The same validation that fires for explicit streaming_mode=True must
        # also fire when streaming is triggered via the size-shorthand: streaming
        # without an output_file is meaningless.
        with self.assertRaises(ValueError):
            self._make_buffer(output_file=None, output_memory_buffer_size=0)

    def test_streaming_skips_in_memory_buffer_allocation(self, verbose=False):
        # Streaming mode must NOT allocate the (potentially huge) rollout
        # tensors. ``rollout_data`` and ``timestamp_data`` are None.
        out_path = os.path.join(self.tmpdir, "no_buf.h5")
        buf = self._make_buffer(output_file=out_path, streaming_mode=True)
        with self.subTest(desc="rollout_data is None"):
            self.assertIsNone(buf.rollout_data)
        with self.subTest(desc="timestamp_data is None"):
            self.assertIsNone(buf.timestamp_data)
        # Drive a no-op rollout so finalize() doesn't trip on an empty file.
        zero = torch.zeros(2, 2, 1, 2, 2, 4)
        self._drive_full_rollout(buf, ic_data_per_batch=[zero], tstamps_per_batch=[torch.tensor([0.0, 1.0])])
        buf.finalize()

    def test_streaming_round_trip_matches_buffered(self, verbose=False):
        # The same input, fed through streaming-mode and buffered-mode buffers,
        # must produce byte-identical output files. This is the strongest
        # correctness check: it pins down that streaming hits the same disk
        # coordinates as the bulk-flush path.
        num_samples = 6
        R = 1
        H, W = 2, 3
        img_shape = (H, W)

        # build identical inputs
        batches = []
        for b in range(3):
            arr = torch.zeros(2, R + 1, 1, 1, H, W)
            for i in range(2):
                ic = b * 2 + i
                for idt in range(R + 1):
                    arr[i, idt, ...] = float(ic * 10 + idt)
            batches.append(arr)
        tstamps = [torch.tensor([float(b * 2), float(b * 2 + 1)]) for b in range(3)]

        out_streaming = os.path.join(self.tmpdir, "streaming.h5")
        out_buffered = os.path.join(self.tmpdir, "buffered.h5")

        for path, streaming in [(out_streaming, True), (out_buffered, False)]:
            buf = self._make_buffer(
                output_file=path,
                num_samples=num_samples,
                batch_size=2,
                num_rollout_steps=R,
                ensemble_size=1,
                img_shape=img_shape,
                channel_names=("a",),
                streaming_mode=streaming,
            )
            self._drive_full_rollout(buf, ic_data_per_batch=batches, tstamps_per_batch=tstamps)
            buf.finalize()

        with h5.File(out_streaming, "r") as f_s, h5.File(out_buffered, "r") as f_b:
            with self.subTest(desc="fields match buffered mode"):
                self.assertTrue(np.array_equal(f_s["fields"][...], f_b["fields"][...]))
            with self.subTest(desc="timestamps match buffered mode"):
                self.assertTrue(np.array_equal(f_s["timestamp"][...], f_b["timestamp"][...]))

    def test_streaming_zero_buffers_is_noop(self, verbose=False):
        # Without buffers, zero_buffers() must not raise (and must not touch
        # the file_offset bookkeeping that streaming depends on).
        out_path = os.path.join(self.tmpdir, "zero_noop.h5")
        buf = self._make_buffer(output_file=out_path, streaming_mode=True)

        # advance file_offset via a full IC rollout
        zero = torch.zeros(2, 2, 1, 2, 2, 4)
        self._drive_full_rollout(buf, ic_data_per_batch=[zero], tstamps_per_batch=[torch.tensor([0.0, 1.0])])
        offset_before = buf.file_offset

        buf.zero_buffers()
        with self.subTest(desc="no exception"):
            pass
        with self.subTest(desc="file_offset preserved"):
            self.assertEqual(buf.file_offset, offset_before)
        buf.finalize()

    def test_streaming_finalize_idempotent(self, verbose=False):
        # finalize() in streaming mode must close the file once and be safe
        # to call again. Mirrors test_finalize_idempotent for the buffered path.
        out_path = os.path.join(self.tmpdir, "stream_idempotent.h5")
        buf = self._make_buffer(output_file=out_path, streaming_mode=True)

        zero = torch.zeros(2, 2, 1, 2, 2, 4)
        self._drive_full_rollout(buf, ic_data_per_batch=[zero], tstamps_per_batch=[torch.tensor([0.0, 1.0])])

        buf.finalize()
        with self.subTest(desc="file_handle nulled after first finalize"):
            self.assertIsNone(buf.file_handle)

        try:
            buf.finalize()
        except Exception as e:
            self.fail(f"second finalize() raised: {e!r}")

    def test_buffer_device_string_resolves_to_torch_device(self, verbose=False):
        # Passing a string device name should resolve to a torch.device on
        # ``buffer_device``, matching how ``device`` is handled.
        buf = self._make_buffer(output_file=None, buffer_device="cpu")
        self.assertIsInstance(buf.buffer_device, torch.device)
        self.assertEqual(buf.buffer_device.type, "cpu")

    def test_buffer_device_default_is_cpu(self, verbose=False):
        # Constructed without ``buffer_device`` the buffer must land on CPU,
        # preserving pre-port behavior for callers that don't opt in to
        # GPU-resident buffers. Construct ``RolloutBuffer`` directly (bypassing
        # the helper) so this test pins the actual signature default.
        H, W = 2, 4
        lat_lon = (
            [float(90 - 180.0 * i / max(H - 1, 1)) for i in range(H)],
            [float(360.0 * j / W) for j in range(W)],
        )
        buf = RolloutBuffer(
            num_samples=4,
            batch_size=2,
            num_rollout_steps=1,
            rollout_dt=6,
            ensemble_size=1,
            img_shape=(H, W),
            local_shape=(H, W),
            local_offset=(0, 0),
            channel_names=["a", "b"],
            lat_lon=lat_lon,
            device=self.device,
            output_channels=["a", "b"],
            output_file=None,
        )
        self.assertEqual(buf.buffer_device.type, "cpu")
        self.assertEqual(buf.rollout_data.device.type, "cpu")
        self.assertEqual(buf.timestamp_data.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU-resident buffer test")
    def test_buffer_device_cuda_keeps_buffer_on_gpu(self, verbose=False):
        # When buffer_device is CUDA, the buffer tensors must live on GPU and
        # pin_memory must be False (pinning only applies to host memory).
        cuda_dev = torch.device("cuda", torch.cuda.current_device())
        buf = self._make_buffer(output_file=None, buffer_device=cuda_dev)
        with self.subTest(desc="rollout_data on cuda"):
            self.assertEqual(buf.rollout_data.device.type, "cuda")
        with self.subTest(desc="timestamp_data on cuda"):
            self.assertEqual(buf.timestamp_data.device.type, "cuda")
        with self.subTest(desc="not pinned"):
            self.assertFalse(buf.rollout_data.is_pinned())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU-resident buffer round-trip test")
    def test_buffer_device_cuda_round_trip_matches_cpu(self, verbose=False):
        # End-to-end: CUDA-resident buffer must produce the same file as a
        # CPU buffer for identical inputs. This catches mistakes in the
        # GPU→CPU bridge inside _flush_buffer_to_disk.
        cuda_dev = torch.device("cuda", torch.cuda.current_device())
        num_samples = 4
        R = 1
        H, W = 2, 3

        # build identical inputs (place on CUDA so update() copy_ has work to do)
        batches_cpu = []
        for b in range(2):
            arr = torch.zeros(2, R + 1, 1, 1, H, W)
            for i in range(2):
                ic = b * 2 + i
                for idt in range(R + 1):
                    arr[i, idt, ...] = float(ic * 10 + idt)
            batches_cpu.append(arr)
        batches_cuda = [b.to(cuda_dev) for b in batches_cpu]
        tstamps_cpu = [torch.tensor([float(b * 2), float(b * 2 + 1)]) for b in range(2)]
        tstamps_cuda = [t.to(cuda_dev) for t in tstamps_cpu]

        # CPU buffer reference (default buffer_device)
        out_cpu = os.path.join(self.tmpdir, "buf_cpu.h5")
        buf_cpu = self._make_buffer(
            output_file=out_cpu,
            num_samples=num_samples,
            batch_size=2,
            num_rollout_steps=R,
            ensemble_size=1,
            img_shape=(H, W),
            channel_names=("a",),
        )
        self._drive_full_rollout(buf_cpu, ic_data_per_batch=batches_cpu, tstamps_per_batch=tstamps_cpu)
        buf_cpu.finalize()

        # CUDA buffer
        out_cuda = os.path.join(self.tmpdir, "buf_cuda.h5")
        # we want device=cuda too so .copy_(non_blocking=True) into a cuda buffer works
        H_, W_ = H, W
        lat_lon = (
            [float(90 - 180.0 * i / max(H_ - 1, 1)) for i in range(H_)],
            [float(360.0 * j / W_) for j in range(W_)],
        )
        buf_cuda = RolloutBuffer(
            num_samples=num_samples,
            batch_size=2,
            num_rollout_steps=R,
            rollout_dt=6,
            ensemble_size=1,
            img_shape=(H, W),
            local_shape=(H, W),
            local_offset=(0, 0),
            channel_names=["a"],
            lat_lon=lat_lon,
            device=cuda_dev,
            scale=None,
            bias=None,
            output_channels=["a"],
            output_file=out_cuda,
            output_memory_buffer_size=None,
            streaming_mode=False,
            buffer_device=cuda_dev,
        )
        self._drive_full_rollout(buf_cuda, ic_data_per_batch=batches_cuda, tstamps_per_batch=tstamps_cuda)
        buf_cuda.finalize()

        with h5.File(out_cpu, "r") as f_c, h5.File(out_cuda, "r") as f_g:
            with self.subTest(desc="fields match"):
                self.assertTrue(np.array_equal(f_c["fields"][...], f_g["fields"][...]))
            with self.subTest(desc="timestamps match"):
                self.assertTrue(np.array_equal(f_c["timestamp"][...], f_g["timestamp"][...]))

    def test_gds_view_rejects_cpu_tensor(self, verbose=False):
        # The GDS path hands a raw pointer to cuFile, so a host tensor must be
        # rejected rather than silently passed through. Flipping enable_gds on a
        # plain buffer exercises the check without needing the GDS driver.
        buf = self._make_buffer(output_file=None)
        buf.enable_gds = True
        with self.assertRaises(ValueError):
            buf._np_view_for_write(torch.zeros(2, 3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GDS view contiguity test")
    def test_gds_view_rejects_non_contiguous_cuda_tensor(self, verbose=False):
        # A contiguous copy made inside _np_view_for_write would be a temporary
        # nobody holds a reference to, so non-contiguous input must be rejected.
        cuda_dev = torch.device("cuda", torch.cuda.current_device())
        buf = self._make_buffer(output_file=None)
        buf.enable_gds = True
        tensor = torch.zeros(3, 2, device=cuda_dev).t()
        with self.subTest(desc="non-contiguous raises"):
            with self.assertRaises(ValueError):
                buf._np_view_for_write(tensor)
        # the view never dereferences GPU memory on host, so only check its metadata
        with self.subTest(desc="contiguous accepted"):
            arr = buf._np_view_for_write(tensor.contiguous())
            self.assertEqual(arr.shape, (2, 3))
            self.assertEqual(arr.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
