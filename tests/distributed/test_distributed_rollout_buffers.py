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
from parameterized import parameterized

import numpy as np
import torch

from makani.utils import comm
from makani.utils.inference.rollout_buffer import SpectrumAverageBuffer, ZonalSpectrumAverageBuffer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .distributed_helpers import _init_grid, _split_helper, reduce_success, teardown_comms
from ..testutils import disable_tf32, compare_tensors

_CHANNEL_NAMES = ["c0", "c1", "c2", "c3", "c4"]
# at least as many channels as azimuth ranks, which the distributed transforms transpose the channels over
_OUTPUT_CHANNELS = ["c4", "c0", "c2", "c3"]
# the batch and ensemble members held by each data-parallel rank
_LOCAL_BATCH_SIZE = 2
_LOCAL_ENSEMBLE_SIZE = 2
# (rollout step, update) sequence: two updates of the first step, one of the second
_ROLLOUT_STEPS = [0, 0, 1]


class TestDistributedSpectrumBuffers(unittest.TestCase):
    """
    The spectrum buffers of the inferencer, with the field split over the spatial (h x w) ranks,
    the ensemble members over the ensemble ranks and the samples over the batch ranks. Each rank's
    statistics must equal its slice of the statistics a serial buffer accumulates from the whole
    field, all ensemble members and all samples.

    Run with e.g.::

        GRID_H=2 GRID_W=2 GRID_E=2 mpirun -n 8 pytest tests/distributed/test_distributed_rollout_buffers.py
    """

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)
        cls.brank = comm.get_rank("batch")

    @classmethod
    def tearDownClass(cls):
        teardown_comms()

    def setUp(self):
        disable_tf32()

    def _split_spatial(self, tensor):
        tensor_local = _split_helper(tensor, dim=-1, group=self.w_group)
        return _split_helper(tensor_local, dim=-2, group=self.h_group)

    def _ensemble_range(self):
        # where this rank's members sit in the serial layout [target, member 0, member 1, ...];
        # the first ensemble rank also holds the target
        if self.erank == 0:
            return slice(0, _LOCAL_ENSEMBLE_SIZE + 1)
        start = self.erank * _LOCAL_ENSEMBLE_SIZE + 1
        return slice(start, start + _LOCAL_ENSEMBLE_SIZE)

    def _run(self, make_buffer, nlat, nlon):
        """Update and finalize a serial and a distributed buffer with the same fields."""
        global_batch_size = _LOCAL_BATCH_SIZE * self.grid_size_b
        global_ensemble_size = _LOCAL_ENSEMBLE_SIZE * self.grid_size_e
        scale = torch.linspace(0.5, 2.5, len(_CHANNEL_NAMES)).reshape(1, -1, 1, 1)
        bias = torch.linspace(-1.0, 1.0, len(_CHANNEL_NAMES)).reshape(1, -1, 1, 1)

        # the same fields on every rank, from a generator of their own, so that no rank-dependent use of
        # the default generator, e.g. by an uneven weight initialization, shifts them
        generator = torch.Generator().manual_seed(333)
        updates = []
        for idt in _ROLLOUT_STEPS:
            shape = (global_batch_size, global_ensemble_size, len(_CHANNEL_NAMES), nlat, nlon)
            data = torch.randn(shape, generator=generator).to(self.device)
            targ = torch.randn((global_batch_size, 1, *shape[2:]), generator=generator).to(self.device)
            updates.append((idt, data, targ))

        # the serial buffer has to be updated in the serial context too, since it queries the comms when
        # it is called
        with comm.serial():
            buffer_serial = make_buffer(global_ensemble_size, scale, bias)
            for idt, data, targ in updates:
                buffer_serial.update(data, targ, idt)
            buffer_serial.finalize()

        buffer_dist = make_buffer(_LOCAL_ENSEMBLE_SIZE, scale, bias)
        batch_range = slice(self.brank * _LOCAL_BATCH_SIZE, (self.brank + 1) * _LOCAL_BATCH_SIZE)
        ensemble_range = slice(self.erank * _LOCAL_ENSEMBLE_SIZE, (self.erank + 1) * _LOCAL_ENSEMBLE_SIZE)
        for idt, data, targ in updates:
            data_local = self._split_spatial(data[batch_range, ensemble_range])
            targ_local = self._split_spatial(targ[batch_range])
            buffer_dist.update(data_local, targ_local, idt)
        buffer_dist.finalize()

        return buffer_serial, buffer_dist

    def _compare(self, buffer_serial, buffer_dist, index, tol, verbose=False):
        mean_serial = buffer_serial.running_mean[index]
        m2_serial = buffer_serial.running_var[index]

        with self.subTest(desc="mean"):
            atol = tol * mean_serial.abs().max().item()
            ok = compare_tensors("mean", buffer_dist.running_mean, mean_serial, atol, tol, verbose, shape_check=True)
            self.assertTrue(reduce_success(ok, self.device))
        with self.subTest(desc="m2"):
            atol = tol * m2_serial.abs().max().item()
            ok = compare_tensors("m2", buffer_dist.running_var, m2_serial, atol, tol, verbose, shape_check=True)
            self.assertTrue(reduce_success(ok, self.device))
        with self.subTest(desc="count"):
            ok = torch.equal(buffer_dist.num_samples_tracked, buffer_serial.num_samples_tracked)
            self.assertTrue(reduce_success(ok, self.device))

    # at most 128 longitudes for the CPU runners of the CI; the odd number of latitudes keeps the shards uneven
    @parameterized.expand(
        [
            [32, 64, "legendre-gauss", 1e-4],
            [33, 64, "equiangular", 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_spectrum_buffer(self, nlat, nlon, grid, tol, verbose=False):
        def make_buffer(ensemble_size, scale, bias):
            return SpectrumAverageBuffer(
                num_rollout_steps=2,
                rollout_dt=6,
                img_shape=(nlat, nlon),
                ensemble_size=ensemble_size,
                grid_type=grid,
                channel_names=_CHANNEL_NAMES,
                device=self.device,
                scale=scale,
                bias=bias,
                output_channels=_OUTPUT_CHANNELS,
                spatial_distributed=True,
            )

        buffer_serial, buffer_dist = self._run(make_buffer, nlat, nlon)

        # the degrees are split over the polar ranks; the orders are summed over the azimuth ranks
        l_range = slice(buffer_dist.lmax_offset, buffer_dist.lmax_offset + buffer_dist.lmax_local)
        self._compare(buffer_serial, buffer_dist, (Ellipsis, self._ensemble_range(), l_range), tol, verbose=verbose)

    @parameterized.expand(
        [
            [32, 64, 1e-4],
            [33, 63, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_zonal_spectrum_buffer(self, nlat, nlon, tol, verbose=False):
        lats = np.linspace(90.0, -90.0, nlat).tolist()
        lons = np.linspace(0.0, 360.0, nlon, endpoint=False).tolist()

        def make_buffer(ensemble_size, scale, bias):
            return ZonalSpectrumAverageBuffer(
                num_rollout_steps=2,
                rollout_dt=6,
                img_shape=(nlat, nlon),
                ensemble_size=ensemble_size,
                channel_names=_CHANNEL_NAMES,
                lat_lon=(lats, lons),
                device=self.device,
                scale=scale,
                bias=bias,
                output_channels=_OUTPUT_CHANNELS,
                spatial_distributed=True,
            )

        buffer_serial, buffer_dist = self._run(make_buffer, nlat, nlon)

        # the latitudes are split over the polar ranks, the orders over the azimuth ranks
        lat_range = slice(buffer_dist.nlat_offset, buffer_dist.nlat_offset + buffer_dist.nlat_local)
        m_range = slice(buffer_dist.mmax_offset, buffer_dist.mmax_offset + buffer_dist.mmax_local)
        index = (Ellipsis, self._ensemble_range(), lat_range, m_range)
        self._compare(buffer_serial, buffer_dist, index, tol, verbose=verbose)


if __name__ == "__main__":
    unittest.main()
