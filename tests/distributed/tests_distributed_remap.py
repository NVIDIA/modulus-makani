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

import sys
import os
import unittest
from parameterized import parameterized

import numpy as np
import torch

from torch_harmonics.distributed import compute_split_shapes

from makani.utils.remap import ConservativeRemap, _TargetGrid
from makani.mpu.halo import owner_rank

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .distributed_helpers import _init_grid, _split_helper, _gather_helper, reduce_success, teardown_comms
from ..testutils import compare_tensors, disable_tf32


def _fibonacci_mesh(n_cells: int) -> tuple:
    index = np.arange(n_cells, dtype=np.float64)
    lat = np.degrees(np.arcsin(1.0 - 2.0 * (index + 0.5) / n_cells))
    lon = np.mod(np.degrees(index * np.pi * (3.0 - np.sqrt(5.0))), 360.0)
    return lat, lon


class TestDistributedConservativeRemap(unittest.TestCase):
    """``ConservativeRemap(spatial_distributed=True)`` against the single-process operator.

    Both source kinds run with the exact same coordinate metadata every rank
    already carries in full (`target_lat`/`target_lon`, and for a raster
    source `source_lat`/`source_lon` too) -- construction works out each
    rank's local "h"/"w" block and whatever communication it needs from that
    alone. What differs per source kind is what gets exercised:

    * raster: the data tensor is genuinely h/w-sharded (`_split_helper`); an
      ordinary (non-pole) boundary row/column needs a halo fetched via
      `torch_harmonics`' `polar_halo_exchange` ("h") and this module's
      `azimuthal_halo_exchange` ("w").
    * mesh: cell assignment is a crisp point-in-box test, not an
      interpolation with reach, so there is no halo to fetch -- instead each
      rank is deliberately handed a geographically *wrong* initial partition
      (every cell's true h-owner shifted by one rank) so construction's exact
      P2P redistribution has real work to do, not a no-op.

    Both cases exercise the pole's cross-"w" reduce whenever the sweep's h
    split puts a rank's block on an actual pole row.

    Forward and backward are checked as separate subtests (``self.subTest``,
    matching ``tests_distributed_fft.py``), and both are *exact* comparisons
    against the serial operator -- not just "the gradient is finite". Forward:
    gather the distributed output and diff it against serial's. Backward: a
    single shared upstream gradient (``ograd_full``, generated once so every
    rank agrees on it -- relies on ``_init_grid``'s fixed ``torch.manual_seed``
    the same way the FFT tests do) is split the same way the *output* is
    split and backpropagated locally; the resulting input gradient is
    compared to serial's. For mesh, "split the same way" means indexing
    serial's per-cell gradient by each rank's local-to-global cell mapping
    (`global_indices`, from the same wrong-on-purpose partition mask used for
    the forward pass) rather than a raster-style `_split_helper`, since a
    mesh redistribution reorders cells instead of tiling a shared axis.
    """

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)

    @classmethod
    def tearDownClass(cls):
        teardown_comms()

    def setUp(self):
        disable_tf32()

    def _split_helper(self, tensor):
        tensor_local = _split_helper(tensor, dim=-2, group=self.h_group)
        tensor_local = _split_helper(tensor_local, dim=-1, group=self.w_group)
        return tensor_local

    def _gather_helper(self, tensor):
        tensor_gather = _gather_helper(tensor, dim=-2, group=self.h_group)
        tensor_gather = _gather_helper(tensor_gather, dim=-1, group=self.w_group)
        return tensor_gather

    @parameterized.expand(
        [
            # gradient tolerance follows tests_distributed_model.py's
            # test_distributed_model_fwd_bwd precedent (1e-4 for a float32
            # GPU-vs-serial forward/backward comparison) rather than 1e-5:
            # the distributed path sums via halo-exchange-then-matmul while
            # the serial path sums via one matmul, a different reduction
            # order that is expected to disagree at the float32 rounding
            # level (~1e-5 absolute observed on a real GPU run), not a bug
            [48, 96, 24, 48, 1e-4],
            [64, 128, 16, 32, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_remap_from_raster(self, n_src_lat, n_src_lon, n_tgt_lat, n_tgt_lon, tol, verbose=False):
        source_lat = np.linspace(90.0, -90.0, n_src_lat)
        source_lon = np.linspace(0.0, 360.0, n_src_lon, endpoint=False)
        target_lat = np.linspace(90.0, -90.0, n_tgt_lat)
        target_lon = np.linspace(0.0, 360.0, n_tgt_lon, endpoint=False)

        rng = np.random.default_rng(1234)
        global_field_np = (
            np.sin(np.radians(source_lat))[:, None] * 2.0
            + 0.3 * np.cos(3.0 * np.radians(source_lon))[None, :]
            + rng.normal(scale=0.05, size=(n_src_lat, n_src_lon))
        )
        global_field = torch.as_tensor(global_field_np, dtype=torch.float32, device=self.device)
        global_field.requires_grad = True

        serial_remap = ConservativeRemap.from_raster(source_lat, source_lon, target_lat, target_lon).to(self.device)
        serial_out = serial_remap(global_field)

        with torch.no_grad():
            ograd_full = torch.randn_like(serial_out)

        serial_out.backward(ograd_full)
        igrad_full = global_field.grad.clone()

        local_field = self._split_helper(global_field.detach())
        local_field.requires_grad_(True)

        dist_remap = ConservativeRemap.from_raster(
            source_lat, source_lon, target_lat, target_lon, spatial_distributed=True
        ).to(self.device)
        local_out = dist_remap(local_field)

        ograd_local = self._split_helper(ograd_full)
        local_out.backward(ograd_local)
        igrad_local = local_field.grad.clone()

        with self.subTest(desc="output"):
            gathered = self._gather_helper(local_out.detach())
            ok = compare_tensors("output", gathered, serial_out, atol=tol, rtol=tol, verbose=verbose)
            self.assertTrue(reduce_success(ok, self.device), "distributed vs serial conservative remap (raster)")

        with self.subTest(desc="input gradients"):
            igrad_gathered = self._gather_helper(igrad_local)
            ok = compare_tensors("input gradients", igrad_gathered, igrad_full, atol=tol, rtol=tol, verbose=verbose)
            self.assertTrue(reduce_success(ok, self.device), "distributed vs serial input gradients (raster)")

    @parameterized.expand(
        [
            [20000, 16, 32, 1e-5],
            [20000, 24, 48, 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_distributed_remap_from_mesh(self, n_cells, n_lat, n_lon, tol, verbose=False):
        rng = np.random.default_rng(5678)
        cell_lat, cell_lon = _fibonacci_mesh(n_cells)
        cell_area = np.full(n_cells, 4.0 * np.pi / n_cells)
        values_np = (
            np.sin(np.radians(cell_lat)) * 2.0
            + 0.3 * np.cos(3.0 * np.radians(cell_lon))
            + rng.normal(scale=0.05, size=n_cells)
        )

        target_lat = np.linspace(90.0, -90.0, n_lat)
        target_lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)

        values = torch.as_tensor(values_np, dtype=torch.float32, device=self.device)
        values.requires_grad = True

        serial_remap = ConservativeRemap.from_mesh(cell_lat, cell_lon, cell_area, target_lat, target_lon).to(
            self.device
        )
        serial_out = serial_remap(values)

        with torch.no_grad():
            ograd_full = torch.randn_like(serial_out)

        serial_out.backward(ograd_full)
        igrad_full = values.grad.clone()

        # deliberately wrong initial partition: every cell's true h-owner
        # shifted by one rank, so construction's redistribution has to move
        # real data (rather than confirm a no-op) while staying within the
        # nearest-neighbour-only guarantee. global_indices maps each local
        # cell back to its position in the (unsharded) serial tensors, which
        # is what makes an exact -- not just "is finite" -- gradient
        # comparison possible below despite the redistribution reordering
        # cells rather than tiling a shared axis the way raster's split does
        target = _TargetGrid(target_lat, target_lon)
        flat = target.cell_of(cell_lat, cell_lon)
        row, col = np.divmod(flat, target.n_lon)
        lat_splits = compute_split_shapes(target.n_lat, self.grid_size_h)
        lon_splits = compute_split_shapes(target.n_lon, self.grid_size_w)
        # owner_rank is torch-only (it feeds the P2P index computation in
        # makani.utils.remap directly); this test stays on numpy for its own
        # masking/indexing below, so round-trip through torch just here
        true_owner_h = owner_rank(torch.as_tensor(row, dtype=torch.int64), lat_splits).numpy()
        true_owner_w = owner_rank(torch.as_tensor(col, dtype=torch.int64), lon_splits).numpy()
        # shift toward an adjacent rank without wrapping: "h" is not periodic
        # (see ConservativeRemap's module docstring), so a modulo wraparound
        # shift would move a last-rank cell's initial holder all the way to
        # rank 0 once grid_size_h >= 3 -- a genuine multi-hop displacement
        # the nearest-neighbour-only redistribution is correct to reject,
        # not the "deliberately wrong but still 1 hop away" partition this
        # test wants to exercise
        if self.grid_size_h > 1:
            initial_owner_h = np.where(true_owner_h == self.grid_size_h - 1, true_owner_h - 1, true_owner_h + 1)
        else:
            initial_owner_h = true_owner_h
        mask = (initial_owner_h == self.hrank) & (true_owner_w == self.wrank)
        global_indices = np.flatnonzero(mask)

        local_values = torch.as_tensor(values_np[mask], dtype=torch.float32, device=self.device).requires_grad_(True)

        dist_remap = ConservativeRemap.from_mesh(
            cell_lat[mask], cell_lon[mask], cell_area[mask], target_lat, target_lon, spatial_distributed=True
        ).to(self.device)
        local_out = dist_remap(local_values)

        ograd_local = self._split_helper(ograd_full)
        local_out.backward(ograd_local)
        igrad_local = local_values.grad.clone()

        with self.subTest(desc="output"):
            gathered = self._gather_helper(local_out.detach())
            ok = compare_tensors("output", gathered, serial_out, atol=tol, rtol=tol, verbose=verbose)
            self.assertTrue(reduce_success(ok, self.device), "distributed vs serial conservative remap (mesh)")

        with self.subTest(desc="input gradients"):
            expected_igrad_local = igrad_full[torch.as_tensor(global_indices, dtype=torch.int64, device=self.device)]
            ok = compare_tensors(
                "input gradients", igrad_local, expected_igrad_local, atol=tol, rtol=tol, verbose=verbose
            )
            self.assertTrue(reduce_success(ok, self.device), "distributed vs serial input gradients (mesh)")


if __name__ == "__main__":
    unittest.main()
