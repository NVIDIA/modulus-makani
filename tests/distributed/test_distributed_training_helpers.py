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

import unittest

import torch
import torch.nn as nn
import torch.distributed as dist

from makani.utils import comm
from makani.mpu.helpers import sync_params
from makani.mpu.mappings import init_gradient_reduction_hooks

from .distributed_helpers import _init_grid, reduce_success, sync_and_barrier, teardown_comms

_NUM_FEATURES = 6


class _TinyModel(nn.Module):
    """
    Linear model with one parameter shared across the model ranks and one parameter that is
    rank-local, like a sharded spectral weight. The loss is linear in the weights, so the local
    gradient of each weight is its input.
    """

    def __init__(self, all_shared=False):
        super().__init__()
        self.shared = nn.Parameter(torch.zeros(_NUM_FEATURES, dtype=torch.float64))
        self.shared.is_shared_mp = ["model"]

        self.local = nn.Parameter(torch.zeros(_NUM_FEATURES, dtype=torch.float64))
        self.local.is_shared_mp = ["model"] if all_shared else ["matmul"]
        if not all_shared:
            self.local.sharded_dims_mp = ["h"]

    def forward(self, x_shared, x_local):
        return (self.shared * x_shared).sum() + (self.local * x_local).sum()


class TestDistributedTrainingHelpers(unittest.TestCase):
    """
    Weight synchronization (``sync_params``) and gradient reduction (``init_gradient_reduction_hooks``)
    across the data and model groups. Every rank starts from different values, and the expected
    result is replayed from the values of all ranks and the group layout, so the tests do not
    depend on how the groups are laid out.

    Run with e.g.::

        GRID_H=2 GRID_W=2 GRID_E=2 mpirun -n 8 pytest tests/distributed/test_distributed_training_helpers.py
    """

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)

    @classmethod
    def tearDownClass(cls):
        sync_and_barrier()
        teardown_comms()

    def _rank_values(self, offset):
        # distinct per rank and per element
        return torch.arange(_NUM_FEATURES, dtype=torch.float64, device=self.device) + 10.0 * self.world_rank + offset

    def _all_gather(self, tensor):
        if not dist.is_initialized():
            return tensor.unsqueeze(0).clone()
        tlist = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tlist, tensor.contiguous())
        return torch.stack(tlist, dim=0)

    def _group_keys(self, name):
        # world rank of each rank's group root; a rank in a group of size 1 is its own root
        root = comm.get_root(name) if comm.get_size(name) > 1 else self.world_rank
        return self._all_gather(torch.tensor([root], dtype=torch.long, device=self.device)).flatten()

    def _replay(self, values, name, op):
        """Apply a broadcast, mean or sum over the group ``name`` to the values of all ranks."""
        keys = self._group_keys(name)
        result = values.clone()
        for root in keys.unique().tolist():
            members = keys == root
            if op == "broadcast":
                result[members] = values[root]
            elif op == "mean":
                result[members] = values[members].mean(dim=0)
            elif op == "sum":
                result[members] = values[members].sum(dim=0)
            else:
                raise ValueError(f"Unknown op {op}")
        return result

    def _check(self, name, actual, expected, verbose=False):
        ok = torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)
        if verbose and not ok:
            print(f"rank {self.world_rank}: {name} {actual.tolist()} vs expected {expected.tolist()}")
        self.assertTrue(reduce_success(ok, self.device), name)

    def _test_sync_params(self, mode, verbose=False):
        model = _TinyModel().to(self.device)
        with torch.no_grad():
            model.shared.copy_(self._rank_values(0.0))
            model.local.copy_(self._rank_values(0.5))

        initial = {pname: self._all_gather(p.detach()) for pname, p in model.named_parameters()}

        sync_params(model, mode=mode)

        for pname, param in model.named_parameters():
            # sync_params goes over the data group first, then over each group the parameter is shared across
            expected = self._replay(initial[pname], "data", mode)
            for group in param.is_shared_mp:
                expected = self._replay(expected, group, mode)
            self._check(f"{mode} {pname}", param.detach(), expected[self.world_rank], verbose=verbose)

    def test_sync_params_broadcast(self, verbose=False):
        self._test_sync_params("broadcast", verbose=verbose)

    def test_sync_params_mean(self, verbose=False):
        self._test_sync_params("mean", verbose=verbose)

    def _test_gradient_reduction(self, all_shared, verbose=False):
        model = _TinyModel(all_shared=all_shared).to(self.device)
        model_ddp = init_gradient_reduction_hooks(model, self.device, verbose=False)

        x_shared = self._rank_values(1.0)
        x_local = self._rank_values(2.0)
        model_ddp(x_shared, x_local).backward()

        # the local gradient of each weight is its input; it is averaged over the data group and
        # summed over each group the parameter is shared across, whose ranks all contribute to it
        local_grads = {"shared": x_shared, "local": x_local}
        for pname, param in model.named_parameters():
            expected = self._replay(self._all_gather(local_grads[pname]), "data", "mean")
            for group in param.is_shared_mp:
                expected = self._replay(expected, group, "sum")
            self._check(f"gradient {pname}", param.grad, expected[self.world_rank], verbose=verbose)

    def test_gradient_reduction_all_shared(self, verbose=False):
        # with model parallelism, plain DDP over the world with the gradients scaled by the model size
        self._test_gradient_reduction(all_shared=True, verbose=verbose)

    def test_gradient_reduction_mixed(self, verbose=False):
        # with model parallelism, DDP over the data group with the custom per-parameter comm hook
        self._test_gradient_reduction(all_shared=False, verbose=verbose)


if __name__ == "__main__":
    unittest.main()
