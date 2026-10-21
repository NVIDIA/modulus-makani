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
import shutil
import tempfile
import unittest
from parameterized import parameterized

import torch
import torch.distributed as dist

from makani.utils import comm
from makani.utils import driver
from makani.utils import checkpoint_helpers
from makani.models import model_registry
from makani.mpu.helpers import sync_params
from makani.mpu.mappings import init_gradient_reduction_hooks, reduce_from_parallel_region

from .distributed_helpers import (
    get_default_parameters,
    set_image_shape,
    _init_grid,
    _split_helper,
    _gather_helper,
    reduce_success,
    sync_and_barrier,
    teardown_comms,
)
from ..testutils import disable_tf32, set_seed, compare_tensors, single_step_rollout


def _update_scale(updates, key):
    """
    The reference an update's absolute tolerance scales with: the layer's largest update entry.

    The update of a bias sums the same incoming gradients as the one of the weight of its
    layer, so its rounding error follows that scale even when the sum itself cancels, as it
    does exactly for a bias followed by an instance norm. See test_distributed_model.py.
    """
    scale = updates[key].abs().max()
    if key.endswith(".bias"):
        weight_key = key[: -len(".bias")] + ".weight"
        if weight_key in updates:
            scale = torch.maximum(scale, updates[weight_key].abs().max())
    return max(1.0, scale.item())


class TestDistributedTrainingStep(unittest.TestCase):
    """
    A few optimizer steps of a distributed model against the same steps of the serial model.

    This runs what the trainers do around the model as a whole: the weight synchronization
    after construction, the gradient reductions of DDP and its hooks over the data and the
    model groups, and an optimizer stepping sharded and shared weights. The ranks of the data
    group train on different samples; the serial reference trains on all of them at once, with
    the loss averaged over them as DDP averages the gradients.

    Run with e.g.::

        GRID_H=2 GRID_W=2 GRID_E=2 mpirun -n 8 pytest tests/distributed/test_distributed_training_step.py
    """

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)

        # shared scratch directory: rank 0 picks the path, every rank reads and writes there
        tmpdir = [tempfile.mkdtemp(prefix="makani_training_step_test_", dir="/tmp") if cls.world_rank == 0 else None]
        if dist.is_initialized():
            dist.broadcast_object_list(tmpdir, src=0)
        cls.tmpdir = tmpdir[0]

        sync_and_barrier()

    @classmethod
    def tearDownClass(cls):
        # no rank may still be reading when rank 0 removes the directory
        sync_and_barrier()
        if cls.world_rank == 0:
            shutil.rmtree(cls.tmpdir, ignore_errors=True)
        teardown_comms()

    def setUp(self):
        disable_tf32()

        # the same small setup as test_distributed_model.py, see there for the halo and channel constraints
        self.params = get_default_parameters()
        self.params.history_normalization_mode = "none"
        set_image_shape(self.params, h=36, w=72)
        self.params.scale_factor = 2
        channel_names = ["u10m", "v10m", "t2m", "msl", "u500", "v500", "z500", "t500"]
        self.params.channel_names = channel_names
        self.params.in_channels = self.params.out_channels = list(range(len(channel_names)))
        self.params.N_in_channels = self.params.N_out_channels = len(channel_names)

        # per data-parallel rank; the serial reference trains on the samples of all of them
        self.params.batch_size = 2

        sync_and_barrier()

    def _split_helper(self, tensor, hdim=-2, wdim=-1):
        tensor_local = _split_helper(tensor, dim=hdim, group=self.h_group)
        return _split_helper(tensor_local, dim=wdim, group=self.w_group)

    def _gather_helper(self, tensor, hdim=-2, wdim=-1):
        tensor_gather = _gather_helper(tensor, dim=hdim, group=self.h_group)
        return _gather_helper(tensor_gather, dim=wdim, group=self.w_group)

    def _is_sync_source(self, param):
        # sync_params broadcasts over the data group first, then over each group the parameter is
        # shared across, so the weights of a rank that roots all of them end up everywhere
        groups = ["data"] + list(getattr(param, "is_shared_mp", ["model"]))
        return all((comm.get_size(group) == 1) or (self.world_rank == comm.get_root(group)) for group in groups)

    @staticmethod
    def _loss(out, tar, numel):
        return torch.sum(torch.square(out - tar)) / numel

    @parameterized.expand(
        [
            ("SFNO", 2, 1e-4),
        ],
        skip_on_empty=True,
    )
    def test_distributed_training_step(self, nettype, num_steps, tol, verbose=False):
        self.params.nettype = nettype

        # SGD with momentum: the update is linear in the gradients, so it carries their rounding
        # errors unamplified, unlike Adam, which normalizes gradients that cancel to zero
        lr = 0.1
        momentum = 0.9

        # the serial reference, built on every rank alike; only rank 0 writes the checkpoint
        set_seed(333)
        checkpoint_path = os.path.join(self.tmpdir, f"serial_{nettype}.pt")
        with comm.serial():
            model = model_registry.get_model(self.params).to(self.device)
            if self.world_rank == 0:
                driver.Driver.save_checkpoint(checkpoint_path=checkpoint_path, model=model, checkpoint_mode="flexible")
        sync_and_barrier()

        # the samples of all data-parallel ranks, drawn before the distributed model is built: its
        # initialization consumes a different amount of random numbers on each rank
        num_data = comm.get_size("data")
        data_rank = comm.get_rank("data")
        batch_size = self.params.batch_size
        shape = (num_data * batch_size, self.params.N_in_channels, self.params.img_shape_x, self.params.img_shape_y)
        set_seed(333)
        inp_all = torch.randn(*shape, dtype=torch.float32, device=self.device)
        tar_all = torch.randn(*shape, dtype=torch.float32, device=self.device)

        # each data-parallel rank normalizes its loss by the size of its own samples
        numel = batch_size * shape[1] * shape[2] * shape[3]

        #############################################################
        # serial reference
        #############################################################
        with comm.serial():
            weights_initial = checkpoint_helpers.gather_model_state_dict(model)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            losses_full = []
            for _ in range(num_steps):
                optimizer.zero_grad(set_to_none=True)
                out = model(single_step_rollout(inp_all.unsqueeze(1))).squeeze(1)
                # DDP averages the gradients over the data group, the loss of all samples at once averages here
                loss = self._loss(out, tar_all, numel) / num_data
                loss.backward()
                optimizer.step()
                losses_full.append(loss.detach())
            weights_full = checkpoint_helpers.gather_model_state_dict(model)

        del model, optimizer

        #############################################################
        # distributed
        #############################################################
        model_dist = model_registry.get_model(self.params).to(self.device)
        driver.Driver.restore_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model_dist,
            loss=None,
            optimizer=None,
            scheduler=None,
            counters=None,
            checkpoint_mode="flexible",
            strict=True,
        )

        # throw off the weights of every rank whose weights the synchronization overwrites, so that
        # the comparison below fails unless sync_params restores them
        with torch.no_grad():
            for param in model_dist.parameters():
                if not self._is_sync_source(param):
                    param.add_(0.1 * (self.world_rank + 1))

        # what the trainers do after building the model
        sync_params(model_dist, mode="broadcast")
        weights_synced = checkpoint_helpers.gather_model_state_dict(model_dist)

        model_dist = init_gradient_reduction_hooks(
            model_dist,
            device=self.device,
            reduction_buffer_count=1,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            verbose=False,
        )

        inp_local = self._split_helper(inp_all[data_rank * batch_size : (data_rank + 1) * batch_size])
        tar_local = self._split_helper(tar_all[data_rank * batch_size : (data_rank + 1) * batch_size])

        optimizer = torch.optim.SGD(model_dist.parameters(), lr=lr, momentum=momentum)
        losses_dist = []
        for _ in range(num_steps):
            optimizer.zero_grad(set_to_none=True)
            out_local = model_dist(single_step_rollout(inp_local.unsqueeze(1))).squeeze(1)
            # the loss of the whole grid, from the sums over the shards of the model ranks
            loss = reduce_from_parallel_region(self._loss(out_local, tar_local, numel), "model")
            loss.backward()
            optimizer.step()

            # the loss over all samples is the mean of the ones of the data-parallel ranks
            loss_mean = loss.detach().clone()
            if num_data > 1:
                dist.all_reduce(loss_mean, group=comm.get_group("data"))
                loss_mean = loss_mean / num_data
            losses_dist.append(loss_mean)

        weights_dist = checkpoint_helpers.gather_model_state_dict(model_dist)

        #############################################################
        # compare
        #############################################################
        for key in weights_initial.keys():
            with self.subTest(desc=f"synchronized weight {key}"):
                self.assertTrue(
                    reduce_success(
                        compare_tensors(
                            f"synchronized weight {key}",
                            weights_synced[key],
                            weights_initial[key],
                            0.0,
                            0.0,
                            verbose=verbose,
                        ),
                        self.device,
                    )
                )

        for step, (loss_dist, loss_full) in enumerate(zip(losses_dist, losses_full)):
            with self.subTest(desc=f"loss step {step}"):
                self.assertTrue(
                    reduce_success(
                        compare_tensors(f"loss step {step}", loss_dist, loss_full, tol, tol, verbose=verbose),
                        self.device,
                    )
                )

        # the weights differ from the initial ones by the steps over the learning rate, which are of
        # the scale of the gradients; compare those, the weights themselves would hide their errors
        updates_full = {key: (weights_initial[key] - weights_full[key]) / lr for key in weights_initial.keys()}
        updates_dist = {
            key: (weights_initial[key] - weights_dist["module." + key]) / lr for key in weights_initial.keys()
        }
        for key in weights_initial.keys():
            with self.subTest(desc=f"weight update {key}"):
                self.assertTrue(
                    reduce_success(
                        compare_tensors(
                            f"weight update {key}",
                            updates_dist[key],
                            updates_full[key],
                            tol * _update_scale(updates_full, key),
                            tol,
                            verbose=verbose,
                        ),
                        self.device,
                    )
                )


if __name__ == "__main__":
    unittest.main()
