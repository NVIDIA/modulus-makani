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


import sys
import os
import tempfile
import unittest
from parameterized import parameterized

import torch
import torch_harmonics.distributed as thd

from makani.utils import comm
from makani.utils import driver
from makani.utils import checkpoint_helpers
from makani.utils import LossHandler
from makani.models import model_registry
from makani.mpu.mappings import init_gradient_reduction_hooks
from makani.mpu.mappings import reduce_from_parallel_region

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from .distributed_helpers import (
    get_default_parameters,
    set_image_shape,
    _split_helper,
    _gather_helper,
    reduce_success,
    sync_and_barrier,
)
from ..testutils import disable_tf32, set_seed, compare_tensors


class TestDistributedModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from mpi4py import MPI

        cls.mpi_comm = MPI.COMM_WORLD.Dup()
        cls.mpi_comm_rank = cls.mpi_comm.Get_rank()
        cls.mpi_comm_size = cls.mpi_comm.Get_size()

        if torch.cuda.is_available():
            if cls.mpi_comm_rank == 0:
                print("Running test on GPU")
            local_rank = cls.mpi_comm_rank % torch.cuda.device_count()
            cls.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(cls.device)
        else:
            if cls.mpi_comm_rank == 0:
                print("Running test on CPU")
            cls.device = torch.device("cpu")
        set_seed(333)

        sync_and_barrier()

        return

    @classmethod
    def tearDownClass(cls):
        sync_and_barrier()
        # Free the duplicated communicator. mpi_comm is an mpi4py Intracomm
        # (returned by MPI.COMM_WORLD.Dup()), whose disposal API is Free()
        # — release the duplicated comm explicitly rather than relying on
        # interpreter shutdown to clean it up.
        cls.mpi_comm.Free()

    def setUp(self):

        disable_tf32()

        self.params = get_default_parameters()

        self.params.history_normalization_mode = "none"

        # set image shape, local/crop shapes, offsets, and resampled shapes
        # consistently — see set_image_shape for the rationale.
        set_image_shape(self.params, h=36, w=72)

        # also set the batch size for testing
        self.params.batch_size = 4

    def _init_comms(self):

        # set up distributed
        self.grid_size_h = int(os.getenv("GRID_H", 1))
        self.grid_size_w = int(os.getenv("GRID_W", 1))
        self.grid_size_m = int(os.getenv("GRID_M", 1))
        self.grid_size_e = int(os.getenv("GRID_E", 1))
        self.world_size = self.grid_size_h * self.grid_size_w * self.grid_size_m * self.grid_size_e

        # init groups
        comm.init(
            model_parallel_sizes=[self.grid_size_h, self.grid_size_w, self.grid_size_m],
            model_parallel_names=["h", "w", "matmul"],
            data_parallel_sizes=[self.grid_size_e, -1],
            data_parallel_names=["ensemble", "batch"],
        )
        self.world_rank = comm.get_world_rank()

        # store comm group parameters
        self.wrank = comm.get_rank("w")
        self.hrank = comm.get_rank("h")
        self.erank = comm.get_rank("ensemble")
        self.w_group = comm.get_group("w")
        self.h_group = comm.get_group("h")
        self.e_group = comm.get_group("ensemble")

        # initializing sht process groups just to be sure
        thd.init(self.h_group, self.w_group)

        if self.world_rank == 0:
            print(
                f"Running distributed tests on grid H x W x E = {self.grid_size_h} x {self.grid_size_w} x {self.grid_size_e}"
            )

        return

    def _destroy_comms(self):
        comm.cleanup()
        return

    def _split_helper(self, tensor, hdim=None, wdim=None):
        tensor_local = _split_helper(tensor, dim=hdim, group=self.h_group)
        tensor_local = _split_helper(tensor_local, dim=wdim, group=self.w_group)

        return tensor_local

    def _gather_helper(self, tensor, hdim=-2, wdim=-1):
        tensor_gather = _gather_helper(tensor, dim=hdim, group=self.h_group)
        tensor_gather = _gather_helper(tensor_gather, dim=wdim, group=self.w_group)

        return tensor_gather

    @parameterized.expand(
        [
            # "SNO",
            # "FCN3",
        ],
        skip_on_empty=True,
    )
    def test_distributed_model_checkpoint_restore(self, nettype, verbose=False):
        """
        Tests initialization of all the models and the forward and backward pass
        """
        self.params.nettype = nettype
        if nettype == "DebugNet":
            return

        # fix seed
        set_seed(333)

        # create temporary dir
        tmp_path = None
        if self.mpi_comm_rank == 0:
            tmpdir = tempfile.TemporaryDirectory(dir="/tmp")
            tmp_path = tmpdir.name
        tmp_path = self.mpi_comm.bcast(tmp_path, root=0)

        model = model_registry.get_model(self.params, multistep=False).to(self.device)

        # get state dict
        state_dict_full = checkpoint_helpers.gather_model_state_dict(model, grads=False)

        if self.mpi_comm_rank == 0:
            driver.Driver.save_checkpoint(
                checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"), model=model, checkpoint_mode="flexible"
            )
        self.mpi_comm.Barrier()

        # now init comms
        self._init_comms()

        model_dist = model_registry.get_model(self.params, multistep=False).to(self.device)

        # load checkpoint
        driver.Driver.restore_from_checkpoint(
            checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"),
            model=model_dist,
            loss=None,
            optimizer=None,
            scheduler=None,
            counters=None,
            checkpoint_mode="flexible",
            strict=False,
        )

        # compare parameters
        state_dict_gather_full = checkpoint_helpers.gather_model_state_dict(model_dist, grads=False)
        for key in state_dict_full.keys():
            with self.subTest(desc=f"parameter {key}"):
                param_full = state_dict_full[key].cpu()
                param_gather_full = state_dict_gather_full[key]
                self.assertTrue(
                    reduce_success(
                        compare_tensors(f"parameter {key}", param_full, param_gather_full, verbose=verbose), self.device
                    )
                )

        self.mpi_comm.Barrier()

        # cleanup
        self._destroy_comms()

    @parameterized.expand(
        [
            # ("SNO", 1e-4),
            ("FCN3", 1e-4),
        ],
        skip_on_empty=True,
    )
    def test_distributed_model_fwd_bwd(self, nettype, tol, verbose=False):
        """
        Tests forward backward pass of distributed model vs serial model
        """

        self.params.nettype = nettype
        if nettype == "DebugNet":
            return

        # fix seed
        set_seed(333)

        # create temporary dir
        tmp_path = None
        if self.mpi_comm_rank == 0:
            tmpdir = tempfile.TemporaryDirectory(dir="/tmp")
            tmp_path = tmpdir.name
        tmp_path = self.mpi_comm.bcast(tmp_path, root=0)

        multistep = self.params.n_future > 0
        model = model_registry.get_model(self.params, multistep=multistep).to(self.device)

        inp_shape = (
            self.params.batch_size,
            self.params.N_in_channels,
            self.params.img_shape_x,
            self.params.img_shape_y,
        )

        # prepare some dummy data
        inp_full = torch.randn(*inp_shape, dtype=torch.float32, device=self.device)
        inp_full.requires_grad = True

        # forward pass and save
        out_full = model(inp_full.unsqueeze(1)).flatten(1, 2).clone()
        loss_full = torch.sum(out_full)

        # perform backward pass
        loss_full.backward()
        igrad_full = inp_full.grad.clone()

        # store output:
        if self.mpi_comm_rank == 0:
            torch.save(out_full, os.path.join(tmp_path, "out_full.pt"))
            torch.save(igrad_full, os.path.join(tmp_path, "igrad_full.pt"))
            driver.Driver.save_checkpoint(
                checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"), model=model, checkpoint_mode="flexible"
            )
        self.mpi_comm.Barrier()

        # get also grad output
        state_dict_full = checkpoint_helpers.gather_model_state_dict(model, grads=True)

        # delete local model
        del model

        # now init comms
        self._init_comms()

        # create model, this times distributed
        model_dist = model_registry.get_model(self.params, multistep=multistep).to(self.device)

        # save reduction hooks
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

        # load checkpoint
        driver.Driver.restore_from_checkpoint(
            checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"),
            model=model_dist,
            loss=None,
            optimizer=None,
            scheduler=None,
            counters=None,
            checkpoint_mode="flexible",
            strict=False,
        )

        # split input
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local = model_dist(inp_local.unsqueeze(1)).flatten(1, 2)
        loss_dist = reduce_from_parallel_region(torch.sum(out_local), "model")
        loss_dist.backward()
        igrad_local = inp_local.grad.clone()

        # get weights and wgrads
        state_dict_gather_full = checkpoint_helpers.gather_model_state_dict(model_dist, grads=True)

        #############################################################
        # evaluate FWD pass
        #############################################################
        # output
        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose), self.device
                )
            )

        # loss
        with self.subTest(desc="loss"):
            self.assertTrue(
                reduce_success(compare_tensors("loss", loss_dist, loss_full, tol, tol, verbose=verbose), self.device)
            )

        #############################################################
        # evaluate BWD pass
        #############################################################
        # dgrad
        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        # wgrads
        for key in state_dict_full.keys():
            if key.endswith(".grad"):
                with self.subTest(desc=f"weight gradient {key}"):
                    wgrad_full = state_dict_full[key]
                    wgrad_gather_full = state_dict_gather_full["module." + key]
                    self.assertTrue(
                        reduce_success(
                            compare_tensors(
                                f"weight gradient {key}", wgrad_gather_full, wgrad_full, tol, tol, verbose=verbose
                            ),
                            self.device,
                        )
                    )

        # cleanup
        self._destroy_comms()

    @parameterized.expand(
        [
            # ("SFNO", 1e-6, 1e-6),
            # ("SNO", 1e-6, 1e-6),
            # ("FCN3", 1e-6, 1e-6),
        ],
        skip_on_empty=True,
    )
    def test_distributed_model_gradient_accumulation(self, nettype, atol, rtol, verbose=False):
        """
        Tests gradient accumulation with distributed models
        """

        self.params.nettype = nettype
        if nettype == "DebugNet":
            return

        # fix seed
        set_seed(333)

        # now init comms
        self._init_comms()

        # create model, this times distributed
        model_dist = model_registry.get_model(self.params, multistep=False).to(self.device)

        # save reduction hooks
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

        # get loss object
        self.params.losses = [{"type": "l2", "channel_weights": "constant"}]
        loss_obj = LossHandler(self.params).to(self.device)

        # input shape
        batch_size = self.params.batch_size * 2
        inp_shape = (batch_size, self.params.N_in_channels, self.params.img_shape_x, self.params.img_shape_y)

        # prepare some dummy data and split across ranks
        inp_full = torch.randn(*inp_shape, dtype=torch.float32, device=self.device)
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        tar_full = torch.randn_like(inp_full)
        tar_local = self._split_helper(tar_full, hdim=-2, wdim=-1)
        tar_local.requires_grad = False

        # perform a single forward:
        model_dist.zero_grad(set_to_none=True)
        out_single_local = model_dist(inp_local.unsqueeze(1)).flatten(1, 2)
        loss = loss_obj(out_single_local, tar_local)
        # backward pass
        loss.backward()

        # store the gradients
        state_dict_single_step = checkpoint_helpers.gather_model_state_dict(model_dist, grads=True)

        # split input
        inp_local_split = torch.split(inp_local, self.params.batch_size, dim=0)
        tar_local_split = torch.split(tar_local, self.params.batch_size, dim=0)

        # now perform multiple steps with gradient accumulation
        model_dist.zero_grad(set_to_none=True)
        inp_local_tmp = inp_local_split[0].detach().clone()
        inp_local_tmp.requires_grad = True
        tar_local_tmp = tar_local_split[0].detach().clone()

        # step 1
        with model_dist.no_sync():
            out_double_local = model_dist(inp_local_tmp.unsqueeze(1)).flatten(1, 2)
            loss = loss_obj(out_double_local, tar_local_tmp) / 2.0
        loss.backward()

        inp_local_tmp = inp_local_split[1].detach().clone()
        inp_local_tmp.requires_grad = True
        tar_local_tmp = tar_local_split[1].detach().clone()

        # step 2
        out = model_dist(inp_local_tmp.unsqueeze(1)).flatten(1, 2)
        loss = loss_obj(out, tar_local_tmp) / 2.0
        loss.backward()
        out_double_local = torch.cat([out_double_local, out], dim=0)

        # store the gradients
        state_dict_double_step = checkpoint_helpers.gather_model_state_dict(model_dist, grads=True)

        #############################################################
        # evaluate FWD pass
        #############################################################
        # output
        with self.subTest(desc="output"):
            out_single_gather = self._gather_helper(out_single_local, hdim=-2, wdim=-1)
            out_double_gather = self._gather_helper(out_double_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("output", out_double_gather, out_single_gather, atol, rtol, verbose=verbose),
                    self.device,
                )
            )

        #############################################################
        # evaluate BWD pass
        #############################################################
        # wgrads
        for key in state_dict_single_step.keys():
            if key.endswith(".grad"):
                with self.subTest(desc=f"weight gradient {key}"):
                    wgrad_single = state_dict_single_step[key]
                    wgrad_double = state_dict_double_step[key]
                    self.assertTrue(
                        reduce_success(
                            compare_tensors(
                                f"weight gradient {key}", wgrad_double, wgrad_single, atol, rtol, verbose=verbose
                            ),
                            self.device,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
