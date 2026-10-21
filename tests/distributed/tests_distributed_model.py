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
import shutil
import tempfile
import unittest
from parameterized import parameterized

import torch
import torch.distributed as dist

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
    _init_grid,
    _split_helper,
    _gather_helper,
    reduce_success,
    sync_and_barrier,
    teardown_comms,
)
from ..testutils import disable_tf32, set_seed, compare_tensors, single_step_rollout


def _count_distributed_modules(model):
    # the distributed layers of makani.mpu and torch_harmonics.distributed all carry the prefix
    return sum(type(module).__name__.startswith("Distributed") for module in model.modules())


def _scaled_atol(reference, tol):
    # the rounding error of a sum scales with its summands, not with its result: an entry
    # that cancels to a small value carries the error of the tensor's scale, so the
    # absolute tolerance follows the largest entry; never below tol itself
    return tol * max(1.0, reference.abs().max().item())


def _layer_gradient_scale(gradients, key):
    """
    The reference a gradient's absolute tolerance scales with: the layer's largest gradient entry.

    A bias gradient sums the same incoming gradients as the weight gradient of
    its layer, so its rounding error follows that scale even when the sum
    itself cancels -- as it does exactly for a bias followed by an instance
    norm, which removes the per-channel mean and with it the bias.
    """
    reference = gradients[key]
    if key.endswith(".bias.grad"):
        weight_key = key[: -len(".bias.grad")] + ".weight.grad"
        if weight_key in gradients:
            return torch.stack([reference.abs().max(), gradients[weight_key].abs().max().to(reference.device)])
    return reference


class TestDistributedModel(unittest.TestCase):
    """
    Compares distributed models against the same model built serially.

    The comms are set up once for the class. The serial reference is built
    inside ``comm.serial()`` next to the distributed model, so both live in the
    same process group and no test sets up or tears down the comms itself.
    """

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)

        # shared scratch directory: rank 0 picks the path, every rank reads and writes there
        tmpdir = [tempfile.mkdtemp(prefix="makani_model_test_", dir="/tmp") if cls.world_rank == 0 else None]
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

        self.params = get_default_parameters()

        self.params.history_normalization_mode = "none"

        # set image shape, local/crop shapes, offsets, and resampled shapes
        # consistently — see set_image_shape for the rationale.
        set_image_shape(self.params, h=36, w=72)

        # also set the batch size for testing
        self.params.batch_size = 4

        # start every test with matched ranks and a drained device
        sync_and_barrier()

    def _checkpoint_path(self, name):
        return os.path.join(self.tmpdir, f"{name}_{self.id().split('.')[-1]}.pt")

    def _build_serial_reference(self):
        """Build the serial model and save its checkpoint, on every rank alike."""
        # the checkpoint gathers over the model group when it is distributed, so
        # saving the serial model is serial too, and only rank 0 writes
        checkpoint_path = self._checkpoint_path("serial")
        with comm.serial():
            model = model_registry.get_model(self.params).to(self.device)
            if self.world_rank == 0:
                driver.Driver.save_checkpoint(checkpoint_path=checkpoint_path, model=model, checkpoint_mode="flexible")
        sync_and_barrier()
        return model, checkpoint_path

    def _build_distributed(self, checkpoint_path=None, hooks=False):
        model = model_registry.get_model(self.params).to(self.device)
        if hooks:
            model = init_gradient_reduction_hooks(
                model,
                device=self.device,
                reduction_buffer_count=1,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
                verbose=False,
            )
        if checkpoint_path is not None:
            driver.Driver.restore_from_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                loss=None,
                optimizer=None,
                scheduler=None,
                counters=None,
                checkpoint_mode="flexible",
                # every parameter has to come from the serial reference: a missing one would
                # silently keep this rank's own random init
                strict=True,
            )
        return model

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
            "SFNO",
            "SNO",
            "FCN3",
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

        model, checkpoint_path = self._build_serial_reference()
        with comm.serial():
            state_dict_full = checkpoint_helpers.gather_model_state_dict(model, grads=False)

        model_dist = self._build_distributed(checkpoint_path=checkpoint_path)

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

    @parameterized.expand(["SFNO", "SNO", "FCN3"], skip_on_empty=True)
    def test_serial_build_has_no_distributed_layers(self, nettype):
        """
        A model built inside comm.serial() is serial, while the comms are up.

        The serial references of the other tests rely on this: a model has to
        choose its parallel layout when it is constructed, not when it runs.
        """
        self.params.nettype = nettype

        with comm.serial():
            model = model_registry.get_model(self.params)
        self.assertEqual(_count_distributed_modules(model), 0)

        # without model parallelism the distributed build is serial as well, and there is nothing to contrast
        if comm.get_size("model") > 1:
            model_dist = model_registry.get_model(self.params)
            self.assertGreater(_count_distributed_modules(model_dist), 0)

    @parameterized.expand(
        [
            ("SFNO", 1e-4),
            ("SNO", 1e-4),
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

        model, checkpoint_path = self._build_serial_reference()

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
        out_full = model(single_step_rollout(inp_full.unsqueeze(1))).squeeze(1).clone()
        loss_full = torch.sum(out_full)

        # perform backward pass
        loss_full.backward()
        igrad_full = inp_full.grad.clone()

        # get also grad output
        with comm.serial():
            state_dict_full = checkpoint_helpers.gather_model_state_dict(model, grads=True)

        # delete local model
        del model

        # create model, this times distributed
        model_dist = self._build_distributed(checkpoint_path=checkpoint_path, hooks=True)

        # split input
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local = model_dist(single_step_rollout(inp_local.unsqueeze(1))).squeeze(1)
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
                    compare_tensors(
                        "output", out_gather_full, out_full, _scaled_atol(out_full, tol), tol, verbose=verbose
                    ),
                    self.device,
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
                    compare_tensors(
                        "input gradients",
                        igrad_gather_full,
                        igrad_full,
                        _scaled_atol(igrad_full, tol),
                        tol,
                        verbose=verbose,
                    ),
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
                                f"weight gradient {key}",
                                wgrad_gather_full,
                                wgrad_full,
                                _scaled_atol(_layer_gradient_scale(state_dict_full, key), tol),
                                tol,
                                verbose=verbose,
                            ),
                            self.device,
                        )
                    )

    @parameterized.expand(
        [
            ("SFNO", 1e-6, 1e-6),
            ("SNO", 1e-6, 1e-6),
            ("FCN3", 1e-6, 1e-6),
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

        # create model, this times distributed
        model_dist = self._build_distributed(hooks=True)

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
        out_single_local = model_dist(single_step_rollout(inp_local.unsqueeze(1))).squeeze(1)
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
            out_double_local = model_dist(single_step_rollout(inp_local_tmp.unsqueeze(1))).squeeze(1)
            loss = loss_obj(out_double_local, tar_local_tmp) / 2.0
        loss.backward()

        inp_local_tmp = inp_local_split[1].detach().clone()
        inp_local_tmp.requires_grad = True
        tar_local_tmp = tar_local_split[1].detach().clone()

        # step 2
        out = model_dist(single_step_rollout(inp_local_tmp.unsqueeze(1))).squeeze(1)
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
