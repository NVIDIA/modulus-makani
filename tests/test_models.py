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

import sys
import os
import unittest
import warnings
from parameterized import parameterized, parameterized_class

import torch

from makani.models import model_registry
from makani.utils import checkpoint_helpers
from makani.utils import LossHandler

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, get_default_parameters, compare_tensors, single_step_rollout

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


@parameterized_class(("device",), _devices)
class TestModels(unittest.TestCase):

    def setUp(self):

        disable_tf32()

        self.params = get_default_parameters()

        self.params.history_normalization_mode = "none"

        # generating the image logic that is typically used by the dataloader
        self.params.img_shape_x = 36
        self.params.img_shape_y = 72
        self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
        self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
        self.params.img_shape_x_resampled = self.params.img_shape_x
        self.params.img_shape_y_resampled = self.params.img_shape_y
        self.params.img_local_offset_x = 0
        self.params.img_local_offset_y = 0
        self.params.img_crop_offset_x = 0
        self.params.img_crop_offset_y = 0

        # also set the batch size for testing
        self.params.batch_size = 4

        # set seed
        set_seed(333)

        return

    @parameterized.expand(model_registry.list_models(), skip_on_empty=True)
    def test_model(self, nettype):
        """
        Tests initialization of all the models and the forward and backward pass
        """
        self.params.nettype = nettype
        if nettype == "DebugNet":
            return

        model = model_registry.get_model(self.params).to(self.device)

        inp_shape = (
            self.params.batch_size,
            self.params.N_in_channels,
            self.params.img_shape_x,
            self.params.img_shape_y,
        )
        out_shape = (
            self.params.batch_size,
            self.params.N_out_channels,
            self.params.img_shape_x,
            self.params.img_shape_y,
        )

        # prepare some dummy data
        inp = torch.randn(*inp_shape, dtype=torch.float32, device=self.device)
        inp.requires_grad = True

        # forward pass and check shapes
        out = model(single_step_rollout(inp.unsqueeze(1))).squeeze(1)
        self.assertEqual(out.shape, out_shape)

        # backward pass and check gradients are not None
        out = torch.sum(out)
        out.backward()
        self.assertTrue(inp.grad is not None)
        self.assertEqual(inp.grad.shape, inp_shape)

    @parameterized.expand(
        [
            # Split-batch vs. single-batch gradient accumulation sums a full
            # model's forward/backward in a different order (batch 4 vs. 2+2),
            # which cuBLAS/cuDNN can resolve with a different reduction/tiling
            # algorithm even at full IEEE float32 precision -- confirmed this
            # is not a residual TF32 effect (disable_tf32() already runs in
            # setUp, and its new-style flags -- torch.backends.cuda.matmul.fp32_precision
            # == "ieee", allow_tf32 == False -- were checked directly). That
            # noise compounds across a full model's many layers and varies by
            # GPU architecture (observed passing on H100/V100, failing on
            # GB200/GB10 at the tolerances this list used to have -- AFNO's
            # worst violation was 1.22e-5 against a 1.05e-5 bound, AFNOv2's
            # was 2.98e-6 against 2.02e-6, FNO's raw worst-case was ~4.9e-6
            # against a 1.36e-6 bound). Loosened uniformly with headroom over
            # those, rather than tuned per model to one architecture's numbers.
            ("AFNO", 2e-5, 2e-5),
            ("AFNOv2", 2e-5, 2e-5),
            ("FNO", 2e-5, 2e-5),
            ("ViT", 2e-5, 2e-5),
            ("SFNO", 2e-5, 2e-5),
            ("SNO", 2e-5, 2e-5),
            ("FCN3", 2e-5, 2e-5),
            ("Pangu", 2e-5, 2e-5),
        ],
        skip_on_empty=True,
    )
    def test_gradient_accumulation(self, nettype, atol, rtol, verbose=False):
        """
        Tests initialization of all the models and the forward and backward pass
        """
        self.params.nettype = nettype
        if nettype == "DebugNet":
            return

        # get loss object
        self.params.losses = [{"type": "l2", "channel_weights": "constant"}]
        loss_obj = LossHandler(self.params).to(self.device)

        # get model
        model = model_registry.get_model(self.params).to(self.device)

        batch_size = self.params.batch_size * 2
        inp_shape = (batch_size, self.params.N_in_channels, self.params.img_shape_x, self.params.img_shape_y)

        # prepare some dummy data
        inp = torch.randn(*inp_shape, dtype=torch.float32, device=self.device)
        tar = torch.randn_like(inp, dtype=torch.float32, device=self.device)
        inp.requires_grad = True
        tar.requires_grad = False

        # forward pass
        model.zero_grad(set_to_none=True)

        out_single = model(single_step_rollout(inp.unsqueeze(1))).squeeze(1)
        loss = loss_obj(out_single, tar)
        # backward pass
        loss.backward()
        # igrad
        igrad_single = inp.grad.clone()

        # store the gradients
        state_dict_single_step = checkpoint_helpers.gather_model_state_dict(model, grads=True)

        # split input
        inp_split = torch.split(inp, self.params.batch_size, dim=0)
        tar_split = torch.split(tar, self.params.batch_size, dim=0)

        # forward pass
        model.zero_grad(set_to_none=True)
        inp_tmp = inp_split[0].detach().clone()
        inp_tmp.requires_grad = True
        tar_tmp = tar_split[0].detach().clone()

        # step 1
        out_double = model(single_step_rollout(inp_tmp.unsqueeze(1))).squeeze(1)
        loss = loss_obj(out_double, tar_tmp) / 2.0
        loss.backward()
        igrad_double = inp_tmp.grad.clone()

        inp_tmp = inp_split[1].detach().clone()
        inp_tmp.requires_grad = True
        tar_tmp = tar_split[1].detach().clone()

        # step 2
        out = model(single_step_rollout(inp_tmp.unsqueeze(1))).squeeze(1)
        loss = loss_obj(out, tar_tmp) / 2.0
        loss.backward()
        out_double = torch.cat([out_double, out], dim=0)
        igrad_double = torch.cat([igrad_double, inp_tmp.grad.clone()], dim=0)

        # store gradients
        state_dict_double_step = checkpoint_helpers.gather_model_state_dict(model, grads=True)

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="output"):
            self.assertTrue(compare_tensors("output", out_double, out_single, atol, rtol, verbose))

        #############################################################
        # evaluate BWD pass
        #############################################################
        # igrads
        with self.subTest(desc="input gradient"):
            self.assertTrue(compare_tensors("input gradient", igrad_double, igrad_single, atol, rtol, verbose))

        # wgrads
        with torch.no_grad():
            for key in state_dict_single_step.keys():
                if key.endswith(".grad"):
                    wgrad_single = state_dict_single_step[key]
                    wgrad_double = state_dict_double_step[key]
                    with self.subTest(desc=f"weight gradient {key}"):
                        self.assertTrue(
                            compare_tensors(f"weight gradient {key}", wgrad_double, wgrad_single, atol, rtol, verbose)
                        )


class TestGetModelMultistep(unittest.TestCase):
    """``multistep`` no longer selects a wrapper; asking for it warns rather than doing nothing silently."""

    def setUp(self):
        self.params = get_default_parameters()
        self.params.nettype = "SFNO"

    def test_multistep_warns(self):
        with self.assertWarnsRegex(FutureWarning, "multistep is deprecated"):
            model_registry.get_model(self.params, multistep=True)

    def test_default_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            model_registry.get_model(self.params)
            model_registry.get_model(self.params, multistep=False)

    def test_unknown_arguments_are_refused(self):
        with self.assertRaises(TypeError):
            model_registry.get_model(self.params, multi_step=True)


if __name__ == "__main__":
    disable_tf32()
    unittest.main()
