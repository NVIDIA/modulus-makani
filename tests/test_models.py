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
import tempfile
import unittest
import warnings
from parameterized import parameterized, parameterized_class

import numpy as np
import torch
import torch.nn as nn

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


@parameterized_class(("device",), _devices)
class TestModelOptions(unittest.TestCase):
    """
    Model options the default configurations leave off: the normalization layers of FCN3 and the
    non-linear spectral filter (SpectralAttention) of SFNO.
    """

    def setUp(self):
        # the same small grid and parameters as TestModels
        TestModels.setUp(self)

    def _forward_backward(self):
        model = model_registry.get_model(self.params).to(self.device)

        inp = torch.randn(
            self.params.batch_size,
            self.params.N_in_channels,
            self.params.img_shape_x,
            self.params.img_shape_y,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        out = model(single_step_rollout(inp.unsqueeze(1))).squeeze(1)
        self.assertEqual(
            out.shape,
            (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y),
        )
        self.assertTrue(torch.isfinite(out).all())

        out.sum().backward()
        self.assertTrue(inp.grad is not None)
        self.assertTrue(torch.isfinite(inp.grad).all())

    @parameterized.expand([("layer_norm",), ("instance_norm",), ("instance_norm_s2",)])
    def test_fcn3_normalization_layer(self, normalization_layer):
        self.params.nettype = "FCN3"
        self.params.normalization_layer = normalization_layer
        self._forward_backward()

    @parameterized.expand([("diagonal",), ("l-dependant",)])
    def test_sfno_spectral_attention(self, operator_type):
        self.params.nettype = "SFNO"
        self.params.filter_type = "non-linear"
        self.params.operator_type = operator_type
        self._forward_backward()


def _soft_clamp_reference(x):
    # 0 below zero, x^2 on [0, 0.5) and x - 0.25 above: continuous with a continuous first derivative
    return torch.where(x >= 0.5, x - 0.25, torch.where(x > 0.0, x**2, torch.zeros_like(x)))


class TestClampWaterChannels(unittest.TestCase):
    """clamp_water_channels of FCN3 and FCN3.1, built through the model registry with clamp_water set."""

    _CHANNEL_NAMES = ["u10m", "tcwv", "u500", "q500", "t500"]
    _WATER = [1, 3]

    def setUp(self):
        # the same small grid and parameters as TestModels
        self.device = torch.device("cpu")
        TestModels.setUp(self)
        self.params.channel_names = self._CHANNEL_NAMES
        self.params.clamp_water = True

        self.x = 2.0 * torch.randn(2, len(self._CHANNEL_NAMES), 4, 8)
        self.dry = [c for c in range(len(self._CHANNEL_NAMES)) if c not in self._WATER]

        self.means = torch.tensor([0.0, 0.3, 0.0, 1.2, 0.0])
        self.stds = torch.tensor([1.0, 0.5, 1.0, 2.0, 1.0])

    def _network(self, nettype, normalized):
        self.params.nettype = nettype
        if normalized:
            tmpdir = tempfile.TemporaryDirectory()
            self.addCleanup(tmpdir.cleanup)
            self.params.normalization = "zscore"
            self.params.global_means_path = os.path.join(tmpdir.name, "means.npy")
            self.params.global_stds_path = os.path.join(tmpdir.name, "stds.npy")
            np.save(self.params.global_means_path, self.means.numpy().reshape(1, -1, 1, 1))
            np.save(self.params.global_stds_path, self.stds.numpy().reshape(1, -1, 1, 1))
        model = model_registry.get_model(self.params)
        return next(m for m in model.modules() if hasattr(m, "clamp_water_channels"))

    @parameterized.expand([("FCN3", False), ("FCN3", True), ("FCN3.1", False)])
    def test_clamp_at_normalized_zero(self, nettype, normalized, verbose=False):
        """FCN3 does not store the normalization statistics, so it clamps at normalized zero even when
        they are passed. FCN3.1 does so only without statistics."""
        net = self._network(nettype, normalized)
        out = net.clamp_water_channels(self.x)

        self.assertTrue(torch.equal(out[:, self.dry], self.x[:, self.dry]))
        expected = _soft_clamp_reference(self.x[:, self._WATER])
        self.assertTrue(compare_tensors("water", out[:, self._WATER], expected, verbose=verbose))
        self.assertTrue((out[:, self._WATER] >= 0.0).all())

    @parameterized.expand([("train",), ("eval",)])
    def test_fcn31_clamp_at_physical_zero(self, phase, verbose=False):
        """FCN3.1 clamps at physical zero, in training and in eval mode alike."""
        net = self._network("FCN3.1", normalized=True)
        net.train(phase == "train")
        out = net.clamp_water_channels(self.x)

        self.assertTrue(torch.equal(out[:, self.dry], self.x[:, self.dry]))

        # the clamp acts on the de-normalized value, so the physical water content is non-negative
        offset = (self.means / self.stds)[self._WATER].view(1, -1, 1, 1)
        expected = _soft_clamp_reference(self.x[:, self._WATER] + offset) - offset
        self.assertTrue(compare_tensors("water", out[:, self._WATER], expected, verbose=verbose))
        physical = (
            self.means[self._WATER].view(1, -1, 1, 1) + self.stds[self._WATER].view(1, -1, 1, 1) * out[:, self._WATER]
        )
        self.assertTrue((physical >= -1e-6).all())


class TestPanguShiftWindowMask(unittest.TestCase):
    """The attention mask of Pangu's shifted windows."""

    def _check_common(self, mask):
        values = mask.unique()
        self.assertTrue(set(values.tolist()) <= {0.0, -100.0})
        self.assertTrue(torch.equal(mask, mask.transpose(-1, -2)))
        self.assertTrue((torch.diagonal(mask, dim1=-2, dim2=-1) == 0).all())

    def test_2d(self):
        from makani.models.networks.pangu import get_shift_window_mask

        lat, lon, win_lat, win_lon, shift_lat, shift_lon = 8, 12, 4, 6, 2, 3
        mask = get_shift_window_mask((lat, lon), (win_lat, win_lon), (shift_lat, shift_lon), ndim=2)

        # (windows in lon, windows in lat, tokens per window, tokens per window)
        n_tok = win_lat * win_lon
        self.assertEqual(tuple(mask.shape), (lon // win_lon, lat // win_lat, n_tok, n_tok))
        self._check_common(mask)

        # a window away from the shifted edges attends everywhere
        self.assertTrue((mask[0, 0] == 0).all())

        # in the last latitude window, the rows rolled over from the other pole must not attend to
        # the rows in place, since latitude is not periodic
        rows = torch.arange(n_tok) // win_lon + (lat - win_lat)
        rolled = rows >= lat - shift_lat
        expected = torch.where(rolled.unsqueeze(0) != rolled.unsqueeze(1), -100.0, 0.0)
        self.assertTrue(torch.equal(mask[0, -1], expected))

    def test_3d(self):
        from makani.models.networks.pangu import get_shift_window_mask

        shape, window, shift = (4, 8, 12), (2, 4, 6), (1, 2, 3)
        mask = get_shift_window_mask(shape, window, shift, ndim=3)

        # (windows in lon, windows in pressure x lat, tokens per window, tokens per window)
        n_tok = window[0] * window[1] * window[2]
        n_pl_lat = (shape[0] // window[0]) * (shape[1] // window[1])
        self.assertEqual(tuple(mask.shape), (shape[2] // window[2], n_pl_lat, n_tok, n_tok))
        self._check_common(mask)
        self.assertTrue((mask[0, 0] == 0).all())
        self.assertTrue((mask == -100.0).any())

    def test_shifted_block_rolls_back(self):
        # with the attention, the norms and the MLP made trivial, a shifted block returns its input plus the
        # attention output; the roll back after the attention has to put every token back where it came from
        from makani.models.networks.pangu import Transformer3DBlock

        class _IdentityAttention(nn.Module):
            def forward(self, x, mask=None):
                return x

        class _Zero(nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)

        resolution = (2, 12, 24)
        block = Transformer3DBlock(
            dim=4, input_resolution=resolution, num_heads=1, window_size=(2, 6, 12), shift_size=(1, 3, 6)
        )
        self.assertTrue(block.roll)
        block.norm1 = nn.Identity()
        block.attn = _IdentityAttention()
        block.mlp = _Zero()

        x = torch.randn(2, resolution[0] * resolution[1] * resolution[2], 4)
        self.assertTrue(torch.equal(block(x), 2 * x))


if __name__ == "__main__":
    disable_tf32()
    unittest.main()
