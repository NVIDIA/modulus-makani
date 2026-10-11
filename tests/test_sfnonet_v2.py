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

"""Tests for the SFNOv2 network: channel layout, cutoff, symmetry and initialization."""

import unittest

import torch

from makani.models.model_registry import _check_vector_wind_normalization
from makani.models.networks.sfnonet_v2 import SphericalFourierNeuralOperatorNetV2, _compute_channel_layout
from .testutils import disable_tf32, set_seed


CHANNEL_NAMES = ["u10m", "v10m", "t2m", "u500", "v500", "z500"]


class TestSFNOv2(unittest.TestCase):
    def setUp(self):
        disable_tf32()
        set_seed(333)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inp_shape = (32, 64)

    def _model(self, **kwargs):
        args = dict(
            inp_shape=self.inp_shape,
            out_shape=self.inp_shape,
            inp_chans=len(CHANNEL_NAMES),
            out_chans=len(CHANNEL_NAMES),
            channel_names=CHANNEL_NAMES,
            scale_factor=2,
            embed_dim=16,
            num_layers=2,
        )
        args.update(kwargs)
        return SphericalFourierNeuralOperatorNetV2(**args).to(self.device)

    def test_channel_layout(self):
        layout = _compute_channel_layout(CHANNEL_NAMES, ["xzen", "xoro"], 1, 2 * 7 + 1, 6)
        self.assertEqual(layout["wind_out"], [0, 1, 3, 4])
        self.assertEqual(layout["scalar_out"], [2, 5])
        # wind pairs repeat with the dynamic channel period 7 for the second history step
        self.assertEqual(layout["wind_in"], [0, 1, 3, 4, 7, 8, 10, 11])
        self.assertEqual(sorted(layout["scalar_in"] + layout["wind_in"]), list(range(15)))
        with self.assertRaises(ValueError):
            _compute_channel_layout(CHANNEL_NAMES, [], 0, 7, 6)

    def test_cutoff(self):
        # 16x32 Legendre-Gauss internal grid: lmax = min(16 - 1, 32 // 2) = 15, used as mmax everywhere
        model = self._model()
        self.assertEqual(model.lmax, 15)
        for t in (model.project_in.sht, model.project_in.vsht, model.synthesize_internal.isht, model.synthesize_out.ivsht, model.sht, model.isht):
            self.assertEqual((t.lmax, t.mmax), (15, 15))
        # an explicit cutoff may exceed the exact limit, but not what the 16x32 internal grid resolves
        self.assertEqual(self._model(lmax=16).lmax, 16)
        with self.assertRaises(ValueError):
            self._model(lmax=17)
        # the LESS setting: internal grid = data grid, equiangular with Clenshaw-Curtis, cutoff beyond the
        # exact limit of 15 (resolvable up to 32)
        model = self._model(scale_factor=1, sht_grid_type="equiangular", lmax=24)
        self.assertEqual((model.lmax, model.project_in.sht.lmax, model.sht.lmax), (24, 24, 24))
        self.assertEqual((model.sht.nlat, model.sht.nlon, model.sht.grid), (32, 64, "equiangular"))
        with self.assertRaises(ValueError):
            self._model(scale_factor=1, sht_grid_type="equiangular", lmax=33)

    def test_forward_backward(self):
        for kwargs in (
            {},
            {"n_history": 1, "inp_chans": 2 * len(CHANNEL_NAMES)},
            {"channel_names": ["t2m", "z500"], "inp_chans": 2, "out_chans": 2},
            {"pos_embed": "none", "use_film": False, "skip": "linear", "normalization_layer": "instance_norm_s2"},
            {"out_shape": (16, 32)},
            {"scale_factor": 1, "sht_grid_type": "equiangular", "lmax": 24},
            {"num_groups": 4},
            {"separable": True},
        ):
            with self.subTest(**kwargs):
                model = self._model(**kwargs)
                inp = torch.randn(2, model.inp_chans, *self.inp_shape, device=self.device, requires_grad=True)
                out = model(inp)
                self.assertEqual(tuple(out.shape), (2, model.out_chans, *model.out_shape))
                out.sum().backward()
                self.assertTrue(torch.isfinite(inp.grad).all())

    def test_longitude_rotation_equivariance(self):
        # rotating the input by 180 degrees about the polar axis must rotate the output, exactly up to round-off,
        # also with a non-trivial latitude embedding and FiLM
        model = self._model()
        with torch.no_grad():
            model.pos_embed.normal_()
            for blk in model.blocks:
                blk.film.fc2.weight.normal_(std=0.1)
        model.eval()
        inp = torch.randn(2, model.inp_chans, *self.inp_shape, device=self.device)
        shift = self.inp_shape[1] // 2
        with torch.no_grad():
            out = model(inp)
            out_rot = model(torch.roll(inp, shift, dims=-1))
        self.assertTrue(torch.allclose(torch.roll(out, shift, dims=-1), out_rot, atol=1e-4, rtol=1e-4))

    def test_filter_parametrization(self):
        # spectral weight count: embed_dim^2 * lmax / num_groups, or embed_dim * lmax when separable
        def n_filter_params(model):
            return sum(p.numel() for p in model.blocks[0].filter.parameters())

        dense = n_filter_params(self._model())
        self.assertEqual(dense, 16 * 16 * 15)
        self.assertEqual(n_filter_params(self._model(num_groups=4)), dense // 4)
        self.assertEqual(n_filter_params(self._model(separable=True)), 16 * 15)
        # transforms are shared when the internal grid is the data grid
        model = self._model(scale_factor=1, sht_grid_type="equiangular", lmax=24)
        self.assertIs(model.project_internal, model.project_in)
        self.assertIs(model.synthesize_out, model.synthesize_internal)
        model = self._model()
        self.assertIsNot(model.project_internal, model.project_in)

    def test_vector_wind_normalization_guard(self):
        self.assertTrue(SphericalFourierNeuralOperatorNetV2.requires_vector_wind_normalization)
        scale = torch.tensor([3.0, 3.0, 1.0, 9.0, 9.0, 1.0])
        bias = torch.tensor([0.0, 0.0, 280.0, 0.0, 0.0, 5000.0])
        _check_vector_wind_normalization(bias, scale, CHANNEL_NAMES, "SFNOv2")
        # stats without wind pairs or without normalization pass trivially
        _check_vector_wind_normalization(bias, scale, ["t2m"] * 6, "SFNOv2")
        _check_vector_wind_normalization(None, None, CHANNEL_NAMES, "SFNOv2")
        # componentwise wind means, as plain z-scoring produces
        with self.assertRaisesRegex(ValueError, "u500"):
            bad_bias = bias.clone()
            bad_bias[3] = 6.65
            _check_vector_wind_normalization(bad_bias, scale, CHANNEL_NAMES, "SFNOv2")
        # componentwise wind scales
        with self.assertRaisesRegex(ValueError, "u10m"):
            bad_scale = scale.clone()
            bad_scale[1] = 2.5
            _check_vector_wind_normalization(bias, bad_scale, CHANNEL_NAMES, "SFNOv2")

    def test_wind_roundtrip(self):
        # a band-limited vector field passes projection and synthesis unchanged
        model = self._model()
        x = torch.randn(2, 6, *self.inp_shape, device=self.device)
        with torch.no_grad():
            coeffs = model.project_in(x, model.scalar_out, model.wind_out)
            x_bl = model.synthesize_out(*coeffs, model.out_inverse_perm)
            coeffs = model.project_in(x_bl, model.scalar_out, model.wind_out)
            x_rt = model.synthesize_out(*coeffs, model.out_inverse_perm)
        self.assertTrue(torch.allclose(x_bl, x_rt, atol=1e-4, rtol=1e-4))

    def test_initialization(self):
        # unit-variance input keeps roughly unit variance through the residual stream, and the fresh network
        # predicts the input (big skip) plus a small correction
        model = self._model(embed_dim=64, num_layers=6)
        model.eval()
        stds = []
        hooks = [blk.register_forward_hook(lambda m, i, o: stds.append(o.float().std().item())) for blk in model.blocks]
        # band-limited input with unit variance: white noise would lose most of its variance to the cutoff,
        # which says nothing about the initialization
        with torch.no_grad():
            inp = torch.randn(4, model.inp_chans, *self.inp_shape, device=self.device)
            coeffs = model.project_in(inp, model.scalar_out, model.wind_out)
            inp = model.synthesize_out(*coeffs, model.out_inverse_perm)
            inp = inp / inp.std(dim=(-2, -1), keepdim=True)
            out = model(inp)
        for h in hooks:
            h.remove()
        for s in stds:
            self.assertGreater(s, 0.5)
            self.assertLess(s, 2.0)
        self.assertLess((out - inp).std().item(), 1.0)
        self.assertGreater((out - inp).std().item(), 0.1)


if __name__ == "__main__":
    unittest.main()
