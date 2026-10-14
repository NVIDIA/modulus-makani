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
import unittest
from parameterized import parameterized

import torch
import torch.nn as nn
import torch.distributed as dist

import torch_harmonics as th
import torch_harmonics.distributed as thd

from makani.utils import comm

from makani.mpu.mappings import init_gradient_reduction_hooks

# distributed mlp + transformer engine availability flag
from makani.mpu.layers import DistributedMLP, _TE_AVAILABLE

# layer norm imports
from makani.models.common.layer_norm import GeometricInstanceNormS2
from makani.mpu.layer_norm import DistributedGeometricInstanceNormS2, DistributedInstanceNorm2d

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from .distributed_helpers import _init_grid, _split_helper, _gather_helper, reduce_success, teardown_comms
from ..testutils import disable_tf32, set_seed, compare_tensors


class TestDistributedLayers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)

    @classmethod
    def tearDownClass(cls):
        teardown_comms()

    def setUp(self):
        disable_tf32()

    def _split_helper(self, tensor, hdim=-2, wdim=-1):
        tensor_local = _split_helper(tensor, dim=hdim, group=self.h_group)
        tensor_local = _split_helper(tensor_local, dim=wdim, group=self.w_group)
        return tensor_local

    def _gather_helper(self, tensor, hdim=-2, wdim=-1):
        tensor_gather = _gather_helper(tensor, dim=hdim, group=self.h_group)
        tensor_gather = _gather_helper(tensor_gather, dim=wdim, group=self.w_group)

        return tensor_gather

    @parameterized.expand(
        [
            [180, 360, 256, 512, 32, 8, 1e-3],
            [181, 360, 181, 360, 1, 10, 1e-3],
            [180, 360, 128, 256, 32, 8, 1e-4],
            [181, 360, 91, 180, 1, 10, 1e-4],
            [128, 256, 256, 512, 32, 8, 1e-4],
            [91, 180, 181, 360, 1, 10, 1e-4],
            # the three filters of SFNO with scale_factor=8 on a 36 x 72 grid: truncated modes,
            # mixed grid types and an internal grid too small to split evenly
            [36, 72, 4, 9, 4, 8, 1e-4, "equiangular", "legendre-gauss", 4, 5],
            [4, 9, 4, 9, 4, 8, 1e-4, "legendre-gauss", "legendre-gauss", 4, 5],
            [4, 9, 36, 72, 4, 8, 1e-4, "legendre-gauss", "equiangular", 4, 5],
        ],
        skip_on_empty=True,
    )
    def test_distributed_spectral_conv(
        self,
        nlat_in,
        nlon_in,
        nlat_out,
        nlon_out,
        batch_size,
        num_chan,
        tol,
        grid_in="equiangular",
        grid_out="equiangular",
        lmax=None,
        mmax=None,
        verbose=False,
    ):
        B, C, Hi, Wi, Ho, Wo = batch_size, num_chan, nlat_in, nlon_in, nlat_out, nlon_out

        from makani.models.common import SpectralConv

        # set up handles; without explicit modes the inverse takes those of the forward
        forward_transform_local = th.RealSHT(nlat=Hi, nlon=Wi, lmax=lmax, mmax=mmax, grid=grid_in).to(self.device)
        inverse_transform_local = th.InverseRealSHT(
            nlat=Ho, nlon=Wo, lmax=forward_transform_local.lmax, mmax=forward_transform_local.mmax, grid=grid_out
        ).to(self.device)
        forward_transform_dist = thd.DistributedRealSHT(nlat=Hi, nlon=Wi, lmax=lmax, mmax=mmax, grid=grid_in).to(
            self.device
        )
        inverse_transform_dist = thd.DistributedInverseRealSHT(
            nlat=Ho, nlon=Wo, lmax=forward_transform_dist.lmax, mmax=forward_transform_dist.mmax, grid=grid_out
        ).to(self.device)

        set_seed(333)

        spect_conv_local = SpectralConv(
            forward_transform_local,
            inverse_transform_local,
            C,
            C,
            operator_type="dhconv",
            num_groups=1,
            bias=True,
            gain=1.0,
        ).to(self.device)

        spect_conv_dist = SpectralConv(
            forward_transform_dist,
            inverse_transform_dist,
            C,
            C,
            operator_type="dhconv",
            num_groups=1,
            bias=True,
            gain=1.0,
        ).to(self.device)

        # set up wgrad reductions
        spect_conv_dist = init_gradient_reduction_hooks(
            spect_conv_dist,
            device=self.device,
            reduction_buffer_count=1,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            verbose=False,
        )

        # make sure weights are the same:
        with torch.no_grad():
            weight = self._split_helper(spect_conv_local.weight, hdim=-1, wdim=None)
            spect_conv_dist.module.weight.copy_(weight)
            spect_conv_dist.module.bias.copy_(spect_conv_local.bias)

        # input
        inp_full = torch.randn((B, C, Hi, Wi), dtype=torch.float32, device=self.device)

        #############################################################
        # local transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full, res_full = spect_conv_local(inp_full)

        # create grads for backward; the residual is part of it, since the blocks use it as their skip
        # connection, and when the grids differ it is resampled through the transforms
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)
            rgrad_full = torch.randn_like(res_full)

        # BWD pass
        torch.autograd.backward([out_full, res_full], [ograd_full, rgrad_full])
        igrad_full = inp_full.grad.clone()
        wgrad_full = spect_conv_local.weight.grad.clone()
        bgrad_full = spect_conv_local.bias.grad.clone()

        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local, res_local = spect_conv_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full, hdim=-2, wdim=-1)
        rgrad_local = self._split_helper(rgrad_full, hdim=-2, wdim=-1)
        torch.autograd.backward([out_local, res_local], [ograd_local, rgrad_local])
        igrad_local = inp_local.grad.clone()
        wgrad_local = spect_conv_dist.module.weight.grad.clone()
        bgrad_local = spect_conv_dist.module.bias.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose), self.device
                )
            )

        with self.subTest(desc="residual"):
            res_gather_full = self._gather_helper(res_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("residual", res_gather_full, res_full, tol, tol, verbose=verbose), self.device
                )
            )

        #############################################################
        # evaluate input grads
        #############################################################
        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        #############################################################
        # evaluate Weight grads
        #############################################################
        with self.subTest(desc="weight gradients"):
            wgrad_gather_full = self._gather_helper(wgrad_local, hdim=-1, wdim=None)
            self.assertTrue(
                reduce_success(
                    compare_tensors("weight gradients", wgrad_gather_full, wgrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        with self.subTest(desc="bias gradients"):
            bgrad_gather_list = [torch.empty_like(bgrad_local) for _ in range(self.world_size)]
            bgrad_gather_list[self.world_rank] = bgrad_local
            dist.all_gather(bgrad_gather_list, bgrad_local, group=None)
            for idb, bgrad_gather_full in enumerate(bgrad_gather_list):
                self.assertTrue(
                    reduce_success(
                        compare_tensors(
                            f"bias gradient {idb}", bgrad_gather_full, bgrad_full, tol, tol, verbose=verbose
                        ),
                        self.device,
                    )
                )

    @parameterized.expand(
        [
            [256, 512, 32, 8, True, 1e-4],
            [181, 360, 1, 10, True, 1e-4],
            [256, 512, 32, 8, False, 1e-4],
            [181, 360, 1, 10, False, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_instance_norm_2d(self, nlat, nlon, batch_size, num_chan, affine, tol, verbose=False):
        B, C, H, W = batch_size, num_chan, nlat, nlon

        set_seed(333)

        # create local (serial) instance norm - using PyTorch's standard InstanceNorm2d
        norm_local = nn.InstanceNorm2d(
            num_features=C,
            eps=1e-5,
            affine=affine,
            track_running_stats=False,
        ).to(self.device)

        # create distributed instance norm
        norm_dist = DistributedInstanceNorm2d(
            num_features=C,
            eps=1e-5,
            affine=affine,
        ).to(self.device)

        # set up gradient reduction hooks for distributed version if affine=True
        if affine:
            norm_dist = init_gradient_reduction_hooks(
                norm_dist,
                device=self.device,
                reduction_buffer_count=1,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
                verbose=False,
            )
            norm_dist_handle = norm_dist.module
        else:
            norm_dist_handle = norm_dist

        # make sure weights are the same if affine=True
        if affine:
            with torch.no_grad():
                norm_dist_handle.weight.copy_(norm_local.weight)
                norm_dist_handle.bias.copy_(norm_local.bias)

        # input
        inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)

        #############################################################
        # local (serial) transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full = norm_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        if affine:
            wgrad_full = norm_local.weight.grad.clone()
            bgrad_full = norm_local.bias.grad.clone()

        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local = norm_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full, hdim=-2, wdim=-1)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        if affine:
            wgrad_local = norm_dist_handle.weight.grad.clone()
            bgrad_local = norm_dist_handle.bias.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose), self.device
                )
            )

        #############################################################
        # evaluate input grads
        #############################################################
        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        #############################################################
        # evaluate weight and bias grads
        #############################################################
        # weight gradients should be the same across all processes
        if affine:
            with self.subTest(desc="weight gradients"):
                wgrad_gather_list = [torch.empty_like(wgrad_local) for _ in range(self.world_size)]
                wgrad_gather_list[self.world_rank] = wgrad_local
                dist.all_gather(wgrad_gather_list, wgrad_local, group=None)
                for idw, wgrad_gather_full in enumerate(wgrad_gather_list):
                    self.assertTrue(
                        reduce_success(
                            compare_tensors(
                                f"weight gradient {idw}", wgrad_gather_full, wgrad_full, tol, tol, verbose=verbose
                            ),
                            self.device,
                        )
                    )

        # bias gradients should be the same across all processes
        if affine:
            with self.subTest(desc="bias gradients"):
                bgrad_gather_list = [torch.empty_like(bgrad_local) for _ in range(self.world_size)]
                bgrad_gather_list[self.world_rank] = bgrad_local
                dist.all_gather(bgrad_gather_list, bgrad_local, group=None)
                for idb, bgrad_gather_full in enumerate(bgrad_gather_list):
                    self.assertTrue(
                        reduce_success(
                            compare_tensors(
                                f"bias gradient {idb}", bgrad_gather_full, bgrad_full, tol, tol, verbose=verbose
                            ),
                            self.device,
                        )
                    )

    @parameterized.expand(
        [
            [181, 360, 1, 4, "equiangular", True, 1e-4],
            [181, 360, 1, 4, "equiangular", False, 1e-4],
            [180, 360, 1, 10, "legendre-gauss", True, 1e-4],
            [180, 360, 1, 10, "legendre-gauss", False, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_geometric_instance_norm_s2(
        self, nlat, nlon, batch_size, num_chan, grid_type, affine, tol, verbose=False
    ):
        B, C, H, W = batch_size, num_chan, nlat, nlon

        # set up layer norm parameters
        img_shape = (H, W)
        crop_shape = (H, W)
        crop_offset = (0, 0)
        eps = 1e-5

        set_seed(333)

        # create local (serial) layer norm
        norm_local = GeometricInstanceNormS2(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            grid_type=grid_type,
            num_features=C,
            eps=eps,
            affine=affine,
        ).to(self.device)

        # create distributed layer norm
        norm_dist = DistributedGeometricInstanceNormS2(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            grid_type=grid_type,
            num_features=C,
            eps=eps,
            affine=affine,
        ).to(self.device)

        # set up gradient reduction hooks for distributed version
        if affine:
            norm_dist = init_gradient_reduction_hooks(
                norm_dist,
                device=self.device,
                reduction_buffer_count=1,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
                verbose=False,
            )

        # make sure weights are the same if affine=True
        if affine:
            with torch.no_grad():
                norm_dist.module.weight.copy_(norm_local.weight)
                norm_dist.module.bias.copy_(norm_local.bias)

        # input
        inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)

        #############################################################
        # local (serial) transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full = norm_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        if affine:
            wgrad_full = norm_local.weight.grad.clone()
            bgrad_full = norm_local.bias.grad.clone()

        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local = norm_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full, hdim=-2, wdim=-1)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        if affine:
            wgrad_local = norm_dist.module.weight.grad.clone()
            bgrad_local = norm_dist.module.bias.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose), self.device
                )
            )

        #############################################################
        # evaluate input grads
        #############################################################
        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        #############################################################
        # evaluate weight and bias grads
        #############################################################
        # weight gradients should be the same across all processes
        if affine:
            with self.subTest(desc="weight gradients"):
                wgrad_gather_list = [torch.empty_like(wgrad_local) for _ in range(self.world_size)]
                wgrad_gather_list[self.world_rank] = wgrad_local
                dist.all_gather(wgrad_gather_list, wgrad_local, group=None)
                for idw, wgrad_gather_full in enumerate(wgrad_gather_list):
                    self.assertTrue(
                        reduce_success(
                            compare_tensors(
                                f"weight gradient {idw}", wgrad_gather_full, wgrad_full, tol, tol, verbose=verbose
                            ),
                            self.device,
                        )
                    )

        # bias gradients should be the same across all processes
        if affine:
            with self.subTest(desc="bias gradients"):
                bgrad_gather_list = [torch.empty_like(bgrad_local) for _ in range(self.world_size)]
                bgrad_gather_list[self.world_rank] = bgrad_local
                dist.all_gather(bgrad_gather_list, bgrad_local, group=None)
                for idb, bgrad_gather_full in enumerate(bgrad_gather_list):
                    self.assertTrue(
                        reduce_success(
                            compare_tensors(
                                f"bias gradient {idb}", bgrad_gather_full, bgrad_full, tol, tol, verbose=verbose
                            ),
                            self.device,
                        )
                    )

    @unittest.skipUnless(_TE_AVAILABLE and torch.cuda.is_available(), "transformer_engine (with CUDA) is not available")
    @parameterized.expand(
        [
            # nlat, nlon, batch, channels, hidden, input_format, bias, amp_mode, atol, rtol
            [32, 64, 2, 16, 64, "nchw", True, "bf16", 4e-2, 4e-2],
            [32, 64, 2, 16, 64, "nchw", True, "fp16", 2e-2, 2e-2],
            [32, 64, 2, 16, 64, "nchw", False, "bf16", 4e-2, 4e-2],
            [1, 64, 2, 16, 64, "traditional", True, "bf16", 4e-2, 4e-2],
            [1, 64, 2, 16, 64, "traditional", True, "fp16", 2e-2, 2e-2],
        ],
        skip_on_empty=True,
    )
    def test_distributed_mlp_te(
        self, nlat, nlon, batch_size, num_chan, hidden_dim, input_format, bias, amp_mode, atol, rtol, verbose=False
    ):
        """The transformer-engine MLP path must match the native distributed MLP under
        the same (reduced) precision.

        ``use_te`` only swaps the local GEMM for a ``te.Linear`` while the column/row
        tensor-parallel communication and the bias-after-reduce semantics stay in
        makani. We build the TE and native ``DistributedMLP`` with identical weights and
        run both forward passes inside the SAME ``autocast(bf16/fp16)`` context, then
        compare rank-locally: output, input grad and per-layer weight grad must agree.
        Running both sides at the same precision is the fair comparison (te.Linear has no
        fp32 compute path); the residual gap is the te.Linear-vs-conv2d/linear kernel
        difference at bf16/fp16, hence the loose, precision-appropriate tolerances. A
        wiring bug (bad transpose/bias/reduction) would produce O(1) errors instead.
        """
        # column-parallel fc1 shards the hidden dim over the matmul group
        if hidden_dim % comm.get_size("matmul") != 0:
            self.skipTest(f"hidden_dim {hidden_dim} not divisible by matmul size {comm.get_size('matmul')}")

        B, C, H, W = batch_size, num_chan, nlat, nlon
        amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16

        set_seed(333)

        common = dict(
            in_features=C,
            hidden_features=hidden_dim,
            out_features=C,
            output_bias=bias,
            input_format=input_format,
            act_layer=nn.GELU,
            drop_rate=0.0,  # this is important, otherwise RNG differences will spoil the test
            drop_type="iid",
            comm_name="matmul",
        )
        mlp_native = DistributedMLP(**common, use_te=False).to(self.device)
        mlp_te = DistributedMLP(**common, use_te=True).to(self.device)

        # copy the native weights/biases into the te module so the only difference is the
        # GEMM implementation. params stay fp32; autocast casts them on the fly. the te
        # path runs channels-last ("traditional"), so its weight is 2D (out_local, in_local)
        # and its bias is 1D (out_local,); the native nchw weight/bias carry trailing
        # singleton dims -- reshape into the target shape on copy.
        with torch.no_grad():
            mlp_te.fc1.weight.copy_(mlp_native.fc1.weight.reshape(mlp_te.fc1.weight.shape))
            mlp_te.fc2.weight.copy_(mlp_native.fc2.weight.reshape(mlp_te.fc2.weight.shape))
            if mlp_native.fc1.bias is not None:
                mlp_te.fc1.bias.copy_(mlp_native.fc1.bias.reshape(mlp_te.fc1.bias.shape))
            if mlp_native.fc2.bias is not None:
                mlp_te.fc2.bias.copy_(mlp_native.fc2.bias.reshape(mlp_te.fc2.bias.shape))

        # identical input to both modules
        if input_format == "nchw":
            inp = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)
        else:
            inp = torch.randn((B, H * W, C), dtype=torch.float32, device=self.device)

        inp_native = inp.clone().requires_grad_(True)
        inp_te = inp.clone().requires_grad_(True)

        # FWD under a shared bf16/fp16 autocast so both GEMMs run at the same precision
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out_native = mlp_native(inp_native)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out_te = mlp_te(inp_te)

        # BWD with identical upstream grad (backward runs outside autocast)
        ograd = torch.randn_like(out_native)
        out_native.backward(ograd)
        out_te.backward(ograd.clone())

        # cast to fp32 for the comparison (grads are fp32 already; outputs are low precision)
        with self.subTest(desc="te output"):
            self.assertTrue(
                reduce_success(
                    compare_tensors("te output", out_te.float(), out_native.float(), atol, rtol, verbose=verbose),
                    self.device,
                )
            )

        with self.subTest(desc="te input gradients"):
            self.assertTrue(
                reduce_success(
                    compare_tensors("te input gradients", inp_te.grad, inp_native.grad, atol, rtol, verbose=verbose),
                    self.device,
                )
            )

        with self.subTest(desc="te fc1 weight gradients"):
            te_wgrad = mlp_te.fc1.weight.grad.reshape(mlp_native.fc1.weight.grad.shape)
            self.assertTrue(
                reduce_success(
                    compare_tensors(
                        "te fc1 weight gradients", te_wgrad, mlp_native.fc1.weight.grad, atol, rtol, verbose=verbose
                    ),
                    self.device,
                )
            )

        with self.subTest(desc="te fc2 weight gradients"):
            te_wgrad = mlp_te.fc2.weight.grad.reshape(mlp_native.fc2.weight.grad.shape)
            self.assertTrue(
                reduce_success(
                    compare_tensors(
                        "te fc2 weight gradients", te_wgrad, mlp_native.fc2.weight.grad, atol, rtol, verbose=verbose
                    ),
                    self.device,
                )
            )

        if bias:
            with self.subTest(desc="te fc2 bias gradients"):
                te_bgrad = mlp_te.fc2.bias.grad.reshape(mlp_native.fc2.bias.grad.shape)
                self.assertTrue(
                    reduce_success(
                        compare_tensors(
                            "te fc2 bias gradients", te_bgrad, mlp_native.fc2.bias.grad, atol, rtol, verbose=verbose
                        ),
                        self.device,
                    )
                )


if __name__ == "__main__":
    unittest.main()
