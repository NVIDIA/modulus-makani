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
import math
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

    # the DISCO convolutions of SNO with scale_factor=8 on a 36 x 72 grid: the encoder onto the internal
    # 4 x 9 grid, the local convolution of the blocks there, whose support covers most of the sphere,
    # and the convolution of the decoder on the output grid
    @parameterized.expand(
        [
            [36, 72, 4, 9, "equiangular", "legendre-gauss", (3, 3), 2.0 * math.pi / 35.0, 4, 8, 1e-4],
            [4, 9, 4, 9, "legendre-gauss", "legendre-gauss", (3, 3), math.sqrt(2.0) * math.pi / 3.0, 4, 8, 1e-4],
            [36, 72, 36, 72, "equiangular", "equiangular", (3, 3), 2.0 * math.pi / 35.0, 4, 8, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_disco_conv(
        self,
        nlat_in,
        nlon_in,
        nlat_out,
        nlon_out,
        grid_in,
        grid_out,
        kernel_shape,
        theta_cutoff,
        batch_size,
        num_chan,
        tol,
        verbose=False,
    ):
        B, C = batch_size, num_chan

        conv_args = dict(
            in_shape=(nlat_in, nlon_in),
            out_shape=(nlat_out, nlon_out),
            kernel_shape=kernel_shape,
            basis_type="harmonic",
            basis_norm_mode="mean",
            grid_in=grid_in,
            grid_out=grid_out,
            bias=False,
            theta_cutoff=theta_cutoff,
        )

        set_seed(333)
        conv_local = th.DiscreteContinuousConvS2(C, C, **conv_args).to(self.device)
        conv_dist = thd.DistributedDiscreteContinuousConvS2(C, C, **conv_args).to(self.device)
        with torch.no_grad():
            conv_dist.weight.copy_(conv_local.weight)

        inp_full = torch.randn((B, C, nlat_in, nlon_in), dtype=torch.float32, device=self.device)

        # local
        inp_full.requires_grad = True
        out_full = conv_local(inp_full)
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()
        wgrad_full = conv_local.weight.grad.clone()

        # distributed
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local = conv_dist(inp_local)
        ograd_local = self._split_helper(ograd_full, hdim=-2, wdim=-1)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()
        # the weight is replicated over the spatial ranks, each holding the contribution of its part of the grid
        wgrad_local = conv_dist.weight.grad.clone()
        if comm.get_size("spatial") > 1:
            dist.all_reduce(wgrad_local, group=comm.get_group("spatial"))

        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose), self.device
                )
            )

        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        with self.subTest(desc="weight gradients"):
            self.assertTrue(
                reduce_success(
                    compare_tensors("weight gradients", wgrad_local, wgrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

    def test_distributed_disco_conv_after_conv_without_halo(self, verbose=False):
        """
        SNO's encoder convolution followed by its decoder convolution, as in the model.

        On a 2 x 2 grid the encoder (36 x 72 -> 4 x 9) needs no latitude halo. torch-harmonics 0.9.3 still sent
        the whole local tensor as a halo, which nobody received; on NCCL the decoder's halo exchange picked it
        up and got the first latitude row of the second polar rank wrong, from B * C_in >= 256 on.
        """
        B, C, tol = 8, 32, 1e-4

        conv_args = dict(kernel_shape=(3, 3), basis_type="harmonic", basis_norm_mode="mean", bias=False)
        encoder_args = dict(
            in_shape=(36, 72),
            out_shape=(4, 9),
            grid_in="equiangular",
            grid_out="legendre-gauss",
            theta_cutoff=2.0 * math.pi / 35.0,
            **conv_args,
        )
        decoder_args = dict(
            in_shape=(36, 72),
            out_shape=(36, 72),
            grid_in="equiangular",
            grid_out="equiangular",
            theta_cutoff=2.0 * math.pi / 35.0,
            **conv_args,
        )

        set_seed(333)
        encoder_local = th.DiscreteContinuousConvS2(5, C, **encoder_args).to(self.device)
        encoder_dist = thd.DistributedDiscreteContinuousConvS2(5, C, **encoder_args).to(self.device)
        decoder_local = th.DiscreteContinuousConvS2(C, 5, **decoder_args).to(self.device)
        decoder_dist = thd.DistributedDiscreteContinuousConvS2(C, 5, **decoder_args).to(self.device)
        with torch.no_grad():
            encoder_dist.weight.copy_(encoder_local.weight)
            decoder_dist.weight.copy_(decoder_local.weight)

        enc_inp_full = torch.randn((B, 5, 36, 72), dtype=torch.float32, device=self.device)
        dec_inp_full = torch.randn((B, C, 36, 72), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            enc_out_full = encoder_local(enc_inp_full)
            dec_out_full = decoder_local(dec_inp_full)
            enc_out_local = encoder_dist(self._split_helper(enc_inp_full, hdim=-2, wdim=-1))
            dec_out_local = decoder_dist(self._split_helper(dec_inp_full, hdim=-2, wdim=-1))

        for desc, out_local, out_full in [
            ("encoder output", enc_out_local, enc_out_full),
            ("decoder output", dec_out_local, dec_out_full),
        ]:
            with self.subTest(desc=desc):
                out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
                self.assertTrue(
                    reduce_success(
                        compare_tensors(desc, out_gather_full, out_full, tol, tol, verbose=verbose), self.device
                    )
                )

    # the bilinear resampling in the decoder of SNO, from the internal 4 x 9 grid back onto 36 x 72, and the
    # opposite direction
    @parameterized.expand(
        [
            [4, 9, 36, 72, "legendre-gauss", "equiangular", 4, 8, 1e-5],
            [36, 72, 4, 9, "equiangular", "legendre-gauss", 4, 8, 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_distributed_resample(
        self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out, batch_size, num_chan, tol, verbose=False
    ):
        B, C = batch_size, num_chan

        resample_args = dict(grid_in=grid_in, grid_out=grid_out, mode="bilinear")
        resample_local = th.ResampleS2(nlat_in, nlon_in, nlat_out, nlon_out, **resample_args).to(self.device)
        resample_dist = thd.DistributedResampleS2(nlat_in, nlon_in, nlat_out, nlon_out, **resample_args).to(self.device)

        set_seed(333)
        inp_full = torch.randn((B, C, nlat_in, nlon_in), dtype=torch.float32, device=self.device)

        # local
        inp_full.requires_grad = True
        out_full = resample_local(inp_full)
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # distributed
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local = resample_dist(inp_local)
        ograd_local = self._split_helper(ograd_full, hdim=-2, wdim=-1)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose), self.device
                )
            )

        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

    # SNO's distributed layers called several times in a row with changing batch sizes, as the
    # gradient accumulation test does (a batch of 8 and batches of 4): every call has to match the
    # serial layer, whatever the module saw before. The local convolution runs on a 16 latitude grid
    # with SNO's cutoff for it, sqrt(2) pi / (nlat - 1): on 4 latitudes that cutoff spans 85 degrees,
    # a halo of 2 rows, more than the single latitude each of 4 polar ranks holds; on 16 the halo is
    # 1 row against 4 latitudes per rank
    @parameterized.expand(
        [
            ["disco_encoder", 36, 72, 4, 9, "equiangular", "legendre-gauss", 2.0 * math.pi / 35.0],
            ["disco_local", 16, 33, 16, 33, "legendre-gauss", "legendre-gauss", math.sqrt(2.0) * math.pi / 15.0],
            ["disco_decoder", 36, 72, 36, 72, "equiangular", "equiangular", 2.0 * math.pi / 35.0],
            ["resample_decoder", 4, 9, 36, 72, "legendre-gauss", "equiangular", None],
            ["spectral_global", 4, 9, 4, 9, "legendre-gauss", "legendre-gauss", None],
            ["instance_norm", 4, 9, 4, 9, "legendre-gauss", "legendre-gauss", None],
        ],
        skip_on_empty=True,
    )
    def test_distributed_layer_batch_size_sequence(
        self, kind, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out, theta_cutoff, verbose=False
    ):
        # the channels of SNO in the model test: 5 variables in and out, embed_dim 32 inside
        C, tol = 32, 1e-4
        C_in = 5 if kind == "disco_encoder" else C
        C_out = 5 if kind == "disco_decoder" else C

        set_seed(333)
        if kind.startswith("disco"):
            conv_args = dict(
                in_shape=(nlat_in, nlon_in),
                out_shape=(nlat_out, nlon_out),
                kernel_shape=(3, 3),
                basis_type="harmonic",
                basis_norm_mode="mean",
                grid_in=grid_in,
                grid_out=grid_out,
                bias=False,
                theta_cutoff=theta_cutoff,
            )
            layer_local = th.DiscreteContinuousConvS2(C_in, C_out, **conv_args).to(self.device)
            layer_dist = thd.DistributedDiscreteContinuousConvS2(C_in, C_out, **conv_args).to(self.device)
            with torch.no_grad():
                layer_dist.weight.copy_(layer_local.weight)
        elif kind.startswith("resample"):
            resample_args = dict(grid_in=grid_in, grid_out=grid_out, mode="bilinear")
            layer_local = th.ResampleS2(nlat_in, nlon_in, nlat_out, nlon_out, **resample_args).to(self.device)
            layer_dist = thd.DistributedResampleS2(nlat_in, nlon_in, nlat_out, nlon_out, **resample_args).to(
                self.device
            )
        elif kind == "instance_norm":
            # what SNO's blocks normalize with, on its internal grid; affine as in the model, with
            # weights away from the identity
            layer_local = nn.InstanceNorm2d(C, eps=1e-6, affine=True, track_running_stats=False).to(self.device)
            layer_dist = DistributedInstanceNorm2d(C, eps=1e-6, affine=True).to(self.device)
            with torch.no_grad():
                layer_local.weight.normal_()
                layer_local.bias.normal_()
                layer_dist.weight.copy_(layer_local.weight)
                layer_dist.bias.copy_(layer_local.bias)
        else:
            from makani.models.common import SpectralConv

            transforms_local = (
                th.RealSHT(nlat_in, nlon_in, lmax=4, mmax=5, grid=grid_in),
                th.InverseRealSHT(nlat_out, nlon_out, lmax=4, mmax=5, grid=grid_out),
            )
            transforms_dist = (
                thd.DistributedRealSHT(nlat_in, nlon_in, lmax=4, mmax=5, grid=grid_in),
                thd.DistributedInverseRealSHT(nlat_out, nlon_out, lmax=4, mmax=5, grid=grid_out),
            )
            conv_args = dict(operator_type="dhconv", num_groups=1, bias=True, gain=1.0)
            spectral_local = SpectralConv(*[t.to(self.device) for t in transforms_local], C, C, **conv_args)
            spectral_dist = SpectralConv(*[t.to(self.device) for t in transforms_dist], C, C, **conv_args)
            spectral_local, spectral_dist = spectral_local.to(self.device), spectral_dist.to(self.device)
            with torch.no_grad():
                spectral_dist.weight.copy_(self._split_helper(spectral_local.weight, hdim=-1, wdim=None))
                spectral_dist.bias.copy_(spectral_local.bias)

            # the output only; the residual is covered by test_distributed_spectral_conv
            def layer_local(x):
                return spectral_local(x)[0]

            def layer_dist(x):
                return spectral_dist(x)[0]

        # B=4 first, so that a failure at B=8 is not mistaken for one of the first call
        for call, batch_size in enumerate([4, 8, 4]):
            inp_full = torch.randn((batch_size, C_in, nlat_in, nlon_in), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                out_full = layer_local(inp_full)
                out_local = layer_dist(self._split_helper(inp_full, hdim=-2, wdim=-1))
            with self.subTest(desc=f"call {call}, batch {batch_size}: output"):
                out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
                self.assertTrue(
                    reduce_success(
                        compare_tensors(
                            f"{kind} call {call} output", out_gather_full, out_full, tol, tol, verbose=verbose
                        ),
                        self.device,
                    )
                )

            # each layer against itself, batch by batch: the samples are independent, so the larger
            # batch has to give what its halves give. Tells which of the two layers is off
            if batch_size > 4:
                with torch.no_grad():
                    halves_full = torch.cat([layer_local(x) for x in inp_full.split(4, dim=0)], dim=0)
                    inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
                    halves_local = torch.cat([layer_dist(x) for x in inp_local.split(4, dim=0)], dim=0)
                with self.subTest(desc=f"call {call}, batch {batch_size}: serial layer vs its halves"):
                    self.assertTrue(
                        reduce_success(
                            compare_tensors(
                                f"{kind} call {call} serial vs halves", out_full, halves_full, tol, tol, verbose=verbose
                            ),
                            self.device,
                        )
                    )
                with self.subTest(desc=f"call {call}, batch {batch_size}: distributed layer vs its halves"):
                    halves_gather_full = self._gather_helper(halves_local, hdim=-2, wdim=-1)
                    self.assertTrue(
                        reduce_success(
                            compare_tensors(
                                f"{kind} call {call} distributed vs halves",
                                out_gather_full,
                                halves_gather_full,
                                tol,
                                tol,
                                verbose=verbose,
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
