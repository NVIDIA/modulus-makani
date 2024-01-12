# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import os
import unittest
from parameterized import parameterized

import torch
import torch.nn.functional as F
import torch.distributed as dist

import torch_harmonics.distributed as thd

from makani.models.common import RealFFT2, InverseRealFFT2

from makani.utils import comm
from modulus.distributed.utils import split_tensor_along_dim
from modulus.distributed.mappings import gather_from_parallel_region, scatter_to_parallel_region, \
                                         reduce_from_parallel_region
from makani.mpu.layers import DistributedRealFFT2, DistributedInverseRealFFT2
from makani.mpu.fft3d import RealFFT3, InverseRealFFT3, DistributedRealFFT3, DistributedInverseRealFFT3

class TestDistributedRealFFT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # set up distributed
        cls.grid_size_h = int(os.getenv('GRID_H', 1))
        cls.grid_size_w = int(os.getenv('GRID_W', 1))
        #port = int(os.getenv('MASTER_PORT', '29501'))
        #master_address = os.getenv('MASTER_ADDR', 'localhost')
        cls.world_size = cls.grid_size_h * cls.grid_size_w

        # init groups
        comm.init(model_parallel_sizes=[cls.grid_size_h, cls.grid_size_w, 1, 1],
                  model_parallel_names=["h", "w", "fin", "fout"])
        cls.world_rank = comm.get_world_rank()
        
        if torch.cuda.is_available():
            if cls.world_rank == 0:
                print("Running test on GPU")
            local_rank = comm.get_local_rank()
            cls.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(cls.device)
            torch.cuda.manual_seed(333)
        else:
            if cls.world_rank == 0:
                print("Running test on CPU")
            cls.device = torch.device('cpu')
        torch.manual_seed(333)

        # store comm group parameters
        cls.wrank = comm.get_rank("w")
        cls.hrank = comm.get_rank("h")
        cls.w_group = comm.get_group("w")
        cls.h_group = comm.get_group("h")

        # initializing sht process groups
        thd.init(cls.h_group, cls.w_group)

        if cls.world_rank == 0:
            print(f"Running distributed tests on grid H x W = {cls.grid_size_h} x {cls.grid_size_w}")

        
    def _split_helper(self, tensor):
        with torch.no_grad():
            # split in W
            tensor_list_local = split_tensor_along_dim(tensor, dim=-1, num_chunks=self.grid_size_w)
            tensor_local = tensor_list_local[self.wrank]

            # split in H
            tensor_list_local = split_tensor_along_dim(tensor_local, dim=-2, num_chunks=self.grid_size_h)
            tensor_local = tensor_list_local[self.hrank]

        return tensor_local
        
        
    def _gather_helper_fwd(self, tensor, B, C, transform_dist, D=0):
        # we need the shapes
        l_shapes = transform_dist.l_shapes
        m_shapes = transform_dist.m_shapes

        # gather in W
        if self.grid_size_w > 1:
            if D > 0:
                gather_shapes = [(B, C, D, l_shapes[self.hrank], m) for m in m_shapes]
            else:
                gather_shapes = [(B, C, l_shapes[self.hrank], m) for m in m_shapes]
            olist = [torch.empty(shape, dtype=tensor.dtype, device=tensor.device) for shape in gather_shapes]
            olist[self.wrank] = tensor
            dist.all_gather(olist, tensor, group=self.w_group)
            tensor_gather = torch.cat(olist, dim=-1)
        else:
            tensor_gather = tensor

        # gather in H
        if self.grid_size_h > 1:
            if D > 0:
                gather_shapes = [(B, C, D, l, transform_dist.lwmax) for l in l_shapes]
            else:
                gather_shapes = [(B, C, l, transform_dist.mmax) for l in l_shapes]
            olist = [torch.empty(shape, dtype=tensor_gather.dtype, device=tensor_gather.device) for shape in gather_shapes]
            olist[self.hrank] = tensor_gather
            dist.all_gather(olist, tensor_gather, group=self.h_group)
            tensor_gather = torch.cat(olist, dim=-2)

        return tensor_gather

    def _gather_helper_bwd(self, tensor, B, C, transform_dist, D=0):
        # we need the shapes
        lat_shapes = transform_dist.lat_shapes
        lon_shapes = transform_dist.lon_shapes

        # gather in W
        if self.grid_size_w > 1:
            if D > 0:
                gather_shapes = [(B, C, D, lat_shapes[self.hrank], w) for w in lon_shapes]
            else:
                gather_shapes = [(B, C, lat_shapes[self.hrank], w) for w in lon_shapes]
            olist = [torch.empty(shape, dtype=tensor.dtype, device=tensor.device) for shape in gather_shapes]
            olist[self.wrank] = tensor
            dist.all_gather(olist, tensor, group=self.w_group)
            tensor_gather = torch.cat(olist, dim=-1)
        else:
            tensor_gather = tensor

        # gather in H
        if self.grid_size_h > 1:
            if D > 0:
                gather_shapes = [(B, C, D, h, transform_dist.nw) for h in lat_shapes]
            else:
                gather_shapes = [(B, C, h, transform_dist.nlon) for h in lat_shapes]
            olist = [torch.empty(shape, dtype=tensor_gather.dtype, device=tensor_gather.device) for shape in gather_shapes]
            olist[self.hrank] = tensor_gather
            dist.all_gather(olist, tensor_gather, group=self.h_group)
            tensor_gather = torch.cat(olist, dim=-2)

        return tensor_gather


    @parameterized.expand([
        [256, 512, 0, 32,  8, 1e-6],
        [361, 720, 0,  1, 10, 1e-6],
        [256, 512, 4, 32,  8, 1e-6],
        [361, 720, 4,  1, 10, 1e-6],
    ])
    def test_distributed_fft(self, nlat, nlon, nalt, batch_size, num_chan, tol):
        B, C, D, H, W = batch_size, num_chan, nalt, nlat, nlon

        # set up handles
        if D > 0:
            forward_transform_local = RealFFT3(nd=D, nh=H, nw=W).to(self.device)
            forward_transform_dist = DistributedRealFFT3(nd=D, nh=H, nw=W).to(self.device)
        else:
            forward_transform_local = RealFFT2(nlat=H, nlon=W).to(self.device)
            forward_transform_dist = DistributedRealFFT2(nlat=H, nlon=W).to(self.device)

        # create tensors
        if D > 0:
            inp_full = torch.randn((B, C, D, H, W), dtype=torch.float32, device=self.device)
        else:
            inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)

        #############################################################
        # local transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full = forward_transform_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)
            
        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()
        
        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full)
        inp_local.requires_grad = True
        out_local = forward_transform_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = forward_transform_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()
        
        # set eval dims
        dims = (-1,-2,-3) if D > 0 else (-1,-2)
        
        #############################################################
        # evaluate FWD pass
        #############################################################
        with torch.no_grad():
            out_gather_full = self._gather_helper_fwd(out_local, B, C, forward_transform_dist, D)
            err = torch.mean(torch.norm(out_full-out_gather_full, p=2, dim=dims) / torch.norm(out_full, p=2, dim=dims) )
            if self.world_rank == 0:
                print(f"final relative error of output: {err.item()}")
        self.assertTrue(err.item() <= tol)

        #############################################################
        # evaluate BWD pass
        #############################################################
        with torch.no_grad():
            igrad_gather_full = self._gather_helper_bwd(igrad_local, B, C, forward_transform_dist, D)
            err = torch.mean(torch.norm(igrad_full-igrad_gather_full, p=2, dim=dims) / torch.norm(igrad_full, p=2, dim=dims) )
            if self.world_rank == 0:
                print(f"final relative error of gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)


    @parameterized.expand([
        [256, 512, 0, 32,  8, 1e-6],
        [361, 720, 0,  1, 10, 1e-6],
        [256, 512, 4, 32,  8, 1e-6],
        [361, 720, 4,  1, 10, 1e-6],
    ])
    def test_distributed_ifft(self, nlat, nlon, nalt, batch_size, num_chan, tol):
        B, C, D, H, W = batch_size, num_chan, nalt, nlat, nlon

        if D > 0:
            forward_transform_local = RealFFT3(nd=D, nh=H, nw=W).to(self.device)
            backward_transform_local = InverseRealFFT3(nd=D, nh=H, nw=W).to(self.device)
            backward_transform_dist = DistributedInverseRealFFT3(nd=D, nh=H, nw=W).to(self.device)
        else:
            forward_transform_local = RealFFT2(nlat=H, nlon=W).to(self.device)
            backward_transform_local = InverseRealFFT2(nlat=H, nlon=W).to(self.device)
            backward_transform_dist = DistributedInverseRealFFT2(nlat=H, nlon=W).to(self.device)

        # create tensors
        if D > 0:
            dummy_full = torch.randn((B, C, D, H, W), dtype=torch.float32, device=self.device)
        else:
            dummy_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)
        inp_full = forward_transform_local(dummy_full)

        #############################################################
        # local transform
	#############################################################
	# FWD pass
        inp_full.requires_grad = True
        out_full = backward_transform_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)

        # repeat once due to known irfft bug
        inp_full.grad = None
        out_full = backward_transform_local(inp_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full)
        inp_local.requires_grad = True
        out_local = backward_transform_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = backward_transform_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # set eval dims
        dims = (-1,-2,-3) if D > 0 else	(-1,-2)
        
        #############################################################
        # evaluate FWD pass
        #############################################################
        with torch.no_grad():
            out_gather_full = self._gather_helper_bwd(out_local, B, C, backward_transform_dist, D)
            err = torch.mean(torch.norm(out_full-out_gather_full, p=2, dim=dims) / torch.norm(out_full, p=2, dim=dims) )
            if self.world_rank == 0:
                print(f"final relative error of output: {err.item()}")
        self.assertTrue(err.item() <= tol)

        #############################################################
        # evaluate BWD pass
        #############################################################
        with torch.no_grad():
            igrad_gather_full = self._gather_helper_fwd(igrad_local, B, C, backward_transform_dist, D)
            err = torch.mean(torch.norm(igrad_full-igrad_gather_full, p=2, dim=dims) / torch.norm(igrad_full, p=2, dim=dims) )
            if self.world_rank == 0:
                print(f"final relative error of gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)

if __name__ == '__main__':    
    unittest.main()
