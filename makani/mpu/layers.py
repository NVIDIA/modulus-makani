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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import custom_fwd, custom_bwd

from makani.utils import comm

# parallel helpers
from modulus.distributed.utils import compute_split_shapes
from modulus.distributed.mappings import reduce_from_parallel_region
from modulus.distributed.mappings import scatter_to_parallel_region
from modulus.distributed.mappings import gather_from_parallel_region
from modulus.distributed.mappings import copy_to_parallel_region

# use some distributed routines from torch harmonics
from torch_harmonics.distributed import distributed_transpose_azimuth as distributed_transpose_w
from torch_harmonics.distributed import distributed_transpose_polar as distributed_transpose_h


class DistributedRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):
        super(DistributedRealFFT2, self).__init__()

        # get the comms grid:
        self.comm_size_h = comm.get_size("h")
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        # compute half modes
        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

        # shapes
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_h)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_w)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_h)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # store number of chans
        num_chans = x.shape[1]
        
        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_w > 1:
            x = distributed_transpose_w.apply(x, (1, -1), self.lon_shapes)

        # do first FFT
        x = torch.fft.rfft(x, n=self.nlon, dim=-1, norm="ortho")

        # mode truncation
        x = x[..., :self.mmax].contiguous()
        
        # transpose: after this, m is split and c is local
        if self.comm_size_w > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_w)
            x = distributed_transpose_w.apply(x, (-1, 1), chan_shapes)
            
        # transpose: after this, c is split and h is local
        if self.comm_size_h > 1:
            x = distributed_transpose_h.apply(x, (1, -2), self.lat_shapes)

        # do second FFT:
        x = torch.fft.fft(x, n=self.nlat, dim=-2, norm="ortho")

        # apply mode truncation:
        x = torch.cat([x[..., :self.lmax_high,  :],
                       x[..., -self.lmax_low:, :]], dim=-2)
        
        # transpose: after this, l is split and c is local
        if self.comm_size_h > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_h)
            x = distributed_transpose_h.apply(x, (-2, 1), chan_shapes)

        return x


class DistributedInverseRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):
        super(DistributedInverseRealFFT2, self).__init__()

        # get the comms grid:
        self.comm_size_h = comm.get_size("h")
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        # compute half modes
        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

        # shapes
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_h)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_w)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_h)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # store number of channels
        num_chans = x.shape[1]

        # transpose: after that, channels are split, l is local:
        if self.comm_size_h > 1:
            x = distributed_transpose_h.apply(x, (1, -2), self.l_shapes)
            
        # we should pad the middle here manually, so that the inverse FFT is correct
        # TEST THIS
        if self.lmax < self.nlat:
            xh = x[..., :self.lmax_high, :]
            xl = x[..., -self.lmax_low:, :]
            xhp = F.pad(xh, (0, 0, 0, self.nlat-self.lmax), mode="constant")
            x = torch.cat([xhp, xl], dim=-2)

        # do first fft
        x = torch.fft.ifft(x, n=self.nlat, dim=-2, norm="ortho")

        if self.comm_size_h > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_h)
            x = distributed_transpose_h.apply(x, (-2, 1), chan_shapes)

        # transpose: after this, channels are split and m is local
        if self.comm_size_w > 1:
            x = distributed_transpose_w.apply(x, (1, -1), self.m_shapes)

        # apply the inverse (real) FFT
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="ortho")

        # transpose: after this, m is split and channels are local
        if self.comm_size_w > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_w)
            x = distributed_transpose_w.apply(x, (-1, 1), chan_shapes)

        return x


class _DistMatmulHelper(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, X, weight, bias, inp_group_name, out_group_name):
        # store some variables
        ctx.save_for_backward(X, weight, bias)
        ctx.out_group_name = out_group_name

        # matrix multiplication
        xconv = F.conv2d(X, weight, bias=None)

        # reduce
        if comm.get_size(inp_group_name) > 1:
            dist.all_reduce(xconv, group=comm.get_group(inp_group_name))

        # add bias
        if bias is not None:
            xconvbias = xconv + bias
        else:
            xconvbias = xconv

        return xconvbias

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        X, weight, bias = ctx.saved_tensors
        gname = ctx.out_group_name

        # do the bwd pass on dgrad
        grad_input = F.conv_transpose2d(grad_out, weight, bias=None)

        # reduce across nodes
        if comm.get_size(gname) > 1:
            dgrad_handle = dist.all_reduce(grad_input, group=comm.get_group(gname), async_op=True)

        # weight grad
        grad_weight = F.conv2d(X.transpose(0, 1), grad_out.transpose(0, 1), bias=None).transpose(0, 1)

        if bias is not None:
            grad_bias = torch.sum(grad_out, dim=(0, 2, 3), keepdim=True)
        else:
            grad_bias = None

        if comm.get_size(gname) > 1:
            dgrad_handle.wait()

        return grad_input, grad_weight, grad_bias, None, None


class DistributedMatmul(nn.Module):
    def __init__(self, inp_dim, out_dim, input_format="nchw", comm_inp_name="fin", comm_out_name="fout", bias=True):
        super(DistributedMatmul, self).__init__()

        # get sizes
        self.comm_inp_name = comm_inp_name
        self.comm_out_name = comm_out_name
        comm_inp_size = comm.get_size(self.comm_inp_name)
        comm_out_size = comm.get_size(self.comm_out_name)

        # split:
        assert inp_dim % comm_inp_size == 0, f"Error, the size of input feature dim ({inp_dim}) has to be evenly divisible by the input feature comm dim ({comm_inp_size})"
        assert out_dim % comm_out_size == 0, f"Error, the size of output feature dim ({out_dim}) has to be evenly divisible by the output feature comm dim ({comm_out_size})"

        # compute reduced dims
        inp_dim_local = inp_dim // comm_inp_size
        out_dim_local = out_dim // comm_out_size

        # parameters
        if input_format == "nchw":
            self.weight = nn.Parameter(torch.ones(out_dim_local, inp_dim_local, 1, 1))
            self.weight.is_shared_mp = ["spatial"]
            self.weight.sharded_dims_mp = [self.comm_out_name, self.comm_inp_name, None, None]
            self.matmul_handle = F.conv2d
        elif input_format == "traditional":
            self.weight = nn.Parameter(torch.ones(out_dim_local, inp_dim_local))
            self.weight.sharded_dims_mp = [self.comm_out_name, self.comm_inp_name]
            self.matmul_handle = F.linear
        else:
            raise NotImplementedError(f"Error, input format {input_format} not supported.")

        # bias
        self.bias = None
        if bias:
            if input_format == "nchw":
                self.bias = nn.Parameter(torch.zeros(1, out_dim_local, 1, 1))
                self.bias.is_shared_mp = ["spatial"]
                self.bias.sharded_dims_mp = [None, self.comm_out_name, None, None]
            elif input_format == "traditional":
                self.bias = nn.Parameter(torch.zeros(out_dim_local))
                self.bias.sharded_dims_mp = [self.comm_out_name]

    def forward(self, x):
        x_cp = copy_to_parallel_region(x, self.comm_out_name)
        x_loc = self.matmul_handle(x_cp, self.weight, bias=None)
        x_out = reduce_from_parallel_region(x_loc, self.comm_inp_name)
        if self.bias is not None:
            x_out = x_out + self.bias

        return x_out


# distributed encoder/decoder
class DistributedEncoderDecoder(nn.Module):
    def __init__(self, num_layers, input_dim, output_dim, hidden_dim, act_layer, gain=1.0, input_format="nchw", comm_inp_name="fin", comm_out_name="fout"):
        super(DistributedEncoderDecoder, self).__init__()

        # get comms
        comm_inp_size = comm.get_size(comm_inp_name)
        comm_out_size = comm.get_size(comm_out_name)

        # get list of modules
        encoder_modules = []
        current_dim = input_dim
        comm_inp_name_tmp = comm_inp_name
        comm_out_name_tmp = comm_out_name
        for i in range(num_layers - 1):
            encoder_modules.append(
                DistributedMatmul(current_dim, hidden_dim, input_format=input_format, comm_inp_name=comm_inp_name_tmp, comm_out_name=comm_out_name_tmp, bias=True)
            )

            # proper initialization
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(encoder_modules[-1].weight, mean=0.0, std=scale)
            if encoder_modules[-1].bias is not None:
                nn.init.constant_(encoder_modules[-1].bias, 0.0)

            encoder_modules.append(act_layer())
            current_dim = hidden_dim
            comm_inp_name_tmp, comm_out_name_tmp = (comm_out_name_tmp, comm_inp_name_tmp)

        # final layer
        encoder_modules.append(DistributedMatmul(current_dim, output_dim, input_format=input_format, comm_inp_name=comm_inp_name_tmp, comm_out_name=comm_out_name_tmp, bias=False))

        # proper initialization of final layer
        scale = math.sqrt(gain / current_dim)
        nn.init.normal_(encoder_modules[-1].weight, mean=0.0, std=scale)
        if encoder_modules[-1].bias is not None:
            nn.init.constant_(encoder_modules[-1].bias, 0.0)

        # create fwd sequence
        self.fwd = nn.Sequential(*encoder_modules)

        # store the comm names for in and out so that they can be queried
        self.comm_inp_name = comm_inp_name
        self.comm_out_name = comm_out_name_tmp

    def forward(self, x):
        return self.fwd(x)


# more complicated layers
class DistributedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        output_bias=True,
        input_format="nchw",
        comm_inp_name="fin",
        comm_hidden_name="fout",
        act_layer=nn.GELU,
        drop_rate=0.0,
        drop_type="iid",
        checkpointing=False,
        gain=1.0,
    ):
        super(DistributedMLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # sanity checks:
        if (input_format == "traditional") and (drop_type == "features"):
            raise NotImplementedError(f"Error, traditional input format and feature dropout cannot be selected simultaneously")

        # get effective embedding size:
        comm_inp_size = comm.get_size(comm_inp_name)
        comm_hid_size = comm.get_size(comm_hidden_name)

        self.fc1 = DistributedMatmul(in_features, hidden_features, input_format=input_format, comm_inp_name=comm_inp_name, comm_out_name=comm_hidden_name, bias=True)

        # initialize the weights correctly
        scale = math.sqrt(2.0 / in_features)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=scale)
        nn.init.constant_(self.fc1.bias, 0.0)

        self.fc2 = DistributedMatmul(hidden_features, out_features, input_format=input_format, comm_inp_name=comm_hidden_name, comm_out_name=comm_inp_name, bias=output_bias)

        # gain factor for the output determines the scaling of the output init
        scale = math.sqrt(gain / hidden_features)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=scale)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0.0)

        self.act = act_layer()

        if drop_rate > 0.0:
            if drop_type == "iid":
                self.drop = nn.Dropout(drop_rate)
            elif drop_type == "features":
                self.drop = nn.Dropout2d(drop_rate)
            else:
                raise NotImplementedError(f"Error, drop_type {drop_type} not supported")
        else:
            self.drop = nn.Identity()

    def fwd(self, x):
        # do the mlp
        # first layer
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        # second layer
        x = self.fc2(x)
        x = self.drop(x)

        return x

    @torch.jit.ignore
    def _checkpoint_forward(self, x):
        return checkpoint(self.fwd, x, use_reentrant=False)

    def forward(self, x):
        if self.checkpointing:
            return self._checkpoint_forward(x)
        else:
            return self.fwd(x)


class DistributedPatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, input_is_matmul_parallel=False, output_is_matmul_parallel=True):
        super(DistributedPatchEmbed, self).__init__()

        # store params
        self.input_parallel = input_is_matmul_parallel
        self.output_parallel = output_is_matmul_parallel

        # get comm sizes:
        matmul_comm_size = comm.get_size("matmul")
        spatial_comm_size = comm.get_size("spatial")

        # compute parameters
        assert (img_size[1] // patch_size[1]) % spatial_comm_size == 0, "Error, make sure that the spatial comm size evenly divides patched W"
        num_patches = ((img_size[1] // patch_size[1]) // spatial_comm_size) * (img_size[0] // patch_size[0])
        self.img_size = (img_size[0], img_size[1] // spatial_comm_size)
        self.patch_size = patch_size
        self.num_patches = num_patches

        # get effective embedding size:
        if self.output_parallel:
            assert embed_dim % matmul_comm_size == 0, "Error, the embed_dim needs to be divisible by matmul_parallel_size"
            out_chans_local = embed_dim // matmul_comm_size
        else:
            out_chans_local = embed_dim

        # the weights  of this layer is shared across spatial parallel ranks
        self.proj = nn.Conv2d(in_chans, out_chans_local, kernel_size=patch_size, stride=patch_size)

        # make sure we reduce them across rank
        self.proj.weight.is_shared_mp = ["spatial"]
        self.proj.bias.is_shared_mp = ["spatial"]

        # gather shapes
        self.gather_shapes = compute_split_shapes(in_chans, comm.get_size("matmul"))

    def forward(self, x):
        if self.input_parallel:
            x = gather_from_parallel_region(x, 1, self.gather_shapes, "matmul")

        if self.output_parallel:
            x = copy_to_parallel_region(x, "matmul")

        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # new: B, C, H*W
        x = self.proj(x).flatten(2)
        return x


class DistributedAttention(nn.Module):
    """Distributed Attention layer"""

    def __init__(
        self,
        dim,
        input_format="traditional",
        comm_inp_name="fin",
        comm_hidden_name="fout",
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super(DistributedAttention, self).__init__()

        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads

        assert num_heads % comm.get_size(comm_hidden_name) == 0, "heads are not evenly split across model ranks"
        self.num_heads_local = num_heads // comm.get_size(comm_hidden_name)
        self.head_dim = dim // self.num_heads

        self.comm_inp_name = comm_inp_name
        self.comm_hidden_name = comm_hidden_name

        self.qkv = DistributedMatmul(dim, dim * 3, input_format, comm_inp_name=comm_inp_name, comm_out_name=comm_hidden_name, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop_rate = attn_drop_rate
        self.proj = DistributedMatmul(dim, dim, input_format, comm_inp_name=comm_hidden_name, comm_out_name=comm_inp_name, bias=False)
        if proj_drop_rate > 0.0:
            self.proj_drop = nn.Dropout(proj_drop_rate)
        else:
            self.proj_drop = nn.Identity()

        # set up weight sharing, depends on norm type
        if isinstance(self.q_norm, nn.LayerNorm):
            if hasattr(self.q_norm, "weight"):
                self.q_norm.weight.is_shared_mp = []
            if hasattr(self.q_norm, "bias"):
                self.q_norm.bias.is_shared_mp = []

        if isinstance(self.k_norm, nn.LayerNorm):
            if hasattr(self.k_norm, "weight"):
                self.k_norm.weight.is_shared_mp = []
            if hasattr(self.k_norm, "bias"):
                self.k_norm.bias.is_shared_mp = []

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads_local, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_rate)

        # transpose back
        x = x.transpose(1, 2).reshape(B, N, self.num_heads_local * self.head_dim)

        # this is distributed again
        x = self.proj(x)

        # generally we have to be super careful with dropout layers, since
        # those are normalized over the dropouts. That would need to be reduced across nodes
        x = self.proj_drop(x)

        return x


@torch.jit.script
def compl_mul_add_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bkixys,kiot->stbkoxy", a, b)
    res = torch.stack([tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1) + c
    return res


@torch.jit.script
def compl_mul_add_fwd_c(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    cc = torch.view_as_complex(c)
    tmp = torch.einsum("bkixy,kio->bkoxy", ac, bc)
    res = tmp + cc
    return torch.view_as_real(res)


class DistributedAFNO2Dv2(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=False,
        use_complex_kernels=False,
    ):
        super(DistributedAFNO2Dv2, self).__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        # get comm sizes:
        matmul_comm_size = comm.get_size("matmul")
        self.spatial_comm_size = comm.get_size("spatial")

        # select fft function handles
        if self.spatial_comm_size > 1:
            self.fft_handle = distributed_rfft2.apply
            self.ifft_handle = distributed_irfft2.apply
        else:
            self.fft_handle = torch.fft.rfft2
            self.ifft_handle = torch.fft.irfft2

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.gather_shapes = compute_split_shapes(self.num_blocks, matmul_comm_size)
        self.num_blocks_local = self.gather_shapes[comm.get_rank("matmul")]
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.mult_handle = compl_mul_add_fwd_c if use_complex_kernels else compl_mul_add_fwd

        # model paralellism
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel

        # new
        # these weights need to be synced across all spatial ranks!
        self.w1 = nn.Parameter(self.scale * torch.randn(self.num_blocks_local, self.block_size, self.block_size * self.hidden_size_factor, 2))
        self.b1 = nn.Parameter(self.scale * torch.randn(self.num_blocks_local, self.block_size * self.hidden_size_factor, 1, 1, 2))
        self.w2 = nn.Parameter(self.scale * torch.randn(self.num_blocks_local, self.block_size * self.hidden_size_factor, self.block_size, 2))
        self.b2 = nn.Parameter(self.scale * torch.randn(self.num_blocks_local, self.block_size, 1, 1, 2))

        # setting correct sharding and sharing
        self.w1.is_shared_mp = ["spatial"]
        self.w1.sharded_dims_mp = ["matmul", None, None, None]

        self.b1.is_shared_mp = ["spatial"]
        self.b1.sharded_dims_mp = ["matmul", None, None, None, None]

        self.w2.is_shared_mp = ["spatial"]
        self.w2.sharded_dims_mp = ["matmul", None, None, None]

        self.b2.is_shared_mp = ["spatial"]
        self.b2.sharded_dims_mp = ["matmul", None, None, None, None]

    def forward(self, x):
        if not self.input_is_matmul_parallel:
            # distribute data
            x = gather_from_parallel_region(x, 1, self.gather_shapes, "matmul")

        # bias
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W_local = x.shape
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        H_local = H // self.spatial_comm_size
        W = W_local * self.spatial_comm_size
        x = self.fft_handle(x, (H, W), (-2, -1), "ortho")
        x = x.view(B, self.num_blocks_local, self.block_size, H_local, W // 2 + 1)

        # new
        x = torch.view_as_real(x)
        o2 = torch.zeros(x.shape, device=x.device)

        o1 = F.relu(self.mult_handle(x[:, :, :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, :], self.w1, self.b1))
        o2[:, :, :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, :] = self.mult_handle(o1, self.w2, self.b2)

        # finalize
        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, H_local, W // 2 + 1)
        x = self.ifft_handle(x, (H, W), (-2, -1), "ortho")
        x = x.type(dtype) + bias

        # gather
        if not self.output_is_matmul_parallel:
            x = gather_from_parallel_region(x, 1, self.gather_shapes, "matmul")

        return x
