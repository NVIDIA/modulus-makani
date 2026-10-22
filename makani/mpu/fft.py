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

from typing import Optional
import math
import torch
from torch import nn
import torch.nn.functional as F

from makani.utils import comm
from makani.utils.functions import contiguous_complex_safe
from torch_harmonics.distributed import compute_split_shapes
from torch_harmonics.distributed import distributed_transpose_azimuth as distributed_transpose_w
from torch_harmonics.distributed import distributed_transpose_polar as distributed_transpose_h

# zeroes the imaginary parts of the DC and Nyquist bins before the inverse, which cuFFT does not do
from torch_harmonics.fft import irfft


class DistributedRealFFT1(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlon: int, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

        # get the comms grid:
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nlon = nlon
        self.lmax = min(lmax or self.nlon // 2 + 1, self.nlon // 2 + 1)
        self.mmax = min(mmax or self.lmax, self.lmax)

        # compute half modes
        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

        # shapes
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_w)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_w)

    def forward(self, x: torch.Tensor, norm: Optional[str] = "ortho", channel_dim: Optional[int] = -3) -> torch.Tensor:
        # store number of chans
        num_chans = x.shape[channel_dim]

        # We make w local by transposing into channel dim
        if self.comm_size_w > 1:
            x = distributed_transpose_w(x, (channel_dim, -1), self.lon_shapes)

        # do first FFT
        x = torch.fft.rfft(x, n=self.nlon, dim=-1, norm=norm)

        # mode truncation. The slice is over a complex tensor, so the copy goes through the
        # real view -- inductor cannot codegen a copy on a complex buffer (KeyError: 'complex64')
        x = contiguous_complex_safe(x[..., : self.mmax])

        # transpose: after this, m is split and c is local
        if self.comm_size_w > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_w)
            x = distributed_transpose_w(x, (-1, channel_dim), chan_shapes)

        return x


class DistributedInverseRealFFT1(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlon: int, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

        # get the comms grid:
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nlon = nlon
        self.lmax = min(lmax or self.nlon // 2 + 1, self.nlon // 2 + 1)
        self.mmax = min(mmax or self.lmax, self.lmax)

        # compute half modes
        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

        # shapes
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_w)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_w)

    def forward(self, x: torch.Tensor, norm: Optional[str] = "ortho", channel_dim: Optional[int] = -3) -> torch.Tensor:
        # store number of channels
        num_chans = x.shape[channel_dim]

        # transpose: after this, channels are split and m is local
        if self.comm_size_w > 1:
            x = distributed_transpose_w(x, (channel_dim, -1), self.m_shapes)

        # apply the inverse (real) FFT
        x = irfft(x, n=self.nlon, dim=-1, norm=norm)

        # transpose: after this, m is split and channels are local
        if self.comm_size_w > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_w)
            x = distributed_transpose_w(x, (-1, channel_dim), chan_shapes)

        return x


class DistributedRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat: int, nlon: int, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

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

    def forward(self, x: torch.Tensor, norm: Optional[str] = "ortho", channel_dim: Optional[int] = -3) -> torch.Tensor:
        # store number of chans
        num_chans = x.shape[channel_dim]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_w > 1:
            x = distributed_transpose_w(x, (channel_dim, -1), self.lon_shapes)

        # do first FFT
        x = torch.fft.rfft(x, n=self.nlon, dim=-1, norm=norm)

        # mode truncation. The slice is over a complex tensor, so the copy goes through the
        # real view -- inductor cannot codegen a copy on a complex buffer (KeyError: 'complex64')
        x = contiguous_complex_safe(x[..., : self.mmax])

        # transpose: after this, m is split and c is local
        if self.comm_size_w > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_w)
            x = distributed_transpose_w(x, (-1, channel_dim), chan_shapes)

        # transpose: after this, c is split and h is local
        if self.comm_size_h > 1:
            x = distributed_transpose_h(x, (channel_dim, -2), self.lat_shapes)

        # do second FFT:
        x = torch.fft.fft(x, n=self.nlat, dim=-2, norm=norm)

        # apply mode truncation:
        x = torch.cat([x[..., : self.lmax_high, :], x[..., -self.lmax_low :, :]], dim=-2)

        # transpose: after this, l is split and c is local
        if self.comm_size_h > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_h)
            x = distributed_transpose_h(x, (-2, channel_dim), chan_shapes)

        return x


class DistributedInverseRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat: int, nlon: int, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

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

    def forward(self, x: torch.Tensor, norm: Optional[str] = "ortho", channel_dim: Optional[int] = -3) -> torch.Tensor:
        # store number of channels
        num_chans = x.shape[channel_dim]

        # transpose: after that, channels are split, l is local:
        if self.comm_size_h > 1:
            x = distributed_transpose_h(x, (channel_dim, -2), self.l_shapes)

        # we should pad the middle here manually, so that the inverse FFT is correct
        # TEST THIS
        if self.lmax < self.nlat:
            xh = x[..., : self.lmax_high, :]
            xl = x[..., -self.lmax_low :, :]
            xhp = F.pad(xh, (0, 0, 0, self.nlat - self.lmax), mode="constant")
            x = torch.cat([xhp, xl], dim=-2)

        # do first fft
        x = torch.fft.ifft(x, n=self.nlat, dim=-2, norm=norm)

        if self.comm_size_h > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_h)
            x = distributed_transpose_h(x, (-2, channel_dim), chan_shapes)

        # transpose: after this, channels are split and m is local
        if self.comm_size_w > 1:
            x = distributed_transpose_w(x, (channel_dim, -1), self.m_shapes)

        # apply the inverse (real) FFT
        x = irfft(x, n=self.nlon, dim=-1, norm=norm)

        # transpose: after this, m is split and channels are local
        if self.comm_size_w > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_w)
            x = distributed_transpose_w(x, (-1, channel_dim), chan_shapes)

        return x


# 3D routines
# forward
class DistributedRealFFT3(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nd, nh, nw, ldmax=None, lhmax=None, lwmax=None):
        super().__init__()

        # get the comms grid:
        self.comm_size_h = comm.get_size("h")
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nd = nd
        self.nh = nh
        self.nw = nw
        self.ldmax = min(ldmax or self.nd, self.nd)
        self.lhmax = min(lhmax or self.nh, self.nh)
        self.lwmax = min(lwmax or self.nw // 2 + 1, self.nw // 2 + 1)

        # half-modes
        self.ldmax_high = math.ceil(self.ldmax / 2)
        self.ldmax_low = math.floor(self.ldmax / 2)
        self.lhmax_high = math.ceil(self.lhmax / 2)
        self.lhmax_low = math.floor(self.lhmax / 2)

        # shapes, we assume the d-dim is always local
        self.lat_shapes = compute_split_shapes(self.nh, self.comm_size_h)
        self.lon_shapes = compute_split_shapes(self.nw, self.comm_size_w)
        self.l_shapes = compute_split_shapes(self.lhmax, self.comm_size_h)
        self.m_shapes = compute_split_shapes(self.lwmax, self.comm_size_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # make sure input is 5D
        torch._check(x.dim() == 5, lambda: f"expected a 5D input, got {x.dim()}D")

        # store number of chans
        num_chans = x.shape[1]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_w > 1:
            x = distributed_transpose_w(x, (1, -1), self.lon_shapes)

        # do first 2D FFT
        x = torch.fft.rfft2(x, s=(self.nd, self.nw), dim=(-3, -1), norm="ortho")

        # truncate width-modes
        x = x[..., : self.lwmax]

        # truncate depth-modes
        x = torch.cat([x[..., : self.ldmax_high, :, :], x[..., -self.ldmax_low :, :, :]], dim=-3)

        # transpose: after this, m is split and c is local
        if self.comm_size_w > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_w)
            x = distributed_transpose_w(x, (-1, 1), chan_shapes)

        # transpose: after this, c is split and h is local
        if self.comm_size_h > 1:
            x = distributed_transpose_h(x, (1, -2), self.lat_shapes)

        # do second FFT:
        x = torch.fft.fft(x, n=self.nh, dim=-2, norm="ortho")

        # truncate the modes
        x = torch.cat([x[..., : self.lhmax_high, :], x[..., -self.lhmax_low :, :]], dim=-2)

        # transpose: after this, l is split and c is local
        if self.comm_size_h > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_h)
            x = distributed_transpose_h(x, (-2, 1), chan_shapes)

        return x


class DistributedInverseRealFFT3(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nd, nh, nw, ldmax=None, lhmax=None, lwmax=None):
        super().__init__()

        # get the comms grid:
        self.comm_size_h = comm.get_size("h")
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nd = nd
        self.nh = nh
        self.nw = nw
        self.ldmax = min(ldmax or self.nd, self.nd)
        self.lhmax = min(lhmax or self.nh, self.nh)
        self.lwmax = min(lwmax or self.nw // 2 + 1, self.nw // 2 + 1)

        # half-modes
        self.ldmax_high = math.ceil(self.ldmax / 2)
        self.ldmax_low = math.floor(self.ldmax / 2)
        self.lhmax_high = math.ceil(self.lhmax / 2)
        self.lhmax_low = math.floor(self.lhmax / 2)

        # shapes, we assume the d-dim is always local
        self.lat_shapes = compute_split_shapes(self.nh, self.comm_size_h)
        self.lon_shapes = compute_split_shapes(self.nw, self.comm_size_w)
        self.l_shapes = compute_split_shapes(self.lhmax, self.comm_size_h)
        self.m_shapes = compute_split_shapes(self.lwmax, self.comm_size_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # make sure input is 5D
        torch._check(x.dim() == 5, lambda: f"expected a 5D input, got {x.dim()}D")

        # store number of chans
        num_chans = x.shape[1]

        # transpose: after that, channels are split, lh is local:
        if self.comm_size_h > 1:
            x = distributed_transpose_h(x, (1, -2), self.l_shapes)

        # we should pad the middle here manually, so that the inverse FFT is correct
        if self.lhmax < self.nh:
            xh = x[..., : self.lhmax_high, :]
            xl = x[..., -self.lhmax_low :, :]
            xhp = F.pad(xh, (0, 0, 0, self.nh - self.lhmax), mode="constant")
            x = torch.cat([xhp, xl], dim=-2)

        if self.ldmax < self.nd:
            xh = x[..., : self.ldmax_high, :, :]
            xl = x[..., -self.ldmax_low :, :, :]
            xhp = F.pad(xh, (0, 0, 0, 0, 0, self.nd - self.ldmax), mode="constant")
            x = torch.cat([xhp, xl], dim=-3)

        # do first fft
        x = torch.fft.ifft2(x, s=(self.nd, self.nh), dim=(-3, -2), norm="ortho")

        if self.comm_size_h > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_h)
            x = distributed_transpose_h(x, (-2, 1), chan_shapes)

        # transpose: after this, channels are split and m is local
        if self.comm_size_w > 1:
            x = distributed_transpose_w(x, (1, -1), self.m_shapes)

        # apply the inverse (real) FFT
        x = irfft(x, n=self.nw, dim=-1, norm="ortho")

        # transpose: after this, m is split and channels are local
        if self.comm_size_w > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_w)
            x = distributed_transpose_w(x, (-1, 1), chan_shapes)

        return x
