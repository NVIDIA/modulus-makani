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

import torch
import torch.nn as nn
import math
import numpy as np
from collections.abc import Sequence

# patch emebeddings and patch recovery
from makani.models.common import PatchEmbed2D, PatchEmbed3D, PatchRecovery2D, PatchRecovery3D

# down and upsampling layers, Mlp
from makani.models.common import MLP, DownSample3D, UpSample3D

# helpers
from makani.models.common import DropPath

# features
from makani.utils import features

# checkpointing
from torch.utils.checkpoint import checkpoint

### Helpers: Sliding window attention and earth positional bias #####


def get_earth_position_index(window_size, ndim=3):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    This function construct the position index to reuse symmetrical parameters of the position bias.
    implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        window_size (tuple[int]): [pressure levels, latitude, longitude] or [latitude, longitude]
        ndim (int): dimension of tensor, 3 or 2

    Returns:
        position_index (torch.Tensor): [win_pl * win_lat * win_lon, win_pl * win_lat * win_lon] or [win_lat * win_lon, win_lat * win_lon]
    """
    if ndim == 3:
        win_pl, win_lat, win_lon = window_size
    elif ndim == 2:
        win_lat, win_lon = window_size

    if ndim == 3:
        # Index in the pressure level of query matrix
        coords_zi = torch.arange(win_pl)
        # Index in the pressure level of key matrix
        coords_zj = -torch.arange(win_pl) * win_pl

    # Index in the latitude of query matrix
    coords_hi = torch.arange(win_lat)
    # Index in the latitude of key matrix
    coords_hj = -torch.arange(win_lat) * win_lat

    # Index in the longitude of the key-value pair
    coords_w = torch.arange(win_lon)

    # Change the order of the index to calculate the index in total
    if ndim == 3:
        coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w], indexing="ij"))
        coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w], indexing="ij"))
    elif ndim == 2:
        coords_1 = torch.stack(torch.meshgrid([coords_hi, coords_w], indexing="ij"))
        coords_2 = torch.stack(torch.meshgrid([coords_hj, coords_w], indexing="ij"))
    coords_flatten_1 = torch.flatten(coords_1, 1)
    coords_flatten_2 = torch.flatten(coords_2, 1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()

    # Shift the index for each dimension to start from 0
    if ndim == 3:
        coords[:, :, 2] += win_lon - 1
        coords[:, :, 1] *= 2 * win_lon - 1
        coords[:, :, 0] *= (2 * win_lon - 1) * win_lat * win_lat
    elif ndim == 2:
        coords[:, :, 1] += win_lon - 1
        coords[:, :, 0] *= 2 * win_lon - 1

    # Sum up the indexes in two/three dimensions
    position_index = coords.sum(-1)

    return position_index


def get_pad3d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): (Pl, Lat, Lon)
        window_size (tuple[int]): (Pl, Lat, Lon)

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return (
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        padding_front,
        padding_back,
    )


def get_pad2d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): Lat, Lon
        window_size (tuple[int]): Lat, Lon

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom)
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[:4]


def crop2d(x: torch.Tensor, resolution):
    """
    Args:
        x (torch.Tensor): B, C, Lat, Lon
        resolution (tuple[int]): Lat, Lon
    """
    _, _, Lat, Lon = x.shape
    lat_pad = Lat - resolution[0]
    lon_pad = Lon - resolution[1]

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left

    return x[:, :, padding_top : Lat - padding_bottom, padding_left : Lon - padding_right]


def crop3d(x: torch.Tensor, resolution):
    """
    Args:
        x (torch.Tensor): B, C, Pl, Lat, Lon
        resolution (tuple[int]): Pl, Lat, Lon
    """
    _, _, Pl, Lat, Lon = x.shape
    pl_pad = Pl - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]

    padding_front = pl_pad // 2
    padding_back = pl_pad - padding_front

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left
    return x[
        :,
        :,
        padding_front : Pl - padding_back,
        padding_top : Lat - padding_bottom,
        padding_left : Lon - padding_right,
    ]


def window_partition(x: torch.Tensor, window_size, ndim=3):
    """
    Args:
        x: (B, Pl, Lat, Lon, C) or (B, Lat, Lon, C)
        window_size (tuple[int]): [win_pl, win_lat, win_lon] or [win_lat, win_lon]
        ndim (int): dimension of window (3 or 2)

    Returns:
        windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C) or (B*num_lon, num_lat, win_lat, win_lon, C)
    """
    if ndim == 3:
        B, Pl, Lat, Lon, C = x.shape
        win_pl, win_lat, win_lon = window_size
        x = x.view(B, Pl // win_pl, win_pl, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
        windows = (
            x.permute(0, 5, 1, 3, 2, 4, 6, 7)
            .contiguous()
            .view(-1, (Pl // win_pl) * (Lat // win_lat), win_pl, win_lat, win_lon, C)
        )
        return windows
    elif ndim == 2:
        B, Lat, Lon, C = x.shape
        win_lat, win_lon = window_size
        x = x.view(B, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
        windows = x.permute(0, 3, 1, 2, 4, 5).contiguous().view(-1, (Lat // win_lat), win_lat, win_lon, C)
        return windows


def window_reverse(windows, window_size, Pl=1, Lat=1, Lon=1, ndim=3):
    """
    Args:
        windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C) or (B*num_lon, num_lat, win_lat, win_lon, C)
        window_size (tuple[int]): [win_pl, win_lat, win_lon] or [win_lat, win_lon]
        Pl (int): pressure levels
        Lat (int): latitude
        Lon (int): longitude
        ndim (int): dimension of window (3 or 2)

    Returns:
        x: (B, Pl, Lat, Lon, C) or (B, Lat, Lon, C)
    """
    if ndim == 3:
        win_pl, win_lat, win_lon = window_size
        B = int(windows.shape[0] / (Lon / win_lon))
        x = windows.view(
            B,
            Lon // win_lon,
            Pl // win_pl,
            Lat // win_lat,
            win_pl,
            win_lat,
            win_lon,
            -1,
        )
        x = x.permute(0, 2, 4, 3, 5, 1, 6, 7).reshape(B, Pl, Lat, Lon, -1)
        return x
    elif ndim == 2:
        win_lat, win_lon = window_size
        B = int(windows.shape[0] / (Lon / win_lon))
        x = windows.view(B, Lon // win_lon, Lat // win_lat, win_lat, win_lon, -1)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, Lat, Lon, -1)
        return x


def get_shift_window_mask(input_resolution, window_size, shift_size, ndim=3):
    """
    Attention mask of the shifted windows. After the roll, the last window along each dimension holds
    tokens from both ends of that dimension, which the mask keeps from attending to each other. This
    includes longitude: the half windows from the left and right edges are masked from each other,
    even though longitude is periodic.
    Args:
        input_resolution (tuple[int]): [pressure levels, latitude, longitude] or [latitude, longitude]
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude] or [latitude, longitude]
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude] or [latitude, longitude]
        ndim (int): dimension of window (3 or 2)

    Returns:
        attn_mask: (n_lon, n_pl*n_lat, win_pl*win_lat*win_lon, win_pl*win_lat*win_lon) or (n_lon, n_lat, win_lat*win_lon, win_lat*win_lon)
    """
    if ndim == 3:
        Pl, Lat, Lon = input_resolution
        win_pl, win_lat, win_lon = window_size
        shift_pl, shift_lat, shift_lon = shift_size

        img_mask = torch.zeros((1, Pl, Lat, Lon + shift_lon, 1))
    elif ndim == 2:
        Lat, Lon = input_resolution
        win_lat, win_lon = window_size
        shift_lat, shift_lon = shift_size

        img_mask = torch.zeros((1, Lat, Lon + shift_lon, 1))

    if ndim == 3:
        pl_slices = (
            slice(0, -win_pl),
            slice(-win_pl, -shift_pl),
            slice(-shift_pl, None),
        )
    lat_slices = (
        slice(0, -win_lat),
        slice(-win_lat, -shift_lat),
        slice(-shift_lat, None),
    )
    lon_slices = (
        slice(0, -win_lon),
        slice(-win_lon, -shift_lon),
        slice(-shift_lon, None),
    )

    cnt = 0
    if ndim == 3:
        for pl in pl_slices:
            for lat in lat_slices:
                for lon in lon_slices:
                    img_mask[:, pl, lat, lon, :] = cnt
                    cnt += 1
        img_mask = img_mask[:, :, :, :Lon, :]
    elif ndim == 2:
        for lat in lat_slices:
            for lon in lon_slices:
                img_mask[:, lat, lon, :] = cnt
                cnt += 1
        img_mask = img_mask[:, :, :Lon, :]

    mask_windows = window_partition(
        img_mask, window_size, ndim=ndim
    )  # n_lon, n_pl*n_lat, win_pl, win_lat, win_lon, 1 or n_lon, n_lat, win_lat, win_lon, 1
    if ndim == 3:
        win_total = win_pl * win_lat * win_lon
    elif ndim == 2:
        win_total = win_lat * win_lon
    mask_windows = mask_windows.view(mask_windows.shape[0], mask_windows.shape[1], win_total)
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


###############


class EarthAttention3D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    3D window attention with earth position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        window_size (tuple[int]): [pressure levels, latitude, longitude]
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wpl, Wlat, Wlon
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.use_sdpa = use_sdpa
        self.attn_drop = attn_drop

        self.type_of_windows = (input_resolution[0] // window_size[0]) * (input_resolution[1] // window_size[1])
        self.num_lon = input_resolution[2] // window_size[2]

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size[0] ** 2) * (window_size[1] ** 2) * (window_size[2] * 2 - 1),
                self.type_of_windows,
                num_heads,
            )
        )  # Wpl**2 * Wlat**2 * Wlon*2-1, Npl//Wpl * Nlat//Wlat, nH

        earth_position_index = get_earth_position_index(window_size)  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        self.register_buffer("earth_position_index", earth_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_fn = nn.Dropout(self.attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.earth_position_bias_table = nn.init.trunc_normal_(self.earth_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def calculate_attn(self, q, k):
        r"""
        Compute raw attention logits from queries and keys.

        Split out as its own method so subclasses and export wrappers can
        substitute a different attention implementation without reimplementing
        the surrounding bias and masking logic.

        Parameters
        ----------
        q : torch.Tensor
            Queries of shape ``(B_, num_heads, nW_, N, head_dim)``.
        k : torch.Tensor
            Keys of the same shape.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B_, num_heads, nW_, N, N)``. Not yet scaled,
            biased or softmaxed.
        """
        attn = q @ k.transpose(-2, -1)
        return attn

    def add_earth_pos_bias(self, attn, earth_position_bias):
        r"""
        Add the Earth-specific position bias to the attention logits.

        Parameters
        ----------
        attn : torch.Tensor
            Attention logits of shape ``(B_, num_heads, nW_, N, N)``.
        earth_position_bias : torch.Tensor
            Bias of shape ``(num_heads, nW_, N, N)``, broadcast over the batch.

        Returns
        -------
        torch.Tensor
            Biased logits, same shape as ``attn``.
        """
        attn = attn + earth_position_bias.unsqueeze(0)
        return attn

    def apply_attention(self, attn, v, B_, nW_, N, C):
        r"""
        Apply normalized attention weights to the values.

        Parameters
        ----------
        attn : torch.Tensor
            Normalized attention weights of shape ``(B_, num_heads, nW_, N, N)``.
        v : torch.Tensor
            Values of shape ``(B_, num_heads, nW_, N, head_dim)``.
        B_ : int
            Batch dimension, including the longitude windows folded into it.
        nW_ : int
            Number of window groups.
        N : int
            Number of tokens per window.
        C : int
            Feature dimension.

        Returns
        -------
        torch.Tensor
            Attended features of shape ``(B_, nW_, N, C)``, with the heads
            merged back together.
        """
        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        return x

    def extract_earth_pos_bias(self):
        r"""
        Gather the position bias for the current window layout.

        Indexes the learned bias table by the precomputed relative-position
        index and reshapes it to broadcast against the attention logits. The
        bias is Earth-specific: on a lat-lon grid the relationship between two
        cells depends on their absolute latitude and pressure level, not only on
        their offset, so the table is indexed by more than a relative
        displacement.

        Returns
        -------
        torch.Tensor
            Bias of shape ``(num_heads, num_pl * num_lat, N, N)`` where
            ``N = Wpl * Wlat * Wlon``.
        """
        earth_position_bias = self.earth_position_bias_table[self.earth_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows,
            -1,
        )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon, num_pl*num_lat, nH
        earth_position_bias = earth_position_bias.permute(
            3, 2, 0, 1
        ).contiguous()  # nH, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon

        return earth_position_bias

    def apply_mask(self, attn, mask, B_, nW_, N):
        r"""
        Apply the shifted-window attention mask and normalize.

        Under a shifted window partitioning, some windows contain tokens that
        are not actually adjacent on the globe. The mask assigns those pairs
        ``-inf`` so the softmax gives them zero weight.

        Parameters
        ----------
        attn : torch.Tensor
            Attention logits of shape ``(B_, num_heads, nW_, N, N)``.
        mask : torch.Tensor
            Additive mask of shape ``(num_lon, nW_, N, N)``, containing ``0``
            for valid pairs and ``-inf`` for invalid ones.
        B_ : int
            Batch dimension, including the longitude windows folded into it.
        nW_ : int
            Number of window groups.
        N : int
            Number of tokens per window.

        Returns
        -------
        torch.Tensor
            Softmax-normalized attention weights, same shape as ``attn``.
        """
        nLon = mask.shape[0]
        attn = attn.view(B_ // nLon, nLon, self.num_heads, nW_, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, nW_, N, N)
        attn = self.softmax(attn)

        return attn

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: input features with shape of (B * num_lon, num_pl*num_lat, N, C)
            mask: (0/-inf) mask with shape of (num_lon, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon)
        """

        B_, nW_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        earth_position_bias = self.extract_earth_pos_bias()

        if not self.use_sdpa:
            q_new = q * self.scale

            attn = self.calculate_attn(q_new, k)

            attn = self.add_earth_pos_bias(attn, earth_position_bias)

            if mask is not None:
                attn = self.apply_mask(attn, mask, B_, nW_, N)
            else:
                attn = self.softmax(attn)

            attn = self.attn_drop_fn(attn)

            x = self.apply_attention(attn, v, B_, nW_, N, C)

        else:
            if mask is not None:
                bias = mask.unsqueeze(1).unsqueeze(0) + earth_position_bias.unsqueeze(0).unsqueeze(0)
                # squeeze the bias if needed in dim 2
                # bias = bias.squeeze(2)
            else:
                bias = earth_position_bias.unsqueeze(0)

            # extract batch size for q,k,v
            nLon = self.num_lon
            q = q.view(B_ // nLon, nLon, q.shape[1], q.shape[2], q.shape[3], q.shape[4])
            k = k.view(B_ // nLon, nLon, k.shape[1], k.shape[2], k.shape[3], k.shape[4])
            v = v.view(B_ // nLon, nLon, v.shape[1], v.shape[2], v.shape[3], v.shape[4])
            ####
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, scale=self.scale, dropout_p=self.attn_drop
            )
            x = x.permute(0, 1, 3, 4, 2, 5).reshape(B_, nW_, N, C)

        # projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Transformer3DBlock(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    3D Transformer Block
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude].
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=None,
        shift_size=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        checkpointing_level=0,
    ):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size = (1, 3, 6) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.checkpointing_level = checkpointing_level

        self.norm1 = norm_layer(dim)
        padding = get_pad3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += padding[-1] + padding[-2]
        pad_resolution[1] += padding[2] + padding[3]
        pad_resolution[2] += padding[0] + padding[1]

        self.attn = EarthAttention3D(
            dim=dim,
            input_resolution=pad_resolution,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            input_format="traditional",
            drop_rate=drop,
            checkpointing=(checkpointing_level >= 2),
        )

        shift_pl, shift_lat, shift_lon = self.shift_size
        self.roll = shift_pl and shift_lon and shift_lat

        if self.roll:
            attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, x: torch.Tensor):
        r"""
        Apply windowed 3D attention and the channel MLP with residuals.

        Pads the volume to a whole number of windows, optionally rolls it to
        realize the shifted-window partitioning, attends within each window,
        then reverses the roll and crops back to the original resolution.

        Parameters
        ----------
        x : torch.Tensor
            Tokens of shape ``(B, Pl * Lat * Lon, C)`` on the block's
            ``input_resolution``.

        Returns
        -------
        torch.Tensor
            Tokens of the same shape as ``x``.
        """
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        if self.checkpointing_level >= 1:
            x = checkpoint(self.norm1, x, use_reentrant=False)
        else:
            x = self.norm1(x)
        x = x.view(B, Pl, Lat, Lon, C)

        # start pad
        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

        shift_pl, shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3))
            x_windows = window_partition(shifted_x, self.window_size)
            # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
            # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C

        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C)
        # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

        if self.checkpointing_level > 0:
            attn_windows = checkpoint(self.attn, x_windows, self.attn_mask, use_reentrant=False)
        else:
            attn_windows = self.attn(
                x_windows, mask=self.attn_mask
            )  # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

        attn_windows = attn_windows.view(attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C)

        if self.roll:
            shifted_x = window_reverse(attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad)
            # B * Pl * Lat * Lon * C
            x = torch.roll(shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad)
            x = shifted_x

        # crop, end pad
        x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(0, 2, 3, 4, 1)

        x = x.reshape(B, Pl * Lat * Lon, C)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class FuserLayer(nn.Module):
    """Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    A basic 3D Transformer layer for one stage

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        checkpointing_level=0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.checkpointing_level = checkpointing_level

        self.blocks = nn.ModuleList(
            [
                Transformer3DBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    # shift_size=(0, 0, 0) if i % 2 == 0 else None,
                    shift_size=[0 if i % 2 == 0 else w // 2 for w in window_size],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, Sequence) else drop_path,
                    norm_layer=norm_layer,
                    checkpointing_level=checkpointing_level,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        r"""
        Run the tokens through this layer's stack of transformer blocks.

        Parameters
        ----------
        x : torch.Tensor
            Tokens of shape ``(B, Pl * Lat * Lon, C)``.

        Returns
        -------
        torch.Tensor
            Tokens of the same shape as ``x``.
        """
        for blk in self.blocks:
            x = blk(x)
        return x


class Pangu(nn.Module):
    """Pangu-Weather implementation as in Bi et al.; Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast
    - https://arxiv.org/abs/2211.02556
    """

    def __init__(
        self,
        inp_shape=(721, 1440),
        out_shape=(721, 1440),
        grid_in="equiangular",
        grid_out="equiangular",
        inp_chans=5,
        out_chans=5,
        patch_size=(2, 8, 8),
        embed_dim=8,
        depth_layers=(1, 1, 1, 1),
        num_heads=(1, 1, 1, 1),
        window_size=(2, 6, 12),
        num_surface=2,
        num_atmospheric=3,
        num_levels=1,
        channel_names=["u10m", "t2m", "u500", "z500", "t500"],
        aux_channel_names=[],
        drop_path_rate=0.0,
        checkpointing_level=0,
        **kwargs,
    ):

        super().__init__()
        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_surface = num_surface
        self.num_levels = num_levels
        self.num_atmospheric = num_atmospheric
        self.channel_names = channel_names
        self.aux_channel_names = aux_channel_names
        self.checkpointing_level = checkpointing_level

        drop_path = np.linspace(0, drop_path_rate, 8).tolist()

        # Add static channels to surface
        self.num_aux = len(self.aux_channel_names)
        N_total_surface = self.num_aux + self.num_surface
        self.has_surface = N_total_surface > 0

        # compute static permutations to extract
        self._precompute_channel_groups(self.channel_names, self.aux_channel_names)

        # Patch embeddings are 2D or 3D convolutions, mapping the data to the required patches
        self.patchembed2d = PatchEmbed2D(
            img_size=self.inp_shape,
            patch_size=patch_size[1:],
            in_chans=N_total_surface,
            embed_dim=embed_dim,
            padding=True,
            flatten=False,
            norm_layer=None,
        )

        self.patchembed3d = PatchEmbed3D(
            img_size=(num_levels, self.inp_shape[0], self.inp_shape[1]),
            patch_size=patch_size,
            in_chans=num_atmospheric,
            embed_dim=embed_dim,
            norm_layer=None,
            padding=True,
        )

        # Patched shapes
        patched_inp_shape = (
            math.ceil(num_levels / patch_size[0]) + 1,
            math.ceil(self.inp_shape[0] / patch_size[1]),
            math.ceil(self.inp_shape[1] / patch_size[2]),
        )

        self.layer1 = FuserLayer(
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=depth_layers[0],
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2],
            checkpointing_level=self.checkpointing_level,
        )

        patched_inp_shape_downsample = (
            math.ceil(num_levels / patch_size[0]) + 1,
            math.ceil(patched_inp_shape[1] / 2),
            math.ceil(patched_inp_shape[2] / 2),
        )

        self.downsample = DownSample3D(
            in_dim=embed_dim,
            input_resolution=patched_inp_shape,
            output_resolution=patched_inp_shape_downsample,
        )

        self.layer2 = FuserLayer(
            dim=embed_dim * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=depth_layers[1],
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2:],
            checkpointing_level=self.checkpointing_level,
        )

        self.layer3 = FuserLayer(
            dim=embed_dim * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=depth_layers[2],
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:],
            checkpointing_level=self.checkpointing_level,
        )

        self.upsample = UpSample3D(embed_dim * 2, embed_dim, patched_inp_shape_downsample, patched_inp_shape)

        self.layer4 = FuserLayer(
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=depth_layers[3],
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2],
            checkpointing_level=self.checkpointing_level,
        )

        self.patchrecovery2d = PatchRecovery2D(self.inp_shape, patch_size[1:], 2 * embed_dim, num_surface)
        self.patchrecovery3d = PatchRecovery3D(
            (num_levels, self.inp_shape[0], self.inp_shape[1]), patch_size, 2 * embed_dim, num_atmospheric
        )

    def _precompute_channel_groups(
        self,
        channel_names=[],
        aux_channel_names=[],
    ):
        """
        Group the channels appropriately into atmospheric pressure levels and surface variables
        """

        atmo_chans, surf_chans, dyn_aux_chans, stat_aux_chans, pressure_lvls = features.get_channel_groups(
            channel_names, aux_channel_names
        )
        aux_chans = dyn_aux_chans + stat_aux_chans

        # compute how many channel groups will be kept internally
        self.n_atmo_groups = len(pressure_lvls)
        self.n_atmo_chans = len(atmo_chans) // self.n_atmo_groups

        # make sure they are divisible. Attention! This does not guarantee that the grrouping is correct
        if len(atmo_chans) % self.n_atmo_groups:
            raise ValueError(
                f"Expected number of atmospheric variables to be divisible by number of atmospheric groups but got {len(atmo_chans)} and {self.n_atmo_groups}"
            )

        self.register_buffer("atmo_channels", torch.tensor(atmo_chans, dtype=torch.long), persistent=False)
        self.register_buffer("surf_channels", torch.tensor(surf_chans, dtype=torch.long), persistent=False)
        self.register_buffer("aux_channels", torch.tensor(aux_chans, dtype=torch.long), persistent=False)

        self.n_surf_chans = self.surf_channels.shape[0]
        self.n_aux_chans = self.aux_channels.shape[0]

        # Precompute the per-pressure-level channel layout used by prepare_input /
        # prepare_output. This depends only on channel_names (static config), so we
        # build it once here as index buffers instead of recomputing numpy ops
        # (np.unique / sorted) on every forward -- those would graph-break under
        # torch.compile.
        levels = np.unique([value[1:] for value in channel_names[self.num_surface :]]).tolist()
        levels = sorted(levels, key=lambda x: int(x))
        if len(levels) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} pressure levels from channel_names but parsed " f"{len(levels)}: {levels}"
            )
        level_dict = {level: [idx for idx, value in enumerate(channel_names) if value[1:] == level] for level in levels}

        # [vars_per_level, num_levels]: input[:, atmo_level_index, :, :] yields
        # [B, vars_per_level, num_levels, H, W], matching the original stack(dim=2).
        atmo_level_index = torch.tensor([level_dict[level] for level in levels], dtype=torch.long).t().contiguous()
        self.register_buffer("atmo_level_index", atmo_level_index, persistent=False)

        # flat permutation mapping each restructured atmospheric channel back to its
        # original global channel position (used to scatter the output back).
        reordered_ids = [idx for level in levels for idx in level_dict[level]]
        self.register_buffer("reordered_ids", torch.tensor(reordered_ids, dtype=torch.long), persistent=False)

        return

    def prepare_input(self, input):
        """
        Prepares the input tensor for the Pangu model by splitting it into surface * static variables and atmospheric,
        and reshaping the atmospheric variables into the required format.
        """

        # extract surface and static variables
        surface_inp = input[:, self.surf_channels, :, :]
        aux_inp = input[:, self.aux_channels, :, :]
        surface_aux_inp = torch.cat([surface_inp, aux_inp], dim=1)

        # extract atmospheric variables and restructure into [B, vars, levels, H, W]
        # using the precomputed level-index buffer (no numpy in the forward path).
        atmospheric_inp = input[:, self.atmo_level_index, :, :]

        return surface_aux_inp, atmospheric_inp

    def prepare_output(self, output_surface, output_atmospheric):
        """
        The output of the Pangu model gives separate outputs for surface and atmospheric variables
        Also the atmospheric variables are restructured before fed to the network --> see self.prepare_input()
        This functions reverts the restructuring and concatenates the output to a single tensor.
        """
        # Flatten & reorder the output atmospheric back to the original channel order
        # using the precomputed permutation buffer (vectorized scatter; no numpy and
        # no Python loop in the forward path).
        flattened_atmospheric = output_atmospheric.reshape(
            output_atmospheric.shape[0], -1, output_atmospheric.shape[3], output_atmospheric.shape[4]
        )
        reordered_atmospheric = torch.cat(
            [torch.zeros_like(output_surface), torch.zeros_like(flattened_atmospheric)], dim=1
        )
        reordered_atmospheric[:, self.reordered_ids, :, :] = flattened_atmospheric

        # Append the surface output, this has not been reordered.
        if output_surface is not None:
            reordered_atmospheric[:, self.surf_channels, :, :] = output_surface
            output = reordered_atmospheric
        else:
            output = reordered_atmospheric

        return output

    def forward(self, input):
        r"""
        Map an input field to the predicted output field.

        Splits the flat channel stack into surface and atmospheric (pressure
        level) variables, embeds each with its own patch embedding, runs the
        encoder-decoder transformer stack with a skip connection between the two
        levels of resolution, and recombines the two variable groups into a
        single output stack.

        Parameters
        ----------
        input : torch.Tensor
            Input of shape ``(B, inp_chans, nlat, nlon)``, with the surface and
            atmospheric variables concatenated along the channel dimension in
            the order the model was configured with.

        Returns
        -------
        torch.Tensor
            Prediction of shape ``(B, out_chans, nlat, nlon)``, in the same
            channel layout as the input.
        """

        # Prep the input by splitting into surface and atmospheric variables
        surface_aux, atmospheric = self.prepare_input(input)

        # patch embedding
        if self.checkpointing_level > 0:
            surface = checkpoint(self.patchembed2d, surface_aux, use_reentrant=False)
            atmospheric = checkpoint(self.patchembed3d, atmospheric, use_reentrant=False)
        else:
            surface = self.patchembed2d(surface_aux)
            atmospheric = self.patchembed3d(atmospheric)

        if not self.has_surface:
            x = atmospheric
        else:
            x = torch.concat([surface.unsqueeze(2), atmospheric], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        x = self.layer1(x)

        skip = x

        if self.checkpointing_level >= 4:
            x = checkpoint(self.downsample, x, use_reentrant=False)
        else:
            x = self.downsample(x)

        x = self.layer2(x)
        x = self.layer3(x)

        if self.checkpointing_level >= 4:
            x = checkpoint(self.upsample, x, use_reentrant=False)
        else:
            x = self.upsample(x)
        x = self.layer4(x)

        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        if not self.has_surface:
            output_surface = None
            output_atmospheric = output
        else:
            output_surface = output[:, :, 0, :, :]
            output_atmospheric = output[:, :, 1:, :, :]

        if not self.has_surface:
            output_surface = None
            if self.checkpointing_level >= 4:
                output_atmospheric = checkpoint(self.patchrecovery3d, output_atmospheric, use_reentrant=False)
            else:
                output_atmospheric = self.patchrecovery3d(output_atmospheric)
        else:
            if self.checkpointing_level >= 4:
                output_surface = checkpoint(self.patchrecovery2d, output_surface, use_reentrant=False)
                output_atmospheric = checkpoint(self.patchrecovery3d, output_atmospheric, use_reentrant=False)
            else:
                output_surface = self.patchrecovery2d(output_surface)
                output_atmospheric = self.patchrecovery3d(output_atmospheric)

        output = self.prepare_output(output_surface, output_atmospheric)

        return output
