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

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# zeroes the imaginary parts of the DC and Nyquist bins before the inverse, which cuFFT does not do
from torch_harmonics.fft import irfft
from makani.models.common import DropPath, PatchEmbed2D


class PeriodicPad2d(nn.Module):
    r"""
    Pad longitudinal (left-right) circular, pad latitude (top-bottom) with zeros.

    Matches the topology of a global lat-lon grid: longitude wraps around, so
    the left and right edges are genuinely adjacent and are padded circularly,
    while latitude terminates at the poles and is padded with zeros.

    Parameters
    ----------
    pad_width : int
        Number of cells added on each of the four sides.
    """

    def __init__(self, pad_width):
        super(PeriodicPad2d, self).__init__()
        self.pad_width = pad_width

    def forward(self, x):
        r"""
        Pad the input according to the grid topology.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Padded tensor of shape
            ``(B, C, H + 2 * pad_width, W + 2 * pad_width)``.
        """
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular")
        out = F.pad(out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0)
        return out


class Mlp(nn.Module):
    r"""
    Two-layer channels-last feed-forward block.

    The MLP used inside :class:`Block`. Simpler than
    :class:`~makani.models.common.layers.MLP`: linear layers only, no
    model-parallel or TransformerEngine paths.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int, optional
        Hidden width, defaults to ``in_features``.
    out_features : int, optional
        Number of output features, defaults to ``in_features``.
    act_layer : callable, optional
        Activation constructor, by default :class:`torch.nn.GELU`.
    drop : float, optional
        Dropout probability applied after each layer, by default ``0.0``.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        r"""
        Apply the feed-forward block.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(..., in_features)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(..., out_features)``.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    r"""
    Adaptive Fourier Neural Operator token mixer.

    Mixes tokens by transforming to Fourier space, applying a two-layer
    block-diagonal MLP to the complex coefficients, and transforming back:

    .. math::

        y = \mathcal{F}^{-1}\bigl(S_\lambda(\mathrm{MLP}(\mathcal{F}(x)))\bigr) + x

    Three choices make this cheaper than dense spectral mixing. The channel
    dimension is split into ``num_blocks`` groups mixed independently, so the
    weights are block-diagonal rather than full. Only a fraction of the modes is
    retained (``hard_thresholding_fraction``), discarding the high frequencies.
    And a soft-shrink :math:`S_\lambda` sparsifies the coefficients, which acts
    as a learned frequency-domain filter.

    The transform is always taken in fp32 regardless of autocast, since the FFT
    accumulates over the whole grid.

    Parameters
    ----------
    hidden_size : int
        Channel dimension. Must be divisible by ``num_blocks``.
    num_blocks : int, optional
        Number of independently mixed channel blocks, by default ``8``.
    sparsity_threshold : float, optional
        Soft-shrink threshold :math:`\lambda` applied to the output
        coefficients, by default ``0.01``.
    hard_thresholding_fraction : float, optional
        Fraction of Fourier modes retained, by default ``1`` (keep all).
    hidden_size_factor : int, optional
        Width of the spectral MLP's hidden layer as a multiple of the block
        size, by default ``1``.

    Raises
    ------
    ValueError
        If ``hidden_size`` is not divisible by ``num_blocks``.

    References
    ----------
    Guibas, J.; Mardani, M.; Pathak, J.; Vahdat, A.; Kashinath, K.; Catanzaro,
    B.; Anandkumar, A.; Adaptive Fourier Neural Operators: Efficient Token
    Mixers for Transformers; ICLR 2022.
    """

    def __init__(
        self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1
    ):
        super().__init__()
        if hidden_size % num_blocks != 0:
            raise ValueError(f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}")

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor)
        )
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size)
        )
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        r"""
        Mix tokens in Fourier space and add the input back as a residual.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, H, W, hidden_size)``, channels last.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``, with the input added back.
        """
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device
        )
        o1_imag = torch.zeros(
            [B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = F.relu(
            torch.einsum(
                "...bi,bio->...bo",
                x[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes].real,
                self.w1[0],
            )
            - torch.einsum(
                "...bi,bio->...bo",
                x[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes].imag,
                self.w1[1],
            )
            + self.b1[0]
        )

        o1_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = F.relu(
            torch.einsum(
                "...bi,bio->...bo",
                x[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes].imag,
                self.w1[0],
            )
            + torch.einsum(
                "...bi,bio->...bo",
                x[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes].real,
                self.w1[1],
            )
            + self.b1[1]
        )

        o2_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            torch.einsum(
                "...bi,bio->...bo",
                o1_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],
                self.w2[0],
            )
            - torch.einsum(
                "...bi,bio->...bo",
                o1_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],
                self.w2[1],
            )
            + self.b2[0]
        )

        o2_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            torch.einsum(
                "...bi,bio->...bo",
                o1_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],
                self.w2[0],
            )
            + torch.einsum(
                "...bi,bio->...bo",
                o1_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],
                self.w2[1],
            )
            + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        # the inverse in H first, then the real inverse in W: the m=0 column of the mixed spectrum is not
        # Hermitian in general, and backends differ in how they treat its imaginary parts
        x = torch.fft.ifft(x, n=H, dim=1, norm="ortho")
        x = irfft(x, n=W, dim=2, norm="ortho")
        x = x.type(dtype)

        return x + bias


class Block(nn.Module):
    r"""
    AFNO block: spectral token mixing followed by a channel MLP.

    Structurally a transformer block with :class:`AFNO2D` in place of
    attention. ``double_skip`` selects between two residual arrangements: with
    it enabled each sublayer gets its own skip, and with it disabled a single
    skip spans both.

    Parameters
    ----------
    dim : int
        Channel dimension.
    mlp_ratio : float, optional
        Hidden width of the MLP as a multiple of ``dim``, by default ``4.0``.
    drop : float, optional
        Dropout probability inside the MLP, by default ``0.0``.
    drop_path : float, optional
        Stochastic depth probability, by default ``0.0``.
    act_layer : callable, optional
        Activation constructor, by default :class:`torch.nn.GELU`.
    norm_layer : callable, optional
        Normalization constructor, by default :class:`torch.nn.LayerNorm`.
    double_skip : bool, optional
        Use a separate residual connection around each sublayer, by default
        ``True``.
    num_blocks : int, optional
        Number of channel blocks in the AFNO mixer, by default ``8``.
    sparsity_threshold : float, optional
        Soft-shrink threshold in the AFNO mixer, by default ``0.01``.
    hard_thresholding_fraction : float, optional
        Fraction of Fourier modes retained, by default ``1.0``.
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        double_skip=True,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        r"""
        Apply spectral mixing and the channel MLP with residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, H, W, dim)``, channels last.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``.
        """
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class PrecipNet(nn.Module):
    r"""
    Precipitation head on top of a forecasting backbone.

    Runs a backbone and post-processes its output with a periodically padded
    3x3 convolution and a ReLU. The ReLU matters physically: precipitation
    cannot be negative, and clamping it in the architecture is more reliable
    than hoping the loss enforces it. The local convolution smooths the
    backbone's output, which suits a field that is spatially patchy.

    Parameters
    ----------
    backbone : torch.nn.Module
        Model producing the field this head refines.
    patch_size : (int, int), optional
        Recorded for reference, by default ``(16, 16)``.
    inp_chans : int, optional
        Number of backbone input channels, by default ``2``.
    out_chans : int, optional
        Number of output channels, by default ``2``.
    **kwargs
        Ignored; present so model configs can pass extra keys.
    """

    def __init__(self, backbone, patch_size=(16, 16), inp_chans=2, out_chans=2, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.backbone = backbone
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2d(self.out_chans, self.out_chans, kernel_size=3, stride=1, padding=0, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        r"""
        Run the backbone and refine its output into a non-negative field.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, inp_chans, nlat, nlon)``.

        Returns
        -------
        torch.Tensor
            Non-negative prediction of shape ``(B, out_chans, nlat, nlon)``.
        """
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x


class AdaptiveFourierNeuralOperatorNet(nn.Module):
    r"""
    Adaptive Fourier Neural Operator network (AFNO).

    A transformer-shaped architecture in which attention is replaced by the
    :class:`AFNO2D` spectral mixer. The input is tokenized by patch embedding, a
    stack of :class:`Block` layers mixes the tokens, and a linear head decodes
    each token back into its patch. Replacing attention with an FFT-based mixer
    reduces the token-mixing cost from quadratic to ``O(N log N)``, which is
    what makes high-resolution inputs affordable.

    Parameters
    ----------
    inp_shape : (int, int), optional
        Input grid as ``(nlat, nlon)``, by default ``(720, 1440)``.
    patch_size : (int, int), optional
        Patch size, by default ``(6, 6)``.
    inp_chans : int, optional
        Number of input channels, by default ``2``.
    out_chans : int, optional
        Number of output channels, by default ``2``.
    embed_dim : int, optional
        Token dimension, by default ``768``.
    num_layers : int, optional
        Number of AFNO blocks, by default ``12``.
    mlp_ratio : float, optional
        Hidden width of the block MLPs as a multiple of ``embed_dim``, by
        default ``4.0``.
    drop_rate : float, optional
        Dropout probability, by default ``0.0``.
    drop_path_rate : float, optional
        Maximum stochastic depth probability, ramped linearly across the depth,
        by default ``0.0``.
    num_blocks : int, optional
        Number of independently mixed channel blocks in each AFNO mixer, by
        default ``16``.
    sparsity_threshold : float, optional
        Soft-shrink threshold in the AFNO mixers, by default ``0.01``.
    hard_thresholding_fraction : float, optional
        Fraction of Fourier modes retained, by default ``1.0``.
    **kwargs
        Ignored; present so model configs can pass extra keys.

    References
    ----------
    Guibas, J. et al.; Adaptive Fourier Neural Operators: Efficient Token
    Mixers for Transformers; ICLR 2022.
    """

    def __init__(
        self,
        inp_shape=(720, 1440),
        patch_size=(6, 6),
        inp_chans=2,
        out_chans=2,
        embed_dim=768,
        num_layers=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        num_blocks=16,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        **kwargs,
    ):
        super(AdaptiveFourierNeuralOperatorNet, self).__init__()
        self.img_size = inp_shape
        self.patch_size = patch_size
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed2D(
            img_size=self.img_size, patch_size=self.patch_size, in_chans=self.inp_chans, embed_dim=self.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        self.h = self.img_size[0] // self.patch_size[0]
        self.w = self.img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_blocks=num_blocks,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(num_layers)
            ]
        )

        self.head = nn.Linear(self.embed_dim, self.out_chans * self.patch_size[0] * self.patch_size[1], bias=False)

        with torch.no_grad():
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.compiler.disable(recursive=True)
    def no_weight_decay(self):
        r"""
        Parameters that should be excluded from weight decay.

        Position embeddings and class tokens encode location and identity
        rather than a learned transformation, so decaying them toward zero
        degrades the model instead of regularizing it.

        Returns
        -------
        set of str
            Names of the parameters to exclude.
        """
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        r"""
        Tokenize the input and run it through the AFNO blocks.

        Exposed separately from ``forward`` so callers can obtain the latent
        token grid without decoding it back to a field.

        Parameters
        ----------
        x : torch.Tensor
            Input field of shape ``(B, inp_chans, nlat, nlon)``.

        Returns
        -------
        torch.Tensor
            Token grid of shape ``(B, h, w, embed_dim)``, where ``h`` and ``w``
            are the patched grid dimensions.
        """
        B = x.shape[0]
        x = self.patch_embed(x).transpose(1, 2)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        r"""
        Map an input field to the predicted output field.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, inp_chans, nlat, nlon)``.

        Returns
        -------
        torch.Tensor
            Prediction of shape ``(B, out_chans, nlat, nlon)``.
        """
        x = self.forward_features(x)
        x = self.head(x)

        # rearrange
        b = x.shape[0]
        xv = x.view(b, self.h, self.w, self.patch_size[0], self.patch_size[1], -1)
        xvt = torch.permute(xv, (0, 5, 1, 3, 2, 4)).contiguous()
        x = xvt.view(b, -1, (self.h * self.patch_size[0]), (self.w * self.patch_size[1]))

        return x
