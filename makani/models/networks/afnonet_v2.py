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

# helpers
from makani.models.common import ComplexReLU, PatchEmbed2D, DropPath, MLP


@torch.compile
def compl_mul_add_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""
    Block-diagonal complex matmul over real-viewed tensors.

    Computes :math:`y_{b,k,o,x,y} = \sum_i a_{b,k,i,x,y} w_{k,i,o}` with complex
    arithmetic expressed on the real and imaginary components explicitly rather
    than on complex dtypes. Complex kernels are not always supported by the
    compiler backends, so this variant keeps the whole block compilable;
    :func:`compl_mul_add_fwd_c` is the complex-dtype equivalent.

    Parameters
    ----------
    a : torch.Tensor
        Input of shape ``(B, num_blocks, in_block, H, W, 2)``, where the
        trailing axis holds the real and imaginary parts.
    b : torch.Tensor
        Weight of shape ``(num_blocks, in_block, out_block, 2)``.

    Returns
    -------
    torch.Tensor
        Output of shape ``(B, num_blocks, out_block, H, W, 2)``.
    """
    tmp = torch.einsum("bkixys,kior->srbkoxy", a, b)
    res = torch.stack([tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1)
    return res


@torch.compile
def compl_mul_add_fwd_c(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""
    Block-diagonal complex matmul using native complex dtypes.

    Same contraction as :func:`compl_mul_add_fwd`, but performed with
    ``complex64`` tensors instead of explicit real/imaginary bookkeeping. Faster
    where the backend supports complex arithmetic; selected via the
    ``use_complex_kernels`` flag on :class:`AFNO2D`.

    Parameters
    ----------
    a : torch.Tensor
        Input of shape ``(B, num_blocks, in_block, H, W, 2)``, real-viewed.
    b : torch.Tensor
        Weight of shape ``(num_blocks, in_block, out_block, 2)``, real-viewed.

    Returns
    -------
    torch.Tensor
        Output of shape ``(B, num_blocks, out_block, H, W, 2)``, real-viewed.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bkixy,kio->bkoxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


class AFNO2D(nn.Module):
    r"""
    Adaptive Fourier Neural Operator token mixer, channels-first variant.

    Same idea as :class:`makani.models.networks.afnonet.AFNO2D` -- FFT, a
    block-diagonal complex MLP over the coefficients, soft-shrink, inverse FFT
    -- with three practical differences: it operates on ``(B, C, H, W)`` inputs
    instead of channels-last, it uses a complex activation
    (:class:`~makani.models.common.activations.ComplexReLU`) rather than a
    real one applied componentwise, and truncation is two-sided along the
    unhalved axis so both positive and negative frequencies are retained.

    Parameters
    ----------
    hidden_size : int
        Channel dimension. Must be divisible by ``num_blocks``.
    num_blocks : int, optional
        Number of independently mixed channel blocks, by default ``8``.
    sparsity_threshold : float, optional
        Soft-shrink threshold on the output coefficients, by default ``0.0``
        (no sparsification).
    hard_thresholding_fraction : float, optional
        Fraction of Fourier modes retained, by default ``1`` (keep all).
    hidden_size_factor : int, optional
        Width of the spectral MLP's hidden layer as a multiple of the block
        size, by default ``1``.
    use_complex_kernels : bool, optional
        Use native complex arithmetic for the contraction rather than explicit
        real/imaginary bookkeeping, by default ``False``.

    Raises
    ------
    ValueError
        If ``hidden_size`` is not divisible by ``num_blocks``.
    """

    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.0,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
        use_complex_kernels=False,
    ):
        super(AFNO2D, self).__init__()
        if hidden_size % num_blocks != 0:
            raise ValueError(f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}")

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.mult_handle = compl_mul_add_fwd_c if use_complex_kernels else compl_mul_add_fwd

        # new
        self.w1 = nn.Parameter(
            self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor, 2)
        )
        self.b1 = nn.Parameter(self.scale * torch.randn(1, self.num_blocks * self.block_size, 1, 1))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size, 2)
        )
        # self.b2 = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size, 1, 1, 2))

        # self.act = nn.ReLU()
        self.act = ComplexReLU(negative_slope=0.0, mode="cartesian")

    def forward(self, x):
        r"""
        Mix channels in Fourier space and add the input back as a residual.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, hidden_size, H, W)``.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``, with a learned bias and the
            input added back.
        """
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        total_modes_H = H // 2 + 1
        total_modes_W = W // 2 + 1
        kept_modes_H = int(total_modes_H * self.hard_thresholding_fraction)
        kept_modes_W = int(total_modes_W * self.hard_thresholding_fraction)

        x = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        x = x.view(B, self.num_blocks, self.block_size, H, W // 2 + 1)

        # do spectral conv
        x = torch.view_as_real(x)
        x_fft = torch.zeros(x.shape, device=x.device)

        if kept_modes_H == total_modes_H:
            oac = torch.view_as_complex(self.mult_handle(x[:, :, :, :, :kept_modes_W, :], self.w1))
            oa = torch.view_as_real(self.act(oac))
            x_fft[:, :, :, :, :kept_modes_W, :] = self.mult_handle(oa, self.w2)
        else:
            olc = torch.view_as_complex(self.mult_handle(x[:, :, :, :kept_modes_H, :kept_modes_W, :], self.w1))
            ohc = torch.view_as_complex(self.mult_handle(x[:, :, :, -kept_modes_H:, :kept_modes_W, :], self.w1))

            ol = torch.view_as_real(self.act(olc))
            oh = torch.view_as_real(self.act(ohc))

            x_fft[:, :, :, :kept_modes_H, :kept_modes_W, :] = self.mult_handle(ol, self.w2)
            x_fft[:, :, :, -kept_modes_H:, :kept_modes_W, :] = self.mult_handle(oh, self.w2)

        # finalize
        x = F.softshrink(x_fft, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, H, W // 2 + 1)
        # the inverse in H first, then the real inverse in W: the m=0 column of the mixed spectrum is not
        # Hermitian in general, and backends differ in how they treat its imaginary parts
        x = torch.fft.ifft(x, n=H, dim=-2, norm="ortho")
        x = irfft(x, n=W, dim=-1, norm="ortho")
        x = x.type(dtype)

        return x + self.b1 + bias


class Block(nn.Module):
    r"""
    AFNO block with configurable skip connections, channels-first variant.

    Spectral mixing via :class:`AFNO2D` followed by a channel MLP, each behind a
    normalization. Compared with
    :class:`makani.models.networks.afnonet.Block`, the skip around the filter is
    configurable -- it can be a learned 1x1 convolution, an identity, or absent
    entirely -- and ``nested_skip_fno`` controls whether the MLP's residual
    starts from the block input or from the filter output.

    Parameters
    ----------
    h : int
        Height of the token grid. Recorded for the normalization layers.
    w : int
        Width of the token grid.
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
        Normalization constructor, called with no arguments. By default
        :class:`torch.nn.LayerNorm`.
    num_blocks : int, optional
        Number of channel blocks in the AFNO mixer, by default ``8``.
    sparsity_threshold : float, optional
        Soft-shrink threshold in the AFNO mixer, by default ``0.01``.
    hard_thresholding_fraction : float, optional
        Fraction of Fourier modes retained, by default ``1.0``.
    use_complex_kernels : bool, optional
        Use native complex arithmetic in the mixer, by default ``True``.
    skip_fno : str, optional
        Skip connection around the filter: ``"linear"`` (default),
        ``"identity"``, or ``None`` for none. Unrecognized values are treated
        as ``None``.
    nested_skip_fno : bool, optional
        If ``True`` (the default), the MLP's residual is the block input, so
        the two skips nest. If ``False``, it is the filter output, so they are
        sequential.
    checkpointing_level : int, optional
        Gradient checkpointing aggressiveness; the MLP is checkpointed at level
        2 and above. By default ``0``.
    verbose : bool, optional
        Print which skip configuration was selected, by default ``True``.
    """

    def __init__(
        self,
        h,
        w,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        use_complex_kernels=True,
        skip_fno="linear",
        nested_skip_fno=True,
        checkpointing_level=0,
        verbose=True,
    ):
        super(Block, self).__init__()

        # norm layer
        self.norm1 = norm_layer()  # ((h,w))

        if skip_fno is None:
            if verbose:
                print("Using no skip connection around FNO.")

        elif skip_fno == "linear":
            # self.skip_layer = nn.Linear(dim, dim)
            self.skip_layer = nn.Conv2d(dim, dim, 1, 1)
            if verbose:
                print("Using Linear skip connection around FNO.")

        elif skip_fno == "identity":
            self.skip_layer = nn.Identity()
            if verbose:
                print("Using Identity skip connection around FNO.")

        else:
            if verbose:
                print(
                    f"Got skip_fno={skip_fno}, not using any skip around FNO -- use linear or identity to change this."
                )
        self.skip_fno = skip_fno

        self.nested_skip_fno = nested_skip_fno

        # filter
        self.filter = AFNO2D(
            dim, num_blocks, sparsity_threshold, hard_thresholding_fraction, use_complex_kernels=use_complex_kernels
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # norm layer
        self.norm2 = norm_layer()  # ((h,w))

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop_rate=drop,
            checkpointing=(checkpointing_level >= 2),
        )

    def forward(self, x):
        r"""
        Apply spectral mixing and the channel MLP with the configured skips.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, dim, h, w)``.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``.
        """
        residual = x

        x = self.norm1(x)
        x = self.filter(x)

        if self.skip_fno is not None:
            x = x + self.skip_layer(residual)
            if not self.nested_skip_fno:
                residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AdaptiveFourierNeuralOperatorNet(nn.Module):
    r"""
    Adaptive Fourier Neural Operator network, revised implementation.

    Same architecture as
    :class:`makani.models.networks.afnonet.AdaptiveFourierNeuralOperatorNet` --
    patch embedding, a stack of AFNO blocks, a linear decode head -- but working
    in channels-first layout with a configurable normalization layer,
    configurable skip connections around each filter, and gradient
    checkpointing.

    Parameters
    ----------
    inp_shape : (int, int), optional
        Input grid as ``(nlat, nlon)``, by default ``(720, 1440)``. Must be
        divisible by ``patch_size``.
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
        Number of channel blocks in each AFNO mixer, by default ``16``.
    sparsity_threshold : float, optional
        Soft-shrink threshold in the AFNO mixers, by default ``0.01``.
    normalization_layer : str, optional
        ``"instance_norm"`` (default) or ``"layer_norm"``.
    skip_fno : str, optional
        Skip connection around each filter: ``"linear"`` (default),
        ``"identity"``, or ``None``.
    nested_skip_fno : bool, optional
        Whether the block's two skips nest rather than run sequentially, by
        default ``True``.
    hard_thresholding_fraction : float, optional
        Fraction of Fourier modes retained, by default ``1.0``.
    checkpointing_level : int, optional
        Gradient checkpointing aggressiveness, by default ``0``.
    use_complex_kernels : bool, optional
        Use native complex arithmetic in the mixers, by default ``True``.
    verbose : bool, optional
        Print the skip configuration of each block, by default ``False``.
    **kwargs
        Ignored; present so model configs can pass extra keys.

    Raises
    ------
    ValueError
        If ``patch_size`` does not have two entries, if it does not divide the
        image dimensions evenly, or if ``normalization_layer`` is unsupported.
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
        normalization_layer="instance_norm",
        skip_fno="linear",
        nested_skip_fno=True,
        hard_thresholding_fraction=1.0,
        checkpointing_level=0,
        use_complex_kernels=True,
        verbose=False,
        **kwargs,
    ):
        super(AdaptiveFourierNeuralOperatorNet, self).__init__()
        self.img_size = inp_shape
        self.patch_size = patch_size
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        # some sanity checks
        if len(patch_size) != 2:
            raise ValueError(f"Expected patch_size to have two entries but got {patch_size} instead")
        if not ((self.img_size[0] % self.patch_size[0] == 0) and (self.img_size[1] % self.patch_size[1] == 0)):
            raise ValueError(
                f"the patch size {self.patch_size} does not divide the image dimensions {self.img_size} evenly."
            )

        self.patch_embed = PatchEmbed2D(
            img_size=self.img_size, patch_size=self.patch_size, in_chans=self.inp_chans, embed_dim=self.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, num_patches))
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        # compute the downscaled image size
        self.h = self.img_size[0] // self.patch_size[0]
        self.w = self.img_size[1] // self.patch_size[1]

        # pick norm layer
        if normalization_layer == "layer_norm":
            norm_layer = partial(nn.LayerNorm, normalized_shape=(self.h, self.w), eps=1e-6)
        elif normalization_layer == "instance_norm":
            norm_layer = partial(
                nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False
            )
        else:
            raise NotImplementedError(f"Error, normalization {normalization_layer} not implemented.")

        self.blocks = nn.ModuleList(
            [
                Block(
                    h=self.h,
                    w=self.w,
                    dim=self.embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_blocks=num_blocks,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                    use_complex_kernels=use_complex_kernels,
                    skip_fno=skip_fno,
                    nested_skip_fno=nested_skip_fno,
                    checkpointing_level=checkpointing_level,
                    verbose=verbose,
                )
                for i in range(num_layers)
            ]
        )

        # head
        self.head = nn.Conv2d(embed_dim, self.out_chans * self.patch_size[0] * self.patch_size[1], 1, bias=False)

        with torch.no_grad():
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            # nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.InstanceNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.compiler.disable(recursive=False)
    def no_weight_decay(self):
        r"""
        Parameters that should be excluded from weight decay.

        Position embeddings and class tokens encode location and identity
        rather than a learned transformation, so decaying them degrades the
        model instead of regularizing it.

        Returns
        -------
        set of str
            Names of the parameters to exclude.
        """
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        r"""
        Tokenize the input and run it through the AFNO blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input field of shape ``(B, inp_chans, nlat, nlon)``.

        Returns
        -------
        torch.Tensor
            Latent grid of shape ``(B, embed_dim, h, w)``, where ``h`` and ``w``
            are the patched grid dimensions.
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # reshape
        x = x.reshape(B, self.embed_dim, self.h, self.w)

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

        # new: B, C, H, W
        b = x.shape[0]
        xv = x.view(b, self.patch_size[0], self.patch_size[1], -1, self.h, self.w)
        xvt = torch.permute(xv, (0, 3, 4, 1, 5, 2)).contiguous()
        x = xvt.view(b, -1, (self.h * self.patch_size[0]), (self.w * self.patch_size[1]))

        return x
