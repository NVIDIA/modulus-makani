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
from torch.utils.checkpoint import checkpoint
from torch import amp

from functools import partial

# helpers
from makani.models.common import DropPath, MLP, EncoderDecoder

# import global convolution and non-linear spectral layers
from makani.models.common import SpectralConv, SpectralAttention

# get spectral transforms from torch_harmonics
import torch_harmonics as th
import torch_harmonics.distributed as thd

# wrap fft, to unify interface to spectral transforms
from makani.models.common import RealFFT2, InverseRealFFT2, GeometricInstanceNormS2
from makani.mpu.fft import DistributedRealFFT2, DistributedInverseRealFFT2
from makani.mpu.layers import DistributedMLP, DistributedEncoderDecoder

# more distributed stuff
from makani.utils import comm
from makani.utils.grids import get_spectral_transform

# layer normalization
from makani.mpu.layer_norm import DistributedInstanceNorm2d, DistributedLayerNorm, DistributedGeometricInstanceNormS2

# for annotation of models
from dataclasses import dataclass
from physicsnemo import ModelMetaData

from makani.models.physicsnemo_compat import module_from_torch


class SpectralFilterLayer(nn.Module):
    r"""
    Thin dispatcher selecting the spectral filter used inside a block.

    Exists so that :class:`NeuralOperatorBlock` can be written against one
    interface while the choice between a linear spectral convolution and a
    nonlinear spectral MLP stays a config option. Constructor arguments that do
    not apply to the selected filter are ignored.

    Parameters
    ----------
    forward_transform : torch.nn.Module
        Grid-to-spectral transform.
    inverse_transform : torch.nn.Module
        Spectral-to-grid transform.
    embed_dim : int
        Channel width; used for both input and output.
    filter_type : str, optional
        ``"linear"`` (default) selects
        :class:`~makani.models.common.spectral_convolution.SpectralConv`;
        ``"non-linear"`` selects
        :class:`~makani.models.common.spectral_convolution.SpectralAttention`.
    operator_type : str, optional
        Weight layout passed to the selected filter, by default ``"diagonal"``.
    hidden_size_factor : int, optional
        Hidden width multiplier for the nonlinear filter, by default ``1``.
        Ignored by the linear filter.
    rank : float, optional
        Accepted for config compatibility; currently unused.
    separable : bool, optional
        Depthwise (per-channel) filtering for the linear filter, by default
        ``False``. Ignored by the nonlinear filter.
    complex_activation : str, optional
        Complex activation mode for the nonlinear filter, by default ``"real"``.
    spectral_layers : int, optional
        Number of layers in the nonlinear filter's spectral MLP, by default ``1``.
    bias : bool, optional
        Whether the filter carries a bias, by default ``False``.
    drop_rate : float, optional
        Dropout probability inside the nonlinear filter, by default ``0.0``.
    gain : float, optional
        Scales the variance of the filter's output initialization, by default ``1.0``.

    Raises
    ------
    NotImplementedError
        If ``filter_type`` is neither ``"linear"`` nor ``"non-linear"``.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="diagonal",
        hidden_size_factor=1,
        rank=1.0,
        separable=False,
        complex_activation="real",
        spectral_layers=1,
        bias=False,
        drop_rate=0.0,
        gain=1.0,
    ):
        super(SpectralFilterLayer, self).__init__()

        if filter_type == "non-linear":
            self.filter = SpectralAttention(
                forward_transform,
                inverse_transform,
                embed_dim,
                embed_dim,
                operator_type=operator_type,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=bias,
                gain=gain,
            )

        elif filter_type == "linear":
            self.filter = SpectralConv(
                forward_transform,
                inverse_transform,
                embed_dim,
                embed_dim,
                operator_type=operator_type,
                separable=separable,
                bias=bias,
                gain=gain,
            )

        else:
            raise (NotImplementedError)

    def forward(self, x):
        r"""
        Apply the selected spectral filter.

        Parameters
        ----------
        x : torch.Tensor
            Input field of shape ``(B, embed_dim, nlat, nlon)``.

        Returns
        -------
        x : torch.Tensor
            Filtered field, on the inverse transform's grid.
        residual : torch.Tensor
            The input resampled onto the output grid; see
            :meth:`~makani.models.common.spectral_convolution.SpectralConv.forward`.
        """
        return self.filter(x)


class NeuralOperatorBlock(nn.Module):
    r"""
    One block of the SFNO processor: spectral filter, norms, skips, and MLP.

    The structural analogue of a transformer block, with the spectral filter
    playing the role of attention (global mixing across space) and the MLP
    mixing channels pointwise. Both stages sit behind their own normalization
    and skip connection:

    .. code-block:: text

        x -> filter -> norm0 -> [+ inner_skip(residual)] -> act
          -> mlp -> norm1 -> [+ outer_skip(residual)] -> [act]

    Skip weights are initialized with a gain that is halved for each active skip
    path, so the variance of the block's output stays roughly constant no matter
    which skips are enabled -- deep stacks then do not need retuning when the
    block configuration changes.

    Parameters
    ----------
    forward_transform : torch.nn.Module
        Grid-to-spectral transform for the filter.
    inverse_transform : torch.nn.Module
        Spectral-to-grid transform for the filter. May target a different
        resolution, which is how the model changes resolution between blocks.
    embed_dim : int
        Channel width of the block.
    filter_type : str, optional
        ``"linear"`` (default) or ``"non-linear"``; see :class:`SpectralFilterLayer`.
    operator_type : str, optional
        Weight layout for the spectral filter, by default ``"diagonal"``.
    mlp_ratio : float, optional
        Hidden width of the MLP as a multiple of ``embed_dim``, by default ``2.0``.
        Also used as the nonlinear filter's hidden size factor.
    mlp_drop_rate : float, optional
        Dropout probability inside the MLP, by default ``0.0``.
    path_drop_rate : float, optional
        Stochastic depth probability for the block, by default ``0.0``.
    act_layer : callable, optional
        Activation constructor, by default :class:`torch.nn.GELU`. Passing
        :class:`torch.nn.Identity` halves the initialization gain, since no
        activation-induced variance loss needs compensating.
    norm_layer : tuple of callable, optional
        Two normalization constructors, applied after the filter and after the
        MLP respectively. By default both are :class:`torch.nn.Identity`.
    rank : float, optional
        Accepted for config compatibility; currently unused.
    separable : bool, optional
        Depthwise filtering for the linear filter, by default ``False``.
    inner_skip : str, optional
        Skip around the filter: ``"linear"`` (default), ``"identity"``, or
        ``"none"``.
    outer_skip : str, optional
        Skip around the MLP: ``"linear"``, ``"identity"``, or ``"none"``. By
        default ``None``, which raises -- pass ``"none"`` to disable it.
    use_mlp : bool, optional
        Whether to include the channel MLP, by default ``False``. Automatically
        uses the distributed MLP when the ``matmul`` group is larger than one.
    comm_feature_name : str, optional
        Communicator group the distributed MLP shards over, by default ``"matmul"``.
    complex_activation : str, optional
        Complex activation mode for the nonlinear filter, by default ``"real"``.
    spectral_layers : int, optional
        Number of layers in the nonlinear filter's spectral MLP, by default ``1``.
    bias : bool, optional
        Whether the spectral filter carries a bias, by default ``False``.
    final_activation : bool, optional
        Apply an activation after the outer skip, by default ``False``.
    checkpointing_level : int, optional
        Gradient checkpointing aggressiveness; the MLP is checkpointed at level
        2 and above. By default ``0``.

    Raises
    ------
    ValueError
        If ``inner_skip`` or ``outer_skip`` is not one of the supported values.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="diagonal",
        mlp_ratio=2.0,
        mlp_drop_rate=0.0,
        path_drop_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=(nn.Identity, nn.Identity),
        rank=1.0,
        separable=False,
        inner_skip="linear",
        outer_skip=None,
        use_mlp=False,
        comm_feature_name="matmul",
        complex_activation="real",
        spectral_layers=1,
        bias=False,
        final_activation=False,
        checkpointing_level=0,
    ):
        super().__init__()

        # determine some shapes
        if comm.get_size("spatial") > 1:
            self.input_shape_loc = (
                forward_transform.lat_shapes[comm.get_rank("h")],
                forward_transform.lon_shapes[comm.get_rank("w")],
            )
            self.output_shape_loc = (
                inverse_transform.lat_shapes[comm.get_rank("h")],
                inverse_transform.lon_shapes[comm.get_rank("w")],
            )
        else:
            self.input_shape_loc = (forward_transform.nlat, forward_transform.nlon)
            self.output_shape_loc = (inverse_transform.nlat, inverse_transform.nlon)

        # norm layer
        self.norm0 = norm_layer[0]()

        if act_layer == nn.Identity:
            gain_factor = 1.0
        else:
            gain_factor = 2.0

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
            gain_factor /= 2.0
            nn.init.normal_(self.inner_skip.weight, std=math.sqrt(gain_factor / embed_dim))
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()
            gain_factor /= 2.0
        elif inner_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {inner_skip}")

        # convolution layer
        self.filter = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            embed_dim,
            filter_type,
            operator_type,
            hidden_size_factor=mlp_ratio,
            rank=rank,
            separable=separable,
            complex_activation=complex_activation,
            spectral_layers=spectral_layers,
            bias=bias,
            drop_rate=path_drop_rate,
            gain=gain_factor,
        )

        self.act_layer0 = act_layer()

        # norm layer
        self.norm1 = norm_layer[1]()

        if final_activation and act_layer != nn.Identity:
            gain_factor = 2.0
        else:
            gain_factor = 1.0

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
            gain_factor /= 2.0
            torch.nn.init.normal_(self.outer_skip.weight, std=math.sqrt(gain_factor / embed_dim))
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()
            gain_factor /= 2.0
        elif outer_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {outer_skip}")

        if use_mlp == True:
            MLPH = DistributedMLP if (comm.get_size("matmul") > 1) else MLP
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLPH(
                in_features=embed_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=mlp_drop_rate,
                drop_type="features",
                comm_name=comm_feature_name,
                checkpointing=(checkpointing_level >= 2),
                gain=gain_factor,
            )

        # dropout
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()

        if final_activation:
            self.act_layer1 = act_layer()

    def forward(self, x):
        r"""
        Apply the spectral filter, skips, norms and MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, embed_dim, nlat_in, nlon_in)`` matching the
            forward transform's grid.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, embed_dim, nlat_out, nlon_out)`` on the
            inverse transform's grid, which differs from the input shape when
            the block changes resolution.
        """

        x, residual = self.filter(x)

        x = self.norm0(x)

        if hasattr(self, "inner_skip"):
            x = x + self.inner_skip(residual)

        if hasattr(self, "act_layer0"):
            x = self.act_layer0(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.norm1(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            x = x + self.outer_skip(residual)

        if hasattr(self, "act_layer1"):
            x = self.act_layer1(x)

        return x


class SphericalFourierNeuralOperatorNet(nn.Module):
    r"""
    Spherical Fourier Neural Operator (SFNO).

    An encoder-processor-decoder network in which all spatial mixing happens in
    spherical harmonic space. The encoder lifts input variables to ``embed_dim``
    channels, a stack of :class:`NeuralOperatorBlock` layers evolves them, and
    the decoder projects back to ``out_chans``. Because the transforms are
    spherical rather than planar, the operator respects the geometry of the
    globe -- no pole artifacts and no seam at the dateline -- which is what
    makes the rollouts stable over long horizons.

    The first block downsamples from the input grid to an internal grid reduced
    by ``scale_factor``, the interior blocks operate at that resolution, and the
    last block maps back to the output grid. Spectral truncation is controlled
    by ``hard_thresholding_fraction`` or by ``max_modes`` directly.

    Parameters
    ----------
    spectral_transform : str, optional
        ``"sht"`` (default) for spherical harmonic transforms, or ``"fft"`` to
        treat the domain as a periodic plane.
    model_grid_type : str, optional
        Grid the input data lives on, by default ``"equiangular"``.
    sht_grid_type : str, optional
        Grid used internally by the transforms, by default ``"legendre-gauss"``,
        whose quadrature is exact for band-limited fields.
    filter_type : str, optional
        ``"linear"`` (default) or ``"non-linear"`` spectral filter.
    operator_type : str, optional
        Weight layout of the spectral filter, by default ``"dhconv"``
        (rotationally equivariant).
    inp_shape : (int, int), optional
        Input grid as ``(nlat, nlon)``, by default ``(721, 1440)``.
    out_shape : (int, int), optional
        Output grid as ``(nlat, nlon)``, by default ``(721, 1440)``.
    scale_factor : int, optional
        Factor by which the internal grid is coarsened relative to the input,
        by default ``8``.
    inp_chans : int, optional
        Number of input channels, by default ``2``.
    out_chans : int, optional
        Number of output channels, by default ``2``.
    embed_dim : int, optional
        Latent channel width, by default ``32``.
    num_layers : int, optional
        Number of neural operator blocks, by default ``4``.
    use_mlp : bool, optional
        Include the channel MLP in each block, by default ``True``.
    mlp_ratio : float, optional
        Hidden width of the block MLPs as a multiple of ``embed_dim``, by
        default ``2.0``.
    encoder_ratio : int, optional
        Hidden width of the encoder as a multiple of ``embed_dim``, by default ``1``.
    decoder_ratio : int, optional
        Hidden width of the decoder as a multiple of ``embed_dim``, by default ``1``.
    activation_function : str, optional
        One of ``"relu"``, ``"gelu"`` (default) or ``"silu"``.
    encoder_layers : int, optional
        Number of hidden layers in the encoder and decoder, by default ``1``.
    pos_embed : str, optional
        Position embedding type, by default ``"none"``.
    pos_drop_rate : float, optional
        Dropout applied after the position embedding, by default ``0.0``.
    path_drop_rate : float, optional
        Stochastic depth probability, by default ``0.0``.
    mlp_drop_rate : float, optional
        Dropout inside the block MLPs, by default ``0.0``.
    normalization_layer : str, optional
        Normalization used inside the blocks, by default ``"instance_norm"``.
    max_modes : (int, int), optional
        Explicit spectral truncation as ``(lmax, mmax)``. Overrides
        ``hard_thresholding_fraction`` when given.
    hard_thresholding_fraction : float, optional
        Fraction of the available modes retained, by default ``1.0``.
    big_skip : bool, optional
        Add a skip connection from the network input to the decoder input, by
        default ``True``. Lets the network predict a correction rather than the
        full field.
    rank : float, optional
        Accepted for config compatibility; currently unused.
    separable : bool, optional
        Depthwise spectral filtering, by default ``False``.
    complex_activation : str, optional
        Complex activation mode for nonlinear filters, by default ``"real"``.
    spectral_layers : int, optional
        Number of layers in each nonlinear filter's spectral MLP, by default ``3``.
    bias : bool, optional
        Whether the spectral filters carry a bias, by default ``False``.
    checkpointing_level : int, optional
        Gradient checkpointing aggressiveness (higher recomputes more), by
        default ``0``.
    **kwargs
        Ignored; present so model configs can pass extra keys.

    Raises
    ------
    ValueError
        If ``activation_function`` is not one of the supported names.

    References
    ----------
    [1] Bonev, B.; Kurth, T.; Hundt, C.; Pathak, J.; Baust, M.; Kashinath, K.;
    Anandkumar, A.; Spherical Fourier Neural Operators: Learning Stable Dynamics
    on the Sphere; ICML 2023.
    """

    def __init__(
        self,
        spectral_transform="sht",
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        filter_type="linear",
        operator_type="dhconv",
        inp_shape=(721, 1440),
        out_shape=(721, 1440),
        scale_factor=8,
        inp_chans=2,
        out_chans=2,
        embed_dim=32,
        num_layers=4,
        use_mlp=True,
        mlp_ratio=2.0,
        encoder_ratio=1,
        decoder_ratio=1,
        activation_function="gelu",
        encoder_layers=1,
        pos_embed="none",
        pos_drop_rate=0.0,
        path_drop_rate=0.0,
        mlp_drop_rate=0.0,
        normalization_layer="instance_norm",
        max_modes=None,
        hard_thresholding_fraction=1.0,
        big_skip=True,
        rank=1.0,
        separable=False,
        complex_activation="real",
        spectral_layers=3,
        bias=False,
        checkpointing_level=0,
        **kwargs,
    ):
        super(SphericalFourierNeuralOperatorNet, self).__init__()

        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.big_skip = big_skip
        self.checkpointing_level = checkpointing_level

        # compute the downscaled image size
        self.h = int(self.inp_shape[0] // scale_factor)
        self.w = int(self.inp_shape[1] // scale_factor)

        # initialize spectral transforms
        self._init_spectral_transforms(
            spectral_transform, model_grid_type, sht_grid_type, hard_thresholding_fraction, max_modes
        )

        # determine activation function
        if activation_function == "relu":
            activation_function = nn.ReLU
        elif activation_function == "gelu":
            activation_function = nn.GELU
        elif activation_function == "silu":
            activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # set up encoder
        if comm.get_size("matmul") > 1:
            self.encoder = DistributedEncoderDecoder(
                num_layers=encoder_layers,
                input_dim=self.inp_chans,
                output_dim=self.embed_dim,
                hidden_dim=int(encoder_ratio * self.embed_dim),
                act_layer=activation_function,
                input_format="nchw",
                comm_name="matmul",
            )
        else:
            self.encoder = EncoderDecoder(
                num_layers=encoder_layers,
                input_dim=self.inp_chans,
                output_dim=self.embed_dim,
                hidden_dim=int(encoder_ratio * self.embed_dim),
                act_layer=activation_function,
                input_format="nchw",
            )

        # dropout
        self.pos_drop = nn.Dropout(p=pos_drop_rate) if pos_drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, num_layers)]

        # pick norm layer
        if normalization_layer == "layer_norm":
            norm_layer_inp = partial(
                DistributedLayerNorm, normalized_shape=(embed_dim), elementwise_affine=True, eps=1e-6
            )
            norm_layer_out = norm_layer_mid = norm_layer_inp
        elif normalization_layer == "instance_norm":
            if comm.get_size("spatial") > 1:
                norm_layer_inp = partial(DistributedInstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True)
            else:
                norm_layer_inp = partial(
                    nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False
                )
            norm_layer_out = norm_layer_mid = norm_layer_inp
        elif normalization_layer == "instance_norm_s2":
            norm_layer_handle = (
                DistributedGeometricInstanceNormS2 if comm.get_size("spatial") > 1 else GeometricInstanceNormS2
            )
            norm_layer_inp = partial(
                norm_layer_handle,
                img_shape=(self.h, self.w),
                crop_shape=(self.h, self.w),
                crop_offset=(0, 0),
                grid_type=model_grid_type,
                num_features=embed_dim,
                eps=1e-6,
                affine=True,
            )
            norm_layer_mid = norm_layer_inp
            norm_layer_out = partial(
                norm_layer_handle,
                img_shape=out_shape,
                crop_shape=out_shape,
                crop_offset=(0, 0),
                grid_type=model_grid_type,
                num_features=embed_dim,
                eps=1e-6,
                affine=True,
            )
            # norm_layer_mid = partial(norm_layer_handle,
            #                         img_shape=(self.h, self.w), crop_shape=(self.h, self.w), crop_offset=(0,0),
            #                         grid_type=sht_grid_type, pole_mask=0,
        #                         num_features=embed_dim, eps=1e-6, affine=True)
        elif normalization_layer == "none":
            norm_layer_out = norm_layer_mid = norm_layer_inp = nn.Identity
        else:
            raise NotImplementedError(f"Error, normalization {normalization_layer} not implemented.")

        # FNO blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_layers):
            first_layer = i == 0
            last_layer = i == num_layers - 1

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "none"
            outer_skip = "linear"

            if first_layer:
                norm_layer = (norm_layer_inp, norm_layer_mid)
            elif last_layer:
                norm_layer = (norm_layer_out, norm_layer_out)
            else:
                norm_layer = (norm_layer_mid, norm_layer_mid)

            block = NeuralOperatorBlock(
                forward_transform,
                inverse_transform,
                embed_dim,
                filter_type=filter_type,
                operator_type=operator_type,
                mlp_ratio=mlp_ratio,
                mlp_drop_rate=mlp_drop_rate,
                path_drop_rate=dpr[i],
                act_layer=activation_function,
                norm_layer=norm_layer,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=use_mlp,
                comm_feature_name="matmul",
                rank=rank,
                separable=separable,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                bias=bias,
                checkpointing_level=checkpointing_level,
            )

            self.blocks.append(block)

        # decoder takes the output of FNO blocks and the residual from the big skip connection
        if comm.get_size("matmul") > 1:
            self.decoder = DistributedEncoderDecoder(
                num_layers=encoder_layers,
                input_dim=embed_dim,
                output_dim=self.out_chans,
                hidden_dim=int(decoder_ratio * embed_dim),
                act_layer=activation_function,
                gain=0.5 if self.big_skip else 1.0,
                comm_name="matmul",
                input_format="nchw",
            )

        else:
            self.decoder = EncoderDecoder(
                num_layers=encoder_layers,
                input_dim=embed_dim,
                output_dim=self.out_chans,
                hidden_dim=int(decoder_ratio * embed_dim),
                act_layer=activation_function,
                gain=0.5 if self.big_skip else 1.0,
                input_format="nchw",
            )

        # output transform
        if self.big_skip:
            self.residual_transform = nn.Conv2d(self.inp_chans, self.out_chans, 1, bias=False)
            self.residual_transform.weight.is_shared_mp = ["spatial"]
            self.residual_transform.weight.sharded_dims_mp = [None, None, None, None]
            scale = math.sqrt(0.5 / self.inp_chans)
            nn.init.normal_(self.residual_transform.weight, mean=0.0, std=scale)

        # learned position embedding
        if pos_embed == "direct":
            # currently using deliberately a differently shape position embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.inp_shape_loc[0], self.inp_shape_loc[1]))
            # information about how tensors are shared / sharded across ranks
            self.pos_embed.is_shared_mp = []  # no reduction required since pos_embed is already serial
            self.pos_embed.sharded_dims_mp = [None, None, "h", "w"]
            self.pos_embed.type = "direct"
            with torch.no_grad():
                nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed == "frequency":
            if comm.get_size("spatial") > 1:
                lmax_loc = self.itrans_up.l_shapes[comm.get_rank("h")]
                mmax_loc = self.itrans_up.m_shapes[comm.get_rank("w")]
            else:
                lmax_loc = self.itrans_up.lmax
                mmax_loc = self.itrans_up.mmax

            rcoeffs = nn.Parameter(torch.tril(torch.randn(1, embed_dim, lmax_loc, mmax_loc), diagonal=0))
            ccoeffs = nn.Parameter(torch.tril(torch.randn(1, embed_dim, lmax_loc, mmax_loc - 1), diagonal=-1))
            with torch.no_grad():
                nn.init.trunc_normal_(rcoeffs, std=0.02)
                nn.init.trunc_normal_(ccoeffs, std=0.02)
            self.pos_embed = nn.ParameterList([rcoeffs, ccoeffs])
            self.pos_embed.type = "frequency"
            self.pos_embed.is_shared_mp = []
            self.pos_embed.sharded_dims_mp = [None, None, "h", "w"]

        elif pos_embed == "none" or pos_embed == "None" or pos_embed == None:
            pass
        else:
            raise ValueError("Unknown position embedding type")

    @torch.compiler.disable(recursive=True)
    def _init_spectral_transforms(
        self,
        spectral_transform="sht",
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        hard_thresholding_fraction=1.0,
        max_modes=None,
    ):
        """
        Initialize the spectral transforms based on the maximum number of modes to keep. Handles the computation
        of local image shapes and domain parallelism, based on the
        """

        if max_modes is not None:
            modes_lat, modes_lon = max_modes
        else:
            modes_lat = int(self.h * hard_thresholding_fraction)
            modes_lon = int((self.w // 2 + 1) * hard_thresholding_fraction)

        # check for distributed
        if (comm.get_size("spatial") > 1) and (not thd.is_initialized()):
            polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
            azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
            thd.init(polar_group, azimuth_group)

        # prepare the spectral transforms
        if spectral_transform == "sht":
            distributed = comm.get_size("spatial") > 1

            # set up
            self.trans_down = get_spectral_transform("sht", *self.inp_shape, model_grid_type, lmax=modes_lat, mmax=modes_lon, distributed=distributed).float()
            self.itrans_up = get_spectral_transform("isht", *self.out_shape, model_grid_type, lmax=modes_lat, mmax=modes_lon, distributed=distributed).float()
            self.trans = get_spectral_transform("sht", self.h, self.w, sht_grid_type, lmax=modes_lat, mmax=modes_lon, distributed=distributed).float()
            self.itrans = get_spectral_transform("isht", self.h, self.w, sht_grid_type, lmax=modes_lat, mmax=modes_lon, distributed=distributed).float()

        elif spectral_transform == "fft":
            fft_handle = RealFFT2
            ifft_handle = InverseRealFFT2

            if comm.get_size("spatial") > 1:
                fft_handle = DistributedRealFFT2
                ifft_handle = DistributedInverseRealFFT2

            self.trans_down = fft_handle(*self.inp_shape, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans_up = ifft_handle(*self.out_shape, lmax=modes_lat, mmax=modes_lon).float()
            self.trans = fft_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans = ifft_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
        else:
            raise (ValueError("Unknown spectral transform"))

        # use the SHT/FFT to compute the local, downscaled grid dimensions
        if comm.get_size("spatial") > 1:
            self.inp_shape_loc = (
                self.trans_down.lat_shapes[comm.get_rank("h")],
                self.trans_down.lon_shapes[comm.get_rank("w")],
            )
            self.out_shape_loc = (
                self.itrans_up.lat_shapes[comm.get_rank("h")],
                self.itrans_up.lon_shapes[comm.get_rank("w")],
            )
            self.h_loc = self.itrans.lat_shapes[comm.get_rank("h")]
            self.w_loc = self.itrans.lon_shapes[comm.get_rank("w")]
        else:
            self.inp_shape_loc = (self.trans_down.nlat, self.trans_down.nlon)
            self.out_shape_loc = (self.itrans_up.nlat, self.itrans_up.nlon)
            self.h_loc = self.itrans.nlat
            self.w_loc = self.itrans.nlon

    @torch.compiler.disable(recursive=True)
    def no_weight_decay(self):
        r"""
        Parameters that should be excluded from weight decay.

        Position embeddings and class tokens encode absolute location and
        identity rather than a learned transformation, so decaying them toward
        zero degrades the model instead of regularizing it. Optimizer setup
        calls this to build its no-decay parameter group.

        Returns
        -------
        set of str
            Names of the parameters to exclude.
        """
        return {"pos_embed", "cls_token"}

    def _forward_features(self, x):
        for blk in self.blocks:
            if self.checkpointing_level >= 3:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        return x

    def forward(self, x):
        r"""
        Map an input field to the predicted output field.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, inp_chans, nlat, nlon)`` on ``inp_shape``.

        Returns
        -------
        torch.Tensor
            Prediction of shape ``(B, out_chans, nlat, nlon)`` on ``out_shape``.
        """
        # save big skip
        if self.big_skip:
            # if output shape differs, use the spectral transforms to change resolution
            if self.out_shape != self.inp_shape:
                xtype = x.dtype
                # only take the predicted channels as residual
                residual = x.to(torch.float32)
                with amp.autocast(device_type=residual.device.type, enabled=False):
                    residual = self.trans_down(residual)
                    residual = residual.contiguous()
                    residual = self.itrans_up(residual)
                residual = residual.to(dtype=xtype)
            else:
                # only take the predicted channels
                residual = x

        # the column/row fork-join encoder consumes replicated (full-channel) input,
        # so no channel scatter is needed at the model boundary

        if self.checkpointing_level >= 1:
            x = checkpoint(self.encoder, x, use_reentrant=False)
        else:
            x = self.encoder(x)

        if hasattr(self, "pos_embed"):
            if self.pos_embed.type == "frequency":
                pos_embed = torch.stack(
                    [self.pos_embed[0], nn.functional.pad(self.pos_embed[1], (1, 0), "constant", 0)], dim=-1
                )
                with amp.autocast(device_type="cuda", enabled=False):
                    pos_embed = self.itrans_up(torch.view_as_complex(pos_embed))
            else:
                pos_embed = self.pos_embed

            # add pos embed (cast to activation dtype so we don't promote x to fp32)
            x = x + pos_embed.to(dtype=x.dtype)

        # maybe clean the padding just in case
        x = self.pos_drop(x)

        # do the feature extraction
        x = self._forward_features(x)

        if self.checkpointing_level >= 1:
            x = checkpoint(self.decoder, x, use_reentrant=False)
        else:
            x = self.decoder(x)

        # the row-parallel decoder all-reduces to a replicated (full-channel)
        # output, so no channel gather is needed at the model boundary

        if self.big_skip:
            x = x + self.residual_transform(residual)

        return x


# this part exposes the model to modulus by constructing modulus Modules
@dataclass
class SphericalFourierNeuralOperatorNetMetaData(ModelMetaData):
    r"""
    PhysicsNeMo metadata for :class:`SphericalFourierNeuralOperatorNet`.

    Declares which execution features the model supports, so PhysicsNeMo knows
    what it may enable when wrapping it. Consumed by
    :func:`~makani.models.physicsnemo_compat.module_from_torch`, which produces
    the ``SFNO`` module exported from this file.

    Attributes
    ----------
    jit : bool
        TorchScript tracing is not supported. This says nothing about
        ``torch.compile``, which these flags do not govern.
    cuda_graphs : bool
        CUDA graph capture is not supported.
    amp_cpu : bool
        Mixed precision on CPU is not supported.
    amp_gpu : bool
        Mixed precision on GPU is supported.

    Notes
    -----
    The model name is deliberately not set here. ``ModelMetaData.name`` is
    deprecated in PhysicsNeMo 2.x, so it is passed to ``module_from_torch``
    instead and applied only where it still has an effect.
    """

    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True


SFNO = module_from_torch(
    SphericalFourierNeuralOperatorNet,
    SphericalFourierNeuralOperatorNetMetaData(),
    name="SFNO",
)


class FourierNeuralOperatorNet(SphericalFourierNeuralOperatorNet):
    r"""
    Planar Fourier Neural Operator (FNO).

    :class:`SphericalFourierNeuralOperatorNet` with ``spectral_transform="fft"``
    forced, so mixing happens over Fourier modes on a doubly periodic plane
    rather than over spherical harmonics. Useful as a baseline and for domains
    that genuinely are periodic; on global data it treats the poles and the
    dateline as ordinary interior points, which the spherical variant does not.

    Parameters
    ----------
    *args
        Forwarded to :class:`SphericalFourierNeuralOperatorNet`.
    **kwargs
        Forwarded to :class:`SphericalFourierNeuralOperatorNet`. Passing
        ``spectral_transform`` raises, since it is set here.

    See Also
    --------
    SphericalFourierNeuralOperatorNet : the base implementation and full
        parameter documentation.
    """

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, spectral_transform="fft", **kwargs)


@dataclass
class FourierNeuralOperatorNetMetaData(ModelMetaData):
    r"""
    PhysicsNeMo metadata for :class:`FourierNeuralOperatorNet`.

    Same supported-feature declaration as
    :class:`SphericalFourierNeuralOperatorNetMetaData`, under the name ``"FNO"``.
    Consumed by :func:`~makani.models.physicsnemo_compat.module_from_torch` to
    produce the ``FNO`` module exported from this file.

    Attributes
    ----------
    jit : bool
        TorchScript tracing is not supported. This says nothing about
        ``torch.compile``, which these flags do not govern.
    cuda_graphs : bool
        CUDA graph capture is not supported.
    amp_cpu : bool
        Mixed precision on CPU is not supported.
    amp_gpu : bool
        Mixed precision on GPU is supported.

    Notes
    -----
    The model name is deliberately not set here; see
    :class:`SphericalFourierNeuralOperatorNetMetaData`.
    """

    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True


FNO = module_from_torch(
    FourierNeuralOperatorNet,
    FourierNeuralOperatorNetMetaData(),
    name="FNO",
)
