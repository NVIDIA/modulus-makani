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
import torch.amp as amp
from torch.utils.checkpoint import checkpoint

from functools import partial

# helpers
from makani.models.common import DropPath, LayerScale, MLP, EncoderDecoder, SpectralConv
from makani.utils.features import get_water_channels

# get spectral transforms and spherical convolutions from torch_harmonics
import torch_harmonics as th
import torch_harmonics.distributed as thd

# get pre-formulated layers
from makani.models.common import GeometricInstanceNormS2
from makani.mpu.layers import DistributedMLP

# more distributed stuff
from makani.utils import comm
from makani.utils.grids import get_spectral_transform, get_disco_conv, get_resample

# layer normalization
from makani.mpu.layer_norm import DistributedInstanceNorm2d, DistributedLayerNorm, DistributedGeometricInstanceNormS2


# heuristic for finding theta_cutoff
def _compute_cutoff_radius(nlat, kernel_shape, basis_type):
    theta_cutoff_factor = {"piecewise linear": 0.5, "morlet": 0.5, "harmonic": 0.5, "zernike": math.sqrt(2.0)}

    return (kernel_shape[0] + 1) * theta_cutoff_factor[basis_type] * math.pi / float(nlat - 1)


class DiscreteContinuousEncoder(nn.Module):
    r"""
    Encoder built on a discrete-continuous convolution on the sphere.

    Lifts input variables into the model's latent width while resampling from
    the data grid onto the (typically coarser) model grid. Unlike an ordinary
    convolution, a discrete-continuous convolution is defined by a *continuous*
    kernel evaluated at the actual angular positions of the grid points, so
    input and output grids need not agree and the operator remains a genuine
    spherical convolution rather than an approximation on a lat-lon array.

    The kernel cutoff radius is derived from the input resolution and kernel
    shape, so the receptive field stays consistent across resolutions.

    Parameters
    ----------
    inp_shape : (int, int), optional
        Input grid as ``(nlat, nlon)``, by default ``(721, 1440)``.
    out_shape : (int, int), optional
        Output grid as ``(nlat, nlon)``, by default ``(480, 960)``.
    grid_in : str, optional
        Input grid type, by default ``"equiangular"``.
    grid_out : str, optional
        Output grid type, by default ``"equiangular"``.
    inp_chans : int, optional
        Number of input channels, by default ``2``.
    out_chans : int, optional
        Number of output channels, by default ``2``.
    kernel_shape : (int, int), optional
        Shape of the continuous kernel, by default ``(3, 3)``.
    basis_type : str, optional
        Basis the kernel is expanded in, by default ``"harmonic"``.
    basis_norm_mode : str, optional
        Basis normalization mode, by default ``"mean"``.
    use_mlp : bool, optional
        Follow the convolution with an activation and a pointwise MLP, by
        default ``False``. When enabled the convolution weights are scaled by
        :math:`\sqrt{2}` to compensate for the variance the activation removes.
    mlp_ratio : float, optional
        Hidden width of that MLP as a multiple of ``out_chans``, by default ``2.0``.
    activation_function : callable, optional
        Activation constructor, by default :class:`torch.nn.GELU`.
    groups : int, optional
        Number of convolution groups, by default ``1``.
    bias : bool, optional
        Whether the convolution carries a bias, by default ``False``.

    See Also
    --------
    DiscreteContinuousDecoder : the corresponding decoder.
    """

    def __init__(
        self,
        inp_shape=(721, 1440),
        out_shape=(480, 960),
        grid_in="equiangular",
        grid_out="equiangular",
        inp_chans=2,
        out_chans=2,
        kernel_shape=(3, 3),
        basis_type="harmonic",
        basis_norm_mode="mean",
        use_mlp=False,
        mlp_ratio=2.0,
        activation_function=nn.GELU,
        groups=1,
        bias=False,
    ):
        super().__init__()

        # heuristic for finding theta_cutoff
        theta_cutoff = _compute_cutoff_radius(nlat=inp_shape[0], kernel_shape=kernel_shape, basis_type=basis_type)

        # set up local convolution
        self.conv = get_disco_conv(
            inp_chans,
            out_chans,
            distributed=comm.get_size("spatial") > 1,
            in_shape=inp_shape,
            out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            grid_in=grid_in,
            grid_out=grid_out,
            groups=groups,
            bias=bias,
            theta_cutoff=theta_cutoff,
        )
        if comm.get_size("spatial") > 1:
            self.conv.weight.is_shared_mp = ["spatial"]
            self.conv.weight.sharded_dims_mp = [None, None, None]
            if self.conv.bias is not None:
                self.conv.bias.is_shared_mp = ["model"]
                self.conv.bias.sharded_dims_mp = [None]

        if use_mlp:
            with torch.no_grad():
                self.conv.weight *= math.sqrt(2.0)

            self.act = activation_function()

            self.mlp = EncoderDecoder(
                num_layers=1,
                input_dim=out_chans,
                output_dim=out_chans,
                hidden_dim=int(mlp_ratio * out_chans),
                act_layer=activation_function,
                input_format="nchw",
            )

    def forward(self, x):
        r"""
        Encode the input field onto the model grid.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, inp_chans, *inp_shape)``.

        Returns
        -------
        torch.Tensor
            Encoded field of shape ``(B, out_chans, *out_shape)``.
        """

        # perform conv
        x = self.conv(x)

        if hasattr(self, "act"):
            x = self.act(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        return x


class DiscreteContinuousDecoder(nn.Module):
    r"""
    Decoder built on a discrete-continuous convolution on the sphere.

    Mirror of :class:`DiscreteContinuousEncoder`: projects latent features back
    to output variables while resampling from the model grid onto the (typically
    finer) data grid. The upsampling step is applied before the convolution and
    runs in fp32 with autocast disabled, since it is a long accumulation whose
    accuracy the output depends on directly.

    Parameters
    ----------
    inp_shape : (int, int), optional
        Input grid as ``(nlat, nlon)``, by default ``(480, 960)``.
    out_shape : (int, int), optional
        Output grid as ``(nlat, nlon)``, by default ``(721, 1440)``.
    grid_in : str, optional
        Input grid type, by default ``"equiangular"``.
    grid_out : str, optional
        Output grid type, by default ``"equiangular"``.
    inp_chans : int, optional
        Number of input channels, by default ``2``.
    out_chans : int, optional
        Number of output channels, by default ``2``.
    kernel_shape : (int, int), optional
        Shape of the continuous kernel, by default ``(3, 3)``.
    basis_type : str, optional
        Basis the kernel is expanded in, by default ``"harmonic"``.
    basis_norm_mode : str, optional
        Basis normalization mode, by default ``"mean"``.
    use_mlp : bool, optional
        Precede the convolution with a pointwise MLP and activation, by
        default ``False``.
    mlp_ratio : float, optional
        Hidden width of that MLP as a multiple of ``inp_chans``, by default ``2.0``.
    activation_function : callable, optional
        Activation constructor, by default :class:`torch.nn.GELU`.
    groups : int, optional
        Number of convolution groups, by default ``1``.
    bias : bool, optional
        Whether the convolution carries a bias, by default ``False``.
    upsample_sht : bool, optional
        Upsample spectrally via an SHT rather than by interpolation, by default
        ``False``. Spectral upsampling is exact for band-limited fields but
        costs more.

    See Also
    --------
    DiscreteContinuousEncoder : the corresponding encoder.
    """

    def __init__(
        self,
        inp_shape=(480, 960),
        out_shape=(721, 1440),
        grid_in="equiangular",
        grid_out="equiangular",
        inp_chans=2,
        out_chans=2,
        kernel_shape=(3, 3),
        basis_type="harmonic",
        basis_norm_mode="mean",
        use_mlp=False,
        mlp_ratio=2.0,
        activation_function=nn.GELU,
        groups=1,
        bias=False,
        upsample_sht=False,
    ):
        super().__init__()

        if use_mlp:
            self.mlp = EncoderDecoder(
                num_layers=1,
                input_dim=inp_chans,
                output_dim=inp_chans,
                hidden_dim=int(mlp_ratio * inp_chans),
                act_layer=activation_function,
                input_format="nchw",
                gain=2.0,
            )

            self.act = activation_function()

        # init distributed torch-harmonics if needed
        if comm.get_size("spatial") > 1:
            polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
            azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
            thd.init(polar_group, azimuth_group)

        # spatial parallelism in the SHT
        if upsample_sht:
            # set up sht for upsampling

            # set upsampling module
            self.sht = get_spectral_transform("sht", *inp_shape, grid_in, distributed=comm.get_size("spatial") > 1).float()
            self.isht = get_spectral_transform("isht", *out_shape, grid_out, lmax=self.sht.lmax, mmax=self.sht.mmax, distributed=comm.get_size("spatial") > 1).float()
            self.upsample = nn.Sequential(self.sht, self.isht)
        else:

            self.upsample = get_resample(inp_shape, out_shape, grid_in, grid_out, mode="bilinear", distributed=comm.get_size("spatial") > 1)

        # heuristic for finding theta_cutoff
        # nto entirely clear if out or in shape should be used here with a non-conv method for upsampling
        theta_cutoff = _compute_cutoff_radius(nlat=out_shape[0], kernel_shape=kernel_shape, basis_type=basis_type)

        # set up DISCO convolution
        self.conv = get_disco_conv(
            inp_chans,
            out_chans,
            distributed=comm.get_size("spatial") > 1,
            in_shape=out_shape,
            out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            grid_in=grid_out,
            grid_out=grid_out,
            groups=groups,
            bias=False,
            theta_cutoff=theta_cutoff,
        )
        if comm.get_size("spatial") > 1:
            self.conv.weight.is_shared_mp = ["spatial"]
            self.conv.weight.sharded_dims_mp = [None, None, None]
            if self.conv.bias is not None:
                self.conv.bias.is_shared_mp = ["model"]
                self.conv.bias.sharded_dims_mp = [None]

    def forward(self, x):
        r"""
        Decode latent features onto the output grid.

        Parameters
        ----------
        x : torch.Tensor
            Latent field of shape ``(B, inp_chans, *inp_shape)``.

        Returns
        -------
        torch.Tensor
            Output field of shape ``(B, out_chans, *out_shape)``, cast back to
            the dtype of ``x``.
        """
        dtype = x.dtype

        if hasattr(self, "act"):
            x = self.act(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        with amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            x = self.upsample(x)
            x = self.conv(x)
            x = x.to(dtype=dtype)

        return x


class NeuralOperatorBlock(nn.Module):
    r"""
    Processor block mixing spatially via either a local or a global operator.

    The building block of :class:`SphericalNeuralOperatorNet`. Unlike the SFNO
    block, which always mixes globally in spectral space, this one lets
    ``conv_type`` choose between a discrete-continuous convolution with a
    bounded angular footprint (``"local"``) and a spectral convolution acting on
    all modes at once (``"global"``). Stacking both kinds lets a model resolve
    fine local structure and long-range teleconnections without paying global
    cost at every layer.

    Parameters
    ----------
    forward_transform : torch.nn.Module
        Grid-to-spectral transform; also defines the input grid.
    inverse_transform : torch.nn.Module
        Spectral-to-grid transform; also defines the output grid, which may
        differ in resolution.
    inp_chans : int
        Number of input channels.
    out_chans : int
        Number of output channels.
    conv_type : str, optional
        ``"local"`` (default) for a discrete-continuous convolution,
        ``"global"`` for a spectral convolution.
    mlp_ratio : float, optional
        Hidden width of the channel MLP as a multiple of the channel count, by
        default ``2.0``.
    mlp_drop_rate : float, optional
        Dropout probability inside the MLP, by default ``0.0``.
    path_drop_rate : float, optional
        Stochastic depth probability, by default ``0.0``.
    act_layer : callable, optional
        Activation constructor, by default :class:`torch.nn.GELU`.
    norm_layer : callable, optional
        Normalization constructor, by default :class:`torch.nn.Identity`.
    num_groups : int, optional
        Number of channel groups in the convolution, by default ``1``.
    skip : str, optional
        Skip connection type, by default ``"identity"``.
    layer_scale : bool, optional
        Apply a learned per-channel scale to the branch output, by default
        ``True``. Initialized small so the block starts near the identity.
    use_mlp : bool, optional
        Include the channel MLP, by default ``False``.
    kernel_shape : (int, int), optional
        Kernel shape for the local convolution, by default ``(3, 3)``.
    basis_type : str, optional
        Basis for the local convolution kernel, by default ``"harmonic"``.
    basis_norm_mode : str, optional
        Basis normalization mode, by default ``"mean"``.
    checkpointing_level : int, optional
        Gradient checkpointing aggressiveness, by default ``0``.
    bias : bool, optional
        Whether the convolution carries a bias, by default ``False``.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        inp_chans,
        out_chans,
        conv_type="local",
        mlp_ratio=2.0,
        mlp_drop_rate=0.0,
        path_drop_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.Identity,
        num_groups=1,
        skip="identity",
        layer_scale=True,
        use_mlp=False,
        kernel_shape=(3, 3),
        basis_type="harmonic",
        basis_norm_mode="mean",
        checkpointing_level=0,
        bias=False,
    ):
        super().__init__()

        # determine some shapes
        self.input_shape = (forward_transform.nlat, forward_transform.nlon)
        self.output_shape = (inverse_transform.nlat, inverse_transform.nlon)
        self.out_chans = out_chans

        # gain factor for the convolution
        gain_factor = 1.0

        # disco convolution layer
        if conv_type == "local":

            self.local_conv = get_disco_conv(
                inp_chans,
                inp_chans,
                distributed=comm.get_size("spatial") > 1,
                in_shape=self.input_shape,
                out_shape=self.output_shape,
                kernel_shape=kernel_shape,
                basis_type=basis_type,
                basis_norm_mode=basis_norm_mode,
                groups=num_groups,
                grid_in=forward_transform.grid,
                grid_out=inverse_transform.grid,
                bias=False,
                theta_cutoff=math.sqrt(2) * torch.pi / float(self.input_shape[0] - 1),
            )
            if comm.get_size("spatial") > 1:
                self.local_conv.weight.is_shared_mp = ["spatial"]
                self.local_conv.weight.sharded_dims_mp = [None, None, None]
                if self.local_conv.bias is not None:
                    self.local_conv.bias.is_shared_mp = ["model"]
                    self.local_conv.bias.sharded_dims_mp = [None]

            with torch.no_grad():
                self.local_conv.weight *= gain_factor

        elif conv_type == "global":
            # convolution layer
            self.global_conv = SpectralConv(
                forward_transform,
                inverse_transform,
                inp_chans,
                inp_chans,
                operator_type="dhconv",
                num_groups=num_groups,
                bias=bias,
                gain=gain_factor,
            )
        else:
            raise ValueError(f"Unknown convolution type {conv_type}")

        # norm layer
        self.norm = norm_layer()

        if use_mlp == True:
            MLPH = DistributedMLP if (comm.get_size("matmul") > 1) else MLP
            mlp_hidden_dim = int(inp_chans * mlp_ratio)
            self.mlp = MLPH(
                in_features=inp_chans,
                out_features=out_chans,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=mlp_drop_rate,
                drop_type="features",
                checkpointing=(checkpointing_level >= 2),
                gain=gain_factor,
            )

        # dropout
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()

        if layer_scale:
            self.layer_scale = LayerScale(out_chans)
            if comm.get_size("spatial") > 1:
                self.layer_scale.weight.is_shared_mp = ["model"]
                self.layer_scale.weight.sharded_dims_mp = [None, None, None, None]
        else:
            self.layer_scale = nn.Identity()

        # skip connection
        if skip == "linear":
            gain_factor = 1.0
            self.skip = nn.Conv2d(inp_chans, out_chans, 1, 1, bias=False)
            torch.nn.init.normal_(self.skip.weight, std=math.sqrt(gain_factor / inp_chans))
            if comm.get_size("spatial") > 1:
                self.skip.weight.is_shared_mp = ["model"]
                self.skip.weight.sharded_dims_mp = [None, None, None, None]
                if self.skip.bias is not None:
                    self.skip.bias.is_shared_mp = ["model"]
                    self.skip.bias.sharded_dims_mp = [None]

        elif skip == "identity":
            self.skip = nn.Identity()
        elif skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {skip}")

    def forward(self, x):
        """
        Updated NO block
        """

        if hasattr(self, "global_conv"):
            dx, _ = self.global_conv(x)
        elif hasattr(self, "local_conv"):
            dx = self.local_conv(x)

        if hasattr(self, "norm"):
            dx = self.norm(dx)

        if hasattr(self, "mlp"):
            dx = self.mlp(dx)

        dx = self.drop_path(dx)

        if hasattr(self, "skip"):
            x = self.skip(x[..., : self.out_chans, :, :]) + self.layer_scale(dx)
        else:
            x = dx

        return x


class SphericalNeuralOperatorNet(nn.Module):
    """
    Backbone of the FourCastNet2 architecture. Uses a variant of the Spherical Fourier Neural Operator augmented with localized
    spherical Neural Operator Convolutions in encoder and decoder as well as processor layers.

    References:
    [1] Bonev et al., Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere
    [2] Ocampo et al., Scalable and Equivariant Spherical CNNs by Discrete-Continuous (DISCO) Convolutions
    [3] Liu-Schiaffini et al., Neural Operators with Localized Integral and Differential Kernels
    """

    def __init__(
        self,
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        inp_shape=(721, 1440),
        out_shape=(721, 1440),
        kernel_shape=(3, 3),
        filter_basis_type="harmonic",
        filter_basis_norm_mode="mean",
        scale_factor=8,
        encoder_kernel_shape=(3, 3),
        encoder_mlp=False,
        encoder_groups=1,
        channel_names=["z500", "t850"],
        inp_chans=2,
        out_chans=2,
        embed_dim=32,
        num_layers=4,
        num_groups=1,
        use_mlp=True,
        mlp_ratio=2.0,
        activation_function="gelu",
        layer_scale=True,
        pos_drop_rate=0.0,
        path_drop_rate=0.0,
        mlp_drop_rate=0.0,
        normalization_layer="instance_norm",
        max_modes=None,
        hard_thresholding_fraction=1.0,
        sfno_block_frequency=2,
        big_skip=True,
        clamp_water=False,
        bias=False,
        checkpointing_level=0,
        freeze_encoder=False,
        freeze_processor=False,
        **kwargs,
    ):
        super().__init__()

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
        self._init_spectral_transforms(model_grid_type, sht_grid_type, hard_thresholding_fraction, max_modes)

        # determine activation function
        if activation_function == "relu":
            activation_function = nn.ReLU
        elif activation_function == "gelu":
            activation_function = nn.GELU
        elif activation_function == "silu":
            activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # convert kernel shape to tuple
        kernel_shape = tuple(kernel_shape)
        encoder_kernel_shape = tuple(encoder_kernel_shape)

        # set up encoder
        self.encoder = DiscreteContinuousEncoder(
            inp_shape=inp_shape,
            out_shape=(self.h, self.w),
            inp_chans=inp_chans,
            out_chans=embed_dim,
            grid_in=model_grid_type,
            grid_out=sht_grid_type,
            kernel_shape=encoder_kernel_shape,
            basis_type=filter_basis_type,
            basis_norm_mode=filter_basis_norm_mode,
            bias=bias,
            groups=encoder_groups,
            use_mlp=encoder_mlp,
        )

        # dropout
        self.pos_drop = nn.Dropout(p=pos_drop_rate) if pos_drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, num_layers)]

        # get the handle for the normalization layer
        norm_layer = self._get_norm_layer_handle(
            self.h, self.w, embed_dim, normalization_layer=normalization_layer, sht_grid_type=sht_grid_type
        )

        # FNO blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_layers):
            first_layer = i == 0
            last_layer = i == num_layers - 1

            skip = "identity"
            if i % sfno_block_frequency == 0:
                conv_type = "global"
            else:
                conv_type = "local"

            block = NeuralOperatorBlock(
                self.sht,
                self.isht,
                embed_dim,
                embed_dim,
                conv_type=conv_type,
                mlp_ratio=mlp_ratio,
                mlp_drop_rate=mlp_drop_rate,
                path_drop_rate=dpr[i],
                act_layer=activation_function,
                norm_layer=norm_layer,
                skip=skip,
                layer_scale=layer_scale,
                use_mlp=use_mlp,
                kernel_shape=kernel_shape,
                basis_type=filter_basis_type,
                basis_norm_mode=filter_basis_norm_mode,
                bias=bias,
                checkpointing_level=checkpointing_level,
            )

            self.blocks.append(block)

        # set up decoder
        self.decoder = DiscreteContinuousDecoder(
            inp_shape=(self.h, self.w),
            out_shape=out_shape,
            inp_chans=embed_dim,
            out_chans=out_chans,
            grid_in=sht_grid_type,
            grid_out=model_grid_type,
            kernel_shape=encoder_kernel_shape,
            basis_type=filter_basis_type,
            basis_norm_mode=filter_basis_norm_mode,
            bias=bias,
            groups=encoder_groups,
            use_mlp=encoder_mlp,
        )

        # residual prediction
        if self.big_skip:
            self.residual_transform = nn.Conv2d(self.out_chans, self.out_chans, 1, bias=False)
            self.residual_transform.weight.is_shared_mp = ["spatial"]
            self.residual_transform.weight.sharded_dims_mp = [None, None, None, None]
            scale = math.sqrt(0.5 / self.out_chans)
            nn.init.normal_(self.residual_transform.weight, mean=0.0, std=scale)

        # controlled output normalization of q and tcwv
        if clamp_water:
            water_chans = get_water_channels(channel_names)
            if len(water_chans) > 0:
                self.register_buffer("water_channels", torch.tensor(water_chans, dtype=torch.long), persistent=False)
                # boolean mask for out-of-place torch.where clamping
                _mask = torch.zeros(self.out_chans, dtype=torch.bool)
                _mask[water_chans] = True
                self.register_buffer("water_channel_mask", _mask.view(1, -1, 1, 1), persistent=False)

        # finally, freeze part of the model if requested
        if freeze_processor:
            for param in self.blocks.parameters():
                param.requires_grad = False

        # freeze the encoder/decoder
        if freeze_encoder:
            frozen_params = list(self.decoder.parameters()) + list(self.encoder.parameters())
            if self.big_skip:
                frozen_params += list(self.residual_transform.parameters())
            for param in frozen_params:
                param.requires_grad = False

    @torch.compiler.disable(recursive=False)
    def _init_spectral_transforms(
        self,
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        hard_thresholding_fraction=1.0,
        max_modes=None,
    ):
        """
        Initialize the spectral transforms based on the maximum number of modes to keep. Handles the computation
        of local image shapes and domain parallelism, based on the
        """

        # precompute the cutoff frequency on the sphere
        if max_modes is not None:
            modes_lat, modes_lon = max_modes
        else:
            modes_lat = int(self.h * hard_thresholding_fraction)
            modes_lon = int((self.w // 2 + 1) * hard_thresholding_fraction)


        # spatial parallelism in the SHT
        if comm.get_size("spatial") > 1:
            polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
            azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
            thd.init(polar_group, azimuth_group)

        # set up
        self.sht = get_spectral_transform("sht", self.h, self.w, sht_grid_type, lmax=modes_lat, mmax=modes_lon, distributed=comm.get_size("spatial") > 1).float()
        self.isht = get_spectral_transform("isht", self.h, self.w, sht_grid_type, lmax=modes_lat, mmax=modes_lon, distributed=comm.get_size("spatial") > 1).float()

    @torch.compiler.disable(recursive=True)
    def _get_norm_layer_handle(
        self,
        h,
        w,
        embed_dim,
        normalization_layer="none",
        sht_grid_type="legendre-gauss",
    ):
        """
        get the handle for ionitializing normalization layers
        """
        # pick norm layer
        if normalization_layer == "layer_norm":
            norm_layer_handle = partial(
                DistributedLayerNorm, normalized_shape=(embed_dim), elementwise_affine=True, eps=1e-6
            )
        elif normalization_layer == "instance_norm":
            if comm.get_size("spatial") > 1:
                norm_layer_handle = partial(DistributedInstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True)
            else:
                norm_layer_handle = partial(
                    nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False
                )
        elif normalization_layer == "instance_norm_s2":
            norm_layer_handle = (
                DistributedGeometricInstanceNormS2 if comm.get_size("spatial") > 1 else GeometricInstanceNormS2
            )
            norm_layer_handle = partial(
                norm_layer_handle,
                img_shape=(h, w),
                crop_shape=(h, w),
                crop_offset=(0, 0),
                grid_type=sht_grid_type,
                num_features=embed_dim,
                eps=1e-6,
                affine=True,
            )
        elif normalization_layer == "none":
            norm_layer_handle = nn.Identity
        else:
            raise NotImplementedError(f"Error, normalization {normalization_layer} not implemented.")

        return norm_layer_handle

    def _forward_features(self, x):
        for blk in self.blocks:
            if self.checkpointing_level >= 3:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        return x

    def clamp_water_channels(self, x):
        r"""
        Clamp water-related output channels to non-negative values.

        Quantities such as specific humidity and precipitation cannot be
        negative, but nothing in the network guarantees that. Applying a ReLU to
        exactly those channels enforces it in the architecture rather than
        relying on the loss. A no-op if no water channel mask is configured.

        Parameters
        ----------
        x : torch.Tensor
            Output field of shape ``(B, out_chans, nlat, nlon)``.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape, with the masked channels clamped at zero
            and all others untouched.
        """
        if hasattr(self, "water_channel_mask"):
            w = nn.functional.relu(x)
            x = torch.where(self.water_channel_mask, w, x)
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
        dtype = x.dtype

        # save big skip
        if self.big_skip:
            residual = x[..., : self.out_chans, :, :].contiguous()

        if self.checkpointing_level >= 1:
            x = checkpoint(self.encoder, x, use_reentrant=False)
        else:
            x = self.encoder(x)

        # maybe clean the padding just in case
        x = self.pos_drop(x)

        # do the feature extraction
        x = self._forward_features(x)

        if self.checkpointing_level >= 1:
            x = checkpoint(self.decoder, x, use_reentrant=False)
        else:
            x = self.decoder(x)

        if self.big_skip:
            x = x + self.residual_transform(residual)

        # apply output transform
        x = self.clamp_water_channels(x)

        return x
