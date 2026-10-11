# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

r"""
Spherical Fourier Neural Operator, version 2.

A cleaned-up, purely spectral SFNO. Compared to :mod:`makani.models.networks.sfnonet`:

* The input is band-limited *before* anything nonlinear happens. The encoder
  projects every input channel onto spherical harmonics up to one explicit
  cutoff ``lmax`` and evaluates the result on the internal grid; the pointwise
  encoder MLP, all processor blocks and the decoder MLP run at that resolution.
  The decoder maps back through the same cutoff. No block changes resolution.
* Wind components are treated as what they are: a tangent vector field. Every
  ``(u, v)`` pair in ``channel_names`` goes through the vector spherical
  harmonic transform, whose two components (divergent and rotational) are
  carried as latent scalar fields and recombined by the inverse vector
  transform in the decoder. This is the only encoding of ``(u, v)`` that is
  exactly equivariant under rotations about the polar axis.
* The spectral cutoff is one number, ``lmax = mmax``, passed explicitly to
  every transform; nothing relies on a torch-harmonics default. By default it
  is the exact band limit of the grids involved. It can be set higher where
  the quadrature is known to alias little: on the 721-latitude equiangular
  grid with Clenshaw-Curtis weights the exact limit is 360, but at
  ``lmax=640`` (the ERA5 TL639 truncation) the roundtrip aliasing on ERA5
  fields is below 1e-4, while 720 breaks down.
* The only position-dependent parameter is an optional per-latitude embedding,
  which keeps the network exactly equivariant under rotations about the axis.
* Global context enters through :class:`FilmS2`: a feature-wise affine
  modulation whose coefficients are an MLP of the spherical mean of the block
  input. Spherical means are invariant under every isometry, so the layer does
  not break any symmetry the rest of the network has.
* The spectral filter is the linear :class:`~makani.models.common.SpectralConv`
  with one complex weight per degree :math:`\ell` (``"dhconv"``). There is no
  DISCO convolution and no spectral attention.
* The vector transforms assume that normalization did not subtract a constant
  from ``u`` and ``v``: a constant vector field is singular at the poles and
  not band-limited, so a per-component mean would leak into every mode.
  Statistics computed with ``get_stats.py --wind_angle_aware`` have zero wind
  means and one shared scale per pair, which is what this model expects.
* Initialization keeps the variance of the residual stream: blocks are
  pre-norm with a :class:`~makani.models.common.LayerScale` on the branch,
  every linear map is drawn with ``gain / fan_in`` for a unit-variance output,
  and the FiLM layer and the latitude embedding start as exact identities.

What is *not* done here, deliberately: the rotational wind component is a
pseudo-scalar under reflection at the equator, and the processor treats it like
any other channel, so the network is equivariant under axial rotations but not
yet under that reflection. The latitude embedding is not symmetrized either.
"""

import math
import logging
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import amp
from torch.utils.checkpoint import checkpoint

import torch_harmonics as th
import torch_harmonics.distributed as thd

from physicsnemo import ModelMetaData

from makani.models.common import DropPath, MLP, LayerScale, EncoderDecoder, SpectralConv, GeometricInstanceNormS2
from makani.models.physicsnemo_compat import module_from_torch
from makani.mpu.layers import DistributedMLP, DistributedEncoderDecoder
from makani.mpu.layer_norm import DistributedInstanceNorm2d, DistributedLayerNorm, DistributedGeometricInstanceNormS2
from makani.utils import comm
from makani.utils.features import get_channel_groups, get_wind_channels
from makani.utils.grids import GridQuadrature, compute_spherical_bandlimit, get_spectral_transform, grid_to_quadrature_rule


# ---------------------------------------------------------------------------
# spectral transforms
# ---------------------------------------------------------------------------

def _make_transform(kind, nlat, nlon, grid_type, lmax):
    r"""
    Build a (vector) SHT with an explicit triangular cutoff ``lmax = mmax``,
    serial or spatially distributed as the process grid dictates, with fp32
    buffers. Wraps :func:`makani.utils.grids.get_spectral_transform`, which
    handles both torch-harmonics constructor conventions.
    """
    transform = get_spectral_transform(
        kind, nlat, nlon, grid_type, lmax=lmax, mmax=lmax, distributed=(comm.get_size("spatial") > 1), strict=True
    )
    return transform.float()


def _local_degrees(transform):
    r"""
    Degrees held by this rank in the output of ``transform``.

    Distributed transforms shard the spectral coefficients over degree across
    the polar process group and expose the per-rank counts as ``l_shapes``;
    serial transforms hold every degree.
    """
    if hasattr(transform, "l_shapes"):
        rank = comm.get_rank("h")
        offset = sum(transform.l_shapes[:rank])
        return slice(offset, offset + transform.l_shapes[rank])
    return slice(None)


def _wind_spectral_scale(lmax, mode):
    r"""
    Per-degree scaling applied to the vector SHT coefficients of the wind.

    The vector transform returns the coefficients of the velocity potential and
    the streamfunction, i.e. the wind divided by :math:`\sqrt{\ell(\ell+1)}`
    in each degree, which makes the latent fields much redder than the wind
    itself. The scaling chooses which quantity the latent fields represent:

    * ``"potential"``: no scaling; velocity potential and streamfunction.
    * ``"energy"``: multiply by :math:`\sqrt{\ell(\ell+1)}`; the latent fields
      then carry the same energy spectrum as ``(u, v)``.
    * ``"vortdiv"``: multiply by :math:`\ell(\ell+1)`; divergence and vorticity.

    Returns
    -------
    torch.Tensor
        Real tensor of shape ``(lmax, 1)`` broadcasting over orders.
    """
    exponents = {"potential": 0.0, "energy": 0.5, "vortdiv": 1.0}
    if mode not in exponents:
        raise ValueError(f"Unknown wind_spectral_scaling {mode}, expected one of {list(exponents)}")
    ell = torch.arange(lmax, dtype=torch.float64)
    scale = (ell * (ell + 1)).pow(exponents[mode])
    scale[0] = 1.0  # the l=0 vector modes vanish identically; avoid dividing by zero in the decoder
    return scale.to(torch.float32).reshape(lmax, 1)


# ---------------------------------------------------------------------------
# channel bookkeeping
# ---------------------------------------------------------------------------


def _compute_channel_layout(channel_names, aux_channel_names, n_history, inp_chans, out_chans):
    r"""
    Split the input and output channels into scalar fields and wind pairs.

    The input layout follows the makani data pipeline: the predicted channels
    followed by the dynamic auxiliary channels, repeated ``n_history + 1``
    times, followed by the static auxiliary channels. The output channels are
    ``channel_names`` in order.

    Returns
    -------
    dict
        ``scalar_in`` and ``wind_in`` index the input; ``scalar_out`` and
        ``wind_out`` index the output; ``pred_in`` are the input positions of
        the output channels at history offset 0. Wind indices are flat
        ``[u0, v0, u1, v1, ...]`` lists.

    Raises
    ------
    ValueError
        If the channel names do not account for ``inp_chans`` and ``out_chans``.
    """
    channel_names = list(channel_names)
    aux_channel_names = list(aux_channel_names)

    n_pred = len(channel_names)
    if n_pred == 0:
        raise ValueError("SFNOv2 needs channel_names to tell wind components from scalar fields")

    _, _, dyn_aux, stat_aux, _ = get_channel_groups(channel_names, aux_channel_names)
    n_dyn = n_pred + len(dyn_aux)
    expected_inp = n_dyn * (n_history + 1) + len(stat_aux)
    if expected_inp != inp_chans:
        raise ValueError(
            f"channel layout mismatch: {n_pred} predicted + {len(dyn_aux)} dynamic auxiliary channels over "
            f"{n_history + 1} time steps + {len(stat_aux)} static channels = {expected_inp}, but inp_chans={inp_chans}"
        )
    if out_chans != n_pred:
        raise ValueError(f"out_chans={out_chans} does not match the {n_pred} channel names")

    wind_out = get_wind_channels(channel_names)
    wind_in = [c + ih * n_dyn for ih in range(n_history + 1) for c in wind_out]
    scalar_in = [c for c in range(inp_chans) if c not in set(wind_in)]
    scalar_out = [c for c in range(out_chans) if c not in set(wind_out)]

    return dict(scalar_in=scalar_in, wind_in=wind_in, scalar_out=scalar_out, wind_out=wind_out, pred_in=list(range(n_pred)))


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------


class HopfProjection(nn.Module):
    r"""
    Grid-to-spectral projection that handles scalar fields and wind vectors:
    the "Hopf encoding" of the state.

    Scalar channels go through the scalar SHT. Wind pairs go through the
    vector SHT, which is the spectral form of the Helmholtz-Hodge
    decomposition: each tangent vector field becomes a divergent (curl-free)
    and a rotational (divergence-free) scalar component, so that the wind can
    be carried by scalar machinery without losing its vector character. Both
    transforms are truncated at the same ``lmax``. The result is a pair of
    coefficient tensors, so that a matching :class:`HopfSynthesis` on any
    other grid can evaluate the band-limited field there.

    The per-degree ``wind_scale`` decides which quantity the two wind
    components represent (potentials, energy-matched, or vorticity and
    divergence); see :func:`_wind_spectral_scale`. What matters for the MLPs
    downstream is that the resulting fields are of order one like the
    normalized scalar channels.

    Parameters
    ----------
    shape : (int, int)
        Grid resolution of the input.
    grid_type : str
        Grid the input lives on.
    lmax : int
        Cutoff degree.
    wind_scale : torch.Tensor
        Per-degree scaling of the vector coefficients, see :func:`_wind_spectral_scale`.
    """

    def __init__(self, shape, grid_type, lmax, wind_scale):
        super().__init__()
        self.sht = _make_transform("sht", *shape, grid_type, lmax)
        self.vsht = _make_transform("vsht", *shape, grid_type, lmax)
        # the coefficients may be sharded over degree; keep the matching slice of the scaling
        self.register_buffer("wind_scale", wind_scale[_local_degrees(self.vsht)].clone(), persistent=False)

    def forward(self, x, scalar_channels, wind_channels):
        r"""
        Parameters
        ----------
        x : torch.Tensor
            Field of shape ``(B, C, nlat, nlon)``.
        scalar_channels : torch.Tensor
            Indices of the scalar channels.
        wind_channels : torch.Tensor
            Flat indices ``[u0, v0, u1, v1, ...]`` of the wind pairs; may be empty.

        Returns
        -------
        scalar_coeffs : torch.Tensor
            Complex coefficients of shape ``(B, n_scalar, lmax, lmax)``.
        wind_coeffs : torch.Tensor or None
            Complex coefficients of shape ``(B, n_pairs, 2, lmax, lmax)``, the
            spheroidal (index 0) and toroidal (index 1) components of the
            field ``(u_theta, u_phi) := (u, v)``; see the note below.

        Notes
        -----
        The vector SHT takes its components in the spherical frame,
        ``(u_theta, u_phi)`` with ``theta`` the colatitude, so the physical
        wind would be ``(-v, u)``. Passing ``(u, v)`` instead feeds the field
        rotated by 90 degrees at every point (``k x``, up to sign), which is
        a smooth, invertible and equivariant map, so the encoding remains an
        exact, equivariant representation of the wind and the matching
        :class:`HopfSynthesis` undoes it. Only the labels are swapped: index 0
        then carries the physical wind's *rotational* part and index 1 its
        *divergent* part. The ordering is kept as is on purpose, so that
        trained checkpoints stay loadable; changing it would permute the
        encoder's and decoder's wind inputs.
        """
        with amp.autocast(device_type=x.device.type, enabled=False):
            x = x.to(torch.float32)
            scalar_coeffs = self.sht(x[:, scalar_channels])
            wind_coeffs = None
            if wind_channels.numel() > 0:
                B, _, H, W = x.shape
                # (u, v) passed as (u_theta, u_phi): see the note in the docstring
                wind = x[:, wind_channels].reshape(B, -1, 2, H, W)
                wind_coeffs = self.vsht(wind) * self.wind_scale
        return scalar_coeffs, wind_coeffs


class HopfSynthesis(nn.Module):
    r"""
    Spectral-to-grid synthesis, the inverse of :class:`HopfProjection`.

    Parameters
    ----------
    shape : (int, int)
        Grid resolution of the output.
    grid_type : str
        Grid the output lives on.
    lmax : int
        Cutoff degree.
    wind_scale : torch.Tensor
        Must match the scale of the projection the coefficients came from.
    """

    def __init__(self, shape, grid_type, lmax, wind_scale):
        super().__init__()
        self.isht = _make_transform("isht", *shape, grid_type, lmax)
        self.ivsht = _make_transform("ivsht", *shape, grid_type, lmax)
        self.register_buffer("wind_scale", wind_scale[_local_degrees(self.ivsht)].clone(), persistent=False)

    def forward(self, scalar_coeffs, wind_coeffs, inverse_permutation):
        r"""
        Parameters
        ----------
        scalar_coeffs, wind_coeffs
            As returned by :class:`HopfProjection`.
        inverse_permutation : torch.Tensor
            Channel order that restores the original layout from the
            concatenation ``[scalars, u0, v0, u1, v1, ...]``.

        Returns
        -------
        torch.Tensor
            Real field of shape ``(B, C, nlat, nlon)`` in fp32.
        """
        with amp.autocast(device_type=scalar_coeffs.device.type, enabled=False):
            fields = [self.isht(scalar_coeffs)]
            if wind_coeffs is not None:
                # components come back in the same (u, v)-as-(u_theta, u_phi) convention the projection used
                wind = self.ivsht(wind_coeffs / self.wind_scale)
                B, _, _, H, W = wind.shape
                fields.append(wind.reshape(B, -1, H, W))
            x = torch.cat(fields, dim=1)[:, inverse_permutation]
        return x


class FilmS2(nn.Module):
    r"""
    Feature-wise linear modulation conditioned on spherical means.

    Computes :math:`\gamma, \beta = \mathrm{MLP}\big(\tfrac{1}{4\pi}\int x\,d\Omega\big)`
    from one field and applies :math:`y \mapsto (1 + \gamma)\, y + \beta` to
    another, channel by channel. The pooled mean is an invariant of every
    isometry of the sphere, and the modulation is pointwise, so the layer is
    equivariant under all of them. It gives each block access to global
    information that the instance norm removes.

    Initialized to the identity: the hidden layer with ``2 / fan_in`` like the
    other MLPs, the output layer at zero. A freshly built network behaves as if
    the layer were absent.

    Parameters
    ----------
    num_features : int
        Channels of the modulated field.
    quadrature : torch.nn.Module
        Normalized quadrature on the grid the conditioning field lives on,
        mapping ``(B, C, H, W)`` to ``(B, C)``.
    hidden_ratio : float, optional
        Hidden width of the MLP as a multiple of ``num_features``, by default ``1.0``.
    act_layer : callable, optional
        Activation constructor, by default :class:`torch.nn.GELU`.
    cond_features : int, optional
        Channels of the conditioning field, by default ``num_features``.
    """

    def __init__(self, num_features, quadrature, hidden_ratio=1.0, act_layer=nn.GELU, cond_features=None):
        super().__init__()
        cond_features = num_features if cond_features is None else cond_features
        hidden_features = max(int(num_features * hidden_ratio), 1)

        self.quadrature = quadrature
        self.fc1 = nn.Linear(cond_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, 2 * num_features)
        nn.init.normal_(self.fc1.weight, std=math.sqrt(2.0 / cond_features))
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        # global, pointwise parameters: replicated across spatial ranks
        for p in self.parameters():
            p.is_shared_mp = ["spatial"]
            p.sharded_dims_mp = [None] * p.dim()

    def condition(self, x):
        r"""Spherical mean of ``x``: ``(B, C, H, W) -> (B, C)``, in fp32."""
        with amp.autocast(device_type=x.device.type, enabled=False):
            cond = self.quadrature(x.to(torch.float32))
        return cond.to(x.dtype)

    def forward(self, x, cond):
        gamma, beta = self.fc2(self.act(self.fc1(cond))).chunk(2, dim=1)
        return x * (1.0 + gamma[..., None, None]) + beta[..., None, None]


class NeuralOperatorBlock(nn.Module):
    r"""
    One SFNOv2 processor block.

    .. code-block:: text

        x -> norm0 -> SpectralConv -> norm1 -> [FilmS2(., mean(x))] -> MLP -> drop_path -> layer_scale
          -> + skip(x)

    Pre-norm residual block in the style of the FCN3.1 block, with the linear
    spectral convolution as its only spatial operator. Input and output live
    on the same grid.

    Initialization keeps the variance of the residual stream: the spectral
    convolution and the MLP are drawn with unit gain, so their outputs have
    unit variance for unit-variance input, and the branch enters the stream
    through a :class:`~makani.models.common.LayerScale` of ``layer_scale_init``.
    With ``skip="identity"`` the stream is therefore preserved exactly at
    initialization; with ``skip="linear"`` the skip is drawn with unit gain
    and preserves it in expectation.

    Parameters
    ----------
    sht, isht : torch.nn.Module
        Forward and inverse transforms on the internal grid.
    embed_dim : int
        Channel width.
    mlp_ratio : float, optional
        Hidden width of the MLP as a multiple of ``embed_dim``, by default ``2.0``.
    mlp_drop_rate : float, optional
        Dropout inside the MLP, by default ``0.0``.
    path_drop_rate : float, optional
        Stochastic depth probability, by default ``0.0``.
    act_layer : callable, optional
        Activation constructor, by default :class:`torch.nn.GELU`.
    norm_layer : callable, optional
        Normalization constructor, by default :class:`torch.nn.Identity`.
    film : callable, optional
        Constructor returning a :class:`FilmS2` for ``embed_dim`` channels, or
        ``None`` to disable the modulation.
    skip : str, optional
        ``"identity"`` (default) or ``"linear"``, a learned 1x1 convolution.
    layer_scale_init : float, optional
        Initial value of the per-channel branch scale, by default ``0.1``.
    num_groups : int, optional
        Channel groups of the spectral filter, by default ``1``. The filter
        mixes channels per degree within each group; the weight count is
        ``embed_dim**2 * lmax / num_groups``.
    separable : bool, optional
        Depthwise spectral filter (no channel mixing in the filter), by default
        ``False``.
    checkpointing_level : int, optional
        The MLP is checkpointed at level 2 and above, by default ``0``.
    """

    def __init__(
        self,
        sht,
        isht,
        embed_dim,
        mlp_ratio=2.0,
        mlp_drop_rate=0.0,
        path_drop_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.Identity,
        film=None,
        skip="identity",
        layer_scale_init=0.1,
        num_groups=1,
        separable=False,
        checkpointing_level=0,
    ):
        super().__init__()

        self.norm0 = norm_layer()
        self.filter = SpectralConv(
            sht,
            isht,
            embed_dim,
            embed_dim,
            num_groups=num_groups,
            operator_type="dhconv",
            separable=separable,
            bias=False,
            gain=1.0,
        )
        self.norm1 = norm_layer()
        self.film = film() if film is not None else None

        MLPH = DistributedMLP if comm.get_size("matmul") > 1 else MLP
        self.mlp = MLPH(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            drop_rate=mlp_drop_rate,
            drop_type="features",
            comm_name="matmul",
            checkpointing=(checkpointing_level >= 2),
            gain=1.0,
        )
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()

        self.layer_scale = LayerScale(embed_dim, init_value=layer_scale_init)
        self.layer_scale.weight.is_shared_mp = ["spatial"]
        self.layer_scale.weight.sharded_dims_mp = [None, None, None, None]

        if skip == "linear":
            self.skip = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
            nn.init.normal_(self.skip.weight, std=math.sqrt(1.0 / embed_dim))
            self.skip.weight.is_shared_mp = ["spatial"]
            self.skip.weight.sharded_dims_mp = [None, None, None, None]
        elif skip == "identity":
            self.skip = nn.Identity()
        else:
            raise ValueError(f"Unknown skip connection type {skip}")

    def forward(self, x):
        residual = x

        if self.film is not None:
            cond = self.film.condition(x)

        x = self.norm0(x)
        x, _ = self.filter(x)
        x = self.norm1(x)
        if self.film is not None:
            x = self.film(x, cond)
        x = self.mlp(x)
        x = self.drop_path(x)

        return self.skip(residual) + self.layer_scale(x)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class SphericalFourierNeuralOperatorNetV2(nn.Module):
    r"""
    Spherical Fourier Neural Operator, version 2. See the module docstring for
    what distinguishes it from :class:`~makani.models.networks.sfnonet.SphericalFourierNeuralOperatorNet`.

    Parameters
    ----------
    inp_shape, out_shape : (int, int)
        Input and output grids as ``(nlat, nlon)``, by default ``(721, 1440)``.
    inp_chans, out_chans : int
        Number of input and output channels. Must agree with the channel names.
    channel_names : list of str
        Names of the predicted channels. Wind pairs are recognized as
        ``u<suffix>``/``v<suffix>``.
    aux_channel_names : list of str, optional
        Names of the auxiliary input channels, by default ``[]``.
    n_history : int, optional
        Number of additional past time steps in the input, by default ``0``.
    model_grid_type : str, optional
        Grid of the input and output data, by default ``"equiangular"``.
    sht_grid_type : str, optional
        Internal grid, by default ``"legendre-gauss"``. With ``"equiangular"``
        and ``scale_factor=1`` the internal grid is the data grid itself and
        all transforms use Clenshaw-Curtis quadrature.
    scale_factor : int, optional
        Internal grid is the input grid coarsened by this factor, by default ``3``.
    lmax : int, optional
        Explicit spectral cutoff, always used as ``mmax`` too. By default the
        smallest exact band limit of the internal, input and output grids,
        times ``hard_thresholding_fraction``. An explicit value may exceed the
        exact limits (logged), see the module docstring; it may not exceed
        what the grids resolve at all, i.e. their number of latitudes or half
        their number of longitudes.
    hard_thresholding_fraction : float, optional
        Fraction of the band limit to keep when ``lmax`` is not given, by default ``1.0``.
    embed_dim : int, optional
        Latent width, by default ``256``.
    num_layers : int, optional
        Number of processor blocks, by default ``8``.
    mlp_ratio : float, optional
        Hidden width of the block MLPs as a multiple of ``embed_dim``, by default ``2.0``.
    encoder_ratio, decoder_ratio : float, optional
        Hidden width of the encoder and decoder MLPs as a multiple of ``embed_dim``, by default ``1``.
    encoder_layers : int, optional
        Hidden layers in the encoder and decoder MLPs, by default ``1``.
    activation_function : str, optional
        ``"gelu"`` (default), ``"relu"`` or ``"silu"``.
    normalization_layer : str, optional
        ``"instance_norm"`` (default), ``"instance_norm_s2"``, ``"layer_norm"`` or ``"none"``.
    pos_embed : str, optional
        ``"latitude"`` (default) adds a learned per-latitude vector on the
        internal grid after the encoder; ``"none"`` adds nothing.
    use_film : bool, optional
        Enable :class:`FilmS2` in every block, by default ``True``.
    film_ratio : float, optional
        Hidden width of the FiLM MLP as a multiple of ``embed_dim``, by default ``1.0``.
    wind_spectral_scaling : str, optional
        Which quantity the latent wind fields represent; see :func:`_wind_spectral_scale`.
        By default ``"energy"``.
    skip : str, optional
        Block skip connection, ``"identity"`` (default) or ``"linear"``.
    layer_scale_init : float, optional
        Initial value of the per-channel scale on each block's residual
        branch, by default ``0.1``.
    num_groups : int, optional
        Channel groups of the spectral filters, by default ``1`` (dense
        per-degree channel mixing). The spectral weights scale as
        ``embed_dim**2 * lmax / num_groups`` per block, which at full
        resolution is what decides whether the model fits in memory.
    separable : bool, optional
        Depthwise spectral filters, by default ``False``.
    big_skip : bool, optional
        Add the input state to the output so the network predicts a tendency,
        by default ``True``. Resampled spectrally if the grids differ.
    path_drop_rate, mlp_drop_rate : float, optional
        Stochastic depth and MLP dropout, by default ``0.0``.
    checkpointing_level : int, optional
        Gradient checkpointing: 1 for encoder/decoder, 2 adds the MLPs, 3 adds
        whole blocks. By default ``0``.
    **kwargs
        Ignored; present so model configs can pass extra keys.
    """

    # makes get_model refuse normalization statistics with componentwise wind means
    requires_vector_wind_normalization = True

    def __init__(
        self,
        inp_shape=(721, 1440),
        out_shape=(721, 1440),
        inp_chans=2,
        out_chans=2,
        channel_names=["u500", "v500"],
        aux_channel_names=[],
        n_history=0,
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        scale_factor=3,
        lmax=None,
        hard_thresholding_fraction=1.0,
        embed_dim=256,
        num_layers=8,
        mlp_ratio=2.0,
        encoder_ratio=1,
        decoder_ratio=1,
        encoder_layers=1,
        activation_function="gelu",
        normalization_layer="instance_norm",
        pos_embed="latitude",
        use_film=True,
        film_ratio=1.0,
        wind_spectral_scaling="energy",
        skip="identity",
        layer_scale_init=0.1,
        num_groups=1,
        separable=False,
        big_skip=True,
        path_drop_rate=0.0,
        mlp_drop_rate=0.0,
        checkpointing_level=0,
        **kwargs,
    ):
        super().__init__()

        self.inp_shape = tuple(inp_shape)
        self.out_shape = tuple(out_shape)
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.big_skip = big_skip
        self.checkpointing_level = checkpointing_level
        self.model_grid_type = model_grid_type
        self.sht_grid_type = sht_grid_type

        # internal grid
        self.h = int(self.inp_shape[0] // scale_factor)
        self.w = int(self.inp_shape[1] // scale_factor)

        # channel layout
        layout = _compute_channel_layout(channel_names, aux_channel_names, n_history, inp_chans, out_chans)
        for name, idx in layout.items():
            self.register_buffer(name, torch.tensor(idx, dtype=torch.long), persistent=False)
        for side in ("in", "out"):
            order = torch.tensor(layout[f"scalar_{side}"] + layout[f"wind_{side}"], dtype=torch.long)
            self.register_buffer(f"{side}_inverse_perm", torch.argsort(order), persistent=False)
        self.n_wind_pairs = len(layout["wind_out"]) // 2

        # spectral cutoff
        self._init_cutoff(lmax, hard_thresholding_fraction)

        # distributed transforms need the process groups before anything is built
        if (comm.get_size("spatial") > 1) and (not thd.is_initialized()):
            polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
            azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
            thd.init(polar_group, azimuth_group)

        wind_scale = _wind_spectral_scale(self.lmax, wind_spectral_scaling)

        # transforms: input grid -> coefficients -> internal grid, and back out. Each transform holds an
        # (mmax, lmax, nlat) fp32 Legendre table, 1-2 GB at full resolution, so transforms on the same grid
        # are shared rather than built twice.
        internal_grid = ((self.h, self.w), sht_grid_type)
        self.project_in = HopfProjection(self.inp_shape, model_grid_type, self.lmax, wind_scale)
        if internal_grid == (self.inp_shape, model_grid_type):
            self.project_internal = self.project_in
        else:
            self.project_internal = HopfProjection(*internal_grid, self.lmax, wind_scale)
        self.synthesize_internal = HopfSynthesis(*internal_grid, self.lmax, wind_scale)
        if internal_grid == (self.out_shape, model_grid_type):
            self.synthesize_out = self.synthesize_internal
        else:
            self.synthesize_out = HopfSynthesis(self.out_shape, model_grid_type, self.lmax, wind_scale)

        # the processor shares one scalar transform pair
        self.sht = self.project_internal.sht
        self.isht = self.synthesize_internal.isht

        # local shape of the internal grid (spatial parallelism)
        if comm.get_size("spatial") > 1:
            self.h_loc = self.isht.lat_shapes[comm.get_rank("h")]
            self.w_loc = self.isht.lon_shapes[comm.get_rank("w")]
        else:
            self.h_loc, self.w_loc = self.h, self.w

        # activation
        activations = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        if activation_function not in activations:
            raise ValueError(f"Unknown activation function {activation_function}")
        act_layer = activations[activation_function]

        # pointwise encoder and decoder, both on the internal grid
        EncDecH = DistributedEncoderDecoder if comm.get_size("matmul") > 1 else EncoderDecoder
        encdec_kwargs = dict(input_format="nchw")
        if comm.get_size("matmul") > 1:
            encdec_kwargs["comm_name"] = "matmul"
        self.encoder = EncDecH(
            num_layers=encoder_layers,
            input_dim=inp_chans,
            output_dim=embed_dim,
            hidden_dim=int(encoder_ratio * embed_dim),
            act_layer=act_layer,
            **encdec_kwargs,
        )
        self.decoder = EncDecH(
            num_layers=encoder_layers,
            input_dim=embed_dim,
            output_dim=out_chans,
            hidden_dim=int(decoder_ratio * embed_dim),
            act_layer=act_layer,
            gain=0.5 if big_skip else 1.0,
            **encdec_kwargs,
        )

        # latitude embedding: zero-initialized so the network starts exactly equivariant and the
        # encoder output keeps its unit variance
        if pos_embed == "latitude":
            self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.h_loc, 1))
            self.pos_embed.is_shared_mp = ["w"]
            self.pos_embed.sharded_dims_mp = [None, None, "h", None]
        elif pos_embed in ("none", "None", None):
            pass
        else:
            raise ValueError(f"Unknown position embedding type {pos_embed}; SFNOv2 supports 'latitude' and 'none'")

        # normalization on the internal grid
        norm_layer = self._get_norm_layer_handle(normalization_layer)

        # FiLM conditioning: spherical mean on the internal grid, with the quadrature the SHT uses there
        # (Clenshaw-Curtis on equiangular grids, not makani's cell-area default)
        film = None
        if use_film:
            quadrature_rule = "clenshaw-curtiss" if sht_grid_type == "equiangular" else grid_to_quadrature_rule(sht_grid_type)
            quadrature = partial(
                GridQuadrature,
                quadrature_rule,
                img_shape=(self.h, self.w),
                normalize=True,
                distributed=True,
            )
            film = lambda: FilmS2(embed_dim, quadrature(), hidden_ratio=film_ratio, act_layer=act_layer)

        # processor
        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, num_layers)]
        self.blocks = nn.ModuleList(
            [
                NeuralOperatorBlock(
                    self.sht,
                    self.isht,
                    embed_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    path_drop_rate=dpr[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    film=film,
                    skip=skip,
                    layer_scale_init=layer_scale_init,
                    num_groups=num_groups,
                    separable=separable,
                    checkpointing_level=checkpointing_level,
                )
                for i in range(num_layers)
            ]
        )

    def _init_cutoff(self, lmax, hard_thresholding_fraction):
        r"""
        Fix the spectral cutoff ``self.lmax`` (exclusive, used as ``mmax`` too).

        By default the cutoff is the smallest exact band limit of the internal,
        input and output grids. An explicit ``lmax`` may exceed the exact
        limits where the quadrature is known to alias little (see the module
        docstring); it is still bounded by what the transforms can resolve at
        all, i.e. the number of latitudes and half the longitudes.
        """
        grids = [
            ("internal", (self.h, self.w), self.sht_grid_type),
            ("input", self.inp_shape, self.model_grid_type),
            ("output", self.out_shape, self.model_grid_type),
        ]
        exact_limit = min(compute_spherical_bandlimit(shape, grid) for _, shape, grid in grids)
        resolvable = min(min(shape[0], shape[1] // 2) for _, shape, _ in grids)

        if lmax is None:
            lmax = int(exact_limit * hard_thresholding_fraction)
        elif lmax > exact_limit:
            inexact = [f"{name} {shape[0]}x{shape[1]} {grid}" for name, shape, grid in grids if lmax > compute_spherical_bandlimit(shape, grid)]
            logging.info(
                f"lmax={lmax} exceeds the exact band limit of the {', '.join(inexact)} grid(s) ({exact_limit}); "
                f"the transforms on them rely on the quadrature aliasing little at this cutoff"
            )

        if lmax > resolvable:
            raise ValueError(f"lmax={lmax} cannot be resolved on the grids involved (at most {resolvable})")
        if lmax < 1:
            raise ValueError(f"lmax={lmax} must be at least 1")
        self.lmax = lmax

    def _get_norm_layer_handle(self, normalization_layer):
        embed_dim = self.embed_dim
        if normalization_layer == "layer_norm":
            return partial(DistributedLayerNorm, normalized_shape=(embed_dim), elementwise_affine=True, eps=1e-6)
        elif normalization_layer == "instance_norm":
            if comm.get_size("spatial") > 1:
                return partial(DistributedInstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True)
            return partial(nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        elif normalization_layer == "instance_norm_s2":
            handle = DistributedGeometricInstanceNormS2 if comm.get_size("spatial") > 1 else GeometricInstanceNormS2
            return partial(
                handle,
                img_shape=(self.h, self.w),
                crop_shape=(self.h, self.w),
                crop_offset=(0, 0),
                grid_type=self.sht_grid_type,
                num_features=embed_dim,
                eps=1e-6,
                affine=True,
            )
        elif normalization_layer == "none":
            return nn.Identity
        raise NotImplementedError(f"Error, normalization {normalization_layer} not implemented.")

    @torch.compiler.disable(recursive=True)
    def no_weight_decay(self):
        r"""Parameters excluded from weight decay: the latitude embedding."""
        return {"pos_embed"}

    def encode(self, x):
        r"""Band-limit the input onto the internal grid and lift it to ``embed_dim`` channels."""
        coeffs = self.project_in(x, self.scalar_in, self.wind_in)
        x = self.synthesize_internal(*coeffs, self.in_inverse_perm).to(x.dtype)
        if self.checkpointing_level >= 1:
            x = checkpoint(self.encoder, x, use_reentrant=False)
        else:
            x = self.encoder(x)
        if hasattr(self, "pos_embed"):
            x = x + self.pos_embed.to(dtype=x.dtype)
        return x

    def decode(self, x):
        r"""Project to the output channels on the internal grid and evaluate them on the output grid."""
        dtype = x.dtype
        if self.checkpointing_level >= 1:
            x = checkpoint(self.decoder, x, use_reentrant=False)
        else:
            x = self.decoder(x)
        coeffs = self.project_internal(x, self.scalar_out, self.wind_out)
        return self.synthesize_out(*coeffs, self.out_inverse_perm).to(dtype)

    def _forward_features(self, x):
        for blk in self.blocks:
            if self.checkpointing_level >= 3:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

    def forward(self, x):
        r"""
        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, inp_chans, nlat, nlon)`` on ``inp_shape``.

        Returns
        -------
        torch.Tensor
            Prediction of shape ``(B, out_chans, nlat, nlon)`` on ``out_shape``.
        """
        if self.big_skip:
            residual = x[:, self.pred_in]
            if self.out_shape != self.inp_shape:
                # move the skipped state through the same band limit onto the output grid
                coeffs = self.project_in(residual, self.scalar_out, self.wind_out)
                residual = self.synthesize_out(*coeffs, self.out_inverse_perm)

        x = self.encode(x)
        x = self._forward_features(x)
        x = self.decode(x)

        if self.big_skip:
            x = x + residual.to(x.dtype)

        return x


@dataclass
class SphericalFourierNeuralOperatorNetV2MetaData(ModelMetaData):
    r"""
    PhysicsNeMo metadata for :class:`SphericalFourierNeuralOperatorNetV2`.

    Same feature declaration as the other makani models: no TorchScript, no
    CUDA graphs, mixed precision on GPU only. The name is passed to
    :func:`~makani.models.physicsnemo_compat.module_from_torch` instead of being
    set here, since ``ModelMetaData.name`` is deprecated in PhysicsNeMo 2.x.
    """

    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True


SFNOv2 = module_from_torch(
    SphericalFourierNeuralOperatorNetV2,
    SphericalFourierNeuralOperatorNetV2MetaData(),
    name="SFNOv2",
)
