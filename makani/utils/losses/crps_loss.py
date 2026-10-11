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

from typing import Optional, Tuple, List

import numpy as np
import math

import torch
from torch import amp

from makani.utils.losses.base_loss import (
    LossType,
    GeometricBaseLoss,
    SpectralBaseLoss,
    VortDivBaseLoss,
    GradientBaseLoss,
)
from makani.utils import comm
from makani.utils.grids import get_disco_conv

# distributed stuff
from torch_harmonics.distributed import split_tensor_along_dim
from makani.mpu.mappings import scatter_to_parallel_region, reduce_from_parallel_region, distributed_transpose

# torch-harmonics for convolutions
import torch_harmonics as th
import torch_harmonics.distributed as thd


def rankdata(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    ordinal ranking along dimension dim
    """
    ndim = x.dim()
    perm = torch.argsort(x, dim=dim, descending=False, stable=True)

    idx = torch.arange(x.shape[dim], device=x.device).reshape([-1 if i == dim else 1 for i in range(ndim)])
    rank = torch.empty_like(x, dtype=torch.long).scatter_(dim=dim, index=perm, src=idx.expand_as(perm)) + 1
    return rank


# @torch.compile
def _crps_ensemble_kernel(observation: torch.Tensor, forecasts: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    CRPS ensemble score from integrating the PDF piecewise
    compare https://github.com/properscoring/properscoring/blob/master/properscoring/_gufuncs.py#L7
    disabling torch compile for the moment due to very long startup times when training large ensembles with ensemble parallelism

    forecasts: [ensemble, ...], observation: [...], weights: [ensemble, ...]
    Assumes forecasts are sorted along ensemble dimension 0.
    """

    # beware: forecasts are assumed sorted in sorted order
    # compute total weights
    total_weight = torch.sum(weights, dim=0)

    # initial values
    obs_cdf = torch.zeros_like(observation)
    forecast_cdf = torch.zeros_like(observation)
    prev_forecast = torch.zeros_like(observation)
    integral = torch.zeros_like(observation)

    # split lists
    weightslist = torch.split(weights, 1, dim=0)
    forecastlist = torch.split(forecasts, 1, dim=0)
    for forecast, weight in zip(forecastlist, weightslist):

        # compute condition
        condition = torch.logical_and(observation < forecast, torch.abs(obs_cdf) < 1.0e-7)

        # compute terms
        term_true = (observation - prev_forecast) * torch.square(forecast_cdf) + (
            forecast - observation
        ) * torch.square(forecast_cdf - 1)
        term_false = (forecast - prev_forecast) * torch.square(forecast_cdf - obs_cdf)
        increment = torch.where(condition, term_true, term_false)

        # compute integral
        integral = integral + increment

        # update cdf
        obs_cdf = torch.where(condition, 1.0, obs_cdf)
        forecast_cdf = forecast_cdf + weight / total_weight

        # update forcast
        prev_forecast = forecast

    integral = integral + torch.clamp(observation - forecast, min=0.0)

    return torch.squeeze(integral, dim=0)


def _crps_skillspread_kernel(
    observation: torch.Tensor, forecasts: torch.Tensor, weights: torch.Tensor, alpha: float
) -> torch.Tensor:
    """
    fair CRPS variant that uses spread and skill. Assumes pre-sorted ensemble
    """

    observation = observation.unsqueeze(0)

    # get the ranks for the spread computation
    rank = rankdata(forecasts, dim=0)

    #  ensemble size
    num_ensemble = forecasts.shape[0]

    # get the ensemble spread (total_weight is ensemble size here)
    espread = (
        2
        * torch.mean((2 * rank - num_ensemble - 1) * forecasts, dim=0)
        * (float(num_ensemble) - 1.0 + alpha)
        / float(num_ensemble * (num_ensemble - 1))
    )
    eskill = (observation - forecasts).abs().mean(dim=0)

    crps = eskill - 0.5 * espread

    return crps


def _crps_probability_weighted_moment_kernel(
    observation: torch.Tensor, forecasts: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """
    CRPS estimator based on the probability weighted moment. see [1].

    [1] Michael Zamo, Phillippe Naveau. Estimation of the Continuous Ranked Probability Score with Limited Information and Applications to Ensemble Weather Forecasts. Mathematical Geosciences. Volume 50 pp. 209-234. 2018.
    """

    observation = observation.unsqueeze(0)

    #  ensemble size
    num_ensemble = forecasts.shape[0]

    # get the ranks for the pwm computation
    rank = torch.arange(num_ensemble, device=forecasts.device).reshape((num_ensemble,) + (1,) * (forecasts.dim() - 1))

    # get the ensemble spread (total_weight is ensemble size here)
    beta0 = forecasts.mean(dim=0)
    beta1 = (rank * forecasts).sum(dim=0) / float(num_ensemble * (num_ensemble - 1))
    eskill = (observation - forecasts).abs().mean(dim=0)

    crps = eskill + beta0 - 2 * beta1

    return crps


def _crps_naive_skillspread_kernel(
    observation: torch.Tensor, forecasts: torch.Tensor, weights: torch.Tensor, alpha: float
) -> torch.Tensor:
    """
    alternative fair CRPS variant that uses spread and skill. Uses naive computation which is O(N^2) in the number of ensemble members. Useful for complex
    """

    observation = observation.unsqueeze(0)

    #  ensemble size
    num_ensemble = forecasts.shape[0]

    # use broadcasting semantics to compute spread and skill
    espread = (
        (forecasts.unsqueeze(1) - forecasts.unsqueeze(0)).abs().sum(dim=(0, 1))
        * (float(num_ensemble) - 1.0 + alpha)
        / float(num_ensemble * num_ensemble * (num_ensemble - 1))
    )
    eskill = (observation - forecasts).abs().mean(dim=0)

    crps = eskill - 0.5 * espread

    return crps


# @torch.compile
def _crps_gauss_kernel(
    observation: torch.Tensor, forecasts: torch.Tensor, weights: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    CRPS Gauss score, assuming the input ensemble is gaussian distributed
    disabling torch compile for the moment due to very long startup times when training large ensembles with ensemble parallelism
    """

    # compute mean var over observations
    mu = torch.mean(forecasts * weights, dim=0)
    sigma = torch.sqrt(torch.mean(torch.square(forecasts - mu.unsqueeze(0)) * weights, dim=0))

    # protect against too small standard deviations
    sigma = torch.clamp(sigma, min=eps)

    # compute normalized observation
    obs_norm = (observation - mu) / sigma

    # compute normalization
    sqrtpi_inv = 1.0 / np.sqrt(np.pi)
    sqrt2_inv = 1.0 / np.sqrt(2.0)

    # compute PDF and CDF
    pdf = sqrtpi_inv * sqrt2_inv * torch.exp(-0.5 * torch.square(obs_norm))
    cdf2m1 = torch.erf(obs_norm * sqrt2_inv)

    # compute score
    crps = sigma * (obs_norm * cdf2m1 + 2.0 * pdf - sqrtpi_inv)

    return crps


class CRPSLoss(GeometricBaseLoss):

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        crps_type: str = "skillspread",
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        ensemble_weights: Optional[torch.Tensor] = None,
        alpha: Optional[float] = 1.0,
        eps: Optional[float] = 1.0e-6,
        **kwargs,
    ):

        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            spatial_distributed=spatial_distributed,
        )

        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed
        self.ensemble_distributed = (
            comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1) and ensemble_distributed
        )
        self.crps_type = crps_type
        self.alpha = alpha
        self.eps = eps

        if (self.crps_type not in ("skillspread", "naive skillspread")) and (self.alpha < 1.0):
            raise NotImplementedError(
                "The alpha parameter (almost fair CRPS factor) is only supported for the skillspread kernels."
            )

        # we also need a variant of the weights split in ensemble direction:
        quad_weight_split = self.quadrature.quad_weight.reshape(1, 1, -1)
        if self.ensemble_distributed:
            quad_weight_split = split_tensor_along_dim(quad_weight_split, dim=-1, num_chunks=comm.get_size("ensemble"))[
                comm.get_rank("ensemble")
            ]
        quad_weight_split = quad_weight_split.contiguous()
        self.register_buffer("quad_weight_split", quad_weight_split, persistent=False)

        if ensemble_weights is not None:
            self.register_buffer("ensemble_weights", ensemble_weights, persistent=False)
        else:
            self.ensemble_weights = ensemble_weights

    @property
    def type(self):
        return LossType.Probabilistic

    def forward(
        self,
        forecasts: torch.Tensor,
        observations: torch.Tensor,
        spatial_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that spatial_weights have NO ensemble dim
        if (spatial_weights is not None) and (spatial_weights.dim() != observations.dim()):
            spdim = spatial_weights.dim()
            odim = observations.dim()
            raise ValueError(
                f"the weights have to have the same number of dimensions (found {spdim}) as observations (found {odim})."
            )

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, lat, lon
        # observations: batch, channels, lat, lon
        B, E, C, H, W = forecasts.shape

        # if ensemble dim is one dimensional then computing the score is quick:
        if (not self.ensemble_distributed) and (E == 1):
            # in this case, CRPS is straightforward
            crps = torch.abs(observations - forecasts.squeeze(1)).reshape(B, C, H * W)
        else:
            # transpose forecasts: ensemble, batch, channels, lat, lon
            forecasts = torch.moveaxis(forecasts, 1, 0)

            # now we need to transpose the forecasts into ensemble direction.
            # ideally we split spatial dims
            forecasts = forecasts.reshape(E, B, C, H * W)
            if self.ensemble_distributed:
                ensemble_shapes = [forecasts.shape[0] for _ in range(comm.get_size("ensemble"))]
                forecasts = distributed_transpose(forecasts, (-1, 0), ensemble_shapes, "ensemble")
            # observations does not need a transpose, but just a split
            observations = observations.reshape(B, C, H * W)
            if self.ensemble_distributed:
                observations = scatter_to_parallel_region(observations, -1, "ensemble")
            if spatial_weights is not None:
                spatial_weights_split = spatial_weights.flatten(start_dim=-2, end_dim=-1)
                if self.ensemble_distributed:
                    spatial_weights_split = scatter_to_parallel_region(spatial_weights_split, -1, "ensemble")
            else:
                spatial_weights_split = None

            # run appropriate crps kernel to compute it pointwise
            if self.crps_type == "cdf":
                # now, E dimension is local and spatial dim is split further
                # we need to sort the forecasts now
                forecasts, idx = torch.sort(forecasts, dim=0)  # how does the sorting work out if it is batched
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_ensemble_kernel(observations, forecasts, ensemble_weights)
            elif self.crps_type == "probability weighted moment":
                # now, E dimension is local and spatial dim is split further
                # we need to sort the forecasts now
                forecasts, idx = torch.sort(forecasts, dim=0)
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_probability_weighted_moment_kernel(observations, forecasts, ensemble_weights)
            elif self.crps_type == "skillspread":
                if self.ensemble_weights is not None:
                    raise NotImplementedError("currently only constant ensemble weights are supported")
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_skillspread_kernel(observations, forecasts, ensemble_weights, self.alpha)
            elif self.crps_type == "naive skillspread":
                if self.ensemble_weights is not None:
                    raise NotImplementedError("currently only constant ensemble weights are supported")
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_naive_skillspread_kernel(observations, forecasts, ensemble_weights, self.alpha)
            elif self.crps_type == "gauss":
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_gauss_kernel(observations, forecasts, ensemble_weights, self.eps)
            else:
                raise ValueError(f"Unknown CRPS crps_type {self.crps_type}")

        # perform ensemble and spatial average of crps score
        if spatial_weights is not None:
            crps = torch.sum(crps * self.quad_weight_split * spatial_weights_split, dim=-1)
        else:
            crps = torch.sum(crps * self.quad_weight_split, dim=-1)

        # since we split spatial dim into ensemble dim, we need to do an ensemble sum as well
        if self.ensemble_distributed:
            crps = reduce_from_parallel_region(crps, "ensemble")

        # we need to do the spatial averaging manually since
        # we are not calling he quadrature forward function
        if self.spatial_distributed:
            crps = reduce_from_parallel_region(crps, "spatial")

        # the resulting tensor should have dimension B, C, which is what we return
        return crps


class SpectralCRPSLoss(SpectralBaseLoss):

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        lmax: Optional[int] = None,
        crps_type: str = "skillspread",
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        ensemble_weights: Optional[torch.Tensor] = None,
        absolute: Optional[bool] = True,
        alpha: Optional[float] = 1.0,
        eps: Optional[float] = 1.0e-6,
        **kwargs,
    ):

        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            lmax=lmax,
            spatial_distributed=spatial_distributed,
        )

        self.spatial_distributed = spatial_distributed and comm.is_distributed("spatial")
        self.ensemble_distributed = (
            ensemble_distributed and comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)
        )
        self.crps_type = crps_type
        self.alpha = alpha
        self.eps = eps

        if (self.crps_type not in ("skillspread", "naive skillspread")) and (self.alpha < 1.0):
            raise NotImplementedError(
                "The alpha parameter (almost fair CRPS factor) is only supported for the skillspread kernels."
            )

        if ensemble_weights is not None:
            self.register_buffer("ensemble_weights", ensemble_weights, persistent=False)
        else:
            self.ensemble_weights = ensemble_weights

        # if absolute is true, the loss is computed only on the absolute value of the spectral coefficient
        self.absolute = absolute

    @property
    def type(self):
        return LossType.Probabilistic

    def forward(
        self,
        forecasts: torch.Tensor,
        observations: torch.Tensor,
        spectral_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that spatial_weights have NO ensemble dim
        if (spectral_weights is not None) and (spectral_weights.dim() != observations.dim()):
            raise ValueError("the weights have to have the same number of dimensions as observations")

        # get the data type before stripping amp types
        dtype = forecasts.dtype

        # before anything else compute the transform
        # as the CDF definition doesn't generalize well to more than one-dimensional variables, we treat complex and imaginary part as the same
        with amp.autocast(device_type=forecasts.device.type, enabled=False):
            forecasts = forecasts.to(torch.float32)
            observations = observations.to(torch.float32)
            forecasts = self.sht(forecasts) / math.sqrt(4.0 * math.pi)
            observations = self.sht(observations) / math.sqrt(4.0 * math.pi)

        if self.absolute:
            forecasts = torch.abs(forecasts).to(dtype)
            observations = torch.abs(observations).to(dtype)
        else:
            # since the other kernels require sorting, this approach only works with the naive CRPS kernel
            if self.crps_type != "skillspread":
                raise ValueError(
                    f"the non-absolute path only works with the naive 'skillspread' CRPS kernel, but got crps_type {self.crps_type}"
                )

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, mmax, lmax
        # observations: batch, channels, mmax, lmax
        B, E, C, H, W = forecasts.shape

        # always use lm_weights
        if spectral_weights is None:
            spectral_weights = self.lm_weights
        else:
            spectral_weights = spectral_weights * self.lm_weights

        # if ensemble dim is one dimensional then computing the score is quick:
        if (not self.ensemble_distributed) and (E == 1):
            # in this case, CRPS is straightforward
            crps = torch.abs(observations - forecasts.squeeze(1)).reshape(B, C, H * W)
            spectral_weights_split = spectral_weights.reshape(1, 1, H * W)
        else:
            # transpose forecasts: ensemble, batch, channels, lat, lon
            forecasts = torch.movedim(forecasts, 1, 0)

            # now we need to transpose the forecasts into ensemble direction.
            # ideally we split spatial dims
            forecasts = forecasts.reshape(E, B, C, H * W)
            if self.ensemble_distributed:
                ensemble_shapes = [forecasts.shape[0] for _ in range(comm.get_size("ensemble"))]
                forecasts = distributed_transpose(forecasts, (-1, 0), ensemble_shapes, "ensemble")
            # observations does not need a transpose, but just a split
            observations = observations.reshape(B, C, H * W)
            if self.ensemble_distributed:
                observations = scatter_to_parallel_region(observations, -1, "ensemble")

            # tile in complex dim, then flatten last 3 dims
            spectral_weights_split = spectral_weights.reshape(1, 1, H * W)
            if self.ensemble_distributed:
                spectral_weights_split = scatter_to_parallel_region(spectral_weights_split, -1, "ensemble")

            # run appropriate crps kernel to compute it pointwise
            if self.crps_type == "cdf":
                # now, E dimension is local and spatial dim is split further
                # we need to sort the forecasts now
                forecasts, idx = torch.sort(forecasts, dim=0)
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_ensemble_kernel(observations, forecasts, ensemble_weights)
            elif self.crps_type == "probability weighted moment":
                # now, E dimension is local and spatial dim is split further
                # we need to sort the forecasts now
                forecasts, idx = torch.sort(forecasts, dim=0)
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_probability_weighted_moment_kernel(observations, forecasts, ensemble_weights)
            elif self.crps_type == "skillspread":
                if self.ensemble_weights is not None:
                    raise NotImplementedError("currently only constant ensemble weights are supported")
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                if self.absolute:
                    crps = _crps_skillspread_kernel(observations, forecasts, ensemble_weights, self.alpha)
                else:
                    crps = _crps_naive_skillspread_kernel(observations, forecasts, ensemble_weights, self.alpha)
            elif self.crps_type == "gauss":
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_gauss_kernel(observations, forecasts, ensemble_weights, self.eps)
            else:
                raise ValueError(f"Unknown CRPS crps_type {self.crps_type}")

        # perform spatial average of crps score
        crps = torch.sum(crps * spectral_weights_split, dim=-1)

        if self.spatial_distributed:
            crps = reduce_from_parallel_region(crps, "spatial")

        if self.ensemble_distributed:
            crps = reduce_from_parallel_region(crps, "ensemble")

        # the resulting tensor should have dimension B, C, which is what we return
        return crps


class GradientCRPSLoss(GradientBaseLoss):

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        lmax: Optional[int] = None,
        crps_type: str = "skillspread",
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        ensemble_weights: Optional[torch.Tensor] = None,
        absolute: Optional[bool] = True,
        alpha: Optional[float] = 1.0,
        eps: Optional[float] = 1.0e-6,
        **kwargs,
    ):

        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            lmax=lmax,
            spatial_distributed=spatial_distributed,
        )

        # if absolute is true, the loss is computed only on the absolute value of the gradient
        self.absolute = absolute

        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed
        self.ensemble_distributed = (
            comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1) and ensemble_distributed
        )
        self.crps_type = crps_type
        self.alpha = alpha
        self.eps = eps

        if (self.crps_type not in ("skillspread", "naive skillspread")) and (self.alpha < 1.0):
            raise NotImplementedError(
                "The alpha parameter (almost fair CRPS factor) is only supported for the skillspread kernels."
            )

        # we also need a variant of the weights split in ensemble direction:
        quad_weight_split = self.quadrature.quad_weight.reshape(1, 1, -1)
        if self.ensemble_distributed:
            quad_weight_split = split_tensor_along_dim(quad_weight_split, dim=-1, num_chunks=comm.get_size("ensemble"))[
                comm.get_rank("ensemble")
            ]
        quad_weight_split = quad_weight_split.contiguous()
        self.register_buffer("quad_weight_split", quad_weight_split, persistent=False)

        if ensemble_weights is not None:
            self.register_buffer("ensemble_weights", ensemble_weights, persistent=False)
        else:
            self.ensemble_weights = ensemble_weights

    @property
    def type(self):
        return LossType.Probabilistic

    @property
    def n_channels(self):
        if self.absolute:
            return len(self.channel_names)
        else:
            return 2 * len(self.channel_names)

    @torch.compiler.disable(recursive=False)
    def compute_channel_weighting(self, channel_weight_type: str, time_diff_scale: torch.Tensor = None) -> torch.Tensor:
        chw = super().compute_channel_weighting(channel_weight_type, time_diff_scale=time_diff_scale)

        if self.absolute:
            return chw
        else:
            return [weight for weight in chw for _ in range(2)]

    def forward(
        self,
        forecasts: torch.Tensor,
        observations: torch.Tensor,
        spatial_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that spatial_weights have NO ensemble dim
        if (spatial_weights is not None) and (spatial_weights.dim() != observations.dim()):
            spdim = spatial_weights.dim()
            odim = observations.dim()
            raise ValueError(
                f"the weights have to have the same number of dimensions (found {spdim}) as observations (found {odim})."
            )

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, lat, lon
        # observations: batch, channels, lat, lon
        B, E, C, H, W = forecasts.shape

        # get the data type before stripping amp types
        dtype = forecasts.dtype

        # before anything else compute the transform
        # as the CDF definition doesn't generalize well to more than one-dimensional variables, we treat complex and imaginary part as the same
        with amp.autocast(device_type="cuda", enabled=False):

            # compute the SH coefficients of the forecasts and observations
            forecasts = self.sht(forecasts.float()).unsqueeze(-3)
            observations = self.sht(observations.float()).unsqueeze(-3)

            # append zeros, so that we can use the inverse vector SHT
            forecasts = torch.cat([forecasts, torch.zeros_like(forecasts)], dim=-3)
            observations = torch.cat([observations, torch.zeros_like(observations)], dim=-3)

            forecasts = self.ivsht(forecasts)
            observations = self.ivsht(observations)

        forecasts = forecasts.to(dtype)
        observations = observations.to(dtype)

        if self.absolute:
            forecasts = forecasts.pow(2).sum(dim=-3).sqrt()
            observations = observations.pow(2).sum(dim=-3).sqrt()
        else:
            C = 2 * C

        forecasts = forecasts.reshape(B, E, C, H, W)
        observations = observations.reshape(B, C, H, W)

        # if ensemble dim is one dimensional then computing the score is quick:
        if (not self.ensemble_distributed) and (E == 1):
            # in this case, CRPS is straightforward
            crps = torch.abs(observations - forecasts.squeeze(1)).reshape(B, C, H * W)
        else:
            # transpose forecasts: ensemble, batch, channels, lat, lon
            forecasts = torch.moveaxis(forecasts, 1, 0)

            # now we need to transpose the forecasts into ensemble direction.
            # ideally we split spatial dims
            forecasts = forecasts.reshape(E, B, C, H * W)
            if self.ensemble_distributed:
                ensemble_shapes = [forecasts.shape[0] for _ in range(comm.get_size("ensemble"))]
                forecasts = distributed_transpose(forecasts, (-1, 0), ensemble_shapes, "ensemble")
            # observations does not need a transpose, but just a split
            observations = observations.reshape(B, C, H * W)
            if self.ensemble_distributed:
                observations = scatter_to_parallel_region(observations, -1, "ensemble")
            if spatial_weights is not None:
                spatial_weights_split = spatial_weights.flatten(start_dim=-2, end_dim=-1)
                spatial_weights_split = scatter_to_parallel_region(spatial_weights_split, -1, "ensemble")

            # run appropriate crps kernel to compute it pointwise
            if self.crps_type == "cdf":
                # now, E dimension is local and spatial dim is split further
                # we need to sort the forecasts now
                forecasts, idx = torch.sort(forecasts, dim=0)
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_ensemble_kernel(observations, forecasts, ensemble_weights)
            elif self.crps_type == "skillspread":
                if self.ensemble_weights is not None:
                    raise NotImplementedError("currently only constant ensemble weights are supported")
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_skillspread_kernel(observations, forecasts, ensemble_weights, self.alpha)
            elif self.crps_type == "gauss":
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_gauss_kernel(observations, forecasts, ensemble_weights, self.eps)
            else:
                raise ValueError(f"Unknown CRPS crps_type {self.crps_type}")

        # perform ensemble and spatial average of crps score
        if spatial_weights is not None:
            crps = torch.sum(crps * self.quad_weight_split * spatial_weights_split, dim=-1)
        else:
            crps = torch.sum(crps * self.quad_weight_split, dim=-1)

        # since we split spatial dim into ensemble dim, we need to do an ensemble sum as well
        if self.ensemble_distributed:
            crps = reduce_from_parallel_region(crps, "ensemble")

        # we need to do the spatial averaging manually since
        # we are not calling he quadrature forward function
        if self.spatial_distributed:
            crps = reduce_from_parallel_region(crps, "spatial")

        # the resulting tensor should have dimension B, C, which is what we return
        return crps


class VortDivCRPSLoss(VortDivBaseLoss):

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        crps_type: str = "skillspread",
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        ensemble_weights: Optional[torch.Tensor] = None,
        alpha: Optional[float] = 1.0,
        eps: Optional[float] = 1.0e-6,
        **kwargs,
    ):

        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            spatial_distributed=spatial_distributed,
        )

        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed
        self.ensemble_distributed = (
            comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1) and ensemble_distributed
        )
        self.crps_type = crps_type
        self.alpha = alpha
        self.eps = eps

        if (self.crps_type not in ("skillspread", "naive skillspread")) and (self.alpha < 1.0):
            raise NotImplementedError(
                "The alpha parameter (almost fair CRPS factor) is only supported for the skillspread kernels."
            )

        # we also need a variant of the weights split in ensemble direction:
        quad_weight_split = self.quadrature.quad_weight.reshape(1, 1, -1)
        if self.ensemble_distributed:
            quad_weight_split = split_tensor_along_dim(quad_weight_split, dim=-1, num_chunks=comm.get_size("ensemble"))[
                comm.get_rank("ensemble")
            ]
        quad_weight_split = quad_weight_split.contiguous()
        self.register_buffer("quad_weight_split", quad_weight_split, persistent=False)

        if ensemble_weights is not None:
            self.register_buffer("ensemble_weights", ensemble_weights, persistent=False)
        else:
            self.ensemble_weights = ensemble_weights

    @property
    def type(self):
        return LossType.Probabilistic

    def forward(
        self,
        forecasts: torch.Tensor,
        observations: torch.Tensor,
        spatial_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that spatial_weights have NO ensemble dim
        if (spatial_weights is not None) and (spatial_weights.dim() != observations.dim()):
            spdim = spatial_weights.dim()
            odim = observations.dim()
            raise ValueError(
                f"the weights have to have the same number of dimensions (found {spdim}) as observations (found {odim})."
            )

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, lat, lon
        # observations: batch, channels, lat, lon
        B, E, C, H, W = forecasts.shape
        Cw = self.wind_chans.shape[0]

        # Transform the wind channels (u, v pairs) into vorticity/divergence space via
        # the vector-SHT round-trip and scatter them back in place. Every other (scalar)
        # channel is passed through unchanged so it still contributes to the score
        # instead of being silently dropped (see GitHub issue #94).
        # The CDF definition doesn't generalize well to multi-dimensional variables, so
        # the two vector components are treated independently.
        with amp.autocast(device_type="cuda", enabled=False):
            forecasts = forecasts.float()
            observations = observations.float()

            fc_wind = forecasts[..., self.wind_chans, :, :].reshape(B, E, Cw // 2, 2, H, W)
            ob_wind = observations[..., self.wind_chans, :, :].reshape(B, Cw // 2, 2, H, W)
            fc_wind = self.isht(self.vsht(fc_wind)).reshape(B, E, Cw, H, W)
            ob_wind = self.isht(self.vsht(ob_wind)).reshape(B, Cw, H, W)

            # out-of-place scatter back into the full channel set (original order)
            forecasts = forecasts.index_copy(-3, self.wind_chans, fc_wind)
            observations = observations.index_copy(-3, self.wind_chans, ob_wind)

        # if ensemble dim is one dimensional then computing the score is quick:
        if (not self.ensemble_distributed) and (E == 1):
            # in this case, CRPS is straightforward
            crps = torch.abs(observations - forecasts.squeeze(1)).reshape(B, C, H * W)
        else:
            # transpose forecasts: ensemble, batch, channels, lat, lon
            forecasts = torch.moveaxis(forecasts, 1, 0)

            # now we need to transpose the forecasts into ensemble direction.
            # ideally we split spatial dims
            forecasts = forecasts.reshape(E, B, C, H * W)
            if self.ensemble_distributed:
                ensemble_shapes = [forecasts.shape[0] for _ in range(comm.get_size("ensemble"))]
                forecasts = distributed_transpose(forecasts, (-1, 0), ensemble_shapes, "ensemble")
            # observations does not need a transpose, but just a split
            observations = observations.reshape(B, C, H * W)
            if self.ensemble_distributed:
                observations = scatter_to_parallel_region(observations, -1, "ensemble")
            if spatial_weights is not None:
                spatial_weights_split = spatial_weights.flatten(start_dim=-2, end_dim=-1)
                spatial_weights_split = scatter_to_parallel_region(spatial_weights_split, -1, "ensemble")

            # run appropriate crps kernel to compute it pointwise
            if self.crps_type == "cdf":
                # now, E dimension is local and spatial dim is split further
                # we need to sort the forecasts now
                forecasts, idx = torch.sort(forecasts, dim=0)
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_ensemble_kernel(observations, forecasts, ensemble_weights)
            elif self.crps_type == "skillspread":
                if self.ensemble_weights is not None:
                    raise NotImplementedError("currently only constant ensemble weights are supported")
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_skillspread_kernel(observations, forecasts, ensemble_weights, self.alpha)
            elif self.crps_type == "gauss":
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_gauss_kernel(observations, forecasts, ensemble_weights, self.eps)
            else:
                raise ValueError(f"Unknown CRPS crps_type {self.crps_type}")

        # perform ensemble and spatial average of crps score
        if spatial_weights is not None:
            crps = torch.sum(crps * self.quad_weight_split * spatial_weights_split, dim=-1)
        else:
            crps = torch.sum(crps * self.quad_weight_split, dim=-1)

        # since we split spatial dim into ensemble dim, we need to do an ensemble sum as well
        if self.ensemble_distributed:
            crps = reduce_from_parallel_region(crps, "ensemble")

        # we need to do the spatial averaging manually since
        # we are not calling he quadrature forward function
        if self.spatial_distributed:
            crps = reduce_from_parallel_region(crps, "spatial")

        # the resulting tensor should have dimension B, C, which is what we return
        return crps


class KernelScoreLoss(GeometricBaseLoss):
    """
    Computes the kernel score defined in Gneiting and Raftery (2007) with kernels
    defined by the discrete-continuous convolutions.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        crps_type: str = "skillspread",
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        ensemble_weights: Optional[torch.Tensor] = None,
        alpha: Optional[float] = 1.0,
        eps: Optional[float] = 1.0e-6,
        kernel_basis_type: str = "harmonic",
        kernel_basis_norm_mode: str = "nodal",
        kernel_shape: Tuple[int, int] = (3, 3),
        **kwargs,
    ):

        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            spatial_distributed=spatial_distributed,
        )

        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed
        self.ensemble_distributed = (
            comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1) and ensemble_distributed
        )
        self.crps_type = crps_type
        self.alpha = alpha
        self.eps = eps

        if (self.crps_type not in ("skillspread", "naive skillspread")) and (self.alpha < 1.0):
            raise NotImplementedError(
                "The alpha parameter (almost fair CRPS factor) is only supported for the skillspread kernels."
            )

        # we also need a variant of the weights split in ensemble direction:
        quad_weight_split = self.quadrature.quad_weight.reshape(1, 1, -1)
        if self.ensemble_distributed:
            quad_weight_split = split_tensor_along_dim(quad_weight_split, dim=-1, num_chunks=comm.get_size("ensemble"))[
                comm.get_rank("ensemble")
            ]
        quad_weight_split = quad_weight_split.contiguous()
        self.register_buffer("quad_weight_split", quad_weight_split, persistent=False)

        if ensemble_weights is not None:
            self.register_buffer("ensemble_weights", ensemble_weights, persistent=False)
        else:
            self.ensemble_weights = ensemble_weights

        # init distributed torch-harmonics if needed
        if self.spatial_distributed and (comm.get_size("spatial") > 1):
            if not thd.is_initialized():
                polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                thd.init(polar_group, azimuth_group)

        # set up DISCO convolution (one per kernel)

        fb = th.filter_basis.get_filter_basis(tuple(kernel_shape), kernel_basis_type)
        self.kernel_basis_size = fb.kernel_size

        self.conv = get_disco_conv(
            self.n_channels,
            self.n_channels * self.kernel_basis_size,
            distributed=self.spatial_distributed,
            in_shape=img_shape,
            out_shape=img_shape,
            kernel_shape=tuple(kernel_shape),
            basis_type=kernel_basis_type,
            basis_norm_mode=kernel_basis_norm_mode,
            grid_in=grid_type,
            grid_out=grid_type,
            groups=self.n_channels,
            bias=False,
            theta_cutoff=2 * kernel_shape[0] * math.pi / float(img_shape[0] - 1),
        )

        # initialize the weight to identity
        weight = torch.zeros_like(self.conv.weight.data)
        for i in range(self.n_channels):
            for k in range(self.kernel_basis_size):
                weight[i * self.kernel_basis_size + k, 0, k] = 1.0

        # convert weight to buffer to avoid issues with distributed training
        delattr(self.conv, "weight")
        self.conv.register_buffer("weight", weight)

    @property
    def type(self):
        return LossType.Probabilistic

    def forward(
        self,
        forecasts: torch.Tensor,
        observations: torch.Tensor,
        spatial_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that spatial_weights have NO ensemble dim
        if (spatial_weights is not None) and (spatial_weights.dim() != observations.dim()):
            spdim = spatial_weights.dim()
            odim = observations.dim()
            raise ValueError(
                f"the weights have to have the same number of dimensions (found {spdim}) as observations (found {odim})."
            )

        # get the data type before stripping amp types
        dtype = forecasts.dtype

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, lat, lon
        # observations: batch, channels, lat, lon
        B, E, _, H, W = forecasts.shape

        # before anything else compute the transform
        # as the CDF definition doesn't generalize well to more than one-dimensional variables, we treat complex and imaginary part as the same
        with amp.autocast(device_type="cuda", enabled=False):
            forecasts = self.conv(forecasts.float().reshape(B * E, -1, H, W))
            observations = self.conv(observations.float())

        forecasts = forecasts.reshape(B, E, -1, H, W)
        C = forecasts.shape[2]

        # if ensemble dim is one dimensional then computing the score is quick:
        if (not self.ensemble_distributed) and (E == 1):
            # in this case, CRPS is straightforward
            crps = torch.abs(observations - forecasts.squeeze(1)).reshape(B, C, H * W)
        else:
            # transpose forecasts: ensemble, batch, channels, lat, lon
            forecasts = torch.moveaxis(forecasts, 1, 0)

            # now we need to transpose the forecasts into ensemble direction.
            # ideally we split spatial dims
            forecasts = forecasts.reshape(E, B, C, H * W)
            if self.ensemble_distributed:
                ensemble_shapes = [forecasts.shape[0] for _ in range(comm.get_size("ensemble"))]
                forecasts = distributed_transpose(forecasts, (-1, 0), ensemble_shapes, "ensemble")
            # observations does not need a transpose, but just a split
            observations = observations.reshape(B, C, H * W)
            if self.ensemble_distributed:
                observations = scatter_to_parallel_region(observations, -1, "ensemble")
            if spatial_weights is not None:
                spatial_weights_split = spatial_weights.flatten(start_dim=-2, end_dim=-1)
                spatial_weights_split = scatter_to_parallel_region(spatial_weights_split, -1, "ensemble")

            # run appropriate crps kernel to compute it pointwise
            if self.crps_type == "cdf":
                # now, E dimension is local and spatial dim is split further
                # we need to sort the forecasts now
                forecasts, idx = torch.sort(forecasts, dim=0)  # how does the sorting work out if it is batched
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_ensemble_kernel(observations, forecasts, ensemble_weights)
            elif self.crps_type == "probability weighted moment":
                # now, E dimension is local and spatial dim is split further
                # we need to sort the forecasts now
                forecasts, idx = torch.sort(forecasts, dim=0)
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_probability_weighted_moment_kernel(observations, forecasts, ensemble_weights)
            elif self.crps_type == "skillspread":
                if self.ensemble_weights is not None:
                    raise NotImplementedError("currently only constant ensemble weights are supported")
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_skillspread_kernel(observations, forecasts, ensemble_weights, self.alpha)
            elif self.crps_type == "naive skillspread":
                if self.ensemble_weights is not None:
                    raise NotImplementedError("currently only constant ensemble weights are supported")
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_naive_skillspread_kernel(observations, forecasts, ensemble_weights, self.alpha)
            else:
                raise ValueError(f"Unknown CRPS crps_type {self.crps_type}")

        # perform ensemble and spatial average of crps score
        if spatial_weights is not None:
            crps = torch.sum(crps * self.quad_weight_split * spatial_weights_split, dim=-1)
        else:
            crps = torch.sum(crps * self.quad_weight_split, dim=-1)

        # since we split spatial dim into ensemble dim, we need to do an ensemble sum as well
        if self.ensemble_distributed:
            crps = reduce_from_parallel_region(crps, "ensemble")

        # we need to do the spatial averaging manually since
        # we are not calling he quadrature forward function
        if self.spatial_distributed:
            crps = reduce_from_parallel_region(crps, "spatial")

        # reduce the kernel dimensions
        crps = crps.reshape(B, self.n_channels, self.kernel_basis_size).sum(dim=-1)

        # the resulting tensor should have dimension B, C, which is what we return
        return crps
