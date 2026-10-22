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

from typing import Optional, Tuple

import torch

from makani.utils import comm
from makani.mpu.mappings import scatter_to_parallel_region, reduce_from_parallel_region, distributed_transpose
from torch_harmonics.distributed import split_tensor_along_dim

from makani.utils.losses import CRPSLoss, LossType
from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature
from makani.utils.metrics.base_metric import _sanitize_shapes, _welford_reduction_helper, GeometricBaseMetric


class GeometricL1(GeometricBaseMetric):
    def __init__(
        self,
        grid_type: str,
        img_shape: Tuple[int, int],
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        spatial_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            grid_type=grid_type,
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=normalize,
            channel_reduction=channel_reduction,
            batch_reduction=batch_reduction,
            spatial_distributed=spatial_distributed,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:

        if weight is not None:
            diff = self.quadrature(torch.abs(x - y) * weight)
        else:
            diff = self.quadrature(torch.abs(x - y))

        # reduce:
        if self.channel_reduction == "mean":
            diff = torch.mean(diff, dim=1)
        elif self.channel_reduction == "sum":
            diff = torch.sum(diff, dim=1)

        if self.batch_reduction == "mean":
            diff = torch.mean(diff, dim=0)
        elif self.batch_reduction == "sum":
            diff = torch.sum(diff, dim=0)

        return diff


class GeometricRMSE(GeometricBaseMetric):
    def __init__(
        self,
        grid_type: str,
        img_shape: Tuple[int, int],
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        spatial_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            grid_type=grid_type,
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=normalize,
            channel_reduction=channel_reduction,
            batch_reduction=batch_reduction,
            spatial_distributed=spatial_distributed,
        )

    def combine(self, vals, counts, dim=0):
        vals, counts = _sanitize_shapes(vals, counts, dim=dim)
        vals_res, counts_res = _welford_reduction_helper(torch.square(vals), counts, self.batch_reduction, dim=dim)
        vals_res = torch.sqrt(vals_res)
        return vals_res, counts_res

    def finalize(self, vals, counts):
        if self.batch_reduction == "mean":
            return vals
        else:
            return vals / torch.sqrt(counts)

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:

        if weight is not None:
            diff = self.quadrature(torch.square(x - y) * weight)
        else:
            diff = self.quadrature(torch.square(x - y))

        # reduce channels:
        if self.channel_reduction == "mean":
            diff = torch.mean(diff, dim=1)
        elif self.channel_reduction == "sum":
            diff = torch.sum(diff, dim=1)

        # reduce batch:
        if self.batch_reduction == "mean":
            diff = torch.mean(diff, dim=0)
        elif self.batch_reduction == "sum":
            diff = torch.sum(diff, dim=0)

        # compute square root:
        result = torch.sqrt(diff)

        return result


class GeometricACC(GeometricBaseMetric):
    # the ACC of a sample is a ratio of integrals over the same weights
    normalized_by_weights = True

    def __init__(
        self,
        grid_type: str,
        img_shape: Tuple[int, int],
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        method: Optional[str] = "macro",
        bias: Optional[torch.Tensor] = None,
        eps: Optional[float] = 1e-8,
        spatial_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            grid_type=grid_type,
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=normalize,
            channel_reduction=channel_reduction,
            batch_reduction=batch_reduction,
            spatial_distributed=spatial_distributed,
        )

        self.method = method
        self.eps = eps

        if bias is not None:
            if comm.get_size("w") > 1:
                bias = split_tensor_along_dim(bias, dim=-1, num_chunks=comm.get_size("w"))[comm.get_rank("w")]
            if comm.get_size("h") > 1:
                bias = split_tensor_along_dim(bias, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]
            self.register_buffer("bias", bias, persistent=False)

    def compute_counts(self, inp: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        counts = super().compute_counts(inp, weight)
        if self.method == "micro":
            counts = counts.unsqueeze(-1)
        return counts

    def finalize(self, vals, counts):
        if self.method == "micro":
            return vals[..., 0] / torch.sqrt(vals[..., 1] * vals[..., 2])
        else:
            return super().finalize(vals, counts)

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:

        if hasattr(self, "bias"):
            x = x - self.bias
            y = y - self.bias

        if weight is not None:
            cov_xy = self.quadrature(x * y * weight)
            var_x = self.quadrature(torch.square(x) * weight)
            var_y = self.quadrature(torch.square(y) * weight)
        else:
            cov_xy = self.quadrature(x * y)
            var_x = self.quadrature(torch.square(x))
            var_y = self.quadrature(torch.square(y))

        # compute ratio
        if self.method == "macro":
            acc = cov_xy / (torch.sqrt(var_x * var_y) + self.eps)
        else:
            # stack along dim -1:
            # we form the ratio in the finalization step
            acc = torch.stack([cov_xy, var_x, var_y], dim=-1)

        # reduce
        if self.channel_reduction == "mean":
            acc = torch.mean(acc, dim=1)
        elif self.channel_reduction == "sum":
            acc = torch.sum(acc, dim=1)

        if self.batch_reduction == "mean":
            acc = torch.mean(acc, dim=0)
        elif self.batch_reduction == "sum":
            acc = torch.sum(acc, dim=0)

        return acc


class GeometricSpread(GeometricBaseMetric):
    # forward normalizes the weighted spread of a sample by the area of its weights
    normalized_by_weights = True

    def __init__(
        self,
        grid_type: str,
        img_shape: Tuple[int, int],
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        spatial_distributed: Optional[bool] = False,
        **kwargs,
    ):

        super().__init__(
            grid_type=grid_type,
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=normalize,
            channel_reduction=channel_reduction,
            batch_reduction=batch_reduction,
            spatial_distributed=spatial_distributed,
        )

    @property
    def type(self):
        return LossType.Probabilistic

    def compute_counts(self, inp: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight is not None:
            if self.batch_reduction == "mean":
                raise ValueError(
                    "Batch reduction mode 'mean' is not supported when weights are provided. Use 'sum' instead."
                )
            elif self.batch_reduction == "sum":
                counts = torch.sum(self.weighted_counts(weight.to(dtype=inp.dtype)), dim=0)
            else:
                raise ValueError(f"Batch reduction mode '{self.batch_reduction}' is not supported")
        else:
            if self.batch_reduction == "mean":
                counts = torch.ones(size=(inp.shape[2],), device=inp.device, dtype=inp.dtype)
            elif self.batch_reduction == "sum":
                counts = torch.full(size=(inp.shape[2],), fill_value=inp.shape[0], device=inp.device, dtype=inp.dtype)

        if self.channel_reduction == "mean":
            counts = torch.mean(counts, dim=0)
        elif self.channel_reduction == "sum":
            counts = torch.sum(counts, dim=0)

        return counts

    def forward(
        self, forecasts: torch.Tensor, observations: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that the input is 5 dimensional, where dim=1 is the ensemble dim
        # ensemble mean size:
        # by construction, the ensemble members are evenly distributed among the ranks
        ens_size = forecasts.shape[1] * comm.get_size("ensemble")

        # compute ensemble mean:
        predictions = torch.sum(forecasts, dim=1)
        if comm.get_size("ensemble") > 1:
            predictions = reduce_from_parallel_region(predictions, "ensemble")
        predictions = predictions / float(ens_size)

        # compute the local portion of skill and spread
        spread = torch.sum(torch.square(predictions.unsqueeze(1) - forecasts), dim=1)

        # compute quadrature, with weights normalized by the area they cover. A sample without
        # nonzero weights gets zero spread and counts zero, see weighted_counts
        if weight is not None:
            weight = weight.to(dtype=spread.dtype)
            area = torch.clamp(self.quadrature(weight), min=torch.finfo(spread.dtype).tiny)
            spread = self.quadrature(spread * weight) / area
        else:
            spread = self.quadrature(spread)

        # average spread over ensemble
        if comm.get_size("ensemble") > 1:
            spread = reduce_from_parallel_region(spread, "ensemble")
        spread = torch.sqrt(spread / float(ens_size - 1))

        # final reduction
        if self.channel_reduction == "mean":
            spread = torch.mean(spread, dim=1)
        elif self.channel_reduction == "sum":
            spread = torch.sum(spread, dim=1)

        if self.batch_reduction == "mean":
            spread = torch.mean(spread, dim=0)
        elif self.batch_reduction == "sum":
            spread = torch.sum(spread, dim=0)

        return spread


class GeometricSSR(GeometricBaseMetric):
    # the SSR of a sample is a ratio of integrals over the same weights
    normalized_by_weights = True

    def __init__(
        self,
        grid_type: str,
        img_shape: Tuple[int, int],
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        eps: Optional[float] = 1e-6,
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            grid_type=grid_type,
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=normalize,
            channel_reduction=channel_reduction,
            batch_reduction=batch_reduction,
            spatial_distributed=spatial_distributed,
        )

        self.ensemble_distributed = (
            ensemble_distributed and comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)
        )

        # regularizer
        self.eps = eps

    @property
    def type(self):
        return LossType.Probabilistic

    def compute_counts(self, inp: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight is not None:
            if self.batch_reduction == "mean":
                raise ValueError(
                    "Batch reduction mode 'mean' is not supported when weights are provided. Use 'sum' instead."
                )
            elif self.batch_reduction == "sum":
                counts = torch.sum(self.weighted_counts(weight.to(dtype=inp.dtype)), dim=0)
            else:
                raise ValueError(f"Batch reduction mode '{self.batch_reduction}' is not supported")
        else:
            if self.batch_reduction == "mean":
                counts = torch.ones(size=(inp.shape[2],), device=inp.device, dtype=inp.dtype)
            elif self.batch_reduction == "sum":
                counts = torch.full(size=(inp.shape[2],), fill_value=inp.shape[0], device=inp.device, dtype=inp.dtype)

        if self.channel_reduction == "mean":
            counts = torch.mean(counts, dim=0)
        elif self.channel_reduction == "sum":
            counts = torch.sum(counts, dim=0)

        return counts

    def forward(
        self, forecasts: torch.Tensor, observations: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that the input is 5 dimensional, where dim=1 is the ensemble dim
        # ensemble mean size:
        # by construction, the ensemble members are evenly distributed among the ranks
        ens_size = forecasts.shape[1] * comm.get_size("ensemble")

        # compute ensemble mean:
        predictions = torch.sum(forecasts, dim=1)
        if self.ensemble_distributed:
            predictions = reduce_from_parallel_region(predictions, "ensemble")
        predictions = predictions / float(ens_size)

        # compute the local portion of skill and spread
        skill = torch.square(predictions - observations)
        spread = torch.sum(torch.square(predictions.unsqueeze(1) - forecasts), dim=1)

        # average spread over ensemble
        if self.ensemble_distributed:
            spread = reduce_from_parallel_region(spread, "ensemble")
        spread = spread / float(ens_size - 1)

        # compute SSR for each node
        if weight is not None:
            skill = self.quadrature(skill * weight)
            spread = self.quadrature(spread * weight)
        else:
            skill = self.quadrature(skill)
            spread = self.quadrature(spread)

        # compute the SSR including weighting factor
        ssr = torch.sqrt(spread / torch.clamp(skill - spread / float(ens_size), min=self.eps))

        # reduce
        if self.channel_reduction == "mean":
            ssr = torch.mean(ssr, dim=1)
        elif self.channel_reduction == "sum":
            ssr = torch.sum(ssr, dim=1)

        if self.batch_reduction == "mean":
            ssr = torch.mean(ssr, dim=0)
        elif self.batch_reduction == "sum":
            ssr = torch.sum(ssr, dim=0)

        return ssr


class GeometricCRPS(torch.nn.Module):
    def __init__(
        self,
        grid_type: str,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        crps_type: Optional[str] = "skillspread",
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        ensemble_weights: Optional[torch.Tensor] = None,
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__()

        self.metric_func = CRPSLoss(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=[],
            grid_type=grid_type,
            crps_type=crps_type,
            spatial_distributed=spatial_distributed,
            ensemble_distributed=ensemble_distributed,
            ensemble_weights=ensemble_weights,
        )

        self.channel_reduction = channel_reduction
        self.batch_reduction = batch_reduction

        # get the quadrature
        quadrature_rule = grid_to_quadrature_rule(grid_type)
        self.quadrature = GridQuadrature(
            quadrature_rule,
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=True,
            distributed=spatial_distributed,
        )

    @property
    def type(self):
        return self.metric_func.type

    def compute_counts(self, inp: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight is not None:
            if self.batch_reduction == "mean":
                raise ValueError(
                    "Batch reduction mode 'mean' is not supported when weights are provided. Use 'sum' instead."
                )
            elif self.batch_reduction == "sum":
                counts = torch.sum(self.quadrature(weight.to(dtype=inp.dtype)), dim=0)
            else:
                raise ValueError(f"Batch reduction mode '{self.batch_reduction}' is not supported")
        else:
            if self.batch_reduction == "mean":
                counts = torch.ones(size=(inp.shape[2],), device=inp.device, dtype=inp.dtype)
            elif self.batch_reduction == "sum":
                counts = torch.full(size=(inp.shape[2],), fill_value=inp.shape[0], device=inp.device, dtype=inp.dtype)

        if self.channel_reduction == "mean":
            counts = torch.mean(counts, dim=0)
        elif self.channel_reduction == "sum":
            counts = torch.sum(counts, dim=0)

        return counts

    def combine(self, vals, counts, dim=0):
        vals, counts = _sanitize_shapes(vals, counts, dim=dim)
        vals_res, counts_res = _welford_reduction_helper(vals, counts, self.batch_reduction, dim=dim)
        return vals_res, counts_res

    def finalize(self, vals, counts):
        if self.batch_reduction == "mean":
            return vals
        else:
            return vals / counts

    def forward(
        self, forecasts: torch.Tensor, observations: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        crps = self.metric_func(forecasts, observations, weight)

        if self.channel_reduction == "mean":
            crps = torch.mean(crps, dim=1)
        elif self.channel_reduction == "sum":
            crps = torch.sum(crps, dim=1)

        if self.batch_reduction == "mean":
            crps = torch.mean(crps, dim=0)
        elif self.batch_reduction == "sum":
            crps = torch.sum(crps, dim=0)

        return crps


class GeometricRankHistogram(GeometricBaseMetric):
    def __init__(
        self,
        grid_type: str,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            grid_type=grid_type,
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=normalize,
            channel_reduction=channel_reduction,
            batch_reduction=batch_reduction,
            spatial_distributed=spatial_distributed,
        )

        self.ensemble_distributed = (
            comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1) and ensemble_distributed
        )

        # we also need a variant of the weights split in ensemble direction:
        quad_weight_split = self.quadrature.quad_weight.reshape(1, 1, -1, 1)
        if self.ensemble_distributed:
            quad_weight_split = split_tensor_along_dim(quad_weight_split, dim=-2, num_chunks=comm.get_size("ensemble"))[
                comm.get_rank("ensemble")
            ]
        quad_weight_split = quad_weight_split.contiguous()
        self.register_buffer("quad_weight_split", quad_weight_split, persistent=False)

    @property
    def type(self):
        return LossType.Probabilistic

    def compute_counts(self, inp: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight is not None:
            if self.batch_reduction == "mean":
                raise ValueError(
                    "Batch reduction mode 'mean' is not supported when weights are provided. Use 'sum' instead."
                )
            elif self.batch_reduction == "sum":
                counts = torch.sum(self.quadrature(weight.to(dtype=inp.dtype)), dim=0)
            else:
                raise ValueError(f"Batch reduction mode '{self.batch_reduction}' is not supported")
        else:
            if self.batch_reduction == "mean":
                counts = torch.ones(size=(inp.shape[2],), device=inp.device, dtype=inp.dtype)
            elif self.batch_reduction == "sum":
                counts = torch.full(size=(inp.shape[2],), fill_value=inp.shape[0], device=inp.device, dtype=inp.dtype)

        if self.channel_reduction == "mean":
            counts = torch.mean(counts, dim=0)
        elif self.channel_reduction == "sum":
            counts = torch.sum(counts, dim=0)

        # we need to add another dim:
        counts = counts.unsqueeze(-1)

        return counts

    def forward(
        self, forecasts: torch.Tensor, observations: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that weight have NO ensemble dim
        if (weight is not None) and (weight.dim() != observations.dim()):
            spdim = weight.dim()
            odim = observations.dim()
            raise ValueError(
                f"the weights have to have the same number of dimensions (found {spdim}) as observations (found {odim})."
            )

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, lat, lon
        # observations: batch, channels, lat, lon
        B, E, C, H, W = forecasts.shape
        forecasts = forecasts.reshape(B, E, C, H * W)
        observations = observations.reshape(B, C, H * W)

        if self.ensemble_distributed:
            ensemble_shapes = [E for _ in range(comm.get_size("ensemble"))]
            forecasts = distributed_transpose(forecasts, (-1, 1), ensemble_shapes, "ensemble")
            ensemble_size = E * comm.get_size("ensemble")
        else:
            ensemble_size = E

        # observations does not need a transpose, but just a split
        if self.ensemble_distributed:
            observations = scatter_to_parallel_region(observations, -1, "ensemble")
        if weight is not None:
            weight = weight.flatten(start_dim=-2, end_dim=-1)
            if self.ensemble_distributed:
                weight = scatter_to_parallel_region(weight, -1, "ensemble")

        # we need to have ensemble dim innermost
        forecasts = torch.moveaxis(forecasts, 1, -1)

        # sort ensemble members
        forecasts_sorted, _ = torch.sort(forecasts, dim=-1, descending=False, stable=True)

        # make everything contiguous
        forecasts_sorted = forecasts_sorted.contiguous()
        observations = observations.unsqueeze(-1).contiguous()
        insertions = torch.searchsorted(forecasts_sorted, observations, side="right").squeeze(-1)

        # one hot encode
        rankhist = torch.nn.functional.one_hot(insertions, num_classes=ensemble_size + 1).to(dtype=torch.float32)

        # do spatial contraction:
        if weight is not None:
            rankhist = torch.sum(rankhist * self.quad_weight_split * weight.unsqueeze(-1), dim=2)
        else:
            rankhist = torch.sum(rankhist * self.quad_weight_split, dim=2)

        # we need to do the spatial averaging manually since
        # we are not calling he quadrature forward function
        if self.spatial_distributed:
            rankhist = reduce_from_parallel_region(rankhist, "spatial")
        # since we split spatial dim into ensemble dim, we need to do an ensemble sum as well
        if self.ensemble_distributed:
            rankhist = reduce_from_parallel_region(rankhist, "ensemble")

        # reduction over channels
        if self.channel_reduction == "mean":
            rankhist = torch.mean(rankhist, dim=1)
        elif self.channel_reduction == "sum":
            rankhist = torch.sum(rankhist, dim=1)

        if self.batch_reduction == "mean":
            rankhist = torch.mean(rankhist, dim=0)
        elif self.batch_reduction == "sum":
            rankhist = torch.sum(rankhist, dim=0)

        return rankhist


class SimpsonQuadrature(torch.nn.Module):
    def __init__(self, num_intervals, interval_width, device):
        super(SimpsonQuadrature, self).__init__()

        # set up integration weights
        weights = [0.0 for _ in range(num_intervals + 1)]
        if num_intervals % 2 == 0:
            # Simpsons 1/3
            for j in range(1, (num_intervals // 2 + 1)):
                weights[2 * j - 2] += 1.0
                weights[2 * j - 1] += 4.0
                weights[2 * j] += 1.0
            self.weights = torch.tensor(weights, dtype=torch.float32, device=device)
            self.weights *= interval_width / 3.0
        else:
            raise NotImplementedError("Error, please specify an even number of intervals")

    def forward(self, x, dim=1):
        # reshape weights to handle channels
        shape = [1 for _ in range(x.dim())]
        shape[dim] = -1
        weights = torch.reshape(self.weights, shape)

        return torch.sum(x * weights, dim=dim)


class TrapezoidQuadrature(torch.nn.Module):
    def __init__(self, num_intervals, interval_width, device):
        super(TrapezoidQuadrature, self).__init__()

        # set up integration weights
        weights = [interval_width for _ in range(num_intervals + 1)]
        weights[0] *= 0.5
        weights[-1] *= 0.5
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)

    def forward(self, x, dim=1):
        shape = [1 for _ in range(x.dim())]
        shape[dim] = -1
        weights = torch.reshape(self.weights, shape)

        return torch.sum(x * weights, dim=dim)


class Quadrature(torch.nn.Module):
    def __init__(self, num_intervals, interval_width, device):
        super(Quadrature, self).__init__()
        if num_intervals % 2 == 0:
            self.quad = SimpsonQuadrature(num_intervals, interval_width, device)
        else:
            self.quad = TrapezoidQuadrature(num_intervals, interval_width, device)

    def forward(self, x, dim=1):
        return self.quad(x, dim)
