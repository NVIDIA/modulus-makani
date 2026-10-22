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

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from makani.utils.losses.base_loss import LossType
from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature
from makani.utils import comm


def _sanitize_shapes(vals, counts, dim):
    """
    Helper routine to ensure that counts is correctly broadcasted to vals.
    """

    # if vals and counts have the same number of dimensions, we can return them as is
    if vals.dim() == counts.dim():
        # make sure the shapes match or are one
        for vdim, cdim in zip(vals.shape, counts.shape):
            if vdim != cdim and vdim != 1 and cdim != 1:
                raise ValueError("The shape of vals and counts have to match or be one")

        return vals, counts

    # if counts is not a singleton, we need to broadcast it to the shape of vals
    if counts.dim() != 1:
        raise ValueError("The shape of counts has to be exactly 1")
    cshape = [1 for _ in range(vals.dim())]
    cshape[dim] = -1
    counts = counts.reshape(cshape)
    return vals, counts


def _welford_reduction_helper(vals, counts, batch_reduction, dim):
    counts_res = torch.sum(counts, dim=dim)
    if batch_reduction == "mean":
        # results were mean reduced
        vals_res = torch.sum(vals * counts, dim=dim) / counts_res
    elif batch_reduction == "sum":
        # results were sum reduced
        vals_res = torch.sum(vals, dim=dim)
    else:
        # results were not reduced
        vals_res = vals
        counts_res = counts

    return vals_res, counts_res


# geometric base loss class
class GeometricBaseMetric(nn.Module, metaclass=ABCMeta):
    """
    Geometric base loss class used by all geometric losses
    """

    # set by metrics whose result for a sample is normalized by its weights already, see weighted_counts
    normalized_by_weights = False

    def __init__(
        self,
        grid_type: str,
        img_shape: Tuple[int, int],
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = True,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        spatial_distributed: Optional[bool] = False,
    ):
        super().__init__()

        self.img_shape = img_shape
        self.crop_shape = crop_shape
        self.crop_offset = crop_offset
        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed
        self.channel_reduction = channel_reduction
        self.batch_reduction = batch_reduction

        quadrature_rule = grid_to_quadrature_rule(grid_type)

        # get the quadrature
        self.quadrature = GridQuadrature(
            quadrature_rule,
            img_shape=self.img_shape,
            crop_shape=self.crop_shape,
            crop_offset=self.crop_offset,
            normalize=normalize,
            distributed=self.spatial_distributed,
        )

    @property
    def type(self):
        return LossType.Deterministic

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
                counts = torch.ones(size=(inp.shape[1],), device=inp.device, dtype=inp.dtype)
            elif self.batch_reduction == "sum":
                counts = torch.full(size=(inp.shape[1],), fill_value=inp.shape[0], device=inp.device, dtype=inp.dtype)

        if self.channel_reduction == "mean":
            counts = torch.mean(counts, dim=0)
        elif self.channel_reduction == "sum":
            counts = torch.sum(counts, dim=0)

        return counts

    def weighted_counts(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Defines how much every sample counts when weighted results are combined.

        The result of an area average, such as the L1 error, is an integral over the weights, so a sample counts with
        the area its weights integrate to, and finalizing renormalizes the result to that area. A metric whose result
        for a sample is normalized by its weights already, such as a ratio of integrals over the same weights, sets
        normalized_by_weights instead, and every sample with nonzero weights counts once. Counting it with the area
        would normalize it a second time.

        Parameters
        ----------
        weight : torch.Tensor
            Weights of shape (batch, channels, lat, lon).

        Returns
        -------
        counts : torch.Tensor
            Tensor of shape (batch, channels).
        """
        counts = self.quadrature(weight)
        if self.normalized_by_weights:
            counts = (counts > 0).to(dtype=counts.dtype)
        return counts

    def combine(
        self, vals: torch.Tensor, counts: torch.Tensor, dim: Optional[int] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Defines how to combine multiple metrics result using Welford.

        Parameters
        ----------
        vals : torch.Tensor
            Tensor containing metric values, with different measurements stored along dimension dim.

        counts : torch.Tensor
            One-dimensional tensor containing counts, from which vals where computed. The size of counts has
            to match the size of vals in dimension dim.

        dim : int
            Which dimension of vals to aggregate over.

        Returns
        -------
        vals_res : torch.Tensor
            Tensor containing aggretated values. Same shape as vals except for aggregation dimension dim.

        counts_res : torch.Tensor
            Singleton tensor containing the updated count.
        """
        vals, counts = _sanitize_shapes(vals, counts, dim=dim)
        vals_res, counts_res = _welford_reduction_helper(vals, counts, self.batch_reduction, dim=dim)
        return vals_res, counts_res

    def finalize(self, vals: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        Defines how to compute a final average from vals and counts.

        Parameters
        ----------
        vals : torch.Tensor
            Tensor containing values.
        counts : torch.Tensor
            Singleton tensor containing counts.

        Returns
        -------
        vals_res : torch.Tensor
            Values with correct averaging over counts.
        """
        if self.batch_reduction == "mean":
            return vals
        else:
            return vals / counts

    @abstractmethod
    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass
