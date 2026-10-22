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

import importlib.util
import os
import sys
import unittest
from parameterized import parameterized, parameterized_class

import itertools
import tempfile
import math
import numpy as np
import torch
import xarray as xr
import xskillscore as xs

from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature
from makani.utils import MetricsHandler
from makani.utils.metrics.functions import (
    GeometricL1,
    GeometricRMSE,
    GeometricACC,
    GeometricSpread,
    GeometricCRPS,
    GeometricSSR,
    GeometricRankHistogram,
)
from makani.utils.dataloaders.data_helpers import get_lat_lon_grid

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, get_default_parameters, compare_arrays

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))

# check consistency with weatherbench2 if it is installed
_have_wb2 = importlib.util.find_spec("weatherbench2") is not None

# parameters for deterministic metrics
# grid_type, shape
_deterministic_metrics_params = [
    ("euclidean", *(1, 10, 17, 32)),
    ("equiangular", *(1, 10, 17, 32)),
    ("equiangular", *(4, 21, 17, 32)),
    ("equiangular", *(8, 21, 17, 32)),
    ("clenshaw-curtiss", *(1, 10, 17, 32)),
    ("clenshaw-curtiss", *(4, 21, 17, 32)),
    ("legendre-gauss", *(1, 10, 17, 32)),
    ("legendre-gauss", *(4, 21, 17, 32)),
]

# parameters for probabilistic metrics
# grid_type, shape, ensemble_size
_probabilistic_metrics_params = [
    ("equiangular", *(1, 1, 10, 17, 32)),
    ("equiangular", *(1, 21, 10, 17, 32)),
    ("equiangular", *(4, 16, 21, 17, 32)),
    ("clenshaw-curtiss", *(1, 21, 10, 17, 32)),
    ("clenshaw-curtiss", *(4, 16, 21, 17, 32)),
    ("legendre-gauss", *(1, 21, 10, 17, 32)),
    ("legendre-gauss", *(4, 16, 21, 17, 32)),
]

# parameters for weatherbench2 metrics
_wb2_metrics_params = [
    ("equiangular", *(1, 21, 10, 17, 32)),
    ("equiangular", *(4, 16, 21, 17, 32)),
    ("clenshaw-curtiss", *(1, 21, 10, 17, 32)),
    ("clenshaw-curtiss", *(4, 16, 21, 17, 32)),
    ("weatherbench2", *(1, 21, 10, 17, 32)),
    ("weatherbench2", *(4, 16, 21, 17, 32)),
]

_deterministic_metric_aggregation_params = [
    # GeometricL1
    (GeometricL1, "equiangular", *(4, 21, 17, 32), "mean", "mean"),
    (GeometricL1, "equiangular", *(4, 21, 17, 32), "mean", "sum"),
    (GeometricL1, "equiangular", *(4, 21, 17, 32), "sum", "mean"),
    (GeometricL1, "equiangular", *(4, 21, 17, 32), "sum", "sum"),
    (GeometricL1, "equiangular", *(4, 21, 17, 32), "none", "mean"),
    (GeometricL1, "equiangular", *(4, 21, 17, 32), "none", "sum"),
    # GeometricRMSE
    (GeometricRMSE, "equiangular", *(4, 21, 17, 32), "mean", "mean"),
    (GeometricRMSE, "equiangular", *(4, 21, 17, 32), "mean", "sum"),
    (GeometricRMSE, "equiangular", *(4, 21, 17, 32), "sum", "mean"),
    (GeometricRMSE, "equiangular", *(4, 21, 17, 32), "sum", "sum"),
    (GeometricRMSE, "equiangular", *(4, 21, 17, 32), "none", "mean"),
    (GeometricRMSE, "equiangular", *(4, 21, 17, 32), "none", "sum"),
    # ACC
    (GeometricACC, "equiangular", *(4, 21, 17, 32), "mean", "mean"),
    (GeometricACC, "equiangular", *(4, 21, 17, 32), "mean", "sum"),
    (GeometricACC, "equiangular", *(4, 21, 17, 32), "sum", "mean"),
    (GeometricACC, "equiangular", *(4, 21, 17, 32), "sum", "sum"),
    (GeometricACC, "equiangular", *(4, 21, 17, 32), "none", "mean"),
    (GeometricACC, "equiangular", *(4, 21, 17, 32), "none", "sum"),
]

_deterministic_metric_weighted_aggregation_params = [
    # GeometricL1
    (GeometricL1, "equiangular", *(4, 21, 17, 32), "mean", "sum"),
    (GeometricL1, "equiangular", *(4, 21, 17, 32), "sum", "sum"),
    (GeometricL1, "equiangular", *(4, 21, 17, 32), "none", "sum"),
    # # GeometricRMSE
    (GeometricRMSE, "equiangular", *(4, 21, 17, 32), "mean", "sum"),
    (GeometricRMSE, "equiangular", *(4, 21, 17, 32), "sum", "sum"),
    (GeometricRMSE, "equiangular", *(4, 21, 17, 32), "none", "sum"),
    # ACC
    (GeometricACC, "equiangular", *(4, 21, 17, 32), "mean", "sum"),
    (GeometricACC, "equiangular", *(4, 21, 17, 32), "sum", "sum"),
    (GeometricACC, "equiangular", *(4, 21, 17, 32), "none", "sum"),
]

_probabilistic_metric_aggregation_params = [
    # Geometric CRPS
    (GeometricCRPS, "equiangular", *(4, 16, 21, 17, 32), "mean", "mean"),
    (GeometricCRPS, "equiangular", *(4, 16, 21, 17, 32), "mean", "sum"),
    (GeometricCRPS, "equiangular", *(4, 16, 21, 17, 32), "sum", "mean"),
    (GeometricCRPS, "equiangular", *(4, 16, 21, 17, 32), "sum", "sum"),
    (GeometricCRPS, "equiangular", *(4, 16, 21, 17, 32), "none", "mean"),
    (GeometricCRPS, "equiangular", *(4, 16, 21, 17, 32), "none", "sum"),
    # Geometric Spread
    (GeometricSpread, "equiangular", *(4, 16, 21, 17, 32), "mean", "mean"),
    (GeometricSpread, "equiangular", *(4, 16, 21, 17, 32), "mean", "sum"),
    (GeometricSpread, "equiangular", *(4, 16, 21, 17, 32), "sum", "mean"),
    (GeometricSpread, "equiangular", *(4, 16, 21, 17, 32), "sum", "sum"),
    (GeometricSpread, "equiangular", *(4, 16, 21, 17, 32), "none", "mean"),
    (GeometricSpread, "equiangular", *(4, 16, 21, 17, 32), "none", "sum"),
    # Geometric SSR
    (GeometricSSR, "equiangular", *(4, 16, 21, 17, 32), "mean", "mean"),
    (GeometricSSR, "equiangular", *(4, 16, 21, 17, 32), "mean", "sum"),
    (GeometricSSR, "equiangular", *(4, 16, 21, 17, 32), "sum", "mean"),
    (GeometricSSR, "equiangular", *(4, 16, 21, 17, 32), "sum", "sum"),
    (GeometricSSR, "equiangular", *(4, 16, 21, 17, 32), "none", "mean"),
    (GeometricSSR, "equiangular", *(4, 16, 21, 17, 32), "none", "sum"),
    # Geometric Rank Histogram
    (GeometricRankHistogram, "equiangular", *(4, 16, 21, 17, 32), "mean", "mean"),
    (GeometricRankHistogram, "equiangular", *(4, 16, 21, 17, 32), "mean", "sum"),
    (GeometricRankHistogram, "equiangular", *(4, 16, 21, 17, 32), "sum", "mean"),
    (GeometricRankHistogram, "equiangular", *(4, 16, 21, 17, 32), "sum", "sum"),
    (GeometricRankHistogram, "equiangular", *(4, 16, 21, 17, 32), "none", "mean"),
    (GeometricRankHistogram, "equiangular", *(4, 16, 21, 17, 32), "none", "sum"),
]

_probabilistic_metric_weighted_aggregation_params = [
    # Geometric CRPS
    (GeometricCRPS, "equiangular", *(4, 16, 21, 17, 32), "mean", "sum"),
    (GeometricCRPS, "equiangular", *(4, 16, 21, 17, 32), "sum", "sum"),
    (GeometricCRPS, "equiangular", *(4, 16, 21, 17, 32), "none", "sum"),
    # Geometric Spread
    (GeometricSpread, "equiangular", *(4, 16, 21, 17, 32), "mean", "sum"),
    (GeometricSpread, "equiangular", *(4, 16, 21, 17, 32), "sum", "sum"),
    (GeometricSpread, "equiangular", *(4, 16, 21, 17, 32), "none", "sum"),
    # Geometric SSR
    (GeometricSSR, "equiangular", *(4, 16, 21, 17, 32), "mean", "sum"),
    (GeometricSSR, "equiangular", *(4, 16, 21, 17, 32), "sum", "sum"),
    (GeometricSSR, "equiangular", *(4, 16, 21, 17, 32), "none", "sum"),
    # Geometric Rank Histogram
    (GeometricRankHistogram, "equiangular", *(4, 16, 21, 17, 32), "mean", "sum"),
    (GeometricRankHistogram, "equiangular", *(4, 16, 21, 17, 32), "sum", "sum"),
    (GeometricRankHistogram, "equiangular", *(4, 16, 21, 17, 32), "none", "sum"),
]

_metric_handler_params = [
    ("equiangular", 4, 16, 3, "mean"),
    ("equiangular", 4, 16, 3, "sum"),
]


def _valid_area_reference(metric_name, fct, obs, area_weight):
    """
    ACC, spread or SSR of every sample over the area its weights cover, averaged over the samples with valid points.

    area_weight are the quadrature weights times the mask, in the shape of obs.
    """
    # a sample without valid points gives NaN, which nanmean leaves out
    with np.errstate(invalid="ignore"):

        def area_mean(x):
            return np.sum(area_weight * x, axis=(-2, -1)) / np.sum(area_weight, axis=(-2, -1))

        if metric_name == "acc":
            score = area_mean(fct * obs) / np.sqrt(area_mean(fct * fct) * area_mean(obs * obs))
        else:
            ens_size = fct.shape[1]
            fct_mean = np.mean(fct, axis=1)
            spread = area_mean(np.sum(np.square(fct - fct_mean[:, None]), axis=1) / (ens_size - 1))
            if metric_name == "spread":
                score = np.sqrt(spread)
            else:
                score = np.sqrt(spread / (area_mean(np.square(fct_mean - obs)) - spread / ens_size))

    return np.nanmean(score, axis=0)


@parameterized_class(("device",), _devices)
class TestMetrics(unittest.TestCase):
    """
    Testsuite for makani metrics. Compares to properscoring
    """

    def setUp(self):

        disable_tf32()

        set_seed(333)

        return

    @parameterized.expand(_deterministic_metrics_params, skip_on_empty=True)
    def test_weight_normalization(self, grid_type, batch_size, num_channels, nlat, nlon):
        quadrature_type = grid_to_quadrature_rule(grid_type)
        quad = GridQuadrature(quadrature_type, img_shape=(nlat, nlon), normalize=True).to(self.device)
        flat_vector = torch.ones(batch_size, num_channels, nlat, nlon, device=self.device)
        integral = torch.mean(quad(flat_vector)).item()

        self.assertTrue(
            compare_arrays("weight normalization", np.asarray(integral), np.asarray(1.0), rtol=1e-5, atol=0)
        )

    @parameterized.expand(_deterministic_metrics_params, skip_on_empty=True)
    def test_weighted_rmse(self, grid_type, batch_size, num_channels, nlat, nlon, verbose=False):

        # rmse handle
        rmse_func = GeometricRMSE(
            grid_type, img_shape=(nlat, nlon), normalize=True, channel_reduction="none", batch_reduction="none"
        ).to(self.device)

        # generate toy data
        A = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        B = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        rmse = torch.mean(rmse_func(A, B), dim=0).cpu().numpy()

        lwf = torch.squeeze(rmse_func.quadrature.quad_weight).cpu().numpy()
        lwf = xr.DataArray(lwf, dims=["lat", "lon"])
        A = xr.DataArray(A.cpu(), dims=["batch", "channels", "lat", "lon"])
        B = xr.DataArray(B.cpu(), dims=["batch", "channels", "lat", "lon"])
        rmse_xskillscore = xs.rmse(A, B, weights=lwf, dim=["lat", "lon"]).to_numpy().mean(axis=0)

        self.assertTrue(compare_arrays("rmse", rmse, rmse_xskillscore, rtol=1e-5, atol=0, verbose=verbose))

    @parameterized.expand(_deterministic_metrics_params, skip_on_empty=True)
    def test_l1(self, grid_type, batch_size, num_channels, nlat, nlon, verbose=False):

        # l1 handle
        l1_func = GeometricL1(
            grid_type, img_shape=(nlat, nlon), normalize=True, channel_reduction="mean", batch_reduction="mean"
        ).to(self.device)

        # generate toy data
        A = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        B = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        l1 = l1_func(A, B).cpu().numpy()

        lwf = l1_func.quadrature.quad_weight.cpu().numpy()
        lwf = xr.DataArray(np.tile(lwf, (batch_size, num_channels, 1, 1)), dims=["batch", "channels", "lat", "lon"])
        A = xr.DataArray(A.cpu(), dims=["batch", "channels", "lat", "lon"])
        B = xr.DataArray(B.cpu(), dims=["batch", "channels", "lat", "lon"])
        l1_xskillscore = xs.mae(A, B, weights=lwf).to_numpy()

        self.assertTrue(compare_arrays("l1", l1_xskillscore, l1, rtol=1e-5, atol=0, verbose=verbose))

    @parameterized.expand(_deterministic_metrics_params, skip_on_empty=True)
    def test_weighted_acc_macro(self, grid_type, batch_size, num_channels, nlat, nlon, verbose=False):

        # ACC handle
        acc_func = GeometricACC(
            grid_type,
            img_shape=(nlat, nlon),
            normalize=True,
            channel_reduction="none",
            batch_reduction="mean",
            method="macro",
        ).to(self.device)

        # generate toy data
        A = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        B = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)

        # compute means
        A_mean = acc_func.quadrature(A).reshape(batch_size, num_channels, 1, 1)
        B_mean = acc_func.quadrature(B).reshape(batch_size, num_channels, 1, 1)

        # compute score using makani
        acc = acc_func(A - A_mean, B - B_mean).cpu().numpy()

        lwf = torch.squeeze(acc_func.quadrature.quad_weight).cpu().numpy()
        lwf = xr.DataArray(lwf, dims=["lat", "lon"])
        A = xr.DataArray(A.cpu(), dims=["batch", "channels", "lat", "lon"])
        B = xr.DataArray(B.cpu(), dims=["batch", "channels", "lat", "lon"])

        # compute score using xskillscore
        acc_xskillscore = xs.pearson_r(A, B, weights=lwf, dim=["lat", "lon"]).to_numpy().mean(axis=0)

        self.assertTrue(compare_arrays("acc macro", acc, acc_xskillscore, rtol=5e-5, atol=0, verbose=verbose))

    @parameterized.expand(_deterministic_metrics_params, skip_on_empty=True)
    def test_weighted_acc_micro(self, grid_type, batch_size, num_channels, nlat, nlon, verbose=False):

        # ACC handle
        acc_func = GeometricACC(
            grid_type,
            img_shape=(nlat, nlon),
            normalize=True,
            channel_reduction="none",
            batch_reduction="mean",
            method="micro",
        ).to(self.device)

        # generate toy data
        A = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        B = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)

        # compute means
        A_mean = acc_func.quadrature(A).mean(dim=0).reshape(1, num_channels, 1, 1)
        B_mean = acc_func.quadrature(B).mean(dim=0).reshape(1, num_channels, 1, 1)

        # compute score using makani
        acc = acc_func(A - A_mean, B - B_mean).cpu().numpy()
        # finalize score
        acc = acc[..., 0] / np.sqrt(acc[..., 1] * acc[..., 2])

        # we need to ensure that the weights have the correct dimensions
        lwf = torch.tile(torch.squeeze(acc_func.quadrature.quad_weight).unsqueeze(0), (batch_size, 1, 1)).cpu().numpy()
        lwf = xr.DataArray(lwf, dims=["batch", "lat", "lon"])
        A = xr.DataArray(A.cpu(), dims=["batch", "channels", "lat", "lon"])
        B = xr.DataArray(B.cpu(), dims=["batch", "channels", "lat", "lon"])

        # compute score using xskillscore
        acc_xskillscore = xs.pearson_r(A, B, weights=lwf, dim=["batch", "lat", "lon"]).to_numpy()

        self.assertTrue(compare_arrays("acc micro", acc, acc_xskillscore, rtol=5e-5, atol=0, verbose=verbose))

    @parameterized.expand(_probabilistic_metrics_params, skip_on_empty=True)
    def test_weighted_crps(self, grid_type, batch_size, ensemble_size, num_channels, nlat, nlon, verbose=False):

        # CRPS handle
        crps_func = GeometricCRPS(
            grid_type=grid_type,
            img_shape=(nlat, nlon),
            crop_shape=(nlat, nlon),
            crop_offset=(0, 0),
            crps_type="cdf",
            channel_reduction="none",
            batch_reduction="none",
        ).to(self.device)

        # generate toy data
        obs = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        fct = torch.randn(batch_size, ensemble_size, num_channels, nlat, nlon, device=self.device)
        crps = crps_func(fct, obs).cpu().numpy()

        lwf = torch.squeeze(crps_func.metric_func.quadrature.quad_weight).cpu().numpy()
        lwf = xr.DataArray(lwf, dims=["lat", "lon"])
        obs = xr.DataArray(obs.cpu(), dims=["batch", "channels", "lat", "lon"])
        fct = xr.DataArray(fct.cpu(), dims=["batch", "ensemble", "channels", "lat", "lon"])

        # compute score using xskillscore
        crps_xskillscore = xs.crps_ensemble(obs, fct, member_dim="ensemble", weights=lwf, dim=["lat", "lon"]).to_numpy()

        self.assertTrue(compare_arrays("crps", crps, crps_xskillscore, rtol=5e-5, atol=0, verbose=verbose))

    # we need to relax the bounds here because on CPU, for some reason, the deviations are big
    @parameterized.expand(_probabilistic_metrics_params, skip_on_empty=True)
    def test_rank_histogram(
        self, grid_type, batch_size, ensemble_size, num_channels, nlat, nlon, atol=1e-3, rtol=1e-4, verbose=False
    ):
        # Rank histogram Handle
        rankhist_handle = GeometricRankHistogram(
            grid_type=grid_type,
            img_shape=(nlat, nlon),
            crop_shape=(nlat, nlon),
            crop_offset=(0, 0),
            normalize=True,
            channel_reduction="none",
            batch_reduction="mean",
        ).to(self.device)

        # generate toy data
        obs = torch.rand(batch_size, num_channels, nlat, nlon, device=self.device)
        # shift channels by channel index
        obs = obs + torch.linspace(start=0, end=num_channels - 1, steps=num_channels, device=self.device).reshape(
            1, -1, 1, 1
        )
        # assume linear ensemble forecast
        fct = torch.linspace(start=0, end=1, steps=ensemble_size, device=self.device).reshape(1, -1, 1, 1, 1)
        fct = torch.tile(fct, (batch_size, 1, num_channels, nlat, nlon))
        # move also by channel index
        fct = fct + torch.linspace(start=0, end=num_channels - 1, steps=num_channels, device=self.device).reshape(
            1, -1, 1, 1
        )

        # compute rank histogram
        rankhist = rankhist_handle(fct, obs)

        # ensure that means are as expected
        means = torch.mean(rankhist, dim=(1))

        # expected means: always 1 / (ensemble_size+1)
        means_expected = torch.ones([num_channels]) / float(ensemble_size + 1)

        with self.subTest(desc="means"):
            self.assertTrue(
                compare_arrays(
                    "means", means_expected.cpu().numpy(), means.cpu().numpy(), atol=atol, rtol=rtol, verbose=verbose
                )
            )

        # compare to xskillscore:
        obs = xr.DataArray(obs.cpu().numpy(), dims=["batch", "channels", "lat", "lon"])
        fct = xr.DataArray(fct.cpu().numpy(), dims=["batch", "ensemble", "channels", "lat", "lon"])
        rankhist_xskillscore = xs.rank_histogram(obs, fct, dim="batch", member_dim="ensemble").to_numpy()

        # we need to divide by the batch size because xskillscore does not average but just sum
        rankhist_xskillscore = rankhist_xskillscore / float(batch_size)

        # for spatial averaging, flatten the spatial dims
        rankhist_xskillscore = rankhist_xskillscore.reshape(num_channels, nlat * nlon, ensemble_size + 1)
        qw = rankhist_handle.quad_weight_split.squeeze(0).cpu().numpy()
        rankhist_xskillscore = np.sum(rankhist_xskillscore * qw, axis=1)

        with self.subTest(desc="rank histogram"):
            self.assertTrue(
                compare_arrays(
                    "rank histogram",
                    rankhist.cpu().numpy(),
                    rankhist_xskillscore,
                    atol=atol,
                    rtol=rtol,
                    verbose=verbose,
                )
            )

    @parameterized.expand(["acc", "ssr", "spread"], skip_on_empty=True)
    def test_masked_metrics(self, metric_name, verbose=False):
        """
        ACC, SSR and spread of masked data are those over the valid area, whatever the weights integrate to.

        Missing target values, SST over land say, reach the metrics as zero weights. These metrics are normalized by
        the weights of every sample already, so a sample has to count once when they are finalized. Counting it with
        the area its weights integrate to, as for an area average, divides them by that area a second time.
        """
        batch_size, ensemble_size, num_channels, nlat, nlon = 4, 8, 3, 17, 32

        metric_handle = {"acc": GeometricACC, "ssr": GeometricSSR, "spread": GeometricSpread}[metric_name]
        metric_func = metric_handle(
            "equiangular",
            img_shape=(nlat, nlon),
            crop_shape=(nlat, nlon),
            crop_offset=(0, 0),
            normalize=True,
            channel_reduction="none",
            batch_reduction="sum",
        ).to(self.device)

        # a static mask over about a third of the grid, plus a channel that is missing entirely in one sample, which
        # has to stay out of the average
        mask = torch.ones(batch_size, num_channels, nlat, nlon, device=self.device)
        mask[..., : nlon // 3] = 0.0
        mask[0, 1] = 0.0

        truth = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        obs = truth + torch.randn_like(truth)
        if metric_name == "acc":
            fct = truth + torch.randn_like(truth)
        else:
            fct = truth.unsqueeze(1) + torch.randn(
                batch_size, ensemble_size, num_channels, nlat, nlon, device=self.device
            )

        area_weight = (metric_func.quadrature.quad_weight * mask).cpu().numpy()
        reference = _valid_area_reference(metric_name, fct.cpu().numpy(), obs.cpu().numpy(), area_weight)

        # the result must not depend on what the weights integrate to, the inferencer normalizes its masks for example
        for scale in [1.0, 0.25]:
            weight = scale * mask
            result = metric_func(fct, obs, weight)
            result = metric_func.finalize(result, metric_func.compute_counts(fct, weight))

            with self.subTest(desc=f"weights scaled by {scale}"):
                self.assertTrue(
                    compare_arrays(metric_name, result.cpu().numpy(), reference, rtol=1e-5, atol=1e-6, verbose=verbose)
                )


@parameterized_class(("device",), _devices)
class TestMetricsAggregation(unittest.TestCase):
    """
    A set of tests that test the aggregation
    """

    def setUp(self):

        set_seed(333)

        return

    @parameterized.expand(_deterministic_metric_aggregation_params, skip_on_empty=True)
    def test_deterministic_aggregation(
        self, metric_handle, grid_type, batch_size, num_channels, nlat, nlon, cred, bred, verbose=False
    ):

        # inflate batch size
        num_rollout_steps = 10
        batch_size_nsteps = num_rollout_steps * batch_size

        # metric handle
        metric_func = metric_handle(
            grid_type, img_shape=(nlat, nlon), normalize=True, channel_reduction=cred, batch_reduction=bred
        ).to(self.device)

        # input and target:
        inp = torch.randn((batch_size_nsteps, num_channels, nlat, nlon), dtype=torch.float32, device=self.device)
        tar = torch.randn((batch_size_nsteps, num_channels, nlat, nlon), dtype=torch.float32, device=self.device)

        # full metric
        res_full = metric_func(inp, tar)
        counts_full = torch.tensor(inp.shape[0], dtype=torch.float32, device=self.device)
        res_full = metric_func.finalize(res_full, counts_full)

        # split and compute metrics stepwise
        inp_split = torch.split(inp, batch_size, dim=0)
        tar_split = torch.split(tar, batch_size, dim=0)

        res_split = metric_func(inp_split[0], tar_split[0])
        counts_split = torch.tensor(inp_split[0].shape[0], dtype=torch.float32, device=self.device)
        for inps, tars in zip(inp_split[1:], tar_split[1:]):
            res_tmp = metric_func(inps, tars)
            res_tmp = torch.stack([res_split, res_tmp], dim=0)
            counts_tmp = torch.tensor((inps.shape[0]), dtype=torch.float32, device=self.device)
            counts_tmp = torch.stack([counts_split, counts_tmp], dim=0)
            res_tmp, counts_tmp = metric_func.combine(res_tmp, counts_tmp, dim=0)
            # we need to squeeze the counts now since combine returns unsqueezed tensors:
            counts_tmp = counts_tmp.squeeze()
            with torch.no_grad():
                res_split.copy_(res_tmp)
                counts_split.copy_(counts_tmp)

        res_split = metric_func.finalize(res_split, counts_split)

        # compare
        self.assertTrue(
            compare_arrays(
                "deterministic aggregation",
                res_full.cpu().numpy(),
                res_split.cpu().numpy(),
                rtol=1e-6,
                atol=1e-6,
                verbose=verbose,
            )
        )

    @parameterized.expand(_deterministic_metric_weighted_aggregation_params, skip_on_empty=True)
    def test_deterministic_aggregation_weighted(
        self, metric_handle, grid_type, batch_size, num_channels, nlat, nlon, cred, bred, verbose=False
    ):
        """Same as test_deterministic_aggregation but with a spatial weight tensor (0/1 mask) at every step."""
        # inflate batch size
        num_rollout_steps = 10
        batch_size_nsteps = num_rollout_steps * batch_size

        # metric handle
        metric_func = metric_handle(
            grid_type, img_shape=(nlat, nlon), normalize=True, channel_reduction=cred, batch_reduction=bred
        ).to(self.device)

        # input and target
        inp = torch.randn((batch_size_nsteps, num_channels, nlat, nlon), dtype=torch.float32, device=self.device)
        tar = torch.randn((batch_size_nsteps, num_channels, nlat, nlon), dtype=torch.float32, device=self.device)

        # spatial weight: random 0/1 mask (same for all steps so full vs split aggregation are comparable)
        weight_full = torch.where(
            torch.rand(batch_size_nsteps, num_channels, nlat, nlon, device=self.device, dtype=torch.float32) > 0.5,
            1.0,
            0.0,
        )

        # full metric with weight
        res_full = metric_func(inp, tar, weight=weight_full)
        counts_full = metric_func.compute_counts(inp, weight=weight_full)
        res_full = metric_func.finalize(res_full, counts_full)

        # split and compute metrics stepwise, passing the same spatial weight at each step
        inp_split = torch.split(inp, batch_size, dim=0)
        tar_split = torch.split(tar, batch_size, dim=0)
        weight_split = torch.split(weight_full, batch_size, dim=0)

        res_split = metric_func(inp_split[0], tar_split[0], weight=weight_split[0])
        counts_split = metric_func.compute_counts(inp_split[0], weight=weight_split[0])
        for inps, tars, weights in zip(inp_split[1:], tar_split[1:], weight_split[1:]):
            res_tmp = metric_func(inps, tars, weight=weights)
            res_tmp = torch.stack([res_split, res_tmp], dim=0)
            counts_tmp = metric_func.compute_counts(inps, weight=weights)
            counts_tmp = torch.stack([counts_split, counts_tmp], dim=0)
            res_split, counts_split = metric_func.combine(res_tmp, counts_tmp, dim=0)

        res_split = metric_func.finalize(res_split, counts_split)

        self.assertTrue(
            compare_arrays(
                "deterministic aggregation weighted",
                res_full.cpu().numpy(),
                res_split.cpu().numpy(),
                rtol=1e-6,
                atol=1e-6,
                verbose=verbose,
            )
        )

    @parameterized.expand(_deterministic_metric_weighted_aggregation_params, skip_on_empty=True)
    def test_deterministic_aggregation_nan(
        self, metric_handle, grid_type, batch_size, num_channels, nlat, nlon, cred, bred, verbose=False
    ):
        """With NaNs in different positions in output and target, a combined mask (1 where both valid, 0 else) as weight should yield a metric output with no NaNs."""
        # inflate batch size
        num_rollout_steps = 10
        batch_size_nsteps = num_rollout_steps * batch_size
        shape = (batch_size_nsteps, num_channels, nlat, nlon)

        # instantiate metric
        metric_func = metric_handle(
            grid_type, img_shape=(nlat, nlon), normalize=True, channel_reduction=cred, batch_reduction=bred
        ).to(self.device)

        # prediction and target with random values
        inp = torch.randn(shape, dtype=torch.float32, device=self.device)
        tar = torch.randn(shape, dtype=torch.float32, device=self.device)

        # inject NaNs in different positions in inp and tar
        rng = np.random.default_rng(42)
        n_nan = int(math.prod(shape) * 0.1)
        flat_inp = inp.view(-1)
        flat_tar = tar.view(-1)
        idx_inp = rng.choice(flat_inp.numel(), size=n_nan, replace=False)
        idx_tar = rng.choice(flat_tar.numel(), size=n_nan, replace=False)
        flat_inp[idx_inp] = torch.nan
        flat_tar[idx_tar] = torch.nan

        # combined mask: 1 where both are non-NaN, 0 where either is NaN
        valid_inp = torch.logical_not(torch.isnan(inp)).to(torch.float32)
        valid_tar = torch.logical_not(torch.isnan(tar)).to(torch.float32)
        weight_full = valid_inp * valid_tar

        # mask nan values in inp and tar
        inp_masked = torch.where(weight_full > 0.0, inp, 0.0)
        tar_masked = torch.where(weight_full > 0.0, tar, 0.0)

        # full metric with weight
        res_full = metric_func(inp_masked, tar_masked, weight=weight_full)
        counts_full = metric_func.compute_counts(inp_masked, weight=weight_full)
        res_full = metric_func.finalize(res_full, counts_full)

        # split and compute metrics stepwise, passing the same spatial weight at each step
        inp_split = torch.split(inp, batch_size, dim=0)
        tar_split = torch.split(tar, batch_size, dim=0)
        weight_split = torch.split(weight_full, batch_size, dim=0)

        # mask inps and tars
        inp_split_masked = torch.where(weight_split[0] > 0.0, inp_split[0], 0.0)
        tar_split_masked = torch.where(weight_split[0] > 0.0, tar_split[0], 0.0)
        res_split = metric_func(inp_split_masked, tar_split_masked, weight=weight_split[0])
        counts_split = metric_func.compute_counts(inp_split_masked, weight=weight_split[0])
        for inps, tars, weights in zip(inp_split[1:], tar_split[1:], weight_split[1:]):
            # mask inps and tars
            inps_masked = torch.where(weights > 0.0, inps, 0.0)
            tars_masked = torch.where(weights > 0.0, tars, 0.0)
            res_tmp = metric_func(inps_masked, tars_masked, weight=weights)
            res_tmp = torch.stack([res_split, res_tmp], dim=0)
            counts_tmp = metric_func.compute_counts(inps_masked, weight=weights)
            counts_tmp = torch.stack([counts_split, counts_tmp], dim=0)
            res_split, counts_split = metric_func.combine(res_tmp, counts_tmp, dim=0)

        res_split = metric_func.finalize(res_split, counts_split)

        with self.subTest(desc="full result"):
            self.assertFalse(
                torch.isnan(res_full).any(),
                msg="Full metric output must have no NaNs when using combined NaN mask as weight.",
            )
            self.assertFalse(
                torch.isinf(res_full).any(),
                msg="Full metric output must have no Infs when using combined NaN mask as weight.",
            )

        with self.subTest(desc="split result"):
            self.assertFalse(
                torch.isnan(res_split).any(),
                msg="Aggregated metric output must have no NaNs when using combined NaN mask as weight.",
            )
            self.assertFalse(
                torch.isinf(res_split).any(),
                msg="Aggregated metric output must have no Infs when using combined NaN mask as weight.",
            )

        self.assertTrue(
            compare_arrays(
                "deterministic aggregation with nan",
                res_full.cpu().numpy(),
                res_split.cpu().numpy(),
                rtol=1e-6,
                atol=1e-6,
                verbose=verbose,
            )
        )

    @parameterized.expand(_probabilistic_metric_aggregation_params, skip_on_empty=True)
    def test_probabilistic_aggregation(
        self, metric_handle, grid_type, batch_size, ensemble_size, num_channels, nlat, nlon, cred, bred, verbose=False
    ):

        # inflate batch size
        num_rollout_steps = 10
        batch_size_nsteps = num_rollout_steps * batch_size

        # metric handle
        metric_func = metric_handle(
            grid_type=grid_type,
            img_shape=(nlat, nlon),
            crop_shape=(nlat, nlon),
            crop_offset=(0, 0),
            crps_type="skillspread",
            normalize=True,
            channel_reduction=cred,
            batch_reduction=bred,
        ).to(self.device)

        # input and target:
        inp = torch.randn(
            (batch_size_nsteps, ensemble_size, num_channels, nlat, nlon), dtype=torch.float32, device=self.device
        )
        tar = torch.randn((batch_size_nsteps, num_channels, nlat, nlon), dtype=torch.float32, device=self.device)

        # full metric
        res_full = metric_func(inp, tar)
        counts_full = torch.tensor(inp.shape[0], dtype=torch.float32, device=self.device)
        res_full = metric_func.finalize(res_full, counts_full)

        # split and compute metrics stepwise
        inp_split = torch.split(inp, batch_size, dim=0)
        tar_split = torch.split(tar, batch_size, dim=0)

        res_split = metric_func(inp_split[0], tar_split[0])
        counts_split = torch.tensor(inp_split[0].shape[0], dtype=torch.float32, device=self.device)
        for inps, tars in zip(inp_split[1:], tar_split[1:]):
            res_tmp = metric_func(inps, tars)
            res_tmp = torch.stack([res_split, res_tmp], dim=0)
            counts_tmp = torch.tensor(inps.shape[0], dtype=torch.float32, device=self.device)
            counts_tmp = torch.stack([counts_split, counts_tmp], dim=0)
            res_tmp, counts_tmp = metric_func.combine(res_tmp, counts_tmp, dim=0)
            # we need to squeeze the counts now since combine returns unsqueezed tensors:
            counts_tmp = counts_tmp.squeeze()
            with torch.no_grad():
                res_split.copy_(res_tmp)
                counts_split.copy_(counts_tmp)

        res_split = metric_func.finalize(res_split, counts_split)

        # compare
        self.assertTrue(
            compare_arrays(
                "probabilistic aggregation",
                res_full.cpu().numpy(),
                res_split.cpu().numpy(),
                rtol=1e-6,
                atol=1e-6,
                verbose=verbose,
            )
        )

    @parameterized.expand(_probabilistic_metric_weighted_aggregation_params, skip_on_empty=True)
    def test_probabilistic_aggregation_weighted(
        self, metric_handle, grid_type, batch_size, ensemble_size, num_channels, nlat, nlon, cred, bred, verbose=False
    ):
        """Same as test_probabilistic_aggregation but with a spatial weight tensor (0/1 mask) at every step."""
        num_rollout_steps = 10
        batch_size_nsteps = num_rollout_steps * batch_size

        metric_func = metric_handle(
            grid_type=grid_type,
            img_shape=(nlat, nlon),
            crop_shape=(nlat, nlon),
            crop_offset=(0, 0),
            crps_type="skillspread",
            normalize=True,
            channel_reduction=cred,
            batch_reduction=bred,
        ).to(self.device)

        inp = torch.randn(
            (batch_size_nsteps, ensemble_size, num_channels, nlat, nlon), dtype=torch.float32, device=self.device
        )
        tar = torch.randn((batch_size_nsteps, num_channels, nlat, nlon), dtype=torch.float32, device=self.device)

        # spatial weight: 0/1 mask, same ndim as observations (no ensemble dim)
        weight_full = (
            torch.rand(batch_size_nsteps, num_channels, nlat, nlon, device=self.device, dtype=torch.float32) > 0.5
        )

        res_full = metric_func(inp, tar, weight=weight_full)
        counts_full = metric_func.compute_counts(inp, weight=weight_full)
        res_full = metric_func.finalize(res_full, counts_full)

        inp_split = torch.split(inp, batch_size, dim=0)
        tar_split = torch.split(tar, batch_size, dim=0)
        weight_split = torch.split(weight_full, batch_size, dim=0)

        res_split = metric_func(inp_split[0], tar_split[0], weight=weight_split[0])
        counts_split = metric_func.compute_counts(inp_split[0], weight=weight_split[0])
        for inps, tars, weights in zip(inp_split[1:], tar_split[1:], weight_split[1:]):
            res_tmp = metric_func(inps, tars, weight=weights)
            res_tmp = torch.stack([res_split, res_tmp], dim=0)
            counts_tmp = metric_func.compute_counts(inps, weight=weights)
            counts_tmp = torch.stack([counts_split, counts_tmp], dim=0)
            res_split, counts_split = metric_func.combine(res_tmp, counts_tmp, dim=0)

        res_split = metric_func.finalize(res_split, counts_split)

        self.assertTrue(
            compare_arrays(
                "probabilistic aggregation weighted",
                res_full.cpu().numpy(),
                res_split.cpu().numpy(),
                rtol=1e-6,
                atol=1e-6,
                verbose=verbose,
            )
        )

    @parameterized.expand(_probabilistic_metric_weighted_aggregation_params, skip_on_empty=True)
    def test_probabilistic_aggregation_nan(
        self, metric_handle, grid_type, batch_size, ensemble_size, num_channels, nlat, nlon, cred, bred, verbose=False
    ):
        """Same as test_probabilistic_aggregation but with a spatial weight tensor (0/1 mask) at every step."""
        num_rollout_steps = 10
        batch_size_nsteps = num_rollout_steps * batch_size

        # metric handle
        metric_func = metric_handle(
            grid_type=grid_type,
            img_shape=(nlat, nlon),
            crop_shape=(nlat, nlon),
            crop_offset=(0, 0),
            crps_type="skillspread",
            normalize=True,
            channel_reduction=cred,
            batch_reduction=bred,
        ).to(self.device)

        inp = torch.randn(
            (batch_size_nsteps, ensemble_size, num_channels, nlat, nlon), dtype=torch.float32, device=self.device
        )
        tar = torch.randn((batch_size_nsteps, num_channels, nlat, nlon), dtype=torch.float32, device=self.device)

        # inject NaNs in different positions in inp and tar
        rng = np.random.default_rng(42)
        n_nan = int(math.prod((batch_size_nsteps, ensemble_size, num_channels, nlat, nlon)) * 0.1)
        flat_inp = inp.view(-1)
        idx_inp = rng.choice(flat_inp.numel(), size=n_nan, replace=False)
        flat_inp[idx_inp] = torch.nan

        n_nan = int(math.prod((batch_size_nsteps, num_channels, nlat, nlon)) * 0.1)
        flat_tar = tar.view(-1)
        idx_tar = rng.choice(flat_tar.numel(), size=n_nan, replace=False)
        flat_tar[idx_tar] = torch.nan

        # combined mask: 1 where both are non-NaN, 0 where either is NaN
        # we mask all ensemble members if any one is NaN for now
        valid_inp = torch.logical_not(torch.isnan(torch.sum(inp, dim=1))).to(torch.float32)
        valid_tar = torch.logical_not(torch.isnan(tar)).to(torch.float32)
        weight_full = valid_inp * valid_tar

        # mask nan values in inp and tar
        inp_masked = torch.where(weight_full.unsqueeze(1).expand_as(inp) > 0.0, inp, 0.0)
        tar_masked = torch.where(weight_full > 0.0, tar, 0.0)

        res_full = metric_func(inp_masked, tar_masked, weight=weight_full)
        counts_full = metric_func.compute_counts(inp_masked, weight=weight_full)
        res_full = metric_func.finalize(res_full, counts_full)

        inp_split = torch.split(inp, batch_size, dim=0)
        tar_split = torch.split(tar, batch_size, dim=0)
        weight_split = torch.split(weight_full, batch_size, dim=0)

        inp_split_masked = torch.where(weight_split[0].unsqueeze(1).expand_as(inp_split[0]) > 0.0, inp_split[0], 0.0)
        tar_split_masked = torch.where(weight_split[0] > 0.0, tar_split[0], 0.0)
        res_split = metric_func(inp_split_masked, tar_split_masked, weight=weight_split[0])
        counts_split = metric_func.compute_counts(inp_split_masked, weight=weight_split[0])
        for inps, tars, weights in zip(inp_split[1:], tar_split[1:], weight_split[1:]):
            # mask inps and tars
            inps_masked = torch.where(weights.unsqueeze(1).expand_as(inps) > 0.0, inps, 0.0)
            tars_masked = torch.where(weights > 0.0, tars, 0.0)
            res_tmp = metric_func(inps_masked, tars_masked, weight=weights)
            res_tmp = torch.stack([res_split, res_tmp], dim=0)
            counts_tmp = metric_func.compute_counts(inps_masked, weight=weights)
            counts_tmp = torch.stack([counts_split, counts_tmp], dim=0)
            res_split, counts_split = metric_func.combine(res_tmp, counts_tmp, dim=0)

        res_split = metric_func.finalize(res_split, counts_split)

        with self.subTest(desc="full result"):
            self.assertFalse(
                torch.isnan(res_full).any(),
                msg="Full metric output must have no NaNs when using combined NaN mask as weight.",
            )
            self.assertFalse(
                torch.isinf(res_full).any(),
                msg="Full metric output must have no Infs when using combined NaN mask as weight.",
            )

        with self.subTest(desc="split result"):
            self.assertFalse(
                torch.isnan(res_split).any(),
                msg="Aggregated metric output must have no NaNs when using combined NaN mask as weight.",
            )
            self.assertFalse(
                torch.isinf(res_split).any(),
                msg="Aggregated metric output must have no Infs when using combined NaN mask as weight.",
            )

        self.assertTrue(
            compare_arrays(
                "probabilistic aggregation nan",
                res_full.cpu().numpy(),
                res_split.cpu().numpy(),
                rtol=1e-6,
                atol=1e-6,
                verbose=verbose,
            )
        )


@parameterized_class(("device",), _devices)
class TestMetricsHandler(unittest.TestCase):
    """
    A set of tests for the metrics handler
    """

    def setUp(self):

        set_seed(333)

        self.params = get_default_parameters()
        self.params["dhours"] = 1

        # generating the image logic that is typically used by the dataloader
        self.params.img_shape_x = 17
        self.params.img_shape_y = 32
        self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
        self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
        self.params.img_shape_x_resampled = self.params.img_shape_x
        self.params.img_shape_y_resampled = self.params.img_shape_y
        self.params.img_local_offset_x = 0
        self.params.img_local_offset_y = 0
        self.params.img_crop_offset_x = 0
        self.params.img_crop_offset_y = 0

        return

    @parameterized.expand(_metric_handler_params[:1], skip_on_empty=True)
    def test_save_metrics(self, grid_type, batch_size, ensemble_size, num_rollout_steps, bred):

        # create dummy climatology
        num_steps = 1
        num_channels = len(self.params.channel_names)
        clim = torch.zeros(1, num_channels, self.params.img_local_shape_x, self.params.img_local_shape_y)

        # update parameters
        self.params.batch_size = batch_size
        self.params.ensemble_size = ensemble_size

        metric_handler = MetricsHandler(
            self.params,
            clim,
            num_rollout_steps,
            self.device,
            l1_var_names=self.params.channel_names,
            rmse_var_names=self.params.channel_names,
            acc_var_names=self.params.channel_names,
            crps_var_names=self.params.channel_names,
            spread_var_names=self.params.channel_names,
            ssr_var_names=self.params.channel_names,
            rh_var_names=self.params.channel_names,
            wb2_compatible=False,
        )
        metric_handler.initialize_buffers()
        metric_handler.zero_buffers()

        inplist = [
            torch.randn(
                (
                    num_rollout_steps,
                    batch_size,
                    ensemble_size,
                    num_channels,
                    self.params.img_local_shape_x,
                    self.params.img_local_shape_y,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(num_steps)
        ]
        tarlist = [
            torch.randn(
                (
                    num_rollout_steps,
                    batch_size,
                    num_channels,
                    self.params.img_local_shape_x,
                    self.params.img_local_shape_y,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(num_steps)
        ]

        for inp, tar in zip(inplist, tarlist):
            for idt in range(num_rollout_steps):
                inpp = inp[idt, ...]
                tarp = tar[idt, ...]

                # super simple l1 loss
                loss = torch.mean(torch.abs(torch.mean(inpp, dim=1) - tarp))

                # update metric handler
                metric_handler.update(inpp, tarp, loss, idt)

        # finalize
        _ = metric_handler.finalize()

        with tempfile.TemporaryDirectory() as tempdir:
            outfile = os.path.join(tempdir, "metrics.h5")
            metric_handler.save(outfile)

            # make sure file gor written
            self.assertTrue(os.path.isfile(outfile))

    @parameterized.expand(_metric_handler_params, skip_on_empty=True)
    def test_aggregation(self, grid_type, batch_size, ensemble_size, num_rollout_steps, bred, verbose=False):

        # create dummy climatology
        num_steps = 4
        num_channels = len(self.params.channel_names)
        clim = torch.zeros(1, num_channels, self.params.img_local_shape_x, self.params.img_local_shape_y)

        # update parameters
        self.params.batch_size = batch_size
        self.params.ensemble_size = ensemble_size

        metric_handler = MetricsHandler(
            self.params,
            clim,
            num_rollout_steps,
            self.device,
            l1_var_names=self.params.channel_names,
            rmse_var_names=self.params.channel_names,
            acc_var_names=self.params.channel_names,
            crps_var_names=self.params.channel_names,
            spread_var_names=self.params.channel_names,
            ssr_var_names=self.params.channel_names,
            rh_var_names=self.params.channel_names,
            wb2_compatible=False,
        )
        metric_handler.initialize_buffers()
        metric_handler.zero_buffers()

        metric_handler_split = MetricsHandler(
            self.params,
            clim,
            num_rollout_steps,
            self.device,
            l1_var_names=self.params.channel_names,
            rmse_var_names=self.params.channel_names,
            acc_var_names=self.params.channel_names,
            crps_var_names=self.params.channel_names,
            spread_var_names=self.params.channel_names,
            ssr_var_names=self.params.channel_names,
            rh_var_names=self.params.channel_names,
            wb2_compatible=False,
        )
        metric_handler_split.initialize_buffers()
        metric_handler_split.zero_buffers()

        inplist = [
            torch.randn(
                (
                    num_rollout_steps,
                    batch_size,
                    ensemble_size,
                    num_channels,
                    self.params.img_local_shape_x,
                    self.params.img_local_shape_y,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(num_steps)
        ]
        tarlist = [
            torch.randn(
                (
                    num_rollout_steps,
                    batch_size,
                    num_channels,
                    self.params.img_local_shape_x,
                    self.params.img_local_shape_y,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(num_steps)
        ]

        loss_acc = 0.0
        for inp, tar in zip(inplist, tarlist):
            for idt in range(num_rollout_steps):
                inpp = inp[idt, ...]
                tarp = tar[idt, ...]

                # super simple l1 loss
                loss = torch.mean(torch.abs(torch.mean(inpp, dim=1) - tarp))
                if idt == 0:
                    loss_acc += loss.item()

                # update metric handler
                metric_handler.update(inpp, tarp, loss, idt)

        # finalize
        loss_acc /= float(num_steps)
        logs_full = metric_handler.finalize()

        # compare loss
        mloss_full = logs_full["base"]["validation loss"]
        with self.subTest(desc="metrics_handler validation loss"):
            self.assertTrue(
                compare_arrays(
                    "metrics handler validation loss",
                    np.asarray(mloss_full),
                    np.asarray(loss_acc),
                    rtol=1e-6,
                    atol=1e-6,
                    verbose=verbose,
                )
            )

        # metric handler split:
        inplist_split = list(
            itertools.chain.from_iterable([torch.split(tens, batch_size // 2, dim=1) for tens in inplist])
        )
        tarlist_split = list(
            itertools.chain.from_iterable([torch.split(tens, batch_size // 2, dim=1) for tens in tarlist])
        )

        for inp, tar in zip(inplist_split, tarlist_split):
            for idt in range(num_rollout_steps):
                inpp = inp[idt, ...]
                tarp = tar[idt, ...]

                # super simple l1 loss
                loss = torch.mean(torch.abs(torch.mean(inpp, dim=1) - tarp))

                # update metric handler
                metric_handler_split.update(inpp, tarp, loss, idt)

        # finalize
        logs_split = metric_handler_split.finalize()

        # compare loss
        mloss_split = logs_split["base"]["validation loss"]
        with self.subTest(desc="metrics_handler_split validation loss"):
            self.assertTrue(
                compare_arrays(
                    "metrics handler split validation loss",
                    np.asarray(mloss_split),
                    np.asarray(loss_acc),
                    rtol=1e-6,
                    atol=1e-6,
                    verbose=verbose,
                )
            )

        # extract dicts
        metrics_full = logs_full["metrics"]
        metrics_split = logs_split["metrics"]

        # compare scalar metrics
        for key in metrics_full.keys():
            if key == "rollouts":
                continue
            val_full = metrics_full[key]
            val_split = metrics_split[key]
            with self.subTest(desc=f"validation {key}"):
                self.assertTrue(compare_arrays(f"{key}", val_full, val_split, rtol=1e-5, atol=1e-5, verbose=verbose))

        # compare rollouts
        rollouts_full = logs_full["metrics"]["rollouts"]
        rollouts_split = logs_split["metrics"]["rollouts"]

        # aggregate table into data
        data_full = []
        for row in rollouts_full.data:
            data_full.append(row[-1])
        data_full = np.array(data_full)

        data_split = []
        for row in rollouts_split.data:
            data_split.append(row[-1])
        data_split = np.array(data_split)

        with self.subTest(desc="rollouts"):
            self.assertTrue(compare_arrays("rollouts", data_full, data_split, rtol=1e-6, atol=1e-6, verbose=verbose))

    @parameterized.expand(_metric_handler_params, skip_on_empty=True)
    def test_aggregation_weighted(self, grid_type, batch_size, ensemble_size, num_rollout_steps, bred, verbose=False):
        """Test that weighted metric updates aggregate correctly when splitting batches."""
        # create dummy climatology
        num_steps = 4
        num_channels = len(self.params.channel_names)
        clim = torch.zeros(1, num_channels, self.params.img_local_shape_x, self.params.img_local_shape_y)

        # update parameters
        self.params.batch_size = batch_size
        self.params.ensemble_size = ensemble_size

        metric_handler = MetricsHandler(
            self.params,
            clim,
            num_rollout_steps,
            self.device,
            l1_var_names=self.params.channel_names,
            rmse_var_names=self.params.channel_names,
            acc_var_names=self.params.channel_names,
            crps_var_names=self.params.channel_names,
            spread_var_names=self.params.channel_names,
            ssr_var_names=self.params.channel_names,
            rh_var_names=self.params.channel_names,
            wb2_compatible=False,
        )
        metric_handler.initialize_buffers()
        metric_handler.zero_buffers()

        metric_handler_split = MetricsHandler(
            self.params,
            clim,
            num_rollout_steps,
            self.device,
            l1_var_names=self.params.channel_names,
            rmse_var_names=self.params.channel_names,
            acc_var_names=self.params.channel_names,
            crps_var_names=self.params.channel_names,
            spread_var_names=self.params.channel_names,
            ssr_var_names=self.params.channel_names,
            rh_var_names=self.params.channel_names,
            wb2_compatible=False,
        )
        metric_handler_split.initialize_buffers()
        metric_handler_split.zero_buffers()

        inplist = [
            torch.randn(
                (
                    num_rollout_steps,
                    batch_size,
                    ensemble_size,
                    num_channels,
                    self.params.img_local_shape_x,
                    self.params.img_local_shape_y,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(num_steps)
        ]
        tarlist = [
            torch.randn(
                (
                    num_rollout_steps,
                    batch_size,
                    num_channels,
                    self.params.img_local_shape_x,
                    self.params.img_local_shape_y,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(num_steps)
        ]

        # Random weights per (step, rollout_step, batch, channels, x, y) for use in updates
        weightlist = [
            torch.abs(
                torch.randn(
                    num_rollout_steps,
                    batch_size,
                    num_channels,
                    self.params.img_local_shape_x,
                    self.params.img_local_shape_y,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            for _ in range(num_rollout_steps)
        ]

        loss_acc = 0.0
        for i, (inp, tar, wgt) in enumerate(zip(inplist, tarlist, weightlist)):
            for idt in range(num_rollout_steps):
                inpp = inp[idt, ...]
                tarp = tar[idt, ...]
                wgtt = wgt[idt, ...]

                # super simple l1 loss
                loss = torch.mean(torch.abs(torch.mean(inpp, dim=1) - tarp))
                if idt == 0:
                    loss_acc += loss.item()

                metric_handler.update(inpp, tarp, loss, idt, weight=wgtt)

        # finalize
        loss_acc /= float(num_steps)
        logs_full = metric_handler.finalize()

        # metric handler split: same data and weights split in half along batch
        inplist_split = list(
            itertools.chain.from_iterable([torch.split(tens, batch_size // 2, dim=1) for tens in inplist])
        )
        tarlist_split = list(
            itertools.chain.from_iterable([torch.split(tens, batch_size // 2, dim=1) for tens in tarlist])
        )
        # split weights the same way: (num_steps, num_rollout_steps, batch_size, ...) -> per-step chunks of size batch_size//2
        weightlist_split = list(
            itertools.chain.from_iterable([torch.split(tens, batch_size // 2, dim=1) for tens in weightlist])
        )

        for inp, tar, wgt in zip(inplist_split, tarlist_split, weightlist_split):
            for idt in range(num_rollout_steps):
                inpp = inp[idt, ...]
                tarp = tar[idt, ...]
                wgtt = wgt[idt, ...]

                loss = torch.mean(torch.abs(torch.mean(inpp, dim=1) - tarp))
                metric_handler_split.update(inpp, tarp, loss, idt, weight=wgtt)

        # finalize
        logs_split = metric_handler_split.finalize()

        # compare loss
        mloss_split = logs_split["base"]["validation loss"]

        # extract dicts
        metrics_full = logs_full["metrics"]
        metrics_split = logs_split["metrics"]

        # compare scalar metrics
        for key in metrics_full.keys():
            if key == "rollouts":
                continue
            val_full = metrics_full[key]
            val_split = metrics_split[key]
            with self.subTest(desc=f"validation {key}"):
                self.assertTrue(compare_arrays(f"{key}", val_full, val_split, rtol=1e-5, atol=1e-5, verbose=verbose))

        # compare rollouts
        rollouts_full = logs_full["metrics"]["rollouts"]
        rollouts_split = logs_split["metrics"]["rollouts"]

        data_full = []
        for row in rollouts_full.data:
            data_full.append(row[-1])
        data_full = np.array(data_full)

        data_split = []
        for row in rollouts_split.data:
            data_split.append(row[-1])
        data_split = np.array(data_split)

        with self.subTest(desc="rollouts"):
            self.assertTrue(compare_arrays("rollouts", data_full, data_split, rtol=1e-5, atol=1e-5, verbose=verbose))

    @parameterized.expand(_metric_handler_params, skip_on_empty=True)
    def test_aggregation_weighted_vs_unweighted(
        self, grid_type, batch_size, ensemble_size, num_rollout_steps, bred, verbose=False
    ):
        """Unit weights should yield the same result as no weights if weighting is implemented correctly."""
        num_steps = 4
        num_channels = len(self.params.channel_names)
        clim = torch.zeros(1, num_channels, self.params.img_local_shape_x, self.params.img_local_shape_y)

        self.params.batch_size = batch_size
        self.params.ensemble_size = ensemble_size

        handler_unweighted = MetricsHandler(
            self.params,
            clim,
            num_rollout_steps,
            self.device,
            l1_var_names=self.params.channel_names,
            rmse_var_names=self.params.channel_names,
            acc_var_names=self.params.channel_names,
            crps_var_names=self.params.channel_names,
            spread_var_names=self.params.channel_names,
            ssr_var_names=self.params.channel_names,
            rh_var_names=self.params.channel_names,
            wb2_compatible=False,
        )
        handler_unweighted.initialize_buffers()
        handler_unweighted.zero_buffers()

        handler_unit_weights = MetricsHandler(
            self.params,
            clim,
            num_rollout_steps,
            self.device,
            l1_var_names=self.params.channel_names,
            rmse_var_names=self.params.channel_names,
            acc_var_names=self.params.channel_names,
            crps_var_names=self.params.channel_names,
            spread_var_names=self.params.channel_names,
            ssr_var_names=self.params.channel_names,
            rh_var_names=self.params.channel_names,
            wb2_compatible=False,
        )
        handler_unit_weights.initialize_buffers()
        handler_unit_weights.zero_buffers()

        inplist = [
            torch.randn(
                (
                    num_rollout_steps,
                    batch_size,
                    ensemble_size,
                    num_channels,
                    self.params.img_local_shape_x,
                    self.params.img_local_shape_y,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(num_steps)
        ]
        tarlist = [
            torch.randn(
                (
                    num_rollout_steps,
                    batch_size,
                    num_channels,
                    self.params.img_local_shape_x,
                    self.params.img_local_shape_y,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(num_steps)
        ]

        unit_weight = torch.ones(
            batch_size,
            num_channels,
            self.params.img_local_shape_x,
            self.params.img_local_shape_y,
            dtype=torch.float32,
            device=self.device,
        )

        for inp, tar in zip(inplist, tarlist):
            for idt in range(num_rollout_steps):
                inpp = inp[idt, ...]
                tarp = tar[idt, ...]
                loss = torch.mean(torch.abs(torch.mean(inpp, dim=1) - tarp))

                handler_unweighted.update(inpp, tarp, loss, idt)
                handler_unit_weights.update(inpp, tarp, loss, idt, weight=unit_weight)

        logs_unweighted = handler_unweighted.finalize()
        logs_unit_weights = handler_unit_weights.finalize()

        metrics_unweighted = logs_unweighted["metrics"]
        metrics_unit_weights = logs_unit_weights["metrics"]

        for key in metrics_unweighted.keys():
            if key == "rollouts":
                continue
            val_u = metrics_unweighted[key]
            val_w = metrics_unit_weights[key]
            with self.subTest(desc=f"validation {key}"):
                self.assertTrue(
                    compare_arrays(key, val_u, val_w, rtol=1e-5, atol=1e-5, verbose=verbose),
                    msg=f"Unit-weight and unweighted results differ for '{key}'",
                )

        rollouts_u = logs_unweighted["metrics"]["rollouts"]
        rollouts_w = logs_unit_weights["metrics"]["rollouts"]

        data_u = np.array([row[-1] for row in rollouts_u.data])
        data_w = np.array([row[-1] for row in rollouts_w.data])

        with self.subTest(desc="rollouts"):
            self.assertTrue(
                compare_arrays("rollouts", data_u, data_w, rtol=1e-5, atol=1e-5, verbose=verbose),
                msg="Unit-weight and unweighted rollouts differ",
            )


# TODO: ssr test comparing to weatherbench2
@unittest.skipUnless(_have_wb2, "test requires weatherbench2 installation")
@parameterized_class(("device",), _devices)
class ComparetMetricsWB2(unittest.TestCase):
    """
    A set of tests that compare weatherbench2 metrics to makani metrics
    """

    def setUp(self):

        set_seed(333)

    # same as above but compare to wb2
    @parameterized.expand(_wb2_metrics_params, skip_on_empty=True)
    def test_weighted_crps(self, grid_type, batch_size, ensemble_size, num_channels, nlat, nlon, verbose=False):

        # some imports
        from weatherbench2.metrics import CRPS

        # CRPS handle
        crps_func = GeometricCRPS(
            img_shape=(nlat, nlon),
            crop_shape=(nlat, nlon),
            crop_offset=(0, 0),
            channel_names=[],
            grid_type=grid_type,
            crps_type="skillspread",
            channel_reduction="none",
            batch_reduction="none",
        ).to(self.device)

        # generate toy data
        obs = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        fct = torch.randn(batch_size, ensemble_size, num_channels, nlat, nlon, device=self.device)

        # create xarray datasets according to wb2 specification
        # wb2 requires latitude monotonically increasing (south→north);
        # get_lat_lon_grid returns north→south, so flip lat axis.
        latitude, longitude = get_lat_lon_grid((nlat, nlon))
        latitude_sn = latitude[::-1]
        obs_np = obs.cpu().numpy()[..., ::-1, :]
        fct_np = fct.cpu().numpy()[..., ::-1, :]
        xr_obs = xr.Dataset(
            data_vars=dict(var=(["batch", "channel", "latitude", "longitude"], obs_np)),
            coords=dict(latitude=latitude_sn, longitude=longitude),
        )
        xr_fct = xr.Dataset(
            data_vars=dict(var=(["batch", "ensemble", "channel", "latitude", "longitude"], fct_np)),
            coords=dict(latitude=latitude_sn, longitude=longitude),
        )

        # compute and compare CRPS
        crps = crps_func(fct, obs).cpu().numpy()
        crps_wb2 = CRPS(ensemble_dim="ensemble").compute_chunk(xr_fct, xr_obs, region=None)["var"].to_numpy()

        self.assertTrue(compare_arrays("crps", crps, crps_wb2, rtol=1e-3, atol=1e-3, verbose=verbose))

    # same as above but compare to wb2
    @parameterized.expand(_wb2_metrics_params, skip_on_empty=True)
    def test_weighted_ssr(self, grid_type, batch_size, ensemble_size, num_channels, nlat, nlon, verbose=False):

        # some imports
        from weatherbench2.metrics import EnergyScoreSkill, EnergyScoreSpread

        # CRPS handle
        ssr_func = GeometricSSR(
            img_shape=(nlat, nlon),
            crop_shape=(nlat, nlon),
            crop_offset=(0, 0),
            channel_names=[],
            grid_type=grid_type,
            channel_reduction="none",
            batch_reduction="none",
        ).to(self.device)

        # generate toy data
        obs = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        fct = torch.randn(batch_size, ensemble_size, num_channels, nlat, nlon, device=self.device)

        # create xarray datasets according to wb2 specification
        # wb2 requires latitude monotonically increasing (south→north);
        # get_lat_lon_grid returns north→south, so flip lat axis.
        latitude, longitude = get_lat_lon_grid((nlat, nlon))
        latitude_sn = latitude[::-1]
        obs_np = obs.cpu().numpy()[..., ::-1, :]
        fct_np = fct.cpu().numpy()[..., ::-1, :]
        xr_obs = xr.Dataset(
            data_vars=dict(var=(["batch", "channel", "latitude", "longitude"], obs_np)),
            coords=dict(latitude=latitude_sn, longitude=longitude),
        )
        xr_fct = xr.Dataset(
            data_vars=dict(var=(["batch", "ensemble", "channel", "latitude", "longitude"], fct_np)),
            coords=dict(latitude=latitude_sn, longitude=longitude),
        )

        # compute and compare ssr
        ssr = ssr_func(fct, obs).cpu().numpy()
        ssr_skill_wb2 = (
            EnergyScoreSkill(ensemble_dim="ensemble").compute_chunk(xr_fct, xr_obs, region=None)["var"].to_numpy()
        )
        ssr_spread_wb2 = (
            EnergyScoreSpread(ensemble_dim="ensemble").compute_chunk(xr_fct, xr_obs, region=None)["var"].to_numpy()
        )
        ssr_wb2 = ssr_spread_wb2 / ssr_skill_wb2

        self.assertTrue(compare_arrays("ssr", ssr, ssr_wb2, rtol=5e-2, atol=5e-2, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
