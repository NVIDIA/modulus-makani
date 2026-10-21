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

import sys
import os
import math
import tempfile
from typing import Optional
from parameterized import parameterized, parameterized_class

import unittest
import numpy as np
import torch

from makani.utils import LossHandler
from makani.utils.missing_data import fill_missing, missing_values, missing_weights
from makani.utils.losses.base_loss import abs_pow
from makani.utils.losses import (
    CRPSLoss,
    SpectralCRPSLoss,
    GradientCRPSLoss,
    VortDivCRPSLoss,
    GaussianMMDLoss,
    GeometricLpLoss,
    SpectralLpLoss,
    SpectralH1Loss,
    SpectralAMSELoss,
    DriftRegularization,
    SpectralRegularization,
    CoherenceRegularization,
    EnsembleNLLLoss,
    LpEnergyScoreLoss,
    SpectralL2EnergyScoreLoss,
)
from makani.utils.losses.energy_score import (
    SobolevEnergyScoreLoss,
    SpectralCoherenceLoss,
    CorrectedSpectralL2EnergyScoreLoss,
)
from makani.utils.losses.crps_loss import KernelScoreLoss
from makani.utils.losses.base_loss import _compute_channel_weighting_helper, compute_alpha_per_step
from makani.utils.losses.crps_loss import _crps_skillspread_kernel, _crps_gauss_kernel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, get_default_parameters, compare_tensors, compare_arrays

from properscoring import crps_ensemble, crps_gaussian

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))

# compile on/off axis, parameterized the same way as the device axis
_compile_opts = [False, True]

_loss_params = [
    ([{"type": "l1"}], False),
    ([{"type": "l1", "parameters": {"relative": True}}], False),
    ([{"type": "l2", "parameters": {"squared": True}}], False),
    ([{"type": "l2", "channel_weights": "constant"}], True),
    ([{"type": "l2", "channel_weights": "constant"}, {"type": "l2", "channel_weights": "auto"}], True),
    ([{"type": "h1", "channel_weights": "constant"}], True),
    ([{"type": "l2", "channel_weights": "constant", "temp_diff_normalization": True}], True),
    ([{"type": "l2", "channel_weights": "constant"}, {"type": "h1", "channel_weights": "constant"}], True),
    ([{"type": "l2", "channel_weights": "constant"}, {"type": "l1", "channel_weights": "constant"}], True),
    ([{"type": "drift_regularization"}], False),
]

# losses checked for masking of missing (NaN) targets in the LossHandler,
# covering losses on the grid, spectral ones and the relative variant
_missing_target_deterministic_params = [
    ([{"type": "l1", "channel_weights": "constant"}],),
    ([{"type": "l2", "channel_weights": "constant"}],),
    ([{"type": "l2", "channel_weights": "constant", "parameters": {"relative": True}}],),
    ([{"type": "spectral l2", "channel_weights": "constant"}],),
    ([{"type": "h1", "channel_weights": "constant"}],),
    ([{"type": "amse", "channel_weights": "constant"}],),
    ([{"type": "drift_regularization"}],),
]

_missing_target_probabilistic_params = [
    ([{"type": "ensemble_crps", "channel_weights": "constant", "parameters": {"crps_type": "skillspread"}}],),
    ([{"type": "ensemble_crps", "channel_weights": "constant", "parameters": {"crps_type": "cdf"}}],),
    ([{"type": "ensemble_spectral_crps", "channel_weights": "constant"}],),
    ([{"type": "l2_energy_score", "channel_weights": "constant"}],),
]

_loss_weighted_params = [
    ([{"type": "l1"}], False),
    ([{"type": "l1", "parameters": {"relative": True}}], False),
    ([{"type": "l2", "parameters": {"squared": True}}], False),
    ([{"type": "l2", "channel_weights": "constant"}], False),
    ([{"type": "l2", "channel_weights": "constant"}, {"type": "l2", "channel_weights": "auto"}], False),
    ([{"type": "l2", "channel_weights": "constant", "temp_diff_normalization": True}], False),
    ([{"type": "l2", "channel_weights": "constant"}, {"type": "l1", "channel_weights": "constant"}], False),
    ([{"type": "drift_regularization"}], False),
]

_loss_zero_params = [
    ([{"type": "l1"}], False),
    ([{"type": "l2"}], False),
    ([{"type": "l2", "channel_weights": "constant"}], False),
    ([{"type": "l1", "channel_weights": "constant"}], False),
    ([{"type": "h1", "channel_weights": "constant"}], False),
    ([{"type": "drift_regularization"}], False),
]

# ---------------------------------------------------------------------------
# Shared constants for direct loss instantiation tests
# ---------------------------------------------------------------------------
_IMG_H = 32
_IMG_W = 64
_BATCH = 4
_NUM_CH = 5
_CHANNEL_NAMES = ["u10m", "t2m", "u500", "z500", "t500"]

_WIND_CHANNEL_NAMES = ["u500", "v500", "u850", "v850", "t500"]
_NUM_WIND_CH = len(_WIND_CHANNEL_NAMES)

_GEOM_KWARGS = dict(
    img_shape=(_IMG_H, _IMG_W),
    crop_shape=(_IMG_H, _IMG_W),
    crop_offset=(0, 0),
    channel_names=_CHANNEL_NAMES,
    grid_type="equiangular",
)

_WIND_GEOM_KWARGS = dict(
    img_shape=(_IMG_H, _IMG_W),
    crop_shape=(_IMG_H, _IMG_W),
    crop_offset=(0, 0),
    channel_names=_WIND_CHANNEL_NAMES,
    grid_type="equiangular",
)

_SPEC_KWARGS = dict(
    img_shape=(_IMG_H, _IMG_W),
    crop_shape=(_IMG_H, _IMG_W),
    crop_offset=(0, 0),
    channel_names=_CHANNEL_NAMES,
    grid_type="equiangular",
)


def _rand(batch=_BATCH, channels=_NUM_CH, requires_grad=False):
    t = torch.randn(batch, channels, _IMG_H, _IMG_W)
    t.requires_grad_(requires_grad)
    return t


def _imaginary_only_pair(ensemble=5, wavenumber=3, delta=1.0):
    """Build an (forecast, observation) pair differing ONLY in Im(SH coefficients).

    On the uniform longitude grid the SHT's azimuthal transform is a DFT, so for a
    field ``g(theta) * cos(m*lambda)`` the coefficients at order m are purely real,
    while ``g(theta) * sin(m*lambda)`` gives purely imaginary ones. Adding a scaled
    sine component therefore perturbs the imaginary part of the coefficients and
    leaves the real part bit-for-bit identical.

    A spectral loss built on the squared modulus |z|^2 must see this as an error.
    A loss that discards Im(z) scores it as a perfect forecast.

    Every ensemble member is the same perturbed field, so the pairwise spread term
    vanishes and the returned loss is the accuracy (eskill) term alone.
    """

    lon = 2.0 * math.pi * torch.arange(_IMG_W, dtype=torch.float32) / _IMG_W
    profile = torch.sin(torch.linspace(0.0, math.pi, _IMG_H, dtype=torch.float32))

    base = profile.unsqueeze(-1) * torch.cos(wavenumber * lon).unsqueeze(0)
    perturbation = profile.unsqueeze(-1) * torch.sin(wavenumber * lon).unsqueeze(0)

    observations = base.expand(_BATCH, _NUM_CH, _IMG_H, _IMG_W).contiguous()
    forecasts = (base + delta * perturbation).expand(_BATCH, ensemble, _NUM_CH, _IMG_H, _IMG_W).contiguous()

    return forecasts, observations


def _rand_ensemble(ensemble=5, batch=_BATCH, channels=_NUM_CH, requires_grad=False):
    t = torch.randn(batch, ensemble, channels, _IMG_H, _IMG_W)
    t.requires_grad_(requires_grad)
    return t


# ---------------------------------------------------------------------------
# Parameter lists for TestLossCommon
# ---------------------------------------------------------------------------

# Losses expected to be elementwise non-negative.
# EnsembleNLLLoss excluded: proper scoring rule that can be negative.
# GaussianMMDLoss excluded: the unbiased U-statistic MMD² can be negative
#   (e.g., all E members = obs with E=5 gives mmd² = (3-E)/(E-1) = -0.5).
_COMMON_NONNEG = [
    ("geometric_l2",),
    ("geometric_l1",),
    ("spectral_l2",),
    ("spectral_h1",),
    ("drift_regularization",),
    ("crps_cdf",),
    ("crps_gauss",),
    ("crps_pwm",),
    ("crps_naive_skillspread",),
]

# Losses expected to be (near) zero when prd perfectly matches tar.
# EnsembleNLLLoss excluded: with a degenerate ensemble sigma is clipped to eps,
#   leaving a residual log(eps^2)/2 term that is large and negative.
# GaussianMMDLoss excluded: perfect prediction gives mmd² = (3-E)/(E-1) ≠ 0.
# crps_gauss included: with sigma clamped to eps the residual ≈ eps * 0.23 ≈ 2e-6,
#   well within atol=1e-4.
_COMMON_ZERO_PERFECT = [
    ("geometric_l2",),
    ("geometric_l1",),
    ("spectral_l2",),
    ("spectral_h1",),
    ("drift_regularization",),
    ("crps_cdf",),
    ("crps_gauss",),
    ("crps_pwm",),
    ("crps_naive_skillspread",),
]

# All losses participate in the batch-size independence test.
# GaussianMMDLoss is tested with squared=True to avoid sqrt of potentially negative mmd².
_COMMON_BATCHSIZE = [
    ("geometric_l2",),
    ("geometric_l1",),
    ("spectral_l2",),
    ("spectral_h1",),
    ("drift_regularization",),
    ("crps_cdf",),
    ("crps_gauss",),
    ("crps_pwm",),
    ("crps_naive_skillspread",),
    ("nll",),
    ("mmd",),
]


# ===========================================================================
class TestLossCommon(unittest.TestCase):
    """Common property tests executed directly against every loss class.

    Four properties are verified:
      1. ``test_nonneg``                — loss >= 0 elementwise
      2. ``test_zero_on_perfect_prediction`` — loss ≈ 0 when prd == tar
      3. ``test_batchsize_independence``    — loss[i] is unaffected by other samples in the batch
      4. ``test_nan_propagates``            — NaN in prediction or target is not masked
    """

    _E = 5  # ensemble size used for probabilistic losses

    def setUp(self):
        disable_tf32()
        set_seed(333)

    @staticmethod
    def _make(name: str):
        """Return a freshly constructed loss instance for *name*."""
        if name == "geometric_l2":
            return GeometricLpLoss(**_GEOM_KWARGS, p=2.0)
        if name == "geometric_l1":
            return GeometricLpLoss(**_GEOM_KWARGS, p=1.0)
        if name == "spectral_l2":
            return SpectralLpLoss(**_SPEC_KWARGS)
        if name == "spectral_h1":
            return SpectralH1Loss(**_SPEC_KWARGS)
        if name == "drift_regularization":
            return DriftRegularization(**_GEOM_KWARGS)
        if name == "crps_cdf":
            return CRPSLoss(**_GEOM_KWARGS, crps_type="cdf", spatial_distributed=False, ensemble_distributed=False)
        if name == "crps_gauss":
            return CRPSLoss(
                **_GEOM_KWARGS, crps_type="gauss", spatial_distributed=False, ensemble_distributed=False, eps=1e-5
            )
        if name == "crps_pwm":
            return CRPSLoss(
                **_GEOM_KWARGS,
                crps_type="probability weighted moment",
                spatial_distributed=False,
                ensemble_distributed=False,
            )
        if name == "crps_naive_skillspread":
            return CRPSLoss(
                **_GEOM_KWARGS, crps_type="naive skillspread", spatial_distributed=False, ensemble_distributed=False
            )
        if name == "nll":
            return EnsembleNLLLoss(**_GEOM_KWARGS)
        if name == "mmd":
            # squared=True: avoids sqrt of potentially negative mmd² values in common tests
            return GaussianMMDLoss(**_GEOM_KWARGS, squared=True)
        raise ValueError(f"Unknown loss name: {name!r}")

    @classmethod
    def _make_prd_tar(cls, name: str, perfect: bool = False):
        """Return *(prd, tar)*.  Ensemble losses get a 5-D prd; others 4-D."""
        E = cls._E
        tar = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        if name in ("crps_cdf", "crps_gauss", "crps_pwm", "crps_naive_skillspread", "nll", "mmd"):
            if perfect:
                # all E members equal the observation
                prd = tar.unsqueeze(1).expand(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W).clone()
            else:
                prd = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        else:
            prd = tar.clone() if perfect else torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        return prd, tar

    # ------------------------------------------------------------------

    @parameterized.expand(_COMMON_NONNEG)
    def test_nonneg(self, name):
        """Loss output must be elementwise non-negative."""
        fn = self._make(name)
        prd, tar = self._make_prd_tar(name)
        loss = fn(prd, tar)
        self.assertTrue(
            (loss >= -1e-6).all(),
            f"{name}: found negative values, min={loss.min().item():.4e}",
        )

    @parameterized.expand(_COMMON_ZERO_PERFECT)
    def test_zero_on_perfect_prediction(self, name, verbose=False):
        """Loss must be (near) zero when the prediction perfectly matches the target."""
        fn = self._make(name)
        prd, tar = self._make_prd_tar(name, perfect=True)
        loss = fn(prd, tar)
        self.assertTrue(
            compare_tensors(f"{name} zero", loss, torch.zeros_like(loss), atol=1e-4, verbose=verbose),
        )

    @parameterized.expand(_COMMON_BATCHSIZE)
    def test_batchsize_independence(self, name, verbose=False):
        """The loss for sample [0] computed alone must equal loss[0] in a full batch."""
        fn = self._make(name)
        prd, tar = self._make_prd_tar(name)

        loss_single = fn(prd[:1], tar[:1])  # (1, C)
        loss_batch = fn(prd, tar)  # (B, C)

        self.assertTrue(
            compare_tensors(f"{name} batchsize", loss_single[0], loss_batch[0], verbose=verbose),
            f"{name}: loss[0] differs between single-sample and full-batch evaluation",
        )

    @parameterized.expand(_COMMON_BATCHSIZE)
    def test_nan_propagates(self, name):
        """The losses assume finite inputs and contain no NaN handling of their own:
        a NaN in the prediction or the target makes the loss of that sample and
        channel NaN. Missing targets are taken care of before the losses, see
        makani.utils.missing_data."""
        fn = self._make(name)
        for where in ["prediction", "target"]:
            with self.subTest(nan_in=where):
                prd, tar = self._make_prd_tar(name)
                if where == "target":
                    tar[0, 0, 3, 5] = torch.nan
                elif prd.dim() == 5:
                    # a single ensemble member is enough
                    prd[0, 0, 0, 3, 5] = torch.nan
                else:
                    prd[0, 0, 3, 5] = torch.nan
                loss = fn(prd, tar)
                self.assertTrue(torch.isnan(loss[0, 0]), f"{name}: NaN in the {where} was masked")


# ===========================================================================
class TestGeometricLpLoss(unittest.TestCase):
    """Specific tests for GeometricLpLoss beyond what TestLosses covers via LossHandler."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_squared_flag_consistency(self, verbose=False):
        """For p=2, loss(squared=False)^2 must equal loss(squared=True)."""
        fn_unsq = GeometricLpLoss(**_GEOM_KWARGS, p=2.0, squared=False)
        fn_sq = GeometricLpLoss(**_GEOM_KWARGS, p=2.0, squared=True)
        prd, tar = _rand(), _rand()
        self.assertTrue(
            compare_tensors(
                "squared flag", fn_unsq(prd, tar) ** 2, fn_sq(prd, tar), atol=1e-5, rtol=1e-4, verbose=verbose
            ),
        )

    @parameterized.expand([(1.0,), (2.0,), (4.0,)])
    def test_analytic_constant_difference(self, p, verbose=False):
        """Lp loss (squared=False) of a spatially constant difference c over a normalised
        grid equals c, because the quadrature integrates a constant field to 1."""
        c = 2.5
        fn = GeometricLpLoss(**_GEOM_KWARGS, p=p, squared=False)
        prd = torch.full((_BATCH, _NUM_CH, _IMG_H, _IMG_W), c)
        tar = torch.zeros_like(prd)
        loss = fn(prd, tar)
        self.assertTrue(
            compare_tensors(f"analytic L{p}", loss, torch.full_like(loss, c), atol=1e-4, rtol=1e-4, verbose=verbose),
        )

    def test_p_parameter_differentiated(self):
        """L1 and L2 norms must differ for a sparse large-value input."""
        fn_l1 = GeometricLpLoss(**_GEOM_KWARGS, p=1.0)
        fn_l2 = GeometricLpLoss(**_GEOM_KWARGS, p=2.0)
        prd = torch.zeros(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        prd[:, :, _IMG_H // 2, _IMG_W // 2] = 10.0  # single bright pixel per sample
        tar = torch.zeros_like(prd)
        diff = abs(fn_l1(prd, tar).mean().item() - fn_l2(prd, tar).mean().item())
        self.assertGreater(diff, 1e-3, "L1 and L2 norms should differ for a sparse input")

    @parameterized.expand([(1.0,), (4.0,)])
    def test_squared_flag_general(self, p, verbose=False):
        """For arbitrary p: loss(squared=False)^p must equal loss(squared=True)."""
        fn_unsq = GeometricLpLoss(**_GEOM_KWARGS, p=p, squared=False)
        fn_sq = GeometricLpLoss(**_GEOM_KWARGS, p=p, squared=True)
        prd, tar = _rand(), _rand()
        self.assertTrue(
            compare_tensors(
                f"squared flag p={p}", fn_unsq(prd, tar) ** p, fn_sq(prd, tar), atol=1e-4, rtol=1e-3, verbose=verbose
            ),
        )

    @parameterized.expand([(1.0,), (2.0,), (4.0,)])
    def test_relative_loss_double_target(self, p, verbose=False):
        """relative=True: prd = 2*tar gives loss = 1 for any p.

        Proof: relative loss = (∫|2t-t|^p / ∫|t|^p)^(1/p) = (∫|t|^p / ∫|t|^p)^(1/p) = 1.
        """
        fn = GeometricLpLoss(**_GEOM_KWARGS, p=p, relative=True, squared=False)
        set_seed(333)
        tar = _rand() + 2.0  # shift away from zero to keep denominator well-conditioned
        prd = 2.0 * tar
        loss = fn(prd, tar)
        self.assertTrue(
            compare_tensors(
                f"relative L{p} double-target", loss, torch.ones_like(loss), atol=1e-4, rtol=1e-4, verbose=verbose
            ),
        )

    @parameterized.expand([(1.0,), (4.0,)])
    def test_gradient_flow(self, p):
        """abs() mode must produce finite, non-NaN gradients for p=1 and p=4."""
        fn = GeometricLpLoss(**_GEOM_KWARGS, p=p, squared=False)
        prd = _rand(requires_grad=True)
        tar = _rand()
        fn(prd, tar).sum().backward()
        self.assertFalse(torch.isnan(prd.grad).any(), f"p={p}: NaN in gradient")
        self.assertFalse(torch.isinf(prd.grad).any(), f"p={p}: Inf in gradient")


# ===========================================================================
class TestSpectralLpLoss(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_squared_flag_consistency(self, verbose=False):
        """loss(squared=False)^2 must equal loss(squared=True)."""
        fn_unsq = SpectralLpLoss(**_SPEC_KWARGS, squared=False)
        fn_sq = SpectralLpLoss(**_SPEC_KWARGS, squared=True)
        prd, tar = _rand(), _rand()
        self.assertTrue(
            compare_tensors("squared", fn_unsq(prd, tar) ** 2, fn_sq(prd, tar), atol=1e-5, rtol=1e-4, verbose=verbose),
        )

    def test_parseval_consistency_with_geometric_l2(self, verbose=False):
        """SpectralLpLoss and GeometricLpLoss(p=2) both approximate the normalised L2 norm
        on the sphere and should agree within 5 % for a smooth single-mode field."""
        fn_spec = SpectralLpLoss(**_SPEC_KWARGS, squared=True)
        fn_geom = GeometricLpLoss(**_GEOM_KWARGS, p=2.0, squared=True)

        lat = torch.linspace(0, math.pi, _IMG_H)
        lon = torch.linspace(0, 2.0 * math.pi, _IMG_W)
        LAT, LON = torch.meshgrid(lat, lon, indexing="ij")
        smooth = (torch.sin(LAT) * torch.cos(LON)).expand(_BATCH, _NUM_CH, -1, -1).clone()
        zeros = torch.zeros_like(smooth)

        loss_spec = fn_spec(smooth, zeros)
        loss_geom = fn_geom(smooth, zeros)
        self.assertTrue(
            compare_tensors("parseval", loss_spec, loss_geom, atol=1e-3, rtol=0.05, verbose=verbose),
            f"Spectral and geometric L2 should agree within 5 %: "
            f"spec={loss_spec.mean().item():.4f}, geom={loss_geom.mean().item():.4f}",
        )


# ===========================================================================
class TestSpectralH1Loss(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_squared_flag_consistency(self, verbose=False):
        fn_unsq = SpectralH1Loss(**_SPEC_KWARGS, squared=False)
        fn_sq = SpectralH1Loss(**_SPEC_KWARGS, squared=True)
        prd, tar = _rand(), _rand()
        self.assertTrue(
            compare_tensors("squared", fn_unsq(prd, tar) ** 2, fn_sq(prd, tar), atol=1e-5, rtol=1e-4, verbose=verbose),
        )

    def test_parseval_consistency_with_geometric_l2(self, verbose=False):
        """On a single zonal harmonic Y_l^0 (L²-normalized), Parseval gives
        ||Y_l^0||²_{H¹} = l(l+1) · ||Y_l^0||²_{L²}.  With the 1/(4π) Parseval factor
        applied to both losses, SpectralH1Loss(squared=True) must equal
        l(l+1) · GeometricLpLoss(p=2, squared=True)."""
        import torch_harmonics as th
        from makani.utils.grids import compute_spherical_bandlimit

        l_test = 4
        lmax = compute_spherical_bandlimit((_IMG_H, _IMG_W), "equiangular")

        # build single-mode field by inverse SHT of a unit coefficient at (l, m=0)
        isht = th.InverseRealSHT(_IMG_H, _IMG_W, lmax=lmax, mmax=lmax, grid="equiangular").float()
        coeffs = torch.zeros(lmax, lmax, dtype=torch.complex64)
        coeffs[l_test, 0] = 1.0 + 0.0j
        single_mode = isht(coeffs).expand(_BATCH, _NUM_CH, -1, -1).contiguous()
        zeros = torch.zeros_like(single_mode)

        fn_h1 = SpectralH1Loss(**_SPEC_KWARGS, squared=True)
        fn_l2 = GeometricLpLoss(**_GEOM_KWARGS, p=2.0, squared=True)

        h1_val = fn_h1(single_mode, zeros)
        l2_val = fn_l2(single_mode, zeros)

        expected = float(l_test * (l_test + 1)) * l2_val
        self.assertTrue(
            compare_tensors("parseval H1", h1_val, expected, atol=1e-3, rtol=0.05, verbose=verbose),
            f"Expected H1 = l(l+1)·L²: h1={h1_val.mean().item():.4f}, " f"l(l+1)*l2={expected.mean().item():.4f}",
        )

    def test_constant_difference_has_zero_h1_seminorm(self, verbose=False):
        """A spatially constant field lives entirely in the l=0 SHT mode.
        h1_weights[0] = 0*(0+1) = 0, so the H1 seminorm must be exactly zero."""
        fn = SpectralH1Loss(**_SPEC_KWARGS, squared=True)
        prd = torch.full((_BATCH, _NUM_CH, _IMG_H, _IMG_W), 3.0)
        tar = torch.zeros_like(prd)
        loss = fn(prd, tar)
        self.assertTrue(
            compare_tensors("constant diff h1", loss, torch.zeros_like(loss), atol=1e-4, verbose=verbose),
        )

    def test_high_frequency_penalized_more_than_smooth(self):
        """After L2-normalisation, a high-frequency field (l≈8) must score higher H1 than a
        smooth field (l≈1) because h1_weights = l*(l+1) amplifies high modes."""
        fn = SpectralH1Loss(**_SPEC_KWARGS, squared=True)

        lat = torch.linspace(0, math.pi, _IMG_H)
        lon = torch.linspace(0, 2.0 * math.pi, _IMG_W)
        LAT, LON = torch.meshgrid(lat, lon, indexing="ij")

        smooth = torch.sin(LAT).expand(_BATCH, _NUM_CH, -1, -1).clone()
        rough = (torch.sin(8 * LAT) * torch.cos(8 * LON)).expand(_BATCH, _NUM_CH, -1, -1).clone()

        # normalise to same Frobenius norm so only frequency content differs
        rough = rough * smooth.norm() / rough.norm().clamp(min=1e-6)

        tar = torch.zeros_like(smooth)
        h1_smooth = fn(smooth, tar).mean().item()
        h1_rough = fn(rough, tar).mean().item()
        self.assertGreater(
            h1_rough,
            h1_smooth,
            f"Rough H1 ({h1_rough:.4f}) should exceed smooth H1 ({h1_smooth:.4f})",
        )


# ===========================================================================
class TestSpectralRelativeLoss(unittest.TestCase):
    """Tests for the relative mode of SpectralLpLoss and SpectralH1Loss.

    The relative loss is defined as  ||SHT(prd - tar)|| / ||SHT(tar)||
    (with the H1 weighting for SpectralH1Loss).  The tests below verify
    mathematical properties that hold regardless of the spherical geometry.
    """

    def setUp(self):
        disable_tf32()
        set_seed(333)

    @staticmethod
    def _make(cls, squared=False):
        return cls(**_SPEC_KWARGS, relative=True, squared=squared)

    # --- zero on perfect prediction ---

    def test_l2_zero_on_perfect_prediction(self, verbose=False):
        fn = self._make(SpectralLpLoss)
        tar = _rand()
        loss = fn(tar.clone(), tar)
        self.assertTrue(
            compare_tensors("l2 rel perfect", loss, torch.zeros_like(loss), atol=1e-5, verbose=verbose),
        )

    def test_h1_zero_on_perfect_prediction(self, verbose=False):
        fn = self._make(SpectralH1Loss)
        tar = _rand()
        loss = fn(tar.clone(), tar)
        self.assertTrue(
            compare_tensors("h1 rel perfect", loss, torch.zeros_like(loss), atol=1e-5, verbose=verbose),
        )

    # --- unity when prd = 2 * tar  (||2t - t|| = ||t||, so ratio = 1) ---

    def test_l2_unity_when_prd_equals_twice_tar(self, verbose=False):
        fn = self._make(SpectralLpLoss)
        tar = _rand()
        loss = fn(2.0 * tar, tar)
        self.assertTrue(
            compare_tensors("l2 rel 2x", loss, torch.ones_like(loss), atol=1e-4, rtol=1e-3, verbose=verbose),
        )

    def test_h1_unity_when_prd_equals_twice_tar(self, verbose=False):
        fn = self._make(SpectralH1Loss)
        tar = _rand()
        loss = fn(2.0 * tar, tar)
        self.assertTrue(
            compare_tensors("h1 rel 2x", loss, torch.ones_like(loss), atol=1e-4, rtol=1e-3, verbose=verbose),
        )

    # --- squared flag consistency in relative mode ---

    def test_l2_squared_flag_consistency(self, verbose=False):
        """rel(squared=False)² must equal rel(squared=True)."""
        fn_unsq = self._make(SpectralLpLoss, squared=False)
        fn_sq = self._make(SpectralLpLoss, squared=True)
        prd, tar = _rand(), _rand()
        self.assertTrue(
            compare_tensors(
                "l2 rel squared", fn_unsq(prd, tar) ** 2, fn_sq(prd, tar), atol=1e-5, rtol=1e-4, verbose=verbose
            ),
        )

    def test_h1_squared_flag_consistency(self, verbose=False):
        fn_unsq = self._make(SpectralH1Loss, squared=False)
        fn_sq = self._make(SpectralH1Loss, squared=True)
        prd, tar = _rand(), _rand()
        self.assertTrue(
            compare_tensors(
                "h1 rel squared", fn_unsq(prd, tar) ** 2, fn_sq(prd, tar), atol=1e-5, rtol=1e-4, verbose=verbose
            ),
        )

    # --- larger error → larger relative loss ---

    def test_l2_monotone_in_error(self):
        """A prediction farther from the target must have a larger relative loss."""
        fn = self._make(SpectralLpLoss)
        set_seed(333)
        tar = _rand()
        noise = torch.randn_like(tar)
        loss_small = fn(tar + 0.1 * noise, tar).mean().item()
        loss_large = fn(tar + 2.0 * noise, tar).mean().item()
        self.assertLess(
            loss_small,
            loss_large,
            f"L2 rel: small-error loss {loss_small:.4f} should be < large-error {loss_large:.4f}",
        )

    def test_h1_monotone_in_error(self):
        fn = self._make(SpectralH1Loss)
        set_seed(333)
        tar = _rand()
        noise = torch.randn_like(tar)
        loss_small = fn(tar + 0.1 * noise, tar).mean().item()
        loss_large = fn(tar + 2.0 * noise, tar).mean().item()
        self.assertLess(
            loss_small,
            loss_large,
            f"H1 rel: small-error loss {loss_small:.4f} should be < large-error {loss_large:.4f}",
        )


# ===========================================================================
class TestSpectralAMSELoss(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self):
        return SpectralAMSELoss(**_SPEC_KWARGS)

    def test_amplitude_difference_penalized(self):
        """prd = 2*tar has identical phase but double amplitude -> amplitude term > 0, loss > 0."""
        fn = self._fn()
        tar = _rand()
        loss_perfect = fn(tar, tar).sum().item()
        loss_2x = fn(2.0 * tar, tar).sum().item()
        self.assertGreater(loss_2x, loss_perfect + 1e-4)

    def test_phase_difference_penalized(self):
        """cos(phi) and sin(phi) share the same power spectrum but have orthogonal phase:
        their spectral inner product is purely imaginary -> coherence = 0 -> loss > 0."""
        fn = self._fn()

        lon = torch.linspace(0, 2.0 * math.pi, _IMG_W)
        LON = lon.unsqueeze(0).expand(_IMG_H, -1)
        field_cos = torch.cos(LON).expand(_BATCH, _NUM_CH, -1, -1).clone()
        field_sin = torch.sin(LON).expand(_BATCH, _NUM_CH, -1, -1).clone()

        loss = fn(field_cos, field_sin)
        self.assertTrue(
            (loss > 1e-4).all(),
            f"Orthogonal-phase fields must have positive AMSE; min={loss.min().item():.2e}",
        )


# ===========================================================================
class TestDriftRegularization(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, p=1.0):
        return DriftRegularization(**_GEOM_KWARGS, p=p)

    def test_spatial_structure_insensitive(self, verbose=False):
        """Drift measures only the global spatial mean.  Two different spatial patterns
        with equal quadrature integrals must give zero loss."""
        fn = self._fn()
        prd = _rand()
        tar = _rand()

        # shift tar so that quadrature(tar_adjusted) == quadrature(prd) exactly
        prd_mean = fn.quadrature(prd).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        tar_mean = fn.quadrature(tar).unsqueeze(-1).unsqueeze(-1)
        tar_adjusted = tar - tar_mean + prd_mean

        loss = fn(prd, tar_adjusted)
        self.assertTrue(
            compare_tensors("zero drift", loss, torch.zeros_like(loss), atol=1e-6, verbose=verbose),
        )

    @parameterized.expand([(1.0,), (2.0,)])
    def test_scales_with_constant_bias(self, p, verbose=False):
        """For prd = tar + c (uniform bias), drift = c^p.
        The normalised quadrature maps a constant field to itself, so the bias is preserved exactly."""
        c = 2.0
        fn = self._fn(p=p)
        tar = _rand()
        prd = tar + c
        loss = fn(prd, tar)
        self.assertTrue(
            compare_tensors(
                f"drift scale p={p}", loss, torch.full_like(loss, c**p), atol=1e-5, rtol=1e-4, verbose=verbose
            ),
        )

    def test_ensemble_dim_handled(self, verbose=False):
        """5-D input (B, E, C, H, W): loss must equal the mean per-member drift.
        Each member e has constant offset biases[e]; mean drift = mean(biases)."""
        fn = self._fn(p=1.0)
        E = 4
        tar = torch.zeros(_BATCH, _NUM_CH, _IMG_H, _IMG_W)

        # biases = [1, 2, 3, 4] → mean = 2.5
        biases = torch.arange(1, E + 1, dtype=torch.float32)
        prd = biases.reshape(1, E, 1, 1, 1).expand(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W).clone()

        loss = fn(prd, tar)
        expected_mean = biases.mean().item()
        self.assertTrue(
            compare_tensors(
                "ensemble drift", loss, torch.full_like(loss, expected_mean), atol=1e-5, rtol=1e-4, verbose=verbose
            ),
        )


# ===========================================================================
class TestEnsembleNLLLoss(unittest.TestCase):
    """EnsembleNLLLoss requires 5-D (B, E, C, H, W) input and is tested directly."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, eps=1e-5):
        return EnsembleNLLLoss(**_GEOM_KWARGS, eps=eps)

    def test_backward(self):
        """Gradients through the multi-member NLL must be finite and free of NaNs."""
        fn = self._fn()
        E = 5
        forecasts = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W, requires_grad=True)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fn(forecasts, obs).sum().backward()
        self.assertIsNotNone(forecasts.grad)
        self.assertFalse(torch.isnan(forecasts.grad).any(), "NaN in forecasts.grad")
        self.assertFalse(torch.isinf(forecasts.grad).any(), "Inf in forecasts.grad")

    def test_batch_independence(self, verbose=False):
        fn = self._fn()
        E = 5
        fc1 = torch.randn(1, E, _NUM_CH, _IMG_H, _IMG_W)
        obs1 = torch.randn(1, _NUM_CH, _IMG_H, _IMG_W)
        loss1 = fn(fc1, obs1)
        loss4 = fn(fc1.repeat(4, 1, 1, 1, 1), obs1.repeat(4, 1, 1, 1))
        self.assertTrue(compare_tensors("nll batch", loss1.repeat(4, 1), loss4, verbose=verbose))

    def test_single_member_is_finite(self):
        """E=1 forces sigma=0, clamped to eps; result must not be NaN/Inf."""
        fn = self._fn(eps=1e-5)
        forecasts = torch.randn(_BATCH, 1, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        loss = fn(forecasts, obs)
        self.assertTrue(torch.isfinite(loss).all(), "NLL must be finite for a single-member ensemble")

    def test_well_calibrated_lower_nll_than_biased(self):
        """Ensemble centred on the observation has strictly lower NLL than a biased ensemble."""
        fn = self._fn()
        E, sigma = 10, 0.5
        obs = torch.ones(_BATCH, _NUM_CH, _IMG_H, _IMG_W)

        fc_good = obs.unsqueeze(1) + sigma * torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        fc_bad = (2.0 * obs).unsqueeze(1) + sigma * torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)

        nll_good = fn(fc_good, obs).mean().item()
        nll_bad = fn(fc_bad, obs).mean().item()
        self.assertLess(
            nll_good,
            nll_bad,
            f"Centred NLL ({nll_good:.4f}) should be < biased NLL ({nll_bad:.4f})",
        )

    def test_larger_spread_higher_nll_near_truth(self):
        """Same noise pattern scaled by σ=0.1 vs σ=2.0: the (obs-mu)^2/sigma^2 terms cancel,
        leaving only the log(sigma^2) difference, which is larger for the looser ensemble."""
        fn = self._fn()
        E = 20
        obs = torch.ones(_BATCH, _NUM_CH, _IMG_H, _IMG_W)

        set_seed(333)
        noise = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        fc_tight = obs.unsqueeze(1) + 0.1 * noise
        fc_loose = obs.unsqueeze(1) + 2.0 * noise

        nll_tight = fn(fc_tight, obs).mean().item()
        nll_loose = fn(fc_loose, obs).mean().item()
        self.assertGreater(
            nll_loose,
            nll_tight,
            f"Loose NLL ({nll_loose:.4f}) should exceed tight NLL ({nll_tight:.4f})",
        )


# ===========================================================================
class TestGaussianMMDLoss(unittest.TestCase):
    """Specific tests for GaussianMMDLoss.

    Note: the unbiased U-statistic MMD² can be negative (e.g., perfect prediction
    with E=5 gives mmd² = (3-E)/(E-1) = -0.5), so nonneg and zero-on-perfect tests
    do not apply.  Tested properties: squared-flag consistency, spread ordering,
    backward pass, and the E=1 code-path.
    """

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_squared_flag_consistency(self, verbose=False):
        """For a wide ensemble where mmd² > 0, sqrt(mmd²) must equal the unsquared loss."""
        fn_sq = GaussianMMDLoss(**_GEOM_KWARGS, squared=True)
        fn_unsq = GaussianMMDLoss(**_GEOM_KWARGS, squared=False)
        # use a large-offset ensemble so that k(y_m, obs) ≈ 0 → mmd² > 0
        obs = torch.zeros(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc = obs.unsqueeze(1) + 10.0 * torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        mmd2 = fn_sq(fc, obs)
        mmd = fn_unsq(fc, obs)
        self.assertTrue(
            compare_tensors("squared flag", torch.sqrt(mmd2), mmd, atol=1e-5, rtol=1e-4, verbose=verbose),
        )

    def test_spread_increases_mmd(self):
        """A tight ensemble near the observation must have a higher kernel score
        than a wide ensemble far from it."""
        fn = GaussianMMDLoss(**_GEOM_KWARGS, squared=True)
        obs = torch.zeros(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        set_seed(333)
        noise = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        fc_tight = obs.unsqueeze(1) + 0.01 * noise
        fc_wide = obs.unsqueeze(1) + 10.0 * noise
        score_tight = fn(fc_tight, obs).mean().item()
        score_wide = fn(fc_wide, obs).mean().item()
        self.assertGreater(
            score_tight,
            score_wide,
            f"Tight score ({score_tight:.4f}) should be > wide score ({score_wide:.4f})",
        )

    def test_backward(self):
        """Gradients through the double-loop MMD kernel must be finite and free of NaNs."""
        fn = GaussianMMDLoss(**_GEOM_KWARGS, squared=True)
        fc = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W, requires_grad=True)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    def test_e1_special_case(self, verbose=False):
        """With E=1 the code takes a direct kernel path: mmd = k(obs, fc).
        When obs == fc the RBF kernel equals 1, so the spatially-averaged loss
        must equal 1 (squared=True) or 1 (squared=False, sqrt(1)=1)."""
        fn = GaussianMMDLoss(**_GEOM_KWARGS, squared=True)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc = obs.unsqueeze(1)  # E=1, all members = obs
        loss = fn(fc, obs)
        self.assertTrue(
            compare_tensors("e1 perfect", loss, torch.ones_like(loss), atol=1e-5, verbose=verbose),
        )


# ===========================================================================
class TestCRPSLoss(unittest.TestCase):
    """Verifies CRPSLoss against the properscoring reference implementation
    for the CDF and Gaussian kernels."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_cdf_matches_properscoring(self, verbose=False):
        crps_func = CRPSLoss(
            **_GEOM_KWARGS,
            crps_type="cdf",
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=None,
        )

        for ensemble_size in [1, 10]:
            with self.subTest(ensemble_size=ensemble_size):
                inp = torch.empty((_BATCH, ensemble_size, _NUM_CH, _IMG_H, _IMG_W), dtype=torch.float32)
                inp.normal_(1.0, 1.0)
                tar = torch.ones(_BATCH, _NUM_CH, _IMG_H, _IMG_W, dtype=torch.float32)

                result = crps_func(inp, tar).cpu().numpy()

                tar_arr = tar.cpu().numpy()
                inp_arr = inp.cpu().numpy()

                # properscoring uses a different axis convention for the degenerate E=1 case
                if ensemble_size == 1:
                    axis = -1
                    inp_arr = np.squeeze(inp_arr, axis=1)
                else:
                    axis = 1

                result_proper = crps_ensemble(tar_arr, inp_arr, weights=None, issorted=False, axis=axis)
                quad_weight_arr = crps_func.quadrature.quad_weight.cpu().numpy()
                result_proper = np.sum(result_proper * quad_weight_arr, axis=(2, 3))

                self.assertTrue(compare_arrays("output", result, result_proper, verbose=verbose))

    def test_gauss_matches_properscoring(self, verbose=False):
        eps = 1.0e-5
        crps_func = CRPSLoss(
            **_GEOM_KWARGS,
            crps_type="gauss",
            spatial_distributed=False,
            ensemble_distributed=False,
            eps=eps,
        )

        for ensemble_size in [1, 10]:
            with self.subTest(ensemble_size=ensemble_size):
                inp = torch.empty((_BATCH, ensemble_size, _NUM_CH, _IMG_H, _IMG_W), dtype=torch.float32)
                inp.normal_(1.0, 1.0)
                tar = torch.ones(_BATCH, _NUM_CH, _IMG_H, _IMG_W, dtype=torch.float32)

                result = crps_func(inp, tar).cpu().numpy()

                tar_arr = tar.cpu().numpy()
                inp_arr = inp.cpu().numpy()

                # compute mu, sigma; guard against underflows
                mu = np.mean(inp_arr, axis=1)
                sigma = np.maximum(np.sqrt(np.var(inp_arr, axis=1)), eps)

                result_proper = crps_gaussian(tar_arr, mu, sigma, grad=False)
                quad_weight_arr = crps_func.quadrature.quad_weight.cpu().numpy()
                result_proper = np.sum(result_proper * quad_weight_arr, axis=(2, 3))

                self.assertTrue(compare_arrays("output", result, result_proper, verbose=verbose))


# ===========================================================================
class TestSpectralLossWeighted(unittest.TestCase):
    """Spectral losses (SpectralLpLoss, SpectralH1Loss) expect per-mode weights
    shaped (*, lmax, mmax) — not spatial (H, W) weights.
    These tests verify the weighting path using spectral-space weights constructed
    from fn.sht.lmax / fn.sht.mmax."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    @staticmethod
    def _make(loss_type):
        return {
            "l2": SpectralLpLoss(**_SPEC_KWARGS, squared=True),
            "h1": SpectralH1Loss(**_SPEC_KWARGS, squared=True),
        }[loss_type]

    @parameterized.expand([("l2",), ("h1",)])
    def test_ones_weight_unchanged(self, loss_type, verbose=False):
        """All-ones spectral weight must leave the loss identical to no weight."""
        fn = self._make(loss_type)
        prd, tar = _rand(), _rand()
        wgt = torch.ones(1, 1, fn.sht.lmax, fn.sht.mmax)
        self.assertTrue(
            compare_tensors(f"{loss_type} ones wgt", fn(prd, tar, wgt), fn(prd, tar), atol=1e-5, verbose=verbose),
        )

    @parameterized.expand([("l2",), ("h1",)])
    def test_zero_weight_kills_loss(self, loss_type, verbose=False):
        """All-zeros spectral weight must produce a zero loss (no modes contribute)."""
        fn = self._make(loss_type)
        prd, tar = _rand(), _rand()
        wgt = torch.zeros(1, 1, fn.sht.lmax, fn.sht.mmax)
        loss = fn(prd, tar, wgt)
        self.assertTrue(
            compare_tensors(f"{loss_type} zero wgt", loss, torch.zeros_like(loss), atol=1e-5, verbose=verbose),
        )

    def test_l2_dc_only_weight_reduces_loss(self):
        """Keeping only the DC (l=0, m=0) mode gives strictly less L2 loss than
        the full-spectrum loss for a field that has non-trivial spatial structure."""
        fn = SpectralLpLoss(**_SPEC_KWARGS, squared=True)
        lat = torch.linspace(0, math.pi, _IMG_H)
        lon = torch.linspace(0, 2.0 * math.pi, _IMG_W)
        LAT, LON = torch.meshgrid(lat, lon, indexing="ij")
        field = (1.0 + torch.sin(LAT) * torch.cos(LON)).expand(_BATCH, _NUM_CH, -1, -1).clone()
        tar = torch.zeros_like(field)

        loss_full = fn(field, tar).mean().item()

        wgt = torch.zeros(1, 1, fn.sht.lmax, fn.sht.mmax)
        wgt[0, 0, 0, 0] = 1.0  # DC mode only
        loss_dc = fn(field, tar, wgt).mean().item()

        self.assertLess(
            loss_dc,
            loss_full,
            f"DC-only loss ({loss_dc:.4f}) should be < full-spectrum loss ({loss_full:.4f})",
        )


# ===========================================================================
@parameterized_class(("device", "compile"), [(d, c) for (d,) in _devices for c in _compile_opts])
class TestLossHandler(unittest.TestCase):

    @classmethod
    def setUpClass(cls, path: Optional[str] = "/tmp"):

        cls.img_shape_x = 32
        cls.img_shape_y = 64

        cls.tmpdir = tempfile.TemporaryDirectory(dir=path)
        tmp_path = cls.tmpdir.name

        params = get_default_parameters()

        cls.time_diff_stds_path = os.path.join(tmp_path, "time_diff_stds.npy")
        np.save(cls.time_diff_stds_path, np.ones((1, params.N_out_channels, 1, 1), dtype=np.float64))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def setUp(self):

        # clear in-process dynamo state (compiled code, guards, recompile counters)
        # first so each test method compiles from a clean slate — makes the compile
        # subtests order-independent (a prior test can otherwise seed dynamic dims /
        # exhaust the recompile limit on a shared loss forward code object)
        torch._dynamo.reset()

        disable_tf32()

        set_seed(333)

        self.params = get_default_parameters()

        # generating the image logic that is typically used by the dataloader
        self.params.img_shape_x = self.img_shape_x
        self.params.img_shape_y = self.img_shape_y
        self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
        self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
        self.params.img_shape_x_resampled = self.params.img_shape_x
        self.params.img_shape_y_resampled = self.params.img_shape_y
        self.params.img_local_offset_x = self.params.img_crop_offset_x = 0
        self.params.img_local_offset_y = self.params.img_crop_offset_y = 0

        # also set the batch size for testing
        self.params.batch_size = 4

        # set paths
        self.params.time_diff_stds_path = self.time_diff_stds_path

    @parameterized.expand(_loss_params)
    def test_loss(self, losses, uncertainty_weighting=False):
        """
        Tests initialization of loss, as well as the forward and backward pass
        """

        self.params.losses = losses
        self.params.uncertainty_weighting = uncertainty_weighting

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        # test initialization of loss object (self.compile selects the eager vs torch.compile path)
        loss_obj = LossHandler(self.params, compile=self.compile)

        inp = torch.randn(*shape)
        inp.requires_grad = True
        tar = torch.randn(*shape)
        tar.requires_grad = True

        # forward pass and check shapes
        out = loss_obj(tar, inp)
        self.assertEqual(torch.numel(out), 1)
        self.assertTrue(out.item() >= 0.0)

        # backward pass and check gradients are not None
        out.backward()

    @parameterized.expand(_loss_params)
    def test_loss_batchsize_independence(self, losses, uncertainty_weighting=False, verbose=False):
        """
        Tests if losses are independent on batch size, in the sense that proper averaging over batch size
        is performed
        """

        self.params.losses = losses
        # not supported for bs independence:
        self.params.uncertainty_weighting = False

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        # test initialization of loss object (self.compile selects the eager vs torch.compile path;
        # the compiled path also covers a recompile for the doubled batch size below)
        loss_obj = LossHandler(self.params, compile=self.compile)

        inp = torch.randn(*shape)
        tar = torch.randn(*shape)
        out = loss_obj(tar, inp)

        inp2 = torch.cat([inp, inp], dim=0)
        tar2 = torch.cat([tar, tar], dim=0)
        out2 = loss_obj(tar2, inp2)

        self.assertTrue(compare_tensors("loss", out, out2, verbose=verbose))

    @parameterized.expand(_loss_weighted_params)
    def test_loss_weighted(self, losses, uncertainty_weighting=False, verbose=False):
        """
        Tests initialization of loss, as well as the forward and backward pass
        """

        self.params.losses = losses
        self.params.uncertainty_weighting = uncertainty_weighting

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        # test initialization of loss object (self.compile selects the eager vs torch.compile path)
        loss_obj = LossHandler(self.params, compile=self.compile)

        inp = torch.randn(*shape).clone()
        inp.requires_grad = True
        tar = torch.randn(*shape).clone()
        tar.requires_grad = True
        wgt = torch.ones_like(tar)

        # forward pass and check shapes
        out = loss_obj(tar, inp)
        self.assertEqual(torch.numel(out), 1)
        self.assertTrue(out.item() >= 0.0)

        # compute weighted loss
        out_weighted = loss_obj(tar, inp, wgt)

        self.assertTrue(compare_tensors("loss", out, out_weighted, verbose=verbose))

    @parameterized.expand(_loss_weighted_params)
    def test_loss_multistep(self, losses, uncertainty_weighting=False, verbose=False):
        """
        Tests initialization of loss, as well as the forward and backward pass
        """

        self.params.n_future = 2
        self.params.losses = losses
        self.params.uncertainty_weighting = uncertainty_weighting

        shape = (
            self.params.batch_size,
            (self.params.n_future + 1) * self.params.N_out_channels,
            self.params.img_shape_x,
            self.params.img_shape_y,
        )

        # test initialization of loss object (self.compile selects the eager vs torch.compile path)
        loss_obj = LossHandler(self.params, compile=self.compile)

        inp = torch.randn(*shape).clone()
        inp.requires_grad = True
        tar = torch.randn(*shape).clone()
        tar.requires_grad = True
        wgt = torch.ones_like(tar)

        # forward pass and check shapes
        out = loss_obj(tar, inp)
        self.assertEqual(torch.numel(out), 1)
        self.assertTrue(out.item() >= 0.0)

        # compute weighted loss
        out_weighted = loss_obj(tar, inp, wgt)

        self.assertTrue(compare_tensors("loss", out, out_weighted, verbose=verbose))

    @parameterized.expand(_loss_zero_params)
    def test_zero_on_perfect_prediction(self, losses, uncertainty_weighting=False, verbose=False):
        """Loss must be exactly zero when prediction equals target."""
        self.params.losses = losses
        self.params.uncertainty_weighting = uncertainty_weighting
        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)
        # self.compile selects the eager vs torch.compile path
        loss_obj = LossHandler(self.params, compile=self.compile)
        # use a non-zero random field so spectral losses avoid 0/0 in per-mode coherence
        prd = torch.randn(*shape)
        out = loss_obj(prd, prd)
        self.assertTrue(compare_tensors("zero loss", out, torch.zeros_like(out), atol=1e-5, verbose=verbose))

    def test_running_stats(self, verbose=False):
        """
        Tests computation of the running stats
        """

        self.params.losses = [{"type": "l2"}]

        # test initialization of loss object
        loss_obj = LossHandler(self.params, track_running_stats=True)
        loss_obj.train()

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        # this needs to be sufficiently large to mitigarte the bias due to the initialization of the running stats
        num_samples = 100
        for i in range(num_samples):

            inp = i * torch.ones(*shape)
            inp.requires_grad = True
            tar = torch.zeros(*shape)
            tar.requires_grad = True

            # forward pass and check shapes
            out = loss_obj(tar, inp)

        # generate simulated dataset
        data = (
            torch.arange(num_samples)
            .float()
            .reshape(1, 1, -1)
            .repeat(self.params.batch_size, self.params.N_out_channels, 1)
        )
        expected_var, expected_mean = torch.var_mean(data, correction=0, dim=(0, -1))

        var, mean = loss_obj.get_running_stats()

        with self.subTest(desc="mean"):
            self.assertTrue(compare_tensors("mean", mean, expected_mean, verbose=verbose))
        with self.subTest(desc="var"):
            self.assertTrue(compare_tensors("var", var, expected_var, verbose=verbose))

    # ------------------------------------------------------------------
    # multistep_loss_weight modes — _compute_multistep_weight in loss.py:206
    # The default "constant" is exercised by test_loss_multistep above, but
    # "balanced", "linear", "last", "last-n-1", "custom" are all currently dead.
    # ------------------------------------------------------------------

    @parameterized.expand(
        [
            ("constant",),
            ("balanced",),
            ("linear",),
            ("last",),
            ("last-n-1",),
        ]
    )
    def test_multistep_weight_mode(self, weight_type):
        """Each multistep weight mode must build, forward, and backward without error.
        Also asserts the computed multistep_weight buffer has the expected length
        (n_future + 1) — the only invariant common to all modes."""
        self.params.n_future = 2
        self.params.losses = [{"type": "l2", "channel_weights": "constant"}]
        self.params.multistep = {"weight_type": weight_type}

        shape = (
            self.params.batch_size,
            (self.params.n_future + 1) * self.params.N_out_channels,
            self.params.img_shape_x,
            self.params.img_shape_y,
        )

        # self.compile selects the eager vs torch.compile path
        loss_obj = LossHandler(self.params, compile=self.compile)

        # the multistep_weight buffer is tiled by ncw; the per-step prefix is n_future+1 entries
        # tiled to (n_future + 1) * ncw — verify length matches that contract
        expected_len = (self.params.n_future + 1) * loss_obj.channel_weights.shape[1]
        self.assertEqual(loss_obj.multistep_weight.numel(), expected_len)

        inp = torch.randn(*shape, requires_grad=True)
        tar = torch.randn(*shape)
        out = loss_obj(tar, inp)
        self.assertEqual(torch.numel(out), 1)
        self.assertTrue(torch.isfinite(out))
        out.backward()
        self.assertIsNotNone(inp.grad)

    def test_multistep_weight_custom(self):
        """custom mode passes through user-supplied weights and asserts shape match."""
        self.params.n_future = 2
        self.params.losses = [{"type": "l2", "channel_weights": "constant"}]
        self.params.multistep = {"weight_type": "custom", "weights": [0.1, 0.3, 0.6]}

        loss_obj = LossHandler(self.params)
        # the prefix of multistep_weight (before tiling over channels) must equal
        # the user-supplied weights — pull out the per-step values
        ncw = loss_obj.channel_weights.shape[1]
        per_step = loss_obj.multistep_weight.reshape(self.params.n_future + 1, ncw)[:, 0]
        self.assertTrue(compare_tensors("custom multistep weights", per_step, torch.tensor([0.1, 0.3, 0.6])))

    def test_multistep_weight_custom_wrong_length_raises(self):
        """custom weights must match n_future + 1 — validated in _compute_multistep_weight."""
        self.params.n_future = 2
        self.params.losses = [{"type": "l2"}]
        self.params.multistep = {"weight_type": "custom", "weights": [0.1, 0.9]}  # too short
        with self.assertRaises(ValueError):
            LossHandler(self.params)

    def test_multistep_weight_unknown_raises(self):
        """Unknown weight_type must raise ValueError."""
        self.params.n_future = 2
        self.params.losses = [{"type": "l2"}]
        self.params.multistep = {"weight_type": "bogus_mode"}
        with self.assertRaises(ValueError):
            LossHandler(self.params)

    # ------------------------------------------------------------------
    # tendency: True path — loss.py:360-386
    # When ANY loss in the list has tendency=True AND inp is passed to forward,
    # prd/tar are transformed to (prd - inp_state) / (tar - inp_state).
    # ------------------------------------------------------------------

    def test_tendency_loss_changes_output(self, verbose=False):
        """Passing inp= to a tendency-flagged loss must change the loss value.
        With tendency: True the loss is computed on (prd - inp_state) vs (tar - inp_state)
        instead of on prd vs tar directly.

        We must use a loss that is NOT translation-invariant in (prd, tar) —
        plain L2 of the difference is invariant ((prd-inp) - (tar-inp) = prd - tar),
        so it can't distinguish the two paths. Relative L2 divides by ||tar||,
        which becomes ||tar - inp|| under tendency, making the loss differ.
        """
        self.params.losses = [
            {"type": "l2", "channel_weights": "constant", "tendency": True, "parameters": {"relative": True}},
        ]
        loss_obj = LossHandler(self.params)

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)
        prd = torch.randn(*shape)
        tar = torch.randn(*shape)
        inp = torch.randn(*shape)  # non-zero, so the tendency transform is non-trivial

        # without inp, the tendency branch is skipped (inp is None) — falls back to relative L2 on (prd, tar)
        out_no_inp = loss_obj(prd, tar)
        # with inp, tendency transform is active — denominator becomes ||tar - inp||, so the loss differs
        out_with_inp = loss_obj(prd, tar, inp=inp)

        self.assertFalse(
            compare_tensors("tendency vs no-tendency", out_no_inp, out_with_inp, atol=1e-6, rtol=1e-5),
            f"tendency path didn't change the output: no_inp={out_no_inp.item()} with_inp={out_with_inp.item()}",
        )

    def test_tendency_zero_input_recovers_no_tendency(self, verbose=False):
        """When inp is exactly zero the tendency-transformed (prd - 0) is just prd,
        so the loss must equal the no-tendency loss. Sanity check on the formula."""
        self.params.losses = [
            {"type": "l2", "channel_weights": "constant", "tendency": True, "parameters": {"relative": True}},
        ]
        loss_obj = LossHandler(self.params)

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)
        prd = torch.randn(*shape)
        tar = torch.randn(*shape)
        inp_zero = torch.zeros(*shape)

        out_no_inp = loss_obj(prd, tar)
        out_inp_zero = loss_obj(prd, tar, inp=inp_zero)
        self.assertTrue(compare_tensors("tendency w/ zero inp", out_no_inp, out_inp_zero, verbose=verbose))

    # ------------------------------------------------------------------
    # random_slice_loss path — loss.py:330-349
    # Mixes channels through a random orthonormal matrix before computing the loss.
    # ------------------------------------------------------------------

    @parameterized.expand(
        [
            ("random_slice_loss",),  # mixes channels via a random orthonormal slice BEFORE the loss
            ("randomized_loss_weights",),  # multiplies per-channel weights by a random mask AFTER the loss
        ]
    )
    def test_loss_modifier_flag(self, flag_name):
        """Both random-channel modifier flags must run end-to-end without crashing
        and produce a finite scalar with finite gradients. They exercise different
        code paths in LossHandler.forward but share the same smoke-test contract."""
        self.params.losses = [{"type": "l2", "channel_weights": "constant"}]
        setattr(self.params, flag_name, True)

        loss_obj = LossHandler(self.params)

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)
        inp = torch.randn(*shape, requires_grad=True)
        tar = torch.randn(*shape)
        out = loss_obj(tar, inp)
        self.assertTrue(torch.isfinite(out))
        out.backward()
        self.assertIsNotNone(inp.grad)
        self.assertFalse(torch.isnan(inp.grad).any())

    # ------------------------------------------------------------------
    # balanced_weighting path — loss.py:420-424
    # Activated only when track_running_stats=True AND num_batches_tracked > 100.
    # ------------------------------------------------------------------

    def test_balanced_weighting_after_warmup(self):
        """balanced_weighting=True with > 100 tracked batches activates the
        chw / (mean + eps) branch. Forward must remain finite under that path."""
        self.params.losses = [{"type": "l2"}]
        self.params.balanced_weighting = True

        loss_obj = LossHandler(self.params, track_running_stats=True)
        loss_obj.train()

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        # warm up past the 100-batch gate that swaps from ones-like to running mean
        for _ in range(105):
            prd = torch.randn(*shape)
            tar = torch.randn(*shape)
            loss_obj(prd, tar)

        # post-warmup forward — exercises the balanced_weighting branch
        out = loss_obj(torch.randn(*shape), torch.randn(*shape))
        self.assertTrue(torch.isfinite(out))

    # ------------------------------------------------------------------
    # reset_running_stats — loss.py:292-296
    # ------------------------------------------------------------------

    def test_reset_running_stats(self, verbose=False):
        """reset_running_stats restores running_mean=0, running_var=1, and
        num_batches_tracked=0 after warm-up batches have populated them."""
        self.params.losses = [{"type": "l2"}]
        loss_obj = LossHandler(self.params, track_running_stats=True)
        loss_obj.train()

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        # warm up — populate running stats
        for _ in range(5):
            loss_obj(torch.randn(*shape), torch.randn(*shape))
        self.assertGreater(loss_obj.num_batches_tracked.item(), 0, "warm-up failed to populate running stats")

        # reset and verify each buffer is restored to its initial state
        loss_obj.reset_running_stats()
        self.assertTrue(
            compare_tensors(
                "running_mean reset", loss_obj.running_mean, torch.zeros_like(loss_obj.running_mean), verbose=verbose
            )
        )
        self.assertTrue(
            compare_tensors(
                "running_var reset", loss_obj.running_var, torch.ones_like(loss_obj.running_var), verbose=verbose
            )
        )
        self.assertEqual(loss_obj.num_batches_tracked.item(), 0)

    # ------------------------------------------------------------------
    # 5-D prd path of random_slice_loss — loss.py:343-346
    # ------------------------------------------------------------------

    def test_random_slice_loss_ensemble_path(self):
        """random_slice_loss with 5-D prd reshapes to (B*E, ...) for the conv2d
        and reshapes back. Use a probabilistic loss (ensemble_crps) so the
        ensemble dim is preserved through the loss dispatch."""
        self.params.losses = [
            {
                "type": "ensemble_crps",
                "channel_weights": "constant",
                "parameters": {"crps_type": "skillspread"},
            }
        ]
        self.params.random_slice_loss = True

        loss_obj = LossHandler(self.params)

        E = 5
        shape_5d = (
            self.params.batch_size,
            E,
            self.params.N_out_channels,
            self.params.img_shape_x,
            self.params.img_shape_y,
        )
        shape_4d = (
            self.params.batch_size,
            self.params.N_out_channels,
            self.params.img_shape_x,
            self.params.img_shape_y,
        )
        prd = torch.randn(*shape_5d, requires_grad=True)
        tar = torch.randn(*shape_4d)

        out = loss_obj(prd, tar)
        self.assertTrue(torch.isfinite(out))
        out.backward()
        self.assertIsNotNone(prd.grad)
        self.assertFalse(torch.isnan(prd.grad).any())

    # ------------------------------------------------------------------
    # 5-D prd path of tendency — loss.py:377-380
    # ------------------------------------------------------------------

    def test_tendency_loss_ensemble_path(self):
        """tendency: True with 5-D prd hits the prd_tendency = prd - inp_state.unsqueeze(1)
        branch. We assert the path runs cleanly with a probabilistic loss; the
        actual loss value is invariant under the tendency transform for proper-score
        ensemble losses (CRPS depends on differences only), so we don't check value
        change here — covering the code line is the goal."""
        self.params.losses = [
            {
                "type": "ensemble_crps",
                "channel_weights": "constant",
                "tendency": True,
                "parameters": {"crps_type": "skillspread"},
            }
        ]
        loss_obj = LossHandler(self.params)

        E = 5
        shape_5d = (
            self.params.batch_size,
            E,
            self.params.N_out_channels,
            self.params.img_shape_x,
            self.params.img_shape_y,
        )
        shape_4d = (
            self.params.batch_size,
            self.params.N_out_channels,
            self.params.img_shape_x,
            self.params.img_shape_y,
        )

        prd = torch.randn(*shape_5d, requires_grad=True)
        tar = torch.randn(*shape_4d)
        inp = torch.randn(*shape_4d)

        out = loss_obj(prd, tar, inp=inp)
        self.assertTrue(torch.isfinite(out))
        out.backward()
        self.assertIsNotNone(prd.grad)
        self.assertFalse(torch.isnan(prd.grad).any())

    # ------------------------------------------------------------------
    # multistep with empty {} dict — falls back to "constant" weight_type
    # (loss.py:212, _compute_multistep_weight default branch)
    # ------------------------------------------------------------------

    def test_multistep_default_weight_type(self):
        """params.multistep = {} (no weight_type key) must fall back to 'constant'.
        Verifies the else branch in _compute_multistep_weight that picks the
        default when the kwarg dict doesn't have weight_type."""
        self.params.n_future = 2
        self.params.losses = [{"type": "l2", "channel_weights": "constant"}]
        self.params.multistep = {}  # no weight_type → defaults to constant

        loss_obj = LossHandler(self.params)

        # constant mode: each step weighted 1/(n_future+1) before tiling over channels
        ncw = loss_obj.channel_weights.shape[1]
        expected = torch.full(((self.params.n_future + 1) * ncw,), 1.0 / (self.params.n_future + 1))
        self.assertTrue(compare_tensors("default multistep_weight", loss_obj.multistep_weight, expected))

    # ------------------------------------------------------------------
    # channel_weights given as a Python list — loss.py:160-164
    # The handler accepts a nested list (shape (1, N)) in addition to the
    # named-string modes ("constant", "auto", etc.).
    # ------------------------------------------------------------------

    def test_channel_weights_as_list(self):
        """A list-valued channel_weights bypasses the named-string branch and is
        loaded directly. Must be nested (shape (1, N)) so the assert at
        loss.py:164 passes."""
        custom = [[0.1, 0.2, 0.3, 0.4, 0.5]]  # nested → shape (1, 5) for N=5
        self.params.losses = [{"type": "l2", "channel_weights": custom}]
        loss_obj = LossHandler(self.params)

        # before normalization, chw equals the list. The handler ALSO doesn't
        # renormalize here (that's only in compute_channel_weighting paths),
        # so we should see the literal values up to a possible reshape.
        self.assertEqual(loss_obj.channel_weights.shape, (1, 5))
        self.assertTrue(
            compare_tensors(
                "list channel_weights",
                loss_obj.channel_weights,
                torch.tensor(custom, dtype=torch.float32),
            )
        )

    # ------------------------------------------------------------------
    # relative_weight per loss — loss.py:172-173
    # Multiplies the channel weights of one loss by a scalar before they're
    # registered into the channel_weights buffer. Used to balance the
    # contributions of multiple losses without changing their internal weights.
    # ------------------------------------------------------------------

    def test_relative_weight_scales_channel_weights(self, verbose=False):
        """relative_weight on a loss spec multiplies that loss's chw entry.
        Compare two LossHandlers — one with relative_weight=1.0 (identity, the
        default-equivalent), one with 2.0 — and check that the second's
        channel_weights are exactly 2× the first's."""
        base = {"type": "l2", "channel_weights": "constant", "relative_weight": 1.0}
        boosted = {"type": "l2", "channel_weights": "constant", "relative_weight": 2.0}

        self.params.losses = [base]
        loss_obj_base = LossHandler(self.params)

        self.params.losses = [boosted]
        loss_obj_boosted = LossHandler(self.params)

        self.assertTrue(
            compare_tensors(
                "relative_weight scaling",
                loss_obj_boosted.channel_weights,
                2.0 * loss_obj_base.channel_weights,
                verbose=verbose,
            )
        )

    # ------------------------------------------------------------------
    # missing targets: NaN in the target is taken out of every loss
    # ------------------------------------------------------------------

    def _missing_mask(self, shape):
        """Scattered missing points, plus one entirely missing field."""
        generator = torch.Generator().manual_seed(1234)
        missing = torch.rand(shape, generator=generator) < 0.1
        missing[0, 1] = True
        return missing

    def _assert_masked(self, out, prd, missing, verbose=False):
        self.assertTrue(torch.isfinite(out), "loss is not finite with missing targets")
        out.backward()
        self.assertFalse(torch.isnan(prd.grad).any(), "gradient contains NaN")
        # no gradient may reach the prediction from a missing target point
        masked_grad = prd.grad[missing.unsqueeze(1).expand_as(prd)] if prd.dim() == 5 else prd.grad[missing]
        self.assertTrue(
            compare_tensors(
                "gradient at missing points",
                masked_grad,
                torch.zeros_like(masked_grad),
                atol=0.0,
                rtol=0.0,
                verbose=verbose,
                shape_check=True,
            )
        )

    @parameterized.expand(_missing_target_deterministic_params)
    def test_missing_targets_deterministic(self, losses, verbose=False):
        """NaN targets give a finite loss, no gradient from the missing points, and
        the same loss as a target that equals the prediction there."""
        self.params.losses = losses
        loss_obj = LossHandler(self.params, compile=self.compile)

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)
        missing = self._missing_mask(shape)
        prd = torch.randn(*shape, requires_grad=True)
        tar = torch.randn(*shape)

        out = loss_obj(prd, torch.where(missing, torch.nan, tar))

        # missing points count as perfectly predicted
        reference = loss_obj(prd.detach(), torch.where(missing, prd.detach(), tar))
        self.assertTrue(compare_tensors("loss", out.detach(), reference, verbose=verbose))

        self._assert_masked(out, prd, missing, verbose=verbose)

    @parameterized.expand(_missing_target_probabilistic_params)
    def test_missing_targets_probabilistic(self, losses, verbose=False):
        """With an ensemble, the missing points are taken out for every member."""
        self.params.losses = losses
        loss_obj = LossHandler(self.params, compile=self.compile)

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)
        missing = self._missing_mask(shape)
        prd = torch.randn(shape[0], 5, *shape[1:], requires_grad=True)
        tar = torch.where(missing, torch.nan, torch.randn(*shape))

        self._assert_masked(loss_obj(prd, tar), prd, missing, verbose=verbose)

    @parameterized.expand(_missing_target_deterministic_params + _missing_target_probabilistic_params)
    def test_nan_predictions_are_not_masked(self, losses, verbose=False):
        """Only NaN in the target means missing data. NaN coming out of the model is
        a failure and has to reach the loss, whether the target is valid there or not."""
        self.params.losses = losses
        loss_obj = LossHandler(self.params, compile=self.compile)
        probabilistic = "ensemble" in losses[0]["type"] or "energy_score" in losses[0]["type"]

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)
        missing = self._missing_mask(shape)
        tar = torch.where(missing, torch.nan, torch.randn(*shape))
        valid_point = torch.nonzero(torch.logical_not(missing))[0].tolist()
        missing_point = torch.nonzero(missing)[0].tolist()

        for desc, point in [("valid target", valid_point), ("missing target", missing_point)]:
            with self.subTest(at=desc):
                prd = torch.randn(shape[0], 5, *shape[1:]) if probabilistic else torch.randn(*shape)
                if probabilistic:
                    # a single member going NaN is enough
                    prd[point[0], 0, point[1], point[2], point[3]] = torch.nan
                else:
                    prd[tuple(point)] = torch.nan
                self.assertTrue(torch.isnan(loss_obj(prd, tar)), f"NaN prediction at a {desc} point was masked")

    def test_missing_targets_with_random_slices(self, verbose=False):
        """The masking happens before the random slice, which mixes channels and would spread the NaN."""
        self.params.losses = [{"type": "l2", "channel_weights": "constant"}]
        self.params.random_slice_loss = True
        loss_obj = LossHandler(self.params, compile=self.compile)

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)
        missing = self._missing_mask(shape)
        prd = torch.randn(*shape, requires_grad=True)
        tar = torch.where(missing, torch.nan, torch.randn(*shape))

        self._assert_masked(loss_obj(prd, tar), prd, missing, verbose=verbose)


# ===========================================================================
class TestAbsPow(unittest.TestCase):
    """abs_pow must match |x|**e and keep the gradient finite where x vanishes."""

    def test_forward_matches(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.3, 4.0])
        for exponent in [0.5, 1.0, 2.0]:
            with self.subTest(exponent=exponent):
                self.assertTrue(
                    compare_tensors(
                        "abs_pow", abs_pow(x, exponent), x.abs().pow(exponent), atol=0.0, rtol=0.0, shape_check=True
                    )
                )

    def test_gradient_is_finite_at_zero(self):
        for exponent in [0.5, 1.0, 2.0]:
            with self.subTest(exponent=exponent):
                x = torch.tensor([0.0, 0.0, 1.0], requires_grad=True)
                abs_pow(x, exponent).sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all(), f"non-finite gradient for exponent {exponent}")
                self.assertEqual(x.grad[0].item(), 0.0)

    def test_mmd_with_small_exponent_at_perfect_forecast(self):
        """With beta < 1 the per point |x|**beta used to give a NaN gradient where members coincide."""
        fn = GaussianMMDLoss(**_GEOM_KWARGS, beta=0.5)
        tar = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc = tar.unsqueeze(1).expand(_BATCH, 5, _NUM_CH, _IMG_H, _IMG_W).clone().requires_grad_(True)
        fn(fc, tar).sum().backward()
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in the gradient at a perfect forecast")


class TestMissingData(unittest.TestCase):
    """The shared definition of missing target values, used by the losses and the metrics."""

    def setUp(self):
        self.x = torch.tensor([[1.0, torch.nan], [3.0, torch.nan]])
        self.missing = torch.tensor([[False, True], [False, True]])

    def test_missing_values_are_nan(self):
        self.assertTrue(torch.equal(missing_values(self.x), self.missing))

    def test_fill_with_constant(self):
        filled = fill_missing(self.x, 0.0)
        self.assertTrue(
            compare_tensors(
                "filled", filled, torch.tensor([[1.0, 0.0], [3.0, 0.0]]), atol=0.0, rtol=0.0, shape_check=True
            )
        )

    def test_fill_with_tensor_and_given_mask(self):
        fill = torch.full_like(self.x, 7.0)
        filled = fill_missing(self.x, fill, self.missing)
        self.assertTrue(
            compare_tensors(
                "filled", filled, torch.tensor([[1.0, 7.0], [3.0, 7.0]]), atol=0.0, rtol=0.0, shape_check=True
            )
        )

    def test_weights_without_and_with_existing_weights(self):
        expected = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        self.assertTrue(
            compare_tensors("weights", missing_weights(self.missing), expected, atol=0.0, rtol=0.0, shape_check=True)
        )
        wgt = torch.tensor([[2.0, 2.0], [0.5, 0.5]])
        self.assertTrue(
            compare_tensors(
                "combined weights",
                missing_weights(self.missing, wgt),
                wgt * expected,
                atol=0.0,
                rtol=0.0,
                shape_check=True,
            )
        )


class TestComputeChannelWeightingHelper(unittest.TestCase):
    """Tests for _compute_channel_weighting_helper in base_loss.py.

    The helper maps a list of channel names + a mode string to a normalized
    channel-weight vector, optionally multiplied by a time-difference scaling
    tensor. Five named modes ("constant", "auto", "new auto", "custom",
    "pangu") plus the unknown-mode error path.
    """

    def setUp(self):
        set_seed(333)

    def test_constant_uniform(self):
        """constant mode produces uniformly-weighted channels (1/N each)."""
        names = ["u10m", "v10m", "t2m", "z500", "q500"]
        chw = _compute_channel_weighting_helper(names, "constant")
        self.assertEqual(chw.shape, (len(names),))
        expected = torch.full((len(names),), 1.0 / len(names))
        self.assertTrue(compare_tensors("constant chw", chw, expected, atol=1e-7))

    def test_unknown_mode_raises(self):
        """Unknown mode strings must raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            _compute_channel_weighting_helper(["t2m"], "bogus_mode_name")

    def test_weights_sum_to_one_for_all_modes(self):
        """Independent of mode, the returned channel weights must sum to 1
        when no time_diff_scale is supplied. Also serves as branch coverage
        for every named mode in one shot."""
        names = ["u10m", "t2m", "z500", "q200", "msl", "sp"]
        for mode in ["constant", "auto", "new auto", "custom", "pangu"]:
            with self.subTest(mode=mode):
                chw = _compute_channel_weighting_helper(names, mode)
                self.assertAlmostEqual(chw.sum().item(), 1.0, places=5)

    def test_time_diff_scale_multiplies_weights(self):
        """When time_diff_scale is supplied, the final weights equal
        (normalized chw) * time_diff_scale element-wise. No renormalization
        is applied after the multiplication, so the result need not sum to 1.
        """
        names = ["u10m", "t2m", "z500"]
        scale = torch.tensor([2.0, 0.5, 4.0])
        chw_no_scale = _compute_channel_weighting_helper(names, "constant")
        chw_scaled = _compute_channel_weighting_helper(names, "constant", time_diff_scale=scale)
        self.assertTrue(compare_tensors("time_diff_scale", chw_scaled, chw_no_scale * scale, atol=1e-7))


# ===========================================================================
class TestCRPSLossExtended(unittest.TestCase):
    """Additional coverage for CRPSLoss: error paths, weight branches,
    and the skillspread kernel validation against properscoring."""

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, crps_type="cdf", **kw):
        return CRPSLoss(
            **_GEOM_KWARGS,
            crps_type=crps_type,
            spatial_distributed=False,
            ensemble_distributed=False,
            **kw,
        )

    # ------ alpha < 1.0 raises for non-skillspread types (line 210) ------

    def test_alpha_lt1_raises_for_cdf(self):
        with self.assertRaises(NotImplementedError):
            self._fn("cdf", alpha=0.5)

    def test_alpha_lt1_raises_for_gauss(self):
        with self.assertRaises(NotImplementedError):
            self._fn("gauss", alpha=0.5)

    # ------ ensemble_weights registered as buffer (lines 220-221) ------

    def test_ensemble_weights_registered_as_buffer(self):
        """Supplying ensemble_weights must register it as a named buffer."""
        ew = torch.ones(self._E)
        fn = self._fn("cdf", ensemble_weights=ew)
        self.assertIn("ensemble_weights", dict(fn.named_buffers()))

    # ------ dim validation in forward (lines 232, 236-238) ------

    def test_wrong_forecast_dims_raises(self):
        """4-D forecasts tensor must raise ValueError (5-D expected)."""
        fn = self._fn()
        fc = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)  # missing ensemble dim
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        with self.assertRaises(ValueError):
            fn(fc, obs)

    def test_spatial_weight_dim_mismatch_raises(self):
        """spatial_weights with fewer dims than observations must raise ValueError."""
        fn = self._fn()
        fc = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        bad_wgt = torch.ones(_NUM_CH, _IMG_H, _IMG_W)  # 3-D; observations are 4-D
        with self.assertRaises(ValueError):
            fn(fc, obs, spatial_weights=bad_wgt)

    # ------ CDF with custom ensemble_weights (line 276) ------

    def test_cdf_with_custom_ensemble_weights_produces_finite_output(self):
        """CDF kernel must execute without error when ensemble_weights is provided."""
        E = self._E
        ew = torch.ones(E)
        fn = self._fn("cdf", ensemble_weights=ew)
        fc = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))
        self.assertTrue(torch.isfinite(out).all())

    # ------ skillspread + ensemble_weights raises (line 284) ------

    def test_skillspread_with_ensemble_weights_raises(self):
        """skillspread kernel does not support custom ensemble_weights."""
        E = self._E
        ew = torch.ones(E)
        fn = self._fn("skillspread", ensemble_weights=ew)
        fc = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        with self.assertRaises(NotImplementedError):
            fn(fc, obs)

    # ------ gauss + ensemble_weights → NameError (line 292, known bug) ------

    def test_gauss_with_ensemble_weights_raises_nameerror(self):
        """Known bug: gauss branch references undefined `idx` when ensemble_weights is set."""
        E = self._E
        ew = torch.ones(E)
        fn = self._fn("gauss", ensemble_weights=ew)
        fc = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        with self.assertRaises(NameError):
            fn(fc, obs)

    # ------ unknown crps_type raises ValueError (line 299) ------

    def test_unknown_crps_type_raises_in_forward(self):
        """Unknown crps_type must raise ValueError in forward."""
        fn = self._fn("cdf")
        fn.crps_type = "bogus"  # bypass __init__ guard; trigger the forward-time check
        fc = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        with self.assertRaises(ValueError):
            fn(fc, obs)

    # ------ skillspread(alpha=0) matches properscoring.crps_ensemble ------

    def test_skillspread_alpha0_matches_properscoring(self, verbose=False):
        """crps_skillspread(alpha=0.0) is the biased CRPS and must match properscoring."""
        E = self._E
        fn = self._fn("skillspread", alpha=0.0)
        set_seed(333)
        fc = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        result = fn(fc, obs).cpu().numpy()

        fc_np = fc.cpu().numpy()
        obs_np = obs.cpu().numpy()
        result_proper = crps_ensemble(obs_np, fc_np, weights=None, issorted=False, axis=1)
        quad_weight_arr = fn.quadrature.quad_weight.cpu().numpy()
        result_proper = np.sum(result_proper * quad_weight_arr, axis=(2, 3))

        self.assertTrue(
            compare_arrays("skillspread vs properscoring", result, result_proper, atol=1e-5, verbose=verbose)
        )

    # ------ CDF == skillspread(alpha=0) exactly for all ensemble sizes ------

    @parameterized.expand([(2,), (5,), (10,)])
    def test_cdf_equals_skillspread_alpha0(self, ensemble_size, verbose=False):
        """CDF CRPS and skillspread(alpha=0) are the same formula; they must agree
        up to float32 rounding for every ensemble size, including E=2."""
        fn_cdf = self._fn("cdf")
        fn_skill = self._fn("skillspread", alpha=0.0)
        set_seed(333)
        fc = torch.randn(_BATCH, ensemble_size, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        result_cdf = fn_cdf(fc, obs)
        result_skill = fn_skill(fc, obs)
        self.assertTrue(
            compare_tensors(
                "cdf vs skillspread(alpha=0)", result_cdf, result_skill, atol=1e-5, rtol=1e-4, verbose=verbose
            ),
            f"E={ensemble_size}: CDF and skillspread(alpha=0) diverged beyond float32 rounding",
        )

    # ------ CDF == skillspread(alpha=0) gradients ------

    @parameterized.expand([(2,), (5,), (10,)])
    def test_cdf_equals_skillspread_alpha0_gradients(self, ensemble_size, verbose=False):
        """CDF and skillspread(alpha=0) compute the same function; their gradients
        w.r.t. the ensemble forecasts must agree up to float32 rounding."""
        fn_cdf = self._fn("cdf")
        fn_skill = self._fn("skillspread", alpha=0.0)
        set_seed(333)
        fc_cdf = torch.randn(_BATCH, ensemble_size, _NUM_CH, _IMG_H, _IMG_W, requires_grad=True)
        fc_skill = fc_cdf.detach().clone().requires_grad_(True)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)

        fn_cdf(fc_cdf, obs).sum().backward()
        fn_skill(fc_skill, obs).sum().backward()

        self.assertIsNotNone(fc_cdf.grad)
        self.assertIsNotNone(fc_skill.grad)
        self.assertTrue(
            compare_tensors(
                f"cdf vs skillspread(alpha=0) gradients E={ensemble_size}",
                fc_cdf.grad,
                fc_skill.grad,
                atol=1e-4,
                rtol=1e-3,
                verbose=verbose,
            ),
            f"E={ensemble_size}: CDF and skillspread(alpha=0) gradients diverged",
        )

    # ------ skillspread (rank trick) == naive skillspread (O(N²) pairwise) ------
    #
    # Both compute fair CRPS = eskill - 0.5 * espread.  The rank-based version
    # uses the order-statistic identity  sum_{i<j} |F_i - F_j| = sum_k (2k-N-1) F_(k)
    # to collapse the O(N²) pairwise sum into an O(N log N) sort + weighted sum;
    # the naive version evaluates the pairwise sum directly.  On real-valued
    # ensembles they must produce numerically identical results, and identical
    # gradients, for any alpha in [0, 1].
    #
    # Why test E=2: the rank coefficients reduce to (2*1-2-1, 2*2-2-1) = (-1, +1)
    # at N=2, which is also the smallest case where the naive O(N²) pairwise
    # tensor has off-diagonal terms — a known edge-case worth pinning down.

    @parameterized.expand([(2, 1.0), (5, 1.0), (10, 1.0), (5, 0.0), (5, 0.5)])
    def test_skillspread_equals_naive_skillspread(self, ensemble_size, alpha, verbose=False):
        """Rank-based skillspread and naive skillspread must agree on real-valued forecasts
        for any alpha (the (N - 1 + alpha) / (N²(N - 1)) factor is identical in both)."""
        fn_skill = self._fn("skillspread", alpha=alpha)
        fn_naive = self._fn("naive skillspread", alpha=alpha)
        set_seed(333)
        fc = torch.randn(_BATCH, ensemble_size, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        result_skill = fn_skill(fc, obs)
        result_naive = fn_naive(fc, obs)
        self.assertTrue(
            compare_tensors(
                f"skillspread vs naive skillspread E={ensemble_size} alpha={alpha}",
                result_skill,
                result_naive,
                atol=1e-5,
                rtol=1e-4,
                verbose=verbose,
            ),
            f"E={ensemble_size} alpha={alpha}: skillspread and naive skillspread diverged beyond float32 rounding",
        )

    @parameterized.expand([(2, 1.0), (5, 1.0), (10, 1.0), (5, 0.0), (5, 0.5)])
    def test_skillspread_equals_naive_skillspread_gradients(self, ensemble_size, alpha, verbose=False):
        """Rank-based and naive skillspread must produce identical gradients w.r.t. forecasts."""
        fn_skill = self._fn("skillspread", alpha=alpha)
        fn_naive = self._fn("naive skillspread", alpha=alpha)
        set_seed(333)
        fc_skill = torch.randn(_BATCH, ensemble_size, _NUM_CH, _IMG_H, _IMG_W, requires_grad=True)
        fc_naive = fc_skill.detach().clone().requires_grad_(True)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)

        fn_skill(fc_skill, obs).sum().backward()
        fn_naive(fc_naive, obs).sum().backward()

        self.assertIsNotNone(fc_skill.grad)
        self.assertIsNotNone(fc_naive.grad)
        self.assertTrue(
            compare_tensors(
                f"skillspread vs naive skillspread gradients E={ensemble_size} alpha={alpha}",
                fc_skill.grad,
                fc_naive.grad,
                atol=1e-4,
                rtol=1e-3,
                verbose=verbose,
            ),
            f"E={ensemble_size} alpha={alpha}: skillspread and naive skillspread gradients diverged",
        )

    @parameterized.expand([("cdf",), ("skillspread",), ("naive skillspread",), ("probability weighted moment",)])
    def test_gradient_sum_zero_on_perfect_prediction(self, crps_type, verbose=False):
        """Gradients summed over the ensemble dim must be zero at every pixel for a
        perfect forecast (all members == observation).  For the CDF kernel this
        requires the tail-line fix; for skillspread the antisymmetric rank
        coefficients already guarantee a zero sum, so this serves as a regression
        test for both."""
        fn = self._fn(crps_type, alpha=0.0) if crps_type == "skillspread" else self._fn(crps_type)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        fc.requires_grad_(True)
        fn(fc, obs).sum().backward()
        self.assertFalse(torch.isnan(fc.grad).any(), f"NaN in {crps_type} gradient at perfect forecast")
        self.assertFalse(torch.isinf(fc.grad).any(), f"Inf in {crps_type} gradient at perfect forecast")
        grad_sum = fc.grad.sum(dim=1)  # sum over ensemble dim → (B, C, H, W)
        self.assertTrue(
            compare_tensors(
                f"{crps_type} grad ensemble sum at perfect forecast",
                grad_sum,
                torch.zeros_like(grad_sum),
                atol=1e-3,
                verbose=verbose,
            ),
        )

    # ------ fair CRPS (alpha=1) < biased CRPS (alpha=0) for spread ensemble ------

    def test_fair_crps_less_than_biased_for_spread_ensemble(self):
        """Fair CRPS (alpha=1) penalises ensemble spread less than biased CRPS (alpha=0)."""
        E = self._E
        fn_fair = self._fn("skillspread", alpha=1.0)
        fn_biased = self._fn("skillspread", alpha=0.0)
        set_seed(333)
        obs = torch.zeros(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        self.assertLess(
            fn_fair(fc, obs).mean().item(),
            fn_biased(fc, obs).mean().item(),
        )


# ===========================================================================
class TestSpectralCRPSLoss(unittest.TestCase):
    """Full coverage for SpectralCRPSLoss."""

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, crps_type="skillspread", absolute=True, **kw):
        return SpectralCRPSLoss(
            **_SPEC_KWARGS,
            crps_type=crps_type,
            spatial_distributed=False,
            ensemble_distributed=False,
            absolute=absolute,
            **kw,
        )

    # ------ type property (line 393) ------

    def test_type_property(self):
        from makani.utils.losses.base_loss import LossType

        fn = self._fn()
        self.assertEqual(fn.type, LossType.Probabilistic)

    # ------ output shape: (B, C) for all three kernels ------

    @parameterized.expand([("cdf",), ("skillspread",), ("gauss",), ("probability weighted moment",)])
    def test_output_shape(self, crps_type):
        fn = self._fn(crps_type)
        fc = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    # ------ non-negative output ------

    @parameterized.expand([("cdf",), ("skillspread",), ("gauss",), ("probability weighted moment",)])
    def test_nonneg(self, crps_type):
        fn = self._fn(crps_type)
        fc = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        out = fn(fc, obs)
        self.assertTrue(
            (out >= -1e-6).all(),
            f"{crps_type}: found negative values, min={out.min().item():.4e}",
        )

    # ------ zero on perfect prediction for cdf and skillspread ------

    @parameterized.expand([("cdf",), ("skillspread",), ("probability weighted moment",)])
    def test_zero_on_perfect_prediction(self, crps_type, verbose=False):
        """Perfect ensemble (all members = observation) must give zero loss."""
        fn = self._fn(crps_type)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors(f"spectral {crps_type} zero", out, torch.zeros_like(out), atol=1e-4, verbose=verbose),
        )

    # ------ absolute=False path (lines 419-426) ------

    def test_absolute_false_shape(self):
        """absolute=False folds real/imag into channels; output must still be (B, C)."""
        fn = self._fn("skillspread", absolute=False)
        fc = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_absolute_false_nonneg(self):
        fn = self._fn("skillspread", absolute=False)
        fc = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        out = fn(fc, obs)
        self.assertTrue((out >= -1e-6).all())

    def test_absolute_true_and_false_differ(self):
        """absolute=True and absolute=False must give different numerical results.
        Note: absolute=False only works with the skillspread kernel."""
        fn_abs = self._fn("skillspread", absolute=True)
        fn_real = self._fn("skillspread", absolute=False)
        set_seed(333)
        fc = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        self.assertFalse(compare_tensors("abs vs real skillspread", fn_abs(fc, obs), fn_real(fc, obs)))

    def test_absolute_false_zero_on_perfect_prediction(self, verbose=False):
        """absolute=False keeps the spectral coefficients complex; at a perfect
        ensemble both eskill and espread (computed by the naive skillspread
        kernel via abs() of complex differences) collapse to 0, so the loss
        must be (near) zero."""
        fn = self._fn("skillspread", absolute=False)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors(
                "absolute=False zero",
                out,
                torch.zeros_like(out),
                atol=1e-4,
                verbose=verbose,
            )
        )

    def test_absolute_false_backward_finite(self):
        """Gradient through the complex spectral CRPS (absolute=False path) must be
        finite — the .abs() of complex pairwise differences has a kink at zero,
        but with random inputs that's measure-zero."""
        fn = self._fn("skillspread", absolute=False)
        fc = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W, requires_grad=True)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in absolute=False gradient")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in absolute=False gradient")

    def test_absolute_false_e1_path(self):
        """E=1 + absolute=False hits the early-return at loss.py:506-509 with
        complex spectral coefficients; ``torch.abs(complex_diff)`` produces a
        real magnitude that the spatial reduction can sum normally. No crash,
        finite output, near-zero on perfect prediction."""
        fn = self._fn("skillspread", absolute=False)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        # perfect single-member ensemble: forecasts == obs after SHT,
        # so |obs - fc.squeeze(1)| in spectral space is 0 → output ≈ 0
        fc_perfect = obs.unsqueeze(1).clone()  # (B, 1, C, H, W)
        out_perfect = fn(fc_perfect, obs)
        self.assertTrue(
            compare_tensors(
                "absolute=False E=1 zero",
                out_perfect,
                torch.zeros_like(out_perfect),
                atol=1e-4,
            )
        )
        # also assert finite on a random forecast
        fc_random = torch.randn(_BATCH, 1, _NUM_CH, _IMG_H, _IMG_W)
        out_random = fn(fc_random, obs)
        self.assertTrue(torch.isfinite(out_random).all())

    # ------ dim validation in forward (lines 398-403) ------

    def test_wrong_forecast_dims_raises(self):
        fn = self._fn()
        fc = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)  # 4-D, not 5-D
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        with self.assertRaises(ValueError):
            fn(fc, obs)

    def test_spectral_weight_dim_mismatch_raises(self):
        fn = self._fn()
        fc = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        bad_wgt = torch.ones(_NUM_CH, fn.sht.lmax, fn.sht.mmax)  # 3-D; obs are 4-D
        with self.assertRaises(ValueError):
            fn(fc, obs, spectral_weights=bad_wgt)

    # ------ error paths inside forward ------

    def test_alpha_lt1_raises_for_cdf(self):
        with self.assertRaises(NotImplementedError):
            self._fn("cdf", alpha=0.5)

    def test_skillspread_with_ensemble_weights_raises(self):
        E = self._E
        ew = torch.ones(E)
        fn = SpectralCRPSLoss(
            **_SPEC_KWARGS,
            crps_type="skillspread",
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=ew,
            absolute=True,
        )
        fc = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        with self.assertRaises(NotImplementedError):
            fn(fc, obs)

    def test_unknown_crps_type_raises_in_forward(self):
        fn = self._fn("cdf")
        fn.crps_type = "bogus"
        fc = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        with self.assertRaises(ValueError):
            fn(fc, obs)

    # ------ E=1 shortcut (lines 440-443) ------

    def test_e1_gives_zero_for_perfect_prediction(self, verbose=False):
        """With E=1, spectral CRPS = |SHT(obs - fc)| which is 0 when obs == fc."""
        fn = self._fn("skillspread")
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc = obs.unsqueeze(1).clone()  # (B, 1, C, H, W)
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors("spectral e1 zero", out, torch.zeros_like(out), atol=1e-4, verbose=verbose),
        )

    # ------ backward pass produces finite gradients ------

    def test_backward_finite(self):
        fn = self._fn("skillspread")
        fc = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W, requires_grad=True)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    @parameterized.expand([("cdf",), ("skillspread",), ("probability weighted moment",)])
    def test_backward_finite_on_perfect_prediction(self, crps_type):
        """Perfect ensemble (all members == obs) must produce finite gradients."""
        fn = self._fn(crps_type)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone().requires_grad_(True)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), f"NaN in {crps_type} gradient at perfect forecast")
        self.assertFalse(torch.isinf(fc.grad).any(), f"Inf in {crps_type} gradient at perfect forecast")


# ===========================================================================
@parameterized_class(("device",), _devices)
class TestSobolevEnergyScoreLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls, path: Optional[str] = "/tmp"):

        cls.tmpdir = tempfile.TemporaryDirectory(dir=path)
        tmp_path = cls.tmpdir.name
        params = get_default_parameters()
        cls.time_diff_stds_path = os.path.join(tmp_path, "time_diff_stds.npy")
        np.save(cls.time_diff_stds_path, np.ones((1, params.N_out_channels, 1, 1), dtype=np.float64))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def setUp(self):
        disable_tf32()
        set_seed(333)
        self.params = get_default_parameters()
        self.params.img_shape_x = 32
        self.params.img_shape_y = 64
        self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
        self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
        self.params.img_shape_x_resampled = self.params.img_shape_x
        self.params.img_shape_y_resampled = self.params.img_shape_y
        self.params.img_local_offset_x = self.params.img_crop_offset_x = 0
        self.params.img_local_offset_y = self.params.img_crop_offset_y = 0
        self.params.batch_size = 4
        self.params.time_diff_stds_path = self.time_diff_stds_path

    @parameterized.expand(
        [
            # (beta, alpha, offset, fraction, channel_reduction)
            (0.5, 1.0, 1.0, 1.0, True),
            (1.0, 1.0, 1.0, 1.0, True),
            (2.0, 1.0, 1.0, 1.0, True),
            (1.0, 0.5, 1.0, 1.0, True),
            (1.0, 2.0, 1.0, 1.0, True),
            (1.0, 1.0, 0.5, 1.0, True),
            (1.0, 1.0, 2.0, 1.0, True),
            (1.0, 1.0, 1.0, 0.5, True),
            (1.0, 1.0, 1.0, 2.0, True),
            (1.0, 1.0, 1.0, 1.0, False),
            (0.5, 0.5, 0.5, 0.5, True),
            (2.0, 2.0, 2.0, 2.0, True),
        ]
    )
    def test_sobolev_energy_score(self, beta, alpha, offset, fraction, channel_reduction):
        """
        Tests SobolevEnergyScoreLoss for different parameter combinations,
        verifying that output and gradients are not NaN or inf.
        """
        sobolev_loss = SobolevEnergyScoreLoss(
            img_shape=(self.params.img_shape_x, self.params.img_shape_y),
            crop_shape=(self.params.img_shape_x, self.params.img_shape_y),
            crop_offset=(0, 0),
            channel_names=self.params.channel_names,
            grid_type=self.params.model_grid_type,
            lmax=None,
            spatial_distributed=False,
            ensemble_distributed=False,
            channel_reduction=channel_reduction,
            alpha=alpha,
            beta=beta,
            offset=offset,
            fraction=fraction,
        ).to(self.device)

        for ensemble_size in [2, 6]:
            with self.subTest(
                desc=f"beta={beta}, alpha={alpha}, offset={offset}, fraction={fraction}, channel_reduction={channel_reduction}, ensemble_size={ensemble_size}"
            ):
                # Generate forecast tensor: (batch, ensemble, channels, lat, lon)
                forecasts = torch.randn(
                    self.params.batch_size,
                    ensemble_size,
                    self.params.N_in_channels,
                    self.params.img_shape_x,
                    self.params.img_shape_y,
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=True,
                )

                # Generate observation tensor: (batch, channels, lat, lon)
                observations = torch.randn(
                    self.params.batch_size,
                    self.params.N_in_channels,
                    self.params.img_shape_x,
                    self.params.img_shape_y,
                    device=self.device,
                    dtype=torch.float32,
                )

                # Forward pass
                result = sobolev_loss(forecasts, observations)

                # Check output is not NaN or inf
                self.assertFalse(torch.isnan(result).any(), "Output contains NaN values")
                self.assertFalse(torch.isinf(result).any(), "Output contains inf values")

                # Backward pass
                loss = result.sum()
                loss.backward()

                # Check gradients are not NaN or inf
                self.assertIsNotNone(forecasts.grad, "Gradients are None")
                self.assertFalse(torch.isnan(forecasts.grad).any(), "Gradients contain NaN values")
                self.assertFalse(torch.isinf(forecasts.grad).any(), "Gradients contain inf values")

    @parameterized.expand([(0.5,), (1.0,), (2.0,)])
    def test_backward_finite_on_perfect_prediction(self, beta):
        """Perfect ensemble (all members == obs) must produce finite gradients across beta values.
        For beta<1 the |diff|^beta derivative at diff=0 is singular; the eps-mask must neutralize
        that path in the backward pass."""
        sobolev_loss = SobolevEnergyScoreLoss(
            img_shape=(self.params.img_shape_x, self.params.img_shape_y),
            crop_shape=(self.params.img_shape_x, self.params.img_shape_y),
            crop_offset=(0, 0),
            channel_names=self.params.channel_names,
            grid_type=self.params.model_grid_type,
            lmax=None,
            spatial_distributed=False,
            ensemble_distributed=False,
            channel_reduction=True,
            alpha=1.0,
            beta=beta,
            offset=1.0,
            fraction=1.0,
        ).to(self.device)

        ensemble_size = 4
        observations = torch.randn(
            self.params.batch_size,
            self.params.N_in_channels,
            self.params.img_shape_x,
            self.params.img_shape_y,
            device=self.device,
            dtype=torch.float32,
        )
        forecasts = (
            observations.unsqueeze(1)
            .expand(
                self.params.batch_size,
                ensemble_size,
                self.params.N_in_channels,
                self.params.img_shape_x,
                self.params.img_shape_y,
            )
            .clone()
            .requires_grad_(True)
        )

        sobolev_loss(forecasts, observations).sum().backward()
        self.assertIsNotNone(forecasts.grad, f"Gradients are None (beta={beta})")
        self.assertFalse(
            torch.isnan(forecasts.grad).any(), f"NaN in sobolev_es gradient at perfect forecast (beta={beta})"
        )
        self.assertFalse(
            torch.isinf(forecasts.grad).any(), f"Inf in sobolev_es gradient at perfect forecast (beta={beta})"
        )


# ===========================================================================
class TestLpEnergyScoreLoss(unittest.TestCase):
    """Tests for LpEnergyScoreLoss (and the L2EnergyScoreLoss alias)."""

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, channel_reduction=True, p=2.0, **kw):
        return LpEnergyScoreLoss(
            **_GEOM_KWARGS,
            spatial_distributed=False,
            ensemble_distributed=False,
            channel_reduction=channel_reduction,
            p=p,
            **kw,
        )

    def test_output_shape_channel_reduction(self):
        fn = self._fn(channel_reduction=True)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, 1))

    def test_output_shape_no_channel_reduction(self):
        fn = self._fn(channel_reduction=False)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_backward_finite(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E, requires_grad=True)
        obs = _rand()
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    def test_zero_on_perfect_prediction(self, verbose=False):
        fn = self._fn()
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors("lp_es zero", out, torch.zeros_like(out), atol=1e-4, verbose=verbose),
        )

    @parameterized.expand([(1.0,), (2.0,), (4.0,)])
    def test_backward_finite_on_perfect_prediction(self, p):
        """Perfect ensemble (all members == obs) must produce finite gradients across p values.
        The |diff|^p term's derivative is ~p·x^(p-1), which at x=0 with p<1 would be singular;
        the eps-mask must neutralize that path."""
        fn = self._fn(p=p)
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone().requires_grad_(True)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), f"NaN in lp_es gradient at perfect forecast (p={p})")
        self.assertFalse(torch.isinf(fc.grad).any(), f"Inf in lp_es gradient at perfect forecast (p={p})")

    @parameterized.expand([(1.0,), (2.0,), (4.0,)])
    def test_p_parameter_changes_output(self, p):
        """Different p values must produce different loss values for a spread ensemble."""
        fn_ref = self._fn(p=2.0)
        fn_p = self._fn(p=p)
        set_seed(333)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        if p == 2.0:
            self.assertTrue(compare_tensors("same p", fn_ref(fc, obs), fn_p(fc, obs)))
        else:
            self.assertFalse(compare_tensors("diff p", fn_ref(fc, obs), fn_p(fc, obs)))

    def test_batch_independence(self, verbose=False):
        fn = self._fn()
        fc = _rand_ensemble(self._E)
        obs = _rand()
        loss_single = fn(fc[:1], obs[:1])
        loss_batch = fn(fc, obs)
        self.assertTrue(
            compare_tensors("lp_es batch", loss_single[0], loss_batch[0], verbose=verbose),
        )

    @parameterized.expand([(2,), (3,), (5,)])
    def test_combinations_matches_reference(self, ensemble_size, verbose=False):
        """New upper-triangular combinations path must match the O(E^2) reference."""
        fn = self._fn(p=2.0)
        fc = _rand_ensemble(ensemble_size)
        obs = _rand()

        # --- reference: inline O(E^2) outer product (original implementation) ---
        def _reference_loss(forecasts, observations):
            B, E, C, H, W = forecasts.shape
            fc_e = torch.moveaxis(forecasts, 1, 0).reshape(E, B, C, H * W)  # (E, B, C, H*W)
            ob = observations.reshape(1, B, C, H * W)

            espread = (fc_e.unsqueeze(1) - fc_e.unsqueeze(0)).abs().pow(fn.p)  # (E, E, B, C, H*W)
            eskill = (ob - fc_e).abs().pow(fn.p)

            espread = torch.sum(espread * fn.quad_weight_split, dim=-1)  # (E, E, B, C)
            eskill = torch.sum(eskill * fn.quad_weight_split, dim=-1)  # (E, B, C)

            # channel reduction happens before mask/pow — same order as the production code
            if fn.channel_reduction:
                espread = espread.sum(dim=-1, keepdim=True)  # (E, E, B, 1)
                eskill = eskill.sum(dim=-1, keepdim=True)  # (E, B, 1)

            espread_mask = espread < fn.eps
            eskill_mask = eskill < fn.eps
            espread = torch.where(espread_mask, fn.eps, espread)
            eskill = torch.where(eskill_mask, fn.eps, eskill)

            espread = espread.float().pow(1.0 / fn.p).pow(fn.beta)
            eskill = eskill.float().pow(1.0 / fn.p).pow(fn.beta)

            espread = torch.where(espread_mask, 0.0, espread)
            eskill = torch.where(eskill_mask, 0.0, eskill)

            espread = espread.sum(dim=(0, 1)) * (float(E) - 1.0 + fn.alpha) / float(E * E * (E - 1))
            eskill = eskill.sum(dim=0) / float(E)

            return eskill - 0.5 * espread

        with torch.no_grad():
            loss_ref = _reference_loss(fc, obs)
            loss_new = fn(fc, obs)

        self.assertTrue(
            compare_tensors(f"lp_es combinations E={ensemble_size}", loss_ref, loss_new, verbose=verbose),
        )

        # also check that gradients agree
        fc_ref = fc.clone().requires_grad_(True)
        fc_new = fc.clone().requires_grad_(True)
        _reference_loss(fc_ref, obs).sum().backward()
        fn(fc_new, obs).sum().backward()
        self.assertTrue(
            compare_tensors(f"lp_es combinations grad E={ensemble_size}", fc_ref.grad, fc_new.grad, verbose=verbose),
        )


# ===========================================================================
class TestSpectralL2EnergyScoreLoss(unittest.TestCase):
    """Tests for SpectralL2EnergyScoreLoss."""

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, channel_reduction=True, **kw):
        return SpectralL2EnergyScoreLoss(
            **_SPEC_KWARGS,
            spatial_distributed=False,
            ensemble_distributed=False,
            channel_reduction=channel_reduction,
            **kw,
        )

    def test_output_shape_channel_reduction(self):
        fn = self._fn(channel_reduction=True)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, 1))

    def test_output_shape_no_channel_reduction(self):
        fn = self._fn(channel_reduction=False)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_backward_finite(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E, requires_grad=True)
        obs = _rand()
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    def test_zero_on_perfect_prediction(self, verbose=False):
        fn = self._fn()
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors("spec_l2_es zero", out, torch.zeros_like(out), atol=1e-4, verbose=verbose),
        )

    def test_backward_finite_on_perfect_prediction(self):
        """Perfect ensemble (all members == obs) must produce finite gradients."""
        fn = self._fn()
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone().requires_grad_(True)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in spec_l2_es gradient at perfect forecast")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in spec_l2_es gradient at perfect forecast")

    def test_penalizes_imaginary_only_error(self):
        """An error confined to Im(SH coefficients) must not be scored as perfect.

        Regression test: the forward used to cast the complex SHT output back to the
        real input dtype, which discarded the imaginary part of every coefficient. The
        score is built on |z|^2 (``.abs().square()``) and the m > 0 weights carry the
        Parseval factor 2 for the conjugate-symmetric negative-m modes, so dropping
        Im(z) silently halved the information in every m > 0 mode and made the error
        constructed here invisible -- the loss came out exactly 0.
        """

        fn = self._fn()
        fc, obs = _imaginary_only_pair(ensemble=self._E)

        # verify the construction: the real parts of the coefficients must be identical,
        # so anything the loss reports comes purely from the imaginary part
        with torch.no_grad():
            obs_coeffs = fn.sht(obs)
            fc_coeffs = fn.sht(fc[:, 0])
        with self.subTest(desc="perturbation is imaginary-only"):
            self.assertTrue(
                torch.allclose(obs_coeffs.real, fc_coeffs.real, atol=1e-5),
                "test construction is wrong: the real parts of the coefficients differ",
            )
            self.assertFalse(
                torch.allclose(obs_coeffs.imag, fc_coeffs.imag, atol=1e-5),
                "test construction is wrong: the imaginary parts are identical",
            )

        out = fn(fc, obs)
        with self.subTest(desc="loss is nonzero"):
            self.assertGreater(
                out.abs().max().item(),
                1e-3,
                "imaginary-only error scored as a perfect forecast -- Im(z) is being discarded",
            )

    def test_batch_independence(self, verbose=False):
        fn = self._fn()
        fc = _rand_ensemble(self._E)
        obs = _rand()
        loss_single = fn(fc[:1], obs[:1])
        loss_batch = fn(fc, obs)
        self.assertTrue(
            compare_tensors("spec_l2_es batch", loss_single[0], loss_batch[0], verbose=verbose),
        )


# ===========================================================================
class TestSpectralRegularization(unittest.TestCase):
    """Tests for SpectralRegularization."""

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, **kw):
        return SpectralRegularization(
            **_SPEC_KWARGS,
            spatial_distributed=False,
            ensemble_distributed=False,
            **kw,
        )

    def test_output_shape_5d(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_output_shape_4d(self):
        """4-D input (no ensemble dim) must also work."""
        fn = self._fn()
        fc = _rand()
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_backward_finite(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E, requires_grad=True)
        obs = _rand()
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    def test_zero_on_perfect_prediction(self, verbose=False):
        fn = self._fn()
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors("spec_reg zero", out, torch.zeros_like(out), atol=1e-4, verbose=verbose),
        )

    def test_nonneg(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertTrue((out >= -1e-6).all(), f"found negative values, min={out.min().item():.4e}")

    def test_logarithmic_mode(self):
        """logarithmic=True must produce finite output."""
        fn = self._fn(logarithmic=True)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertTrue(torch.isfinite(out).all())

    def test_wrong_forecast_dims_raises(self):
        fn = self._fn()
        fc = torch.randn(1, 2, 3)
        obs = torch.randn(1, 2, 3)
        with self.assertRaises(ValueError):
            fn(fc, obs)


# ===========================================================================
class TestGradientCRPSLoss(unittest.TestCase):
    """Tests for GradientCRPSLoss."""

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, crps_type="skillspread", absolute=True, **kw):
        return GradientCRPSLoss(
            **_GEOM_KWARGS,
            crps_type=crps_type,
            spatial_distributed=False,
            ensemble_distributed=False,
            absolute=absolute,
            **kw,
        )

    @parameterized.expand([("skillspread",), ("cdf",)])
    def test_output_shape(self, crps_type):
        fn = self._fn(crps_type)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_output_shape_absolute_false(self):
        fn = self._fn("skillspread", absolute=False)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, 2 * _NUM_CH))

    @parameterized.expand([("skillspread",), ("cdf",)])
    def test_nonneg(self, crps_type):
        fn = self._fn(crps_type)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertTrue(
            (out >= -1e-6).all(),
            f"{crps_type}: found negative values, min={out.min().item():.4e}",
        )

    def test_backward_finite(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E, requires_grad=True)
        obs = _rand()
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    @parameterized.expand([("skillspread",), ("cdf",)])
    def test_zero_on_perfect_prediction(self, crps_type, verbose=False):
        fn = self._fn(crps_type)
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors(f"grad_crps {crps_type} zero", out, torch.zeros_like(out), atol=1e-4, verbose=verbose),
        )

    @parameterized.expand([("skillspread",), ("cdf",)])
    def test_backward_finite_on_perfect_prediction(self, crps_type):
        """Perfect ensemble (all members == obs) must produce finite gradients."""
        fn = self._fn(crps_type)
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone().requires_grad_(True)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), f"NaN in {crps_type} gradient at perfect forecast")
        self.assertFalse(torch.isinf(fc.grad).any(), f"Inf in {crps_type} gradient at perfect forecast")

    def test_wrong_forecast_dims_raises(self):
        fn = self._fn()
        fc = _rand()
        obs = _rand()
        with self.assertRaises(ValueError):
            fn(fc, obs)


# ===========================================================================
class TestVortDivCRPSLoss(unittest.TestCase):
    """Tests for VortDivCRPSLoss.

    Requires channel_names with u/v wind pairs. Uses _WIND_GEOM_KWARGS
    which has ["u500", "v500", "u850", "v850", "t500"].
    """

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, crps_type="skillspread", **kw):
        return VortDivCRPSLoss(
            **_WIND_GEOM_KWARGS,
            crps_type=crps_type,
            spatial_distributed=False,
            ensemble_distributed=False,
            **kw,
        )

    @parameterized.expand([("skillspread",), ("cdf",)])
    def test_output_shape(self, crps_type):
        fn = self._fn(crps_type)
        fc = _rand_ensemble(self._E, channels=_NUM_WIND_CH)
        obs = _rand(channels=_NUM_WIND_CH)
        out = fn(fc, obs)
        # the loss now scores all channels (wind in vort/div space + scalars passed through)
        self.assertEqual(tuple(out.shape), (_BATCH, fn.n_channels))

    @parameterized.expand([("skillspread",), ("cdf",)])
    def test_nonneg(self, crps_type):
        fn = self._fn(crps_type)
        fc = _rand_ensemble(self._E, channels=_NUM_WIND_CH)
        obs = _rand(channels=_NUM_WIND_CH)
        out = fn(fc, obs)
        self.assertTrue(
            (out >= -1e-6).all(),
            f"{crps_type}: found negative values, min={out.min().item():.4e}",
        )

    def test_backward_finite(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E, channels=_NUM_WIND_CH, requires_grad=True)
        obs = _rand(channels=_NUM_WIND_CH)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    @parameterized.expand([("skillspread",), ("cdf",)])
    def test_zero_on_perfect_prediction(self, crps_type, verbose=False):
        fn = self._fn(crps_type)
        obs = _rand(channels=_NUM_WIND_CH)
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_WIND_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors(f"vortdiv {crps_type} zero", out, torch.zeros_like(out), atol=1e-4, verbose=verbose),
        )

    @parameterized.expand([("skillspread",), ("cdf",)])
    def test_backward_finite_on_perfect_prediction(self, crps_type):
        """Perfect ensemble (all members == obs) must produce finite gradients."""
        fn = self._fn(crps_type)
        obs = _rand(channels=_NUM_WIND_CH)
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_WIND_CH, _IMG_H, _IMG_W).clone().requires_grad_(True)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), f"NaN in vortdiv {crps_type} gradient at perfect forecast")
        self.assertFalse(torch.isinf(fc.grad).any(), f"Inf in vortdiv {crps_type} gradient at perfect forecast")

    def test_wrong_forecast_dims_raises(self):
        fn = self._fn()
        fc = _rand(channels=_NUM_WIND_CH)
        obs = _rand(channels=_NUM_WIND_CH)
        with self.assertRaises(ValueError):
            fn(fc, obs)

    def test_scores_all_channels(self):
        """Regression for issue #94: non-wind (scalar) channels must contribute to the
        loss instead of being silently dropped."""
        fn = self._fn("skillspread")

        # the loss now covers the whole state, not just the wind channels
        self.assertEqual(fn.n_channels, len(_WIND_CHANNEL_NAMES))
        nonwind = [i for i in range(_NUM_WIND_CH) if i not in fn.wind_chans.tolist()]
        self.assertGreaterEqual(len(nonwind), 1, "test config must contain a non-wind channel")

        obs = _rand(channels=_NUM_WIND_CH)
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_WIND_CH, _IMG_H, _IMG_W).clone()
        loss_perfect = fn(fc, obs).sum().item()

        # perturbing ONLY a non-wind channel must change the loss
        fc_pert = fc.clone()
        fc_pert[:, :, nonwind[0], ...] += 1.0
        loss_pert = fn(fc_pert, obs).sum().item()

        self.assertGreater(
            loss_pert,
            loss_perfect + 1e-3,
            f"non-wind channel perturbation did not affect the loss "
            f"(perfect={loss_perfect}, perturbed={loss_pert})",
        )


# ===========================================================================
class TestSpectralCoherenceLoss(unittest.TestCase):
    """Tests for SpectralCoherenceLoss.

    The loss is a probabilistic spectral score with two terms:
      - PSD skill:        ((P_F - P_T)^2) averaged over the ensemble; senses
                          the wrong *amplitude* at each spherical-harmonic
                          degree l
      - Coherence skill / spread:  energy-score-style decomposition of the
                          phase alignment between forecast↔truth and between
                          ensemble members; senses the wrong *phase* at each l

    These tests verify (a) the output-shape contract, (b) that perfect
    predictions yield zero loss, (c) that the PSD term and the coherence term
    each pick up the kind of error they're designed for.
    """

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, channel_reduction=True, relative=False, **kw):
        return SpectralCoherenceLoss(
            **_SPEC_KWARGS,
            spatial_distributed=False,
            ensemble_distributed=False,
            channel_reduction=channel_reduction,
            relative=relative,
            **kw,
        )

    # -- output-shape contract -----------------------------------------------

    def test_output_shape_channel_reduction(self):
        fn = self._fn(channel_reduction=True)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        # contract: (B, n_channels) where n_channels=1 when reducing
        self.assertEqual(tuple(out.shape), (_BATCH, 1))

    def test_output_shape_no_channel_reduction(self):
        fn = self._fn(channel_reduction=False)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_n_channels_matches_output(self):
        """n_channels must agree with the per-sample output dim under both reduction modes.

        Regression test for the previously-undefined self.channel_reduction:
        before this fix, n_channels would AttributeError before forward was
        even called.
        """
        fn_red = self._fn(channel_reduction=True)
        fn_no_red = self._fn(channel_reduction=False)
        self.assertEqual(fn_red.n_channels, 1)
        self.assertEqual(fn_no_red.n_channels, _NUM_CH)

    def test_wrong_forecast_dims_raises(self):
        fn = self._fn()
        fc = _rand()  # 4-D, missing ensemble dim
        obs = _rand()
        with self.assertRaises(ValueError):
            fn(fc, obs)

    # -- backward / zero-on-perfect-prediction -------------------------------

    def test_backward_finite(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E, requires_grad=True)
        obs = _rand()
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    @parameterized.expand([(True,), (False,)])
    def test_zero_on_perfect_prediction(self, relative, verbose=False):
        """All ensemble members == observation:
          - psd_skill = 0 exactly (same inputs through the same op)
          - coherences = P / sqrt(P² + eps) → 1 only as eps → 0

        Setting eps=0 collapses ``1 - P/sqrt(P²)`` to algebraically zero, but
        the loss uses two paths to compute |X|²: ``X.abs().square()`` for the
        PSD term and ``Re(X.conj() * X)`` for coherence. These are equal in
        real arithmetic but differ by a sqrt-then-square ULP per mode in fp32,
        so after summing 75 (l, channel) modes the residual lands at ~1e-6 for
        the relative case. atol is set above that floor."""
        fn = self._fn(relative=relative, eps=0.0)
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors(
                f"spectral coherence zero (relative={relative})",
                out,
                torch.zeros_like(out),
                atol=5e-6,
                verbose=verbose,
            )
        )

    def test_backward_finite_on_perfect_prediction(self):
        """Perfect forecast must not produce NaN/Inf gradients (eps in the
        normalizer protects against the 0/0 in coherence)."""
        fn = self._fn()
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone().requires_grad_(True)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN at perfect forecast")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf at perfect forecast")

    # -- the decomposition does what it claims ------------------------------

    def test_negated_forecast_isolates_coherence_term(self):
        """fc = -obs has identical PSD as obs but opposite phase, so:
          - psd_skill ≈ 0          (amplitudes match)
          - coherence(fc, obs) ≈ -1 → coh_skill picks up the phase mismatch
        Loss must therefore be strictly positive even though amplitudes are right.
        This proves the coherence/phase term is active and not masked by the
        PSD term.
        """
        obs = _rand()
        fc = (-obs).unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        # ensemble members are identical, so coh_spread ≈ 0; the contribution
        # is entirely from the coh_skill term
        out = self._fn()(fc, obs)
        self.assertTrue((out > 0.0).all(), f"loss not strictly positive on -obs: {out}")

    def test_scaled_forecast_isolates_psd_term(self):
        """fc = 2*obs has the same phases as obs (so coherence ≈ 1) but
        4× the PSD, so:
          - coh_skill ≈ 0          (phases match)
          - psd_skill ≈ (4P - P)^2 = 9P^2  per (b, c, l)
        Loss must therefore be strictly positive even though phases are right.
        This proves the PSD/amplitude term is active and not masked by the
        coherence term.
        """
        obs = _rand()
        fc = (2.0 * obs).unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = self._fn()(fc, obs)
        self.assertTrue((out > 0.0).all(), f"loss not strictly positive on 2*obs: {out}")

    def test_better_forecast_lower_loss(self):
        """Monotonicity in error magnitude: a forecast closer to the obs must
        produce a strictly lower loss than one farther from it. This is the
        weakest-but-most-useful behavioral guarantee against the loss
        accidentally pointing the wrong way after a refactor."""
        obs = _rand()
        # both forecasts are perturbations of obs, replicated across ensemble
        set_seed(101)
        small_noise = 0.05 * torch.randn_like(obs)
        large_noise = 0.50 * torch.randn_like(obs)
        fc_close = (obs + small_noise).unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        fc_far = (obs + large_noise).unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        fn = self._fn()
        loss_close = fn(fc_close, obs).sum().item()
        loss_far = fn(fc_far, obs).sum().item()
        self.assertLess(loss_close, loss_far, f"closer forecast had higher loss: {loss_close} vs {loss_far}")

    def test_relative_flag_changes_output(self):
        """relative=True divides psd_skill by psd_observations and drops the
        psd_observations weight on the coherence term, so the two modes must
        produce different values on a non-trivial input."""
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out_abs = self._fn(relative=False)(fc, obs)
        out_rel = self._fn(relative=True)(fc, obs)
        self.assertFalse(
            compare_tensors("relative vs absolute", out_abs, out_rel, atol=1e-6, rtol=1e-5),
            "relative=True and relative=False produced identical outputs",
        )


# ===========================================================================
class TestCorrectedSpectralL2EnergyScoreLoss(unittest.TestCase):
    """Tests for CorrectedSpectralL2EnergyScoreLoss.

    The corrected variant differs from SpectralL2EnergyScoreLoss only in the
    spread term: the ensemble pairwise spread is scaled by the ratio
    psd_true / (psd_pred + eps). The accuracy term (eskill) is identical, so
    most mechanical tests mirror TestSpectralL2EnergyScoreLoss.

    Two extra semantic tests exercise the *purpose* of the correction:
      - test_equivalence_at_correct_amplitude: when psd_pred ≈ psd_true the
        ratio collapses to 1 and the corrected loss must agree with the
        standard one.
      - test_cheap_spread_penalty: when the ensemble has the right phases but
        inflated amplitudes (the canonical "cheap spread" attack on the
        standard ES), the corrected loss must be strictly larger than the
        standard one — the diversity reward is bounded by truth PSD instead
        of by the model's own (inflated) PSD.
    """

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, channel_reduction=True, **kw):
        return CorrectedSpectralL2EnergyScoreLoss(
            **_SPEC_KWARGS,
            spatial_distributed=False,
            ensemble_distributed=False,
            channel_reduction=channel_reduction,
            **kw,
        )

    def test_penalizes_imaginary_only_error(self):
        """Same regression as TestSpectralL2EnergyScoreLoss -- the corrected variant
        carried an identical complex-to-real cast in its forward."""

        fn = self._fn()
        fc, obs = _imaginary_only_pair(ensemble=self._E)

        out = fn(fc, obs)
        self.assertGreater(
            out.abs().max().item(),
            1e-3,
            "imaginary-only error scored as a perfect forecast -- Im(z) is being discarded",
        )

    def _fn_uncorrected(self, channel_reduction=True, **kw):
        return SpectralL2EnergyScoreLoss(
            **_SPEC_KWARGS,
            spatial_distributed=False,
            ensemble_distributed=False,
            channel_reduction=channel_reduction,
            **kw,
        )

    # -- shape contract ------------------------------------------------------

    def test_output_shape_channel_reduction(self):
        fn = self._fn(channel_reduction=True)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, 1))

    def test_output_shape_no_channel_reduction(self):
        fn = self._fn(channel_reduction=False)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_n_channels_matches_output(self):
        self.assertEqual(self._fn(channel_reduction=True).n_channels, 1)
        self.assertEqual(self._fn(channel_reduction=False).n_channels, _NUM_CH)

    def test_wrong_forecast_dims_raises(self):
        fn = self._fn()
        fc = _rand()  # 4-D, missing ensemble dim
        obs = _rand()
        with self.assertRaises(ValueError):
            fn(fc, obs)

    # -- backward / zero-on-perfect-prediction -------------------------------

    def test_backward_finite(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E, requires_grad=True)
        obs = _rand()
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    def test_zero_on_perfect_prediction(self, verbose=False):
        """All ensemble members == observation: espread and eskill are zero
        before sqrt; the eps-mask path replaces them with eps for the sqrt
        and resets to 0 afterward, so the post-sqrt residual is exactly 0.
        The ratio (psd_true / (psd_pred + eps)) is finite and unused (multiplied
        by zero spread), so the eps in the ratio doesn't leak into the result.
        """
        fn = self._fn()
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors(
                "corrected spectral L2 ES zero",
                out,
                torch.zeros_like(out),
                atol=1e-6,
                verbose=verbose,
            )
        )

    def test_backward_finite_on_perfect_prediction(self):
        """Perfect ensemble must produce finite gradients; the eps-mask
        protects against the sqrt(0) gradient singularity."""
        fn = self._fn()
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone().requires_grad_(True)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN at perfect forecast")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf at perfect forecast")

    def test_batch_independence(self, verbose=False):
        """The loss for sample [0] computed alone must equal loss[0] in a full batch."""
        fn = self._fn()
        fc = _rand_ensemble(self._E)
        obs = _rand()
        loss_single = fn(fc[:1], obs[:1])
        loss_batch = fn(fc, obs)
        self.assertTrue(
            compare_tensors("corrected spectral L2 ES batch", loss_single[0], loss_batch[0], verbose=verbose)
        )

    # -- semantic tests: the correction does what its docstring claims -------

    def test_equivalence_at_correct_amplitude(self, verbose=False):
        """When psd_pred ≈ psd_true the spread-rescaling ratio collapses to 1
        and the corrected loss must agree with the standard SpectralL2 ES.

        Construction: forecasts are observations + small noise per member.
        For small enough noise psd_pred ≈ psd_true to within a small relative
        tolerance, and the two loss formulas should produce nearly identical
        outputs.
        """
        obs = _rand()
        # noise-amplitude ≪ obs-amplitude so psd_pred deviation from psd_true
        # stays in the percent range — atol/rtol below set accordingly
        noise = 0.02 * torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone() + noise

        out_corrected = self._fn()(fc, obs)
        out_standard = self._fn_uncorrected()(fc, obs)

        # absolute tolerance scaled to the typical loss magnitude here (~1e-2);
        # rtol reflects the fact that ratio ≠ 1 exactly even with tiny noise
        self.assertTrue(
            compare_tensors(
                "corrected ≈ standard at correct amplitude",
                out_corrected,
                out_standard,
                atol=1e-4,
                rtol=5e-2,
                verbose=verbose,
            ),
            f"corrected={out_corrected.tolist()} standard={out_standard.tolist()}",
        )

    def test_cheap_spread_penalty(self):
        """The whole point of the correction.

        Setup: ensemble with the right *phases* (members are scaled obs + noise)
        but inflated *amplitudes* (k=3). Then psd_pred ≈ k² · psd_true so the
        ratio psd_true / (psd_pred + eps) ≈ 1/k² ≪ 1.

        The accuracy term (eskill) is identical between the two losses. The
        spread term in the corrected loss is scaled down by ratio ≈ 1/9, so it
        provides much less negative contribution. Therefore:

            corrected_loss > standard_loss

        i.e. the corrected loss does not let the model collect cheap spread
        reward by inflating its predicted PSD.
        """
        k = 3.0
        obs = _rand()
        # different noise per member so espread > 0; otherwise the spread term
        # is zero in BOTH losses and the test is degenerate
        noise = 0.1 * torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        fc_inflated = k * obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W) + noise

        out_corrected = self._fn()(fc_inflated, obs)
        out_standard = self._fn_uncorrected()(fc_inflated, obs)

        # corrected must be strictly larger element-wise — the cheap-spread
        # protection is per (B, n_channels) and not just on the average
        self.assertTrue(
            (out_corrected > out_standard).all(),
            f"corrected loss not strictly above standard: "
            f"corrected={out_corrected.tolist()} standard={out_standard.tolist()}",
        )

    def test_eps_does_not_dominate_when_psd_is_small(self):
        """Defensive: the ratio psd_true / (psd_pred + eps) uses eps as a guard
        against divide-by-zero. For very small psd_pred this can dominate and
        skew the spread-rescaling. We verify that for *unit-scale* random
        inputs (where psd_pred is well above eps everywhere) the corrected and
        standard losses are within a few percent — i.e. eps doesn't leak into
        the result at the default eps=1e-6.
        """
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out_corrected = self._fn()(fc, obs)
        out_standard = self._fn_uncorrected()(fc, obs)
        # both finite, no NaN/Inf
        self.assertTrue(torch.isfinite(out_corrected).all())
        self.assertTrue(torch.isfinite(out_standard).all())
        # not the same — random forecasts have psd_pred ≠ psd_true so ratio ≠ 1
        # and the spread term is rescaled
        self.assertFalse(
            compare_tensors("corrected vs standard random", out_corrected, out_standard, atol=1e-6, rtol=1e-4),
            "corrected and standard agree on random ensemble — ratio≠1 should produce a measurable difference",
        )


# ===========================================================================
class TestKernelScoreLoss(unittest.TestCase):
    """Tests for KernelScoreLoss (CRPS-style score with a fixed DISCO kernel).

    KernelScoreLoss applies a non-trainable spherical convolution to forecasts
    and observations before computing the CRPS, so it scores structural error
    rather than pointwise error. The conv is fixed (registered as a buffer) and
    the loss dispatches on crps_type just like CRPSLoss/SpectralCRPSLoss.

    Tests cover:
      - shape contract (B, n_channels)
      - all four crps_type branches (cdf, skillspread, naive skillspread,
        probability weighted moment)
      - zero on perfect prediction (the conv applies identically to both inputs
        so any deterministic kernel preserves the perfect-forecast → zero
        property regardless of the actual weight values)
      - alpha < 1 guard: rejected for non-skillspread kernels, accepted for
        both skillspread variants (regression test for the recent guard fix)
    """

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, crps_type="skillspread", **kw):
        return KernelScoreLoss(
            **_GEOM_KWARGS,
            crps_type=crps_type,
            spatial_distributed=False,
            ensemble_distributed=False,
            **kw,
        )

    # -- shape contract / dispatch validation --------------------------------

    @parameterized.expand(
        [
            ("cdf",),
            ("skillspread",),
            ("naive skillspread",),
            ("probability weighted moment",),
        ]
    )
    def test_output_shape(self, crps_type):
        """Output must be (B, n_channels) for every crps_type."""
        fn = self._fn(crps_type)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_n_channels_matches_output(self):
        fn = self._fn()
        self.assertEqual(fn.n_channels, _NUM_CH)

    def test_wrong_forecast_dims_raises(self):
        fn = self._fn()
        fc = _rand()  # 4-D, missing ensemble dim
        obs = _rand()
        with self.assertRaises(ValueError):
            fn(fc, obs)

    def test_unknown_crps_type_raises_in_forward(self):
        """Unknown crps_type bypassed init guard must raise ValueError in forward."""
        fn = self._fn("cdf")
        fn.crps_type = "bogus"
        fc = _rand_ensemble(self._E)
        obs = _rand()
        with self.assertRaises(ValueError):
            fn(fc, obs)

    # -- alpha guard ---------------------------------------------------------

    def test_alpha_lt1_raises_for_cdf(self):
        with self.assertRaises(NotImplementedError):
            self._fn("cdf", alpha=0.5)

    def test_alpha_lt1_raises_for_pwm(self):
        with self.assertRaises(NotImplementedError):
            self._fn("probability weighted moment", alpha=0.5)

    def test_alpha_lt1_allowed_for_skillspread(self):
        """Both skillspread variants accept alpha < 1 — the guard widening
        we did earlier must hold for KernelScoreLoss too."""
        fn_skill = self._fn("skillspread", alpha=0.5)
        fn_naive = self._fn("naive skillspread", alpha=0.5)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        # both must run without raising
        out_skill = fn_skill(fc, obs)
        out_naive = fn_naive(fc, obs)
        self.assertTrue(torch.isfinite(out_skill).all())
        self.assertTrue(torch.isfinite(out_naive).all())

    # -- backward / zero-on-perfect-prediction -------------------------------

    def test_backward_finite(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E, requires_grad=True)
        obs = _rand()
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    @parameterized.expand(
        [
            ("cdf",),
            ("skillspread",),
            ("naive skillspread",),
            ("probability weighted moment",),
        ]
    )
    def test_zero_on_perfect_prediction(self, crps_type, verbose=False):
        """Perfect ensemble (all members == obs): the conv applies the same
        deterministic transform to both inputs, so conv(obs) - conv(F_e) = 0
        for every member regardless of the conv weight values. Every CRPS
        kernel reduces to zero on identical-to-truth ensembles."""
        fn = self._fn(crps_type)
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors(
                f"kernel score {crps_type} zero",
                out,
                torch.zeros_like(out),
                atol=1e-4,
                verbose=verbose,
            )
        )

    @parameterized.expand(
        [
            ("cdf",),
            ("skillspread",),
            ("naive skillspread",),
            ("probability weighted moment",),
        ]
    )
    def test_backward_finite_on_perfect_prediction(self, crps_type):
        """Perfect ensemble must produce finite gradients across all crps_type
        branches (the eps-mask paths in each kernel must protect their respective
        sqrt(0) / 0/0 singularities)."""
        fn = self._fn(crps_type)
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone().requires_grad_(True)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), f"NaN in {crps_type} gradient at perfect forecast")
        self.assertFalse(torch.isinf(fc.grad).any(), f"Inf in {crps_type} gradient at perfect forecast")

    def test_batch_independence(self, verbose=False):
        """The kernel-score loss for sample [0] computed alone must equal
        loss[0] in a full batch — the conv and CRPS kernels are per-sample."""
        fn = self._fn()
        fc = _rand_ensemble(self._E)
        obs = _rand()
        loss_single = fn(fc[:1], obs[:1])
        loss_batch = fn(fc, obs)
        self.assertTrue(compare_tensors("kernel score batch", loss_single[0], loss_batch[0], verbose=verbose))

    # -- behavioural ---------------------------------------------------------

    def test_better_forecast_lower_loss(self):
        """Monotonicity sanity: a forecast closer to obs gives a strictly
        lower kernel score than one farther from obs. Holds for any
        deterministic kernel because the kernel is applied identically to
        both inputs."""
        obs = _rand()
        set_seed(101)
        small = 0.05 * torch.randn_like(obs)
        large = 0.50 * torch.randn_like(obs)
        fc_close = (obs + small).unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        fc_far = (obs + large).unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        fn = self._fn()
        loss_close = fn(fc_close, obs).sum().item()
        loss_far = fn(fc_far, obs).sum().item()
        self.assertLess(loss_close, loss_far, f"closer forecast had higher loss: {loss_close} vs {loss_far}")


# ===========================================================================
class TestEnsembleLossE1FastPath(unittest.TestCase):
    """E=1 fast-path coverage for all CRPS-style ensemble losses.

    Each of CRPSLoss, SpectralCRPSLoss, GradientCRPSLoss, VortDivCRPSLoss,
    and KernelScoreLoss has an explicit ``(not ensemble_distributed) and (E == 1)``
    short-circuit in its forward that bypasses the standard CRPS kernel
    dispatch and computes ``|obs - fc.squeeze(1)|`` directly. None of the
    existing tests use E=1 (they all parameterize over E=5), so this branch
    is untested in production use cases that run with a single ensemble member.

    For each class we verify:
      - the E=1 path produces a finite, correctly-shaped output
      - perfect prediction (single member == observation) gives zero loss
      - backward through the E=1 path produces finite gradients

    The energy-score family and the coherence losses are covered here too. They
    have no fast path -- their structure is skill minus spread -- but their
    spread terms divide by N(N-1), which is 0/0 at E=1 and used to make the whole
    score NaN. Per issue #96 they now zero the spread instead, since a sum over
    member pairs is empty when there is only one member, so the score degrades to
    the skill term. GaussianMMDLoss already did this.

    For those, ``test_e1_spread_term_vanishes`` is the semantic check: E=1 must
    agree with a two-member ensemble of *identical* members, whose pairwise
    spread is also zero. MMD is excluded from it because its spread is a kernel
    sum rather than a difference sum -- k(x,x)=1, so identical members carry a
    non-zero spread by construction, and only the E=1 empty-sum case is zero.
    """

    def setUp(self):
        disable_tf32()
        set_seed(333)

    # -- factories: build each loss with default kwargs --------------------

    def _crps(self):
        return CRPSLoss(
            **_GEOM_KWARGS,
            crps_type="skillspread",
            spatial_distributed=False,
            ensemble_distributed=False,
        )

    def _spectral_crps(self):
        return SpectralCRPSLoss(
            **_SPEC_KWARGS,
            crps_type="skillspread",
            spatial_distributed=False,
            ensemble_distributed=False,
            absolute=True,
        )

    def _gradient_crps(self):
        return GradientCRPSLoss(
            **_GEOM_KWARGS,
            crps_type="skillspread",
            spatial_distributed=False,
            ensemble_distributed=False,
            absolute=True,
        )

    def _vortdiv_crps(self):
        return VortDivCRPSLoss(
            **_WIND_GEOM_KWARGS,
            crps_type="skillspread",
            spatial_distributed=False,
            ensemble_distributed=False,
        )

    def _kernel_score(self):
        return KernelScoreLoss(
            **_GEOM_KWARGS,
            crps_type="skillspread",
            spatial_distributed=False,
            ensemble_distributed=False,
        )

    # -- factories: the skill-minus-spread family (issue #96) ---------------
    #
    # eps is shrunk from its 1e-6 default for the two coherence losses: their
    # spread is a ratio, and the eps floor keeps (1 - Coh) slightly above zero for
    # identical members, which would otherwise mask the property under test.

    def _lp_energy_score(self):
        return LpEnergyScoreLoss(**_GEOM_KWARGS, p=2.0, spatial_distributed=False, ensemble_distributed=False)

    def _sobolev_energy_score(self):
        return SobolevEnergyScoreLoss(**_SPEC_KWARGS, spatial_distributed=False, ensemble_distributed=False)

    def _spectral_l2_energy_score(self):
        return SpectralL2EnergyScoreLoss(**_SPEC_KWARGS, spatial_distributed=False, ensemble_distributed=False)

    def _corrected_spectral_l2_energy_score(self):
        return CorrectedSpectralL2EnergyScoreLoss(**_SPEC_KWARGS, spatial_distributed=False, ensemble_distributed=False)

    def _spectral_coherence(self):
        return SpectralCoherenceLoss(**_SPEC_KWARGS, eps=1e-16, spatial_distributed=False, ensemble_distributed=False)

    def _coherence_regularization(self):
        return CoherenceRegularization(
            **_SPEC_KWARGS,
            eps=1e-16,
            ensemble_coherence_weight=0.5,
            spatial_distributed=False,
            ensemble_distributed=False,
        )

    def _gaussian_mmd(self):
        return GaussianMMDLoss(**_GEOM_KWARGS, spatial_distributed=False, ensemble_distributed=False)

    def _make(self, name):
        return {
            "CRPSLoss": (self._crps, _NUM_CH),
            "SpectralCRPSLoss": (self._spectral_crps, _NUM_CH),
            "GradientCRPSLoss": (self._gradient_crps, _NUM_CH),
            "VortDivCRPSLoss": (self._vortdiv_crps, _NUM_WIND_CH),
            "KernelScoreLoss": (self._kernel_score, _NUM_CH),
            "LpEnergyScoreLoss": (self._lp_energy_score, _NUM_CH),
            "SobolevEnergyScoreLoss": (self._sobolev_energy_score, _NUM_CH),
            "SpectralL2EnergyScoreLoss": (self._spectral_l2_energy_score, _NUM_CH),
            "CorrectedSpectralL2EnergyScoreLoss": (self._corrected_spectral_l2_energy_score, _NUM_CH),
            "SpectralCoherenceLoss": (self._spectral_coherence, _NUM_CH),
            "CoherenceRegularization": (self._coherence_regularization, _NUM_CH),
            "GaussianMMDLoss": (self._gaussian_mmd, _NUM_CH),
        }[name]

    # -- tests --------------------------------------------------------------

    @parameterized.expand(
        [
            ("CRPSLoss",),
            ("SpectralCRPSLoss",),
            ("GradientCRPSLoss",),
            ("VortDivCRPSLoss",),
            ("KernelScoreLoss",),
            ("LpEnergyScoreLoss",),
            ("SobolevEnergyScoreLoss",),
            ("SpectralL2EnergyScoreLoss",),
            ("CorrectedSpectralL2EnergyScoreLoss",),
            ("SpectralCoherenceLoss",),
            ("CoherenceRegularization",),
            ("GaussianMMDLoss",),
        ]
    )
    def test_e1_output_shape_and_finite(self, name):
        """E=1 path produces (B, n_channels) and finite values."""
        builder, n_ch = self._make(name)
        fn = builder()
        fc = torch.randn(_BATCH, 1, n_ch, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, n_ch, _IMG_H, _IMG_W)
        out = fn(fc, obs)
        self.assertEqual(out.shape[0], _BATCH)
        self.assertEqual(out.shape[1], fn.n_channels)
        self.assertTrue(torch.isfinite(out).all(), f"{name}: non-finite values in E=1 output")

    @parameterized.expand(
        [
            ("CRPSLoss",),
            ("SpectralCRPSLoss",),
            ("GradientCRPSLoss",),
            ("VortDivCRPSLoss",),
            ("KernelScoreLoss",),
            ("LpEnergyScoreLoss",),
            ("SobolevEnergyScoreLoss",),
            ("SpectralL2EnergyScoreLoss",),
            ("CorrectedSpectralL2EnergyScoreLoss",),
        ]
    )
    def test_e1_zero_on_perfect_prediction(self, name, verbose=False):
        """At E=1, fc[:,0] == obs reduces to |obs - obs| = 0; loss must be (near) zero."""
        builder, n_ch = self._make(name)
        fn = builder()
        obs = torch.randn(_BATCH, n_ch, _IMG_H, _IMG_W)
        fc = obs.unsqueeze(1).clone()  # (B, 1, C, H, W) — single member equals obs
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors(
                f"{name} E=1 zero",
                out,
                torch.zeros_like(out),
                atol=1e-4,
                verbose=verbose,
            )
        )

    @parameterized.expand(
        [
            ("CRPSLoss",),
            ("SpectralCRPSLoss",),
            ("GradientCRPSLoss",),
            ("VortDivCRPSLoss",),
            ("KernelScoreLoss",),
            ("LpEnergyScoreLoss",),
            ("SobolevEnergyScoreLoss",),
            ("SpectralL2EnergyScoreLoss",),
            ("CorrectedSpectralL2EnergyScoreLoss",),
            ("SpectralCoherenceLoss",),
            ("CoherenceRegularization",),
            ("GaussianMMDLoss",),
        ]
    )
    def test_e1_backward_finite(self, name):
        """Gradient through the E=1 path must be finite — no NaN/Inf even though
        the loss reduces to a piecewise-linear |x| at the bottom."""
        builder, n_ch = self._make(name)
        fn = builder()
        fc = torch.randn(_BATCH, 1, n_ch, _IMG_H, _IMG_W, requires_grad=True)
        obs = torch.randn(_BATCH, n_ch, _IMG_H, _IMG_W)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), f"{name}: NaN grad in E=1 backward")
        self.assertFalse(torch.isinf(fc.grad).any(), f"{name}: Inf grad in E=1 backward")

    @parameterized.expand(
        [
            ("LpEnergyScoreLoss",),
            ("SobolevEnergyScoreLoss",),
            ("SpectralL2EnergyScoreLoss",),
            ("CorrectedSpectralL2EnergyScoreLoss",),
            ("SpectralCoherenceLoss",),
            ("CoherenceRegularization",),
        ]
    )
    def test_e1_spread_term_vanishes(self, name, verbose=False):
        """Issue #96: at E=1 the spread is an empty sum and must contribute nothing.

        Rather than reimplementing each score's formula, compare against a
        two-member ensemble whose members are identical: its pairwise spread is
        zero as well, so the two must agree exactly. If the E=1 guard instead
        returned NaN, or dropped the skill term, this would catch it.

        GaussianMMDLoss is excluded: its spread is a kernel sum with k(x,x)=1, so
        identical members have a non-zero spread by construction and only the
        genuinely empty E=1 sum is zero.
        """
        builder, n_ch = self._make(name)
        fn = builder()
        obs = torch.randn(_BATCH, n_ch, _IMG_H, _IMG_W)
        fc1 = torch.randn(_BATCH, 1, n_ch, _IMG_H, _IMG_W)

        out_e1 = fn(fc1, obs)
        out_e2 = fn(fc1.repeat(1, 2, 1, 1, 1), obs)

        self.assertTrue(torch.isfinite(out_e1).all(), f"{name}: non-finite E=1 output")
        self.assertTrue(compare_tensors(f"{name} E=1 vs E=2-identical", out_e1, out_e2, atol=1e-4, verbose=verbose))

    @parameterized.expand(
        [
            ("LpEnergyScoreLoss",),
            ("SobolevEnergyScoreLoss",),
            ("SpectralL2EnergyScoreLoss",),
            ("CorrectedSpectralL2EnergyScoreLoss",),
            ("SpectralCoherenceLoss",),
            ("CoherenceRegularization",),
            ("GaussianMMDLoss",),
        ]
    )
    def test_e1_matches_larger_ensemble_shape(self, name):
        """The E=1 guard must not change the output contract for E > 1."""
        builder, n_ch = self._make(name)
        fn = builder()
        obs = torch.randn(_BATCH, n_ch, _IMG_H, _IMG_W)
        out1 = fn(torch.randn(_BATCH, 1, n_ch, _IMG_H, _IMG_W), obs)
        out4 = fn(torch.randn(_BATCH, 4, n_ch, _IMG_H, _IMG_W), obs)
        self.assertEqual(out1.shape, out4.shape)
        self.assertTrue(torch.isfinite(out4).all(), f"{name}: non-finite E=4 output")


class TestCoherenceRegularization(unittest.TestCase):
    """Tests for CoherenceRegularization.

    The loss penalizes low spectral coherence between each ensemble member and
    the observation over a wavenumber band:

        Coh_l = CrossPSD_l / sqrt(PSD^f_l * PSD^y_l + eps)
        Loss  = mean_{l in band} (1 - mean_e Coh_l)

    Coherence is signed, so the loss has three reference points that anchor most
    of the tests below:

        f = +y  -> Coh = +1 -> loss 0    (perfectly correlated)
        f random-> Coh ~  0 -> loss 1    (uncorrelated)
        f = -y  -> Coh = -1 -> loss 2    (anti-correlated)

    Several tests use a tiny eps. The default eps=1e-6 is a stabilizer that
    biases the ratio toward 0 wherever the PSD is small, which pulls the loss
    toward the "uncorrelated" value of 1: at eps=1e-6 the perfectly-correlated
    case scores 0.086 rather than 0. That is fine for training but makes the
    exact reference points untestable, so the analytic tests shrink eps.

    This class also serves as the regression test for the three bugs reported in
    issue #95, each marked below.
    """

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, **kw):
        return CoherenceRegularization(
            **_SPEC_KWARGS,
            spatial_distributed=False,
            ensemble_distributed=False,
            **kw,
        )

    @staticmethod
    def _broadcast(obs, ensemble, scale=1.0):
        """Build an ensemble in which every member is ``scale * obs``."""
        return scale * obs.unsqueeze(1).repeat(1, ensemble, 1, 1, 1)

    # -- output-shape contract -----------------------------------------------

    def test_output_shape(self):
        out = self._fn()(_rand_ensemble(self._E), _rand())
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_n_channels_matches_output(self):
        fn = self._fn()
        self.assertEqual(fn.n_channels, _NUM_CH)
        self.assertEqual(fn(_rand_ensemble(self._E), _rand()).shape[-1], fn.n_channels)

    def test_wrong_forecast_dims_raises(self):
        with self.assertRaises(ValueError):
            self._fn()(_rand(), _rand())  # 4-D, missing ensemble dim

    # -- analytic reference points -------------------------------------------

    def test_fully_correlated_is_zero(self):
        """f = y is perfect coherence, so the loss must vanish."""
        obs = _rand()
        out = self._fn(eps=1e-16)(self._broadcast(obs, self._E), obs)
        self.assertTrue(
            torch.allclose(out, torch.zeros_like(out), atol=1e-3), f"expected 0, got mean {out.mean().item()}"
        )

    def test_anti_correlated_is_two(self):
        """f = -y is coherence -1, so (1 - Coh) = 2.

        This is the case that distinguishes the signed coherence used here from
        the sign-blind magnitude-squared convention, which would score this
        identically to a perfect forecast.
        """
        obs = _rand()
        out = self._fn(eps=1e-16)(self._broadcast(obs, self._E, scale=-1.0), obs)
        self.assertTrue(
            torch.allclose(out, 2.0 * torch.ones_like(out), atol=1e-3), f"expected 2, got mean {out.mean().item()}"
        )

    def test_uncorrelated_is_one(self):
        """Independent forecasts have zero expected coherence, so the loss ~ 1.

        This one is statistical rather than exact, hence the loose tolerance.
        """
        out = self._fn(eps=1e-16)(_rand_ensemble(self._E), _rand())
        self.assertAlmostEqual(out.mean().item(), 1.0, delta=0.1)

    def test_ordering_of_the_three_regimes(self):
        """correlated < uncorrelated < anti-correlated, without tolerance games."""
        obs = _rand()
        fn = self._fn(eps=1e-16)
        pos = fn(self._broadcast(obs, self._E), obs).mean().item()
        rnd = fn(_rand_ensemble(self._E), obs).mean().item()
        neg = fn(self._broadcast(obs, self._E, scale=-1.0), obs).mean().item()
        self.assertLess(pos, rnd)
        self.assertLess(rnd, neg)

    def test_positive_scaling_is_invariant(self):
        """Coherence is normalized, so f = a*y scores 0 for any a > 0."""
        obs = _rand()
        fn = self._fn(eps=1e-16)
        for scale in (0.5, 1.0, 7.0):
            out = fn(self._broadcast(obs, self._E, scale=scale), obs)
            self.assertTrue(
                torch.allclose(out, torch.zeros_like(out), atol=1e-3),
                f"scale {scale}: expected 0, got {out.mean().item()}",
            )

    # -- issue #95 regressions ------------------------------------------------

    def test_cross_psd_is_computed(self):
        """Issue #95 bug 1: the forecast-observation cross-PSD must be used.

        The previous implementation correlated forecasts with each other and
        never formed the cross term, so flipping the sign of the forecast was
        invisible to the loss. A sign flip must move the result by 2.
        """
        obs = _rand()
        fn = self._fn(eps=1e-16)
        pos = fn(self._broadcast(obs, self._E), obs)
        neg = fn(self._broadcast(obs, self._E, scale=-1.0), obs)
        self.assertTrue(torch.allclose(neg - pos, 2.0 * torch.ones_like(pos), atol=1e-3))

    def test_loss_depends_on_observations(self):
        """Issue #95 bug 1: swapping the observation must change the loss."""
        fc = _rand_ensemble(self._E)
        fn = self._fn()
        self.assertFalse(torch.allclose(fn(fc, _rand()), fn(fc, _rand())))

    def test_band_mask_is_applied(self):
        """Issue #95 bug 2: lmin must restrict the wavenumber band.

        Previously l_band was built and registered but never used in forward, so
        lmin had no effect at all.
        """
        widths = [self._fn(lmin=lmin).band_size.item() for lmin in (0, 4, 8)]
        self.assertEqual(widths, sorted(widths, reverse=True))
        self.assertGreater(widths[0], widths[-1])

    def test_band_mask_changes_the_loss(self):
        """Issue #95 bug 2: a different band must give a different value."""
        fc, obs = _rand_ensemble(self._E), _rand()
        self.assertNotAlmostEqual(
            self._fn(lmin=0)(fc, obs).mean().item(), self._fn(lmin=8)(fc, obs).mean().item(), places=4
        )

    def test_band_restricted_to_single_wavenumber(self):
        fn = self._fn(lmin=14)
        self.assertEqual(fn.band_size.item(), 1.0)
        self.assertEqual(tuple(fn(_rand_ensemble(self._E), _rand()).shape), (_BATCH, _NUM_CH))

    def test_empty_band_raises(self):
        """lmin at or beyond the bandlimit would silently produce an empty band."""
        with self.assertRaises(ValueError):
            self._fn(lmin=10_000)

    def test_ensemble_coherence_weight_is_applied(self):
        """Issue #95 bug 3: the weight was stored but never read.

        The inter-member term enters linearly, so the increment over the
        weight=0 baseline must be proportional to the weight.
        """
        fc, obs = _rand_ensemble(self._E), _rand()
        base = self._fn(ensemble_coherence_weight=0.0)(fc, obs)
        d1 = self._fn(ensemble_coherence_weight=1.0)(fc, obs) - base
        d2 = self._fn(ensemble_coherence_weight=2.0)(fc, obs) - base
        self.assertFalse(torch.allclose(d1, torch.zeros_like(d1)))
        self.assertTrue(torch.allclose(d2, 2.0 * d1, atol=1e-4))

    # -- edge cases -----------------------------------------------------------

    def test_single_member_skips_inter_member_term(self):
        """E=1 would divide by E*(E-1)=0 if the inter-member term ran."""
        out = self._fn(ensemble_coherence_weight=1.0)(_rand_ensemble(1), _rand())
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_backward_is_finite(self):
        fc = _rand_ensemble(self._E, requires_grad=True)
        self._fn(ensemble_coherence_weight=0.5)(fc, _rand()).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any())
        self.assertFalse(torch.isinf(fc.grad).any())


# ===========================================================================
class TestComputeAlphaPerStep(unittest.TestCase):
    """compute_alpha_per_step against the closed-form schedules."""

    @parameterized.expand([(0,), (1,), (4,)])
    def test_linear(self, n_future):
        alpha = compute_alpha_per_step(n_future, schedule="linear", alpha_min=0.2, alpha_max=0.8)
        if n_future == 0:
            expected = [0.8]
        else:
            expected = [0.2 + 0.6 * k / n_future for k in range(n_future + 1)]
        self.assertTrue(torch.allclose(alpha, torch.tensor(expected), atol=1e-6))

    @parameterized.expand([(0, None), (4, None), (4, 0.25)])
    def test_sigmoid(self, n_future, t0_frac):
        alpha = compute_alpha_per_step(
            n_future, schedule="sigmoid", alpha_max=0.8, sigmoid_t0_frac=t0_frac, sigmoid_beta=3.0
        )
        t0 = 0.5 if t0_frac is None else t0_frac
        expected = [0.8 / (1.0 + math.exp(-3.0 * (k / max(n_future, 1) - t0))) for k in range(n_future + 1)]
        self.assertTrue(torch.allclose(alpha, torch.tensor(expected), atol=1e-6))

    def test_unknown_schedule_is_constant(self):
        alpha = compute_alpha_per_step(3, schedule="constant", alpha_min=0.1, alpha_max=0.7)
        self.assertTrue(torch.allclose(alpha, torch.full((4,), 0.7)))

    @parameterized.expand([("linear", 0.5), ("quadratic", 0.25), ("unknown", 0.5)])
    def test_annealing(self, annealing, factor):
        base = compute_alpha_per_step(3, schedule="linear", alpha_min=0.2, alpha_max=1.0)
        alpha = compute_alpha_per_step(
            3, schedule="linear", alpha_min=0.2, alpha_max=1.0, training_progress=0.5, annealing=annealing
        )
        self.assertTrue(torch.allclose(alpha, factor * base, atol=1e-6))


# ===========================================================================
class TestCRPSKernelReference(unittest.TestCase):
    """The skillspread and gauss kernels against independent reference formulas."""

    def setUp(self):
        set_seed(333)

    @parameterized.expand([(2, 0.0), (5, 0.0), (5, 0.5), (5, 1.0), (8, 0.95)])
    def test_skillspread_matches_pairwise_formula(self, ens, alpha, verbose=False):
        obs = torch.randn(3, 7, dtype=torch.float64)
        fc = torch.randn(ens, 3, 7, dtype=torch.float64)
        # the kernel expects the ensemble sorted along dim 0
        fc_sorted, _ = torch.sort(fc, dim=0)
        out = _crps_skillspread_kernel(obs, fc_sorted, torch.ones_like(fc_sorted), alpha)

        # E|X - y| - (N - 1 + alpha) / (N - 1) * sum_ij |x_i - x_j| / (2 N^2): alpha=1 is the fair CRPS,
        # alpha=0 the biased ensemble CRPS
        skill = (fc - obs.unsqueeze(0)).abs().mean(dim=0)
        pairwise = (fc.unsqueeze(0) - fc.unsqueeze(1)).abs().sum(dim=(0, 1))
        expected = skill - (ens - 1 + alpha) / (ens - 1) * pairwise / (2 * ens**2)
        self.assertTrue(compare_tensors("skillspread", out, expected, atol=1e-12, rtol=1e-12, verbose=verbose))

    def test_gauss_matches_normal_crps(self, verbose=False):
        obs = torch.randn(3, 7, dtype=torch.float64)
        fc = 1.5 * torch.randn(16, 3, 7, dtype=torch.float64) + 0.3
        out = _crps_gauss_kernel(obs, fc, torch.ones_like(fc), 1e-6)

        # closed-form CRPS of N(mu, sigma^2): sigma * (z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi))
        mu = fc.mean(dim=0)
        sigma = fc.std(dim=0, unbiased=False)
        normal = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
        z = (obs - mu) / sigma
        expected = sigma * (z * (2 * normal.cdf(z) - 1) + 2 * normal.log_prob(z).exp() - 1 / math.sqrt(math.pi))
        self.assertTrue(compare_tensors("gauss", out, expected, atol=1e-12, rtol=1e-10, verbose=verbose))


# ===========================================================================
class TestTransformedCRPSVariants(unittest.TestCase):
    """The gauss kernel and the spatial weights in the gradient and vorticity/divergence CRPS losses."""

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, cls, crps_type, **kw):
        kwargs = _WIND_GEOM_KWARGS if cls is VortDivCRPSLoss else _GEOM_KWARGS
        return cls(**kwargs, crps_type=crps_type, spatial_distributed=False, ensemble_distributed=False, **kw)

    def _channels(self, cls):
        return _NUM_WIND_CH if cls is VortDivCRPSLoss else _NUM_CH

    @parameterized.expand([(GradientCRPSLoss,), (VortDivCRPSLoss,)])
    def test_gauss_shape_and_nonneg(self, cls):
        fn = self._fn(cls, "gauss")
        nch = self._channels(cls)
        out = fn(_rand_ensemble(self._E, channels=nch), _rand(channels=nch))
        self.assertEqual(tuple(out.shape), (_BATCH, fn.n_channels))
        self.assertTrue((out >= -1e-6).all(), f"negative gauss CRPS, min={out.min().item():.4e}")

    @parameterized.expand(
        [
            (GradientCRPSLoss, "skillspread"),
            (GradientCRPSLoss, "cdf"),
            (GradientCRPSLoss, "gauss"),
            (VortDivCRPSLoss, "skillspread"),
            (VortDivCRPSLoss, "gauss"),
        ]
    )
    def test_unit_spatial_weights_match_none(self, cls, crps_type, verbose=False):
        fn = self._fn(cls, crps_type)
        nch = self._channels(cls)
        fc = _rand_ensemble(self._E, channels=nch)
        obs = _rand(channels=nch)
        out_none = fn(fc, obs)
        out_ones = fn(fc, obs, spatial_weights=torch.ones_like(obs))
        self.assertTrue(compare_tensors(f"{crps_type} weights", out_ones, out_none, atol=1e-6, verbose=verbose))

    @parameterized.expand([(GradientCRPSLoss,), (VortDivCRPSLoss,)])
    def test_skillspread_rejects_ensemble_weights(self, cls):
        fn = self._fn(cls, "skillspread", ensemble_weights=torch.ones(self._E))
        nch = self._channels(cls)
        with self.assertRaises(NotImplementedError):
            fn(_rand_ensemble(self._E, channels=nch), _rand(channels=nch))


if __name__ == "__main__":
    disable_tf32()
    unittest.main()
