# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import os
import sys
import unittest
from parameterized import parameterized

import torch
import torch.distributed as dist

from makani.utils import comm

from makani.utils.grids import GridQuadrature
from makani.utils.losses import (
    CRPSLoss,
    EnsembleNLLLoss,
    SpectralCRPSLoss,
    LpEnergyScoreLoss,
    SpectralL2EnergyScoreLoss,
    SobolevEnergyScoreLoss,
    CoherenceRegularization,
)
from makani.utils.losses.energy_score import SpectralCoherenceLoss
from makani.utils.training.training_helpers import NonFiniteLossCheck

# Add parent directory to path for testutils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from .distributed_helpers import _init_grid, _split_helper, _gather_helper, reduce_success, teardown_comms
from ..testutils import disable_tf32, compare_tensors


class TestDistributedLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)

    @classmethod
    def tearDownClass(cls):
        teardown_comms()

    def setUp(self):
        disable_tf32()

    def _raises_nonfinite(self, check, loss, step):
        """Run the check and report whether it raised, without leaving the test early.

        Leaving early on one rank, as ``assertRaises`` would when that rank does not
        raise, would leave the other ranks waiting in the next collective.
        """
        try:
            check(loss, step=step)
        except FloatingPointError:
            return True
        return False

    @parameterized.expand([["first rank", 0], ["last rank", -1]])
    def test_distributed_nonfinite_loss_check(self, desc, nan_rank, verbose=False):
        """A NaN loss on a single rank stops every rank, and a finite one on all ranks stops none."""
        nan_rank = nan_rank % self.world_size
        finite = torch.tensor(1.0, device=self.device)
        seeded = torch.tensor(float("nan") if self.world_rank == nan_rank else 1.0, device=self.device)

        with self.subTest(desc="finite everywhere"):
            raised = self._raises_nonfinite(NonFiniteLossCheck(), finite, step=1)
            self.assertTrue(reduce_success(not raised, self.device), "a finite loss stopped training")

        with self.subTest(desc=f"NaN on the {desc}"):
            raised = self._raises_nonfinite(NonFiniteLossCheck(), seeded, step=1)
            # every rank has to raise, not only the one that saw the NaN
            self.assertTrue(reduce_success(raised, self.device), f"not every rank stopped on a NaN on the {desc}")

        with self.subTest(desc=f"tolerated NaN on the {desc}"):
            check = NonFiniteLossCheck(max_consecutive=1, grad_scaler_enabled=True)
            raised_first = self._raises_nonfinite(check, seeded, step=1)
            raised_second = self._raises_nonfinite(check, seeded, step=2)
            # the tolerance is counted consistently on every rank
            self.assertTrue(reduce_success(not raised_first, self.device), "a tolerated NaN stopped training")
            self.assertTrue(reduce_success(raised_second, self.device), "not every rank stopped past the tolerance")

    def _split_helper(self, tensor):
        with torch.no_grad():
            # split in W
            tensor_local = _split_helper(tensor, dim=-1, group=self.w_group)

            # split in H
            tensor_local = _split_helper(tensor_local, dim=-2, group=self.h_group)

            # split in E
            if tensor.dim() == 5:
                tensor_local = _split_helper(tensor_local, dim=1, group=self.e_group)

        return tensor_local

    def _gather_helper_fwd(self, tensor):
        # gather in world
        if self.world_size > 1:
            olist = [
                torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device) for _ in range(self.world_size)
            ]
            olist[self.world_rank] = tensor
            dist.all_gather(olist, tensor)
            tensor_gather = torch.stack(olist, dim=-1)
        else:
            tensor_gather = tensor.unsqueeze(-1)

        return tensor_gather

    def _gather_helper_bwd(self, tensor, ensemble=False):
        tensor_gather = _gather_helper(tensor, dim=-1, group=self.w_group)
        tensor_gather = _gather_helper(tensor_gather, dim=-2, group=self.h_group)
        if ensemble:
            tensor_gather = _gather_helper(tensor_gather, dim=1, group=self.e_group)

        return tensor_gather

    @parameterized.expand(
        [
            [128, 256, 32, 8, "naive", False, 1e-6],
            [128, 256, 32, 8, "naive", True, 1e-6],
            [129, 256, 32, 8, "naive", True, 1e-6],
            [129, 256, 32, 8, "clenshaw-curtiss", False, 1e-6],
            [129, 256, 32, 8, "clenshaw-curtiss", True, 1e-6],
            [129, 256, 32, 8, "legendre-gauss", False, 1e-6],
            [129, 256, 32, 8, "legendre-gauss", True, 1e-6],
            [129, 256, 32, 8, "weatherbench2", False, 1e-6],
            [129, 256, 32, 8, "weatherbench2", True, 1e-6],
        ],
        skip_on_empty=True,
    )
    def test_distributed_quadrature(self, nlat, nlon, batch_size, num_chan, quad_rule, normalize, tol, verbose=False):

        # disable tf32 for deterministic comparison# disable tf32# disable tf32 for deterministic comparison# disable tf32
        disable_tf32()

        B, C, H, W = batch_size, num_chan, nlat, nlon

        quad_local = GridQuadrature(
            quadrature_rule=quad_rule, img_shape=(H, W), normalize=normalize, distributed=False
        ).to(self.device)
        quad_dist = GridQuadrature(
            quadrature_rule=quad_rule, img_shape=(H, W), normalize=normalize, distributed=True
        ).to(self.device)

        # input
        inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)
        inp_local = self._split_helper(inp_full)
        inp_full.requires_grad = True
        inp_local.requires_grad = True

        # local
        out_full = quad_local(inp_full)
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # distributed
        out_local = quad_dist(inp_local)
        out_local.backward(ograd_full)
        igrad_local = inp_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="outputs"):
            self.assertTrue(
                reduce_success(compare_tensors("outputs", out_local, out_full, tol, tol, verbose=verbose), self.device)
            )

        #############################################################
        # evaluate BWD pass
        #############################################################
        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper_bwd(igrad_local, False)
            self.assertTrue(
                reduce_success(
                    compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

    @parameterized.expand(
        [
            [128, 256, 32, 8, 4, "cdf", 1e-5],
            [129, 256, 1, 10, 4, "cdf", 1e-5],
            [128, 256, 32, 8, 4, "cdf", 1e-5],
            [129, 256, 1, 10, 4, "cdf", 1e-5],
            [128, 256, 32, 8, 4, "skillspread", 1e-5],
            [129, 256, 1, 10, 4, "skillspread", 1e-5],
            [128, 256, 32, 8, 4, "gauss", 1e-5],
            [129, 256, 1, 10, 4, "gauss", 1e-5],
            [128, 256, 32, 8, 4, "nll", 1e-5],
            [129, 256, 1, 10, 4, "nll", 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_distributed_crps(self, nlat, nlon, batch_size, num_chan, ens_size, loss_type, tol, verbose=False):

        # disable tf32 for deterministic comparison# disable tf32
        disable_tf32()

        B, E, C, H, W = batch_size, ens_size, num_chan, nlat, nlon

        # generate gauss random distributed around 1, with sigma=2
        mean, sigma = (1.0, 2.0)
        inp_full = torch.randn((B, E, C, H, W), dtype=torch.float32, device=self.device) * sigma + mean
        obs_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device) * sigma * 0.001 + mean

        if loss_type != "nll":
            # local loss
            loss_fn_local = CRPSLoss(
                img_shape=(H, W),
                crop_shape=None,
                crop_offset=(0, 0),
                channel_names=(),
                grid_type="equiangular",
                crps_type=loss_type,
                eps=1.0e-5,
                spatial_distributed=False,
                ensemble_distributed=False,
                ensemble_weights=None,
            ).to(self.device)

            # distributed loss
            loss_fn_dist = CRPSLoss(
                img_shape=(H, W),
                crop_shape=None,
                crop_offset=(0, 0),
                channel_names=(),
                grid_type="equiangular",
                crps_type=loss_type,
                eps=1.0e-5,
                spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
                ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
                ensemble_weights=None,
            ).to(self.device)
        else:
            # local loss
            loss_fn_local = EnsembleNLLLoss(
                img_shape=(H, W),
                crop_shape=None,
                crop_offset=(0, 0),
                channel_names=(),
                grid_type="equiangular",
                spatial_distributed=False,
                ensemble_distributed=False,
                eps=1.0e-5,
            ).to(self.device)

            # distributed loss
            loss_fn_dist = EnsembleNLLLoss(
                img_shape=(H, W),
                crop_shape=None,
                crop_offset=(0, 0),
                channel_names=(),
                grid_type="equiangular",
                spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
                ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
                eps=1.0e-5,
            ).to(self.device)

        #############################################################
        # local loss
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        obs_full.requires_grad = True
        loss_full = loss_fn_local(inp_full, obs_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(loss_full)
            ograd_local = ograd_full.clone()

        # BWD pass
        loss_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()
        obsgrad_full = obs_full.grad.clone()

        #############################################################
        # distributed loss
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full)
        obs_local = self._split_helper(obs_full)
        inp_local.requires_grad = True
        obs_local.requires_grad = True

        # BWD pass
        loss_local = loss_fn_dist(inp_local, obs_local)
        loss_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()
        obsgrad_local = obs_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="outputs"):
            self.assertTrue(
                reduce_success(
                    compare_tensors("outputs", loss_local, loss_full, tol, tol, verbose=verbose), self.device
                )
            )

        #############################################################
        # evaluate BWD pass
        #############################################################
        # foreacst grads
        with self.subTest(desc="forecast gradients"):
            igrad_gather_full = self._gather_helper_bwd(igrad_local, True)
            self.assertTrue(
                reduce_success(
                    compare_tensors("forecast gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        # observation grads
        with self.subTest(desc="observation gradients"):
            obsgrad_gather_full = self._gather_helper_bwd(obsgrad_local, False)
            self.assertTrue(
                reduce_success(
                    compare_tensors(
                        "observation gradients", obsgrad_gather_full, obsgrad_full, tol, tol, verbose=verbose
                    ),
                    self.device,
                )
            )

    @parameterized.expand(
        [
            # crps_type was renamed: "ensemble_crps" → "cdf", "skillspread_crps" → "skillspread".
            # SpectralCRPSLoss enforces (absolute=False) ⇒ (crps_type == "skillspread")
            # because cdf requires sorting and sorting complex spectral coefficients
            # only makes sense after abs(). So "cdf" rows must have absolute=True.
            [128, 256, 32, 8, 4, "cdf", True, 1e-4],
            [129, 256, 1, 10, 4, "cdf", True, 1e-4],
            [128, 256, 32, 8, 4, "skillspread", False, 1e-4],
            [129, 256, 1, 10, 4, "skillspread", False, 1e-4],
            [128, 256, 32, 8, 4, "skillspread", True, 1e-4],
            [129, 256, 1, 10, 4, "skillspread", True, 5e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_spectral_crps(
        self, nlat, nlon, batch_size, num_chan, ens_size, loss_type, absolute, tol, verbose=False
    ):
        B, E, C, H, W = batch_size, ens_size, num_chan, nlat, nlon

        # generate gauss random distributed around 1, with sigma=2
        mean, sigma = (1.0, 2.0)
        inp_full = torch.randn((B, E, C, H, W), dtype=torch.float32, device=self.device) * sigma + mean
        obs_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device) * sigma * 0.001 + mean

        # local loss
        loss_fn_local = SpectralCRPSLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            crps_type=loss_type,
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=None,
            eps=1.0e-5,
            absolute=absolute,
        ).to(self.device)

        # distributed loss
        loss_fn_dist = SpectralCRPSLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            crps_type=loss_type,
            spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
            ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
            ensemble_weights=None,
            eps=1.0e-5,
            absolute=absolute,
        ).to(self.device)

        #############################################################
        # local loss
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        obs_full.requires_grad = True
        loss_full = loss_fn_local(inp_full, obs_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(loss_full)
            ograd_local = ograd_full.clone()

        # BWD pass
        loss_full = loss_fn_local(inp_full, obs_full)
        loss_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()
        obsgrad_full = obs_full.grad.clone()

        #############################################################
        # distributed loss
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full.clone())
        obs_local = self._split_helper(obs_full.clone())
        inp_local.requires_grad = True
        obs_local.requires_grad = True

        # BWD pass
        loss_local = loss_fn_dist(inp_local, obs_local)
        loss_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()
        obsgrad_local = obs_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="outputs"):
            self.assertTrue(
                reduce_success(
                    compare_tensors("outputs", loss_local, loss_full, tol, tol, verbose=verbose), self.device
                )
            )

        #############################################################
        # evaluate BWD pass
        #############################################################
        # foreacst grads
        with self.subTest(desc="forecast gradients"):
            igrad_gather_full = self._gather_helper_bwd(igrad_local, True)
            self.assertTrue(
                reduce_success(
                    compare_tensors("forecast gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        # observation grads
        with self.subTest(desc="observation gradients"):
            obsgrad_gather_full = self._gather_helper_bwd(obsgrad_local, False)
            self.assertTrue(
                reduce_success(
                    compare_tensors(
                        "observation gradients", obsgrad_gather_full, obsgrad_full, tol, tol, verbose=verbose
                    ),
                    self.device,
                )
            )

    @parameterized.expand(
        [
            [128, 256, 8, 3, 4, 1e-4],
            [129, 256, 2, 5, 4, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_lp_energy_score(self, nlat, nlon, batch_size, num_chan, ens_size, tol, verbose=False):

        # disable tf32 for deterministic comparison# disable tf32
        disable_tf32()

        B, E, C, H, W = batch_size, ens_size, num_chan, nlat, nlon

        # inputs
        mean, sigma = (1.0, 2.0)
        forecasts_full = torch.randn((B, E, C, H, W), dtype=torch.float32, device=self.device) * sigma + mean
        obs_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device) * sigma * 0.01 + mean

        # local loss
        loss_fn_local = LpEnergyScoreLoss(
            img_shape=(H, W),
            crop_shape=(H, W),
            crop_offset=(0, 0),
            channel_names=tuple(),
            grid_type="equiangular",
            alpha=1.0,
            beta=1.0,
            p=2.0,
            eps=1.0e-5,
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=None,
        ).to(self.device)

        # distributed loss
        loss_fn_dist = LpEnergyScoreLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            alpha=1.0,
            beta=1.0,
            p=2.0,
            eps=1.0e-5,
            spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
            ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
            ensemble_weights=None,
        ).to(self.device)

        #############################################################
        # local loss
        #############################################################
        forecasts_full.requires_grad = True
        obs_full.requires_grad = True
        loss_full = loss_fn_local(forecasts_full, obs_full)

        with torch.no_grad():
            ograd_full = torch.randn_like(loss_full)
            ograd_local = ograd_full.clone()

        loss_full.backward(ograd_full)
        fgrad_full = forecasts_full.grad.clone()
        obsgrad_full = obs_full.grad.clone()

        #############################################################
        # distributed loss
        #############################################################
        forecasts_local = self._split_helper(forecasts_full.clone())
        obs_local = self._split_helper(obs_full.clone())
        forecasts_local.requires_grad = True
        obs_local.requires_grad = True

        loss_local = loss_fn_dist(forecasts_local, obs_local)
        loss_local.backward(ograd_local)
        fgrad_local = forecasts_local.grad.clone()
        obsgrad_local = obs_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="outputs"):
            self.assertTrue(
                reduce_success(
                    compare_tensors("outputs", loss_local, loss_full, tol, tol, verbose=verbose), self.device
                )
            )

        #############################################################
        # evaluate BWD pass
        #############################################################
        with self.subTest(desc="forecast gradients"):
            fgrad_gather_full = self._gather_helper_bwd(fgrad_local, True)
            self.assertTrue(
                reduce_success(
                    compare_tensors("forecast gradients", fgrad_gather_full, fgrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        with self.subTest(desc="observation gradients"):
            obsgrad_gather_full = self._gather_helper_bwd(obsgrad_local, False)
            self.assertTrue(
                reduce_success(
                    compare_tensors(
                        "observation gradients", obsgrad_gather_full, obsgrad_full, tol, tol, verbose=verbose
                    ),
                    self.device,
                )
            )

    @parameterized.expand(
        [
            [128, 256, 8, 13, 4, 1e-4],
            [129, 256, 2, 12, 4, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_spectral_l2_energy_score(self, nlat, nlon, batch_size, num_chan, ens_size, tol, verbose=False):

        # disable tf32 for deterministic comparison
        disable_tf32()

        # shapes
        B, E, C, H, W = batch_size, ens_size, num_chan, nlat, nlon

        mean, sigma = (1.0, 2.0)
        forecasts_full = torch.randn((B, E, C, H, W), dtype=torch.float32, device=self.device) * sigma + mean
        obs_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device) * sigma * 0.01 + mean

        # local loss
        loss_fn_local = SpectralL2EnergyScoreLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            alpha=1.0,
            eps=1.0e-3,
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=None,
        ).to(self.device)

        # distributed loss
        loss_fn_dist = SpectralL2EnergyScoreLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            alpha=1.0,
            eps=1.0e-3,
            spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
            ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
            ensemble_weights=None,
        ).to(self.device)

        #############################################################
        # local loss
        #############################################################
        forecasts_full.requires_grad = True
        obs_full.requires_grad = True
        loss_full = loss_fn_local(forecasts_full, obs_full)

        with torch.no_grad():
            ograd_full = torch.randn_like(loss_full)
            ograd_local = ograd_full.clone()

        loss_full.backward(ograd_full)
        fgrad_full = forecasts_full.grad.clone()
        obsgrad_full = obs_full.grad.clone()

        #############################################################
        # distributed loss
        #############################################################
        forecasts_local = self._split_helper(forecasts_full.clone())
        obs_local = self._split_helper(obs_full.clone())
        forecasts_local.requires_grad = True
        obs_local.requires_grad = True

        loss_local = loss_fn_dist(forecasts_local, obs_local)
        loss_local.backward(ograd_local)
        fgrad_local = forecasts_local.grad.clone()
        obsgrad_local = obs_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="outputs"):
            self.assertTrue(
                reduce_success(
                    compare_tensors("outputs", loss_local, loss_full, tol, tol, verbose=verbose), self.device
                )
            )

        #############################################################
        # evaluate BWD pass
        #############################################################
        with self.subTest(desc="forecast gradients"):
            fgrad_gather_full = self._gather_helper_bwd(fgrad_local, True)
            self.assertTrue(
                reduce_success(
                    compare_tensors("forecast gradients", fgrad_gather_full, fgrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        with self.subTest(desc="observation gradients"):
            obsgrad_gather_full = self._gather_helper_bwd(obsgrad_local, False)
            self.assertTrue(
                reduce_success(
                    compare_tensors(
                        "observation gradients", obsgrad_gather_full, obsgrad_full, tol, tol, verbose=verbose
                    ),
                    self.device,
                )
            )

    @parameterized.expand(
        [
            [128, 256, 8, 13, 4, 1.0, 1.0, 1.0, 1.0, 1e-5],
            [129, 256, 2, 12, 4, 0.8, 1.2, 0.5, 0.7, 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_distributed_sobolev_energy_score(
        self, nlat, nlon, batch_size, num_chan, ens_size, alpha, beta, offset, fraction, tol, verbose=False
    ):

        disable_tf32()

        B, E, C, H, W = batch_size, ens_size, num_chan, nlat, nlon

        mean, sigma = (0.3, 1.1)
        forecasts_full = torch.randn((B, E, C, H, W), dtype=torch.float32, device=self.device) * sigma + mean
        obs_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device) * sigma * 0.05 + mean

        # local loss
        loss_fn_local = SobolevEnergyScoreLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=None,
            alpha=alpha,
            beta=beta,
            offset=offset,
            fraction=fraction,
            eps=1.0e-6,
        ).to(self.device)

        # distributed loss
        loss_fn_dist = SobolevEnergyScoreLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
            ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
            ensemble_weights=None,
            alpha=alpha,
            beta=beta,
            offset=offset,
            fraction=fraction,
            eps=1.0e-6,
        ).to(self.device)

        #############################################################
        # local loss
        #############################################################
        forecasts_full.requires_grad = True
        obs_full.requires_grad = True
        loss_full = loss_fn_local(forecasts_full, obs_full)

        with torch.no_grad():
            ograd_full = torch.randn_like(loss_full)
            ograd_local = ograd_full.clone()

        loss_full.backward(ograd_full)
        fgrad_full = forecasts_full.grad.clone()
        obsgrad_full = obs_full.grad.clone()

        #############################################################
        # distributed loss
        #############################################################
        forecasts_local = self._split_helper(forecasts_full.clone())
        obs_local = self._split_helper(obs_full.clone())
        forecasts_local.requires_grad = True
        obs_local.requires_grad = True

        loss_local = loss_fn_dist(forecasts_local, obs_local)
        loss_local.backward(ograd_local)
        fgrad_local = forecasts_local.grad.clone()
        obsgrad_local = obs_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="outputs"):
            self.assertTrue(
                reduce_success(
                    compare_tensors("outputs", loss_local, loss_full, tol, tol, verbose=verbose), self.device
                )
            )

        #############################################################
        # evaluate BWD pass
        #############################################################
        with self.subTest(desc="forecast gradients"):
            fgrad_gather_full = self._gather_helper_bwd(fgrad_local, True)
            self.assertTrue(
                reduce_success(
                    compare_tensors("forecast gradients", fgrad_gather_full, fgrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        with self.subTest(desc="observation gradients"):
            obsgrad_gather_full = self._gather_helper_bwd(obsgrad_local, False)
            self.assertTrue(
                reduce_success(
                    compare_tensors(
                        "observation gradients", obsgrad_gather_full, obsgrad_full, tol, tol, verbose=verbose
                    ),
                    self.device,
                )
            )

    @parameterized.expand(
        [
            # nlat, nlon, batch, chans, ens, lmin, ens_coh_weight, tol
            [128, 256, 8, 13, 4, 0, 0.0, 1e-4],
            [129, 256, 2, 12, 4, 16, 0.0, 1e-4],
            [129, 256, 2, 12, 4, 16, 0.5, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_coherence_regularization(
        self, nlat, nlon, batch_size, num_chan, ens_size, lmin, ens_coh_weight, tol, verbose=False
    ):
        """Local vs distributed equivalence for CoherenceRegularization.

        The lmin > 0 cases matter most: the wavenumber-band mask is split along h
        exactly like the SHT's l dimension, and the band width used to normalize
        is the global one, so a mismatch between the two would show up here as a
        forward discrepancy. The ens_coh_weight case additionally exercises the
        inter-member term, which is formed after the ensemble transpose.
        """

        # disable tf32 for deterministic comparison
        disable_tf32()

        # shapes
        B, E, C, H, W = batch_size, ens_size, num_chan, nlat, nlon

        mean, sigma = (1.0, 2.0)
        forecasts_full = torch.randn((B, E, C, H, W), dtype=torch.float32, device=self.device) * sigma + mean
        obs_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device) * sigma * 0.01 + mean

        # local loss
        loss_fn_local = CoherenceRegularization(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            lmin=lmin,
            ensemble_coherence_weight=ens_coh_weight,
            eps=1.0e-3,
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=None,
        ).to(self.device)

        # distributed loss
        loss_fn_dist = CoherenceRegularization(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            lmin=lmin,
            ensemble_coherence_weight=ens_coh_weight,
            eps=1.0e-3,
            spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
            ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
            ensemble_weights=None,
        ).to(self.device)

        # the band width is a global quantity and must agree across the decomposition
        with self.subTest(desc="band size"):
            self.assertEqual(loss_fn_local.band_size.item(), loss_fn_dist.band_size.item())

        #############################################################
        # local loss
        #############################################################
        forecasts_full.requires_grad = True
        obs_full.requires_grad = True
        loss_full = loss_fn_local(forecasts_full, obs_full)

        with torch.no_grad():
            ograd_full = torch.randn_like(loss_full)
            ograd_local = ograd_full.clone()

        loss_full.backward(ograd_full)
        fgrad_full = forecasts_full.grad.clone()
        obsgrad_full = obs_full.grad.clone()

        #############################################################
        # distributed loss
        #############################################################
        forecasts_local = self._split_helper(forecasts_full.clone())
        obs_local = self._split_helper(obs_full.clone())
        forecasts_local.requires_grad = True
        obs_local.requires_grad = True

        loss_local = loss_fn_dist(forecasts_local, obs_local)
        loss_local.backward(ograd_local)
        fgrad_local = forecasts_local.grad.clone()
        obsgrad_local = obs_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="outputs"):
            self.assertTrue(
                reduce_success(
                    compare_tensors("outputs", loss_local, loss_full, tol, tol, verbose=verbose), self.device
                )
            )

        #############################################################
        # evaluate BWD pass
        #############################################################
        with self.subTest(desc="forecast gradients"):
            fgrad_gather_full = self._gather_helper_bwd(fgrad_local, True)
            self.assertTrue(
                reduce_success(
                    compare_tensors("forecast gradients", fgrad_gather_full, fgrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        with self.subTest(desc="observation gradients"):
            obsgrad_gather_full = self._gather_helper_bwd(obsgrad_local, False)
            self.assertTrue(
                reduce_success(
                    compare_tensors(
                        "observation gradients", obsgrad_gather_full, obsgrad_full, tol, tol, verbose=verbose
                    ),
                    self.device,
                )
            )

    @parameterized.expand(
        [
            [128, 256, 8, 13, 4, False, 1e-4],
            [129, 256, 2, 12, 4, True, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_spectral_coherence_loss(
        self, nlat, nlon, batch_size, num_chan, ens_size, relative, tol, verbose=False
    ):
        """Local vs distributed equivalence for SpectralCoherenceLoss.

        This could not be written before: the reference instance is built with
        spatial_distributed=False while h/w groups are >1, and __init__ used to
        slice lm_weights on group size alone, pairing local weights with the
        full-size coefficients of the non-distributed SHT.
        """

        # disable tf32 for deterministic comparison
        disable_tf32()

        # shapes
        B, E, C, H, W = batch_size, ens_size, num_chan, nlat, nlon

        mean, sigma = (1.0, 2.0)
        forecasts_full = torch.randn((B, E, C, H, W), dtype=torch.float32, device=self.device) * sigma + mean
        obs_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device) * sigma * 0.01 + mean

        # local loss
        loss_fn_local = SpectralCoherenceLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            relative=relative,
            channel_reduction=False,
            eps=1.0e-3,
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=None,
        ).to(self.device)

        # distributed loss
        loss_fn_dist = SpectralCoherenceLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            relative=relative,
            channel_reduction=False,
            eps=1.0e-3,
            spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
            ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
            ensemble_weights=None,
        ).to(self.device)

        # the non-distributed instance must keep full-size weights regardless of
        # whether h/w groups exist; this is the regression guard for the split
        with self.subTest(desc="local weights are not split"):
            self.assertEqual(tuple(loss_fn_local.lm_weights.shape), (loss_fn_local.sht.lmax, loss_fn_local.sht.mmax))

        #############################################################
        # local loss
        #############################################################
        forecasts_full.requires_grad = True
        obs_full.requires_grad = True
        loss_full = loss_fn_local(forecasts_full, obs_full)

        with torch.no_grad():
            ograd_full = torch.randn_like(loss_full)
            ograd_local = ograd_full.clone()

        loss_full.backward(ograd_full)
        fgrad_full = forecasts_full.grad.clone()
        obsgrad_full = obs_full.grad.clone()

        #############################################################
        # distributed loss
        #############################################################
        forecasts_local = self._split_helper(forecasts_full.clone())
        obs_local = self._split_helper(obs_full.clone())
        forecasts_local.requires_grad = True
        obs_local.requires_grad = True

        loss_local = loss_fn_dist(forecasts_local, obs_local)
        loss_local.backward(ograd_local)
        fgrad_local = forecasts_local.grad.clone()
        obsgrad_local = obs_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="outputs"):
            self.assertTrue(
                reduce_success(
                    compare_tensors("outputs", loss_local, loss_full, tol, tol, verbose=verbose), self.device
                )
            )

        #############################################################
        # evaluate BWD pass
        #############################################################
        with self.subTest(desc="forecast gradients"):
            fgrad_gather_full = self._gather_helper_bwd(fgrad_local, True)
            self.assertTrue(
                reduce_success(
                    compare_tensors("forecast gradients", fgrad_gather_full, fgrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        with self.subTest(desc="observation gradients"):
            obsgrad_gather_full = self._gather_helper_bwd(obsgrad_local, False)
            self.assertTrue(
                reduce_success(
                    compare_tensors(
                        "observation gradients", obsgrad_gather_full, obsgrad_full, tol, tol, verbose=verbose
                    ),
                    self.device,
                )
            )


if __name__ == "__main__":
    disable_tf32()
    unittest.main()
