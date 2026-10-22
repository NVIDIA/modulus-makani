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

import os
import sys
import unittest
from parameterized import parameterized_class

import numpy as np

import torch

from parameterized import parameterized

import makani.utils.constants as const
from makani.utils.losses.hydrostatic_loss import HydrostaticBalanceLoss
from makani.models.parametrizations import ConstraintsWrapper
from makani.utils.constraints import NonNegativeConstraint, HydrostaticBalanceProjection, get_matching_channels_pl

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, compare_tensors

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


@parameterized_class(("device",), _devices)
class TestConstraints(unittest.TestCase):

    def setUp(self):

        disable_tf32()

        set_seed(333)

        # load the data:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        data = np.load(os.path.join(data_dir, "sample_30km_equator.npz"))

        # fields
        self.data = torch.from_numpy(data["data"].astype(np.float32))
        self.bias = torch.from_numpy(data["bias"].astype(np.float32))
        self.scale = torch.from_numpy(data["scale"].astype(np.float32))
        self.data = ((self.data - self.bias) / self.scale).to(self.device)
        # metadata
        self.channel_names = data["channel_names"].tolist()
        self.img_shape = data["img_shape"]
        self.crop_shape = data["crop_shape"]
        self.crop_offset = data["crop_offset"]

    @parameterized.expand([("dry", False), ("moist", True)])
    def test_hydrostatic_balance_loss(self, _name, use_moist_air_formula):
        # loss
        hbloss = HydrostaticBalanceLoss(
            img_shape=self.img_shape,
            crop_shape=self.crop_shape,
            crop_offset=self.crop_offset,
            channel_names=self.channel_names,
            grid_type="equiangular",
            bias=self.bias,
            scale=self.scale,
            p_min=50,
            p_max=900,
            use_moist_air_formula=use_moist_air_formula,
        ).to(self.device)
        loss_tens = hbloss(self.data, None)

        # average over batch and sum over channels
        loss_val = torch.mean(torch.sum(loss_tens, dim=1)).item()

        self.assertTrue(loss_val <= 1e-4)

    @parameterized.expand([("dry", False), ("moist", True)])
    def test_hydrostatic_balance_constraint_wrapper_era5(self, _name, use_moist_air_formula):
        # loss
        hbloss = HydrostaticBalanceLoss(
            img_shape=self.img_shape,
            crop_shape=self.crop_shape,
            crop_offset=self.crop_offset,
            channel_names=self.channel_names,
            grid_type="equiangular",
            bias=self.bias,
            scale=self.scale,
            p_min=50,
            p_max=900,
            use_moist_air_formula=use_moist_air_formula,
        ).to(self.device)

        # constraints wrapper
        constraint_dict = {
            "type": "hydrostatic_balance",
            "options": dict(p_min=50, p_max=900, use_moist_air_formula=use_moist_air_formula),
        }
        cwrap = ConstraintsWrapper(
            constraints=[constraint_dict],
            channel_names=self.channel_names,
            bias=self.bias,
            scale=self.scale,
            model_handle=None,
        ).to(self.device)

        # create a short vector:
        B, C, H, W = self.data.shape
        data_short = torch.empty((B, cwrap.N_in_channels, H, W), dtype=torch.float32, device=self.device)
        # t_idx
        data_short[:, 0, ...] = self.data[:, cwrap.constraint_list[0].t_idx[0], ...]
        # z_idx
        data_short[:, 1 : len(cwrap.constraint_list[0].z_idx) + 1, ...] = self.data[
            :, cwrap.constraint_list[0].z_idx, ...
        ]
        # q_idx
        off_idx = len(cwrap.constraint_list[0].z_idx) + 1
        if use_moist_air_formula:
            data_short[:, off_idx : off_idx + len(cwrap.constraint_list[0].q_idx), ...] = self.data[
                :, cwrap.constraint_list[0].q_idx, ...
            ]
            off_idx += len(cwrap.constraint_list[0].q_idx)
        # remaining channels
        data_short[:, off_idx:, ...] = self.data[:, cwrap.constraint_list[0].aux_idx, ...]
        data_map = cwrap(data_short)

        # check the hb loss
        hb_loss_tens = hbloss(data_map, None)

        # average over batch and sum over channels
        hb_loss_val = torch.mean(torch.sum(hb_loss_tens, dim=1)).item()

        with self.subTest("hydrostatic balance loss"):
            self.assertTrue(hb_loss_val <= 1e-6)

        # now check that the loss on the non-hb components is zero too
        aux_loss_val = torch.nn.functional.mse_loss(
            data_map[:, cwrap.constraint_list[0].aux_idx, ...], self.data[:, cwrap.constraint_list[0].aux_idx, ...]
        ).item()
        with self.subTest("auxiliary channels unchanged"):
            self.assertTrue(aux_loss_val <= 1e-6)

    @parameterized.expand([("dry", False), ("moist", True)])
    def test_hydrostatic_balance_constraint_wrapper_random(self, _name, use_moist_air_formula):
        # loss
        hbloss = HydrostaticBalanceLoss(
            img_shape=self.img_shape,
            crop_shape=self.crop_shape,
            crop_offset=self.crop_offset,
            channel_names=self.channel_names,
            grid_type="equiangular",
            bias=self.bias,
            scale=self.scale,
            p_min=50,
            p_max=900,
            use_moist_air_formula=use_moist_air_formula,
        ).to(self.device)

        # constraints wrapper
        constraint_dict = {
            "type": "hydrostatic_balance",
            "options": dict(p_min=50, p_max=900, use_moist_air_formula=use_moist_air_formula),
        }
        cwrap = ConstraintsWrapper(
            constraints=[constraint_dict],
            channel_names=self.channel_names,
            bias=self.bias,
            scale=self.scale,
            model_handle=None,
        ).to(self.device)

        # create a short vector:
        B, C, H, W = self.data.shape
        data_short = torch.empty((B, cwrap.N_in_channels, H, W), dtype=torch.float32, device=self.device)
        data_short.normal_(0.0, 1.0)
        data_map = cwrap(data_short)

        # check the hb loss
        hb_loss_tens = hbloss(data_map, None)

        # average over batch and sum over channels
        hb_loss_val = torch.mean(torch.sum(hb_loss_tens, dim=1)).item()

        with self.subTest("hydrostatic balance loss"):
            self.assertTrue(hb_loss_val <= 1e-6)

        # now check that the loss on the non-hb components is zero too
        off_idx = len(cwrap.constraint_list[0].z_idx) + 1
        if use_moist_air_formula:
            off_idx += len(cwrap.constraint_list[0].q_idx)
        aux_loss_val = torch.nn.functional.mse_loss(
            data_map[:, cwrap.constraint_list[0].aux_idx, ...], data_short[:, off_idx:, ...]
        ).item()
        with self.subTest("auxiliary channels unchanged"):
            self.assertTrue(aux_loss_val <= 1e-6)

    @parameterized.expand([("dry", False), ("moist", True)])
    def test_hydrostatic_balance_matches_independent_integration(self, _name, use_moist_air_formula):
        """Independent validation of the hydrostatic-balance formula.

        Instead of checking the wrapper output against HydrostaticBalanceLoss (which
        derives from the same formula -> circular), we build a *known* temperature
        profile, integrate the hypsometric equation forward to obtain geopotentials,
        feed [T0, Z...] through the wrapper, and confirm it reconstructs the original
        temperatures (and passes geopotential / humidity / aux channels through).

            Phi_i - Phi_{i-1} = R_d * 0.5 * (Tv_i + Tv_{i-1}) * ln(p_{i-1}/p_i)
        """
        R = const.R_DRY_AIR
        qc = const.Q_CORRECTION_MOIST_AIR

        # synthetic channel set: matching t/z (and q) pressure levels + aux channels
        levels = [925, 850, 700, 500, 300, 100, 50]
        channel_names = [f"t{p}" for p in levels] + [f"z{p}" for p in levels]
        if use_moist_air_formula:
            channel_names += [f"q{p}" for p in levels]
        channel_names += ["u10m", "v10m", "t2m"]  # aux passthrough channels

        constraint_dict = {
            "type": "hydrostatic_balance",
            "options": dict(p_min=0, p_max=2000, use_moist_air_formula=use_moist_air_formula),
        }
        cwrap = ConstraintsWrapper(
            constraints=[constraint_dict], channel_names=channel_names, bias=None, scale=None, model_handle=None
        ).to(self.device)
        con = cwrap.constraint_list[0]

        # the wrapper orders pressures descending (bottom -> top)
        pressures = con.pressures
        n = len(pressures)
        self.assertEqual(n, len(levels))
        self.assertEqual(len(con.t_idx), n)

        B, H, W = 2, 3, 4
        # known temperature profile (physical units, varying over space)
        T = 200.0 + 60.0 * torch.rand(B, n, H, W, device=self.device, dtype=torch.float32)
        if use_moist_air_formula:
            q = 0.02 * torch.rand(B, n, H, W, device=self.device, dtype=torch.float32)
            Tv = T * (1.0 + qc * q)
        else:
            Tv = T

        # integrate the hypsometric equation forward to get geopotential per level
        Z = torch.zeros(B, n, H, W, device=self.device, dtype=torch.float32)
        Z[:, 0, ...] = 1000.0  # arbitrary reference geopotential at the bottom level
        for i in range(1, n):
            plog = float(np.log(pressures[i - 1] / pressures[i]))
            Z[:, i, ...] = Z[:, i - 1, ...] + R * 0.5 * (Tv[:, i, ...] + Tv[:, i - 1, ...]) * plog

        # assemble the reduced input in the wrapper's layout: [T0, Z0..Z_{n-1}, (q...), aux...]
        inp = torch.zeros(B, cwrap.N_in_channels, H, W, device=self.device, dtype=torch.float32)
        inp[:, 0, ...] = T[:, 0, ...]
        inp[:, 1 : n + 1, ...] = Z
        off = n + 1
        if use_moist_air_formula:
            inp[:, off : off + n, ...] = q
            off += n
        n_aux = len(con.aux_idx)
        aux_vals = torch.randn(B, n_aux, H, W, device=self.device, dtype=torch.float32)
        inp[:, off : off + n_aux, ...] = aux_vals

        out = cwrap(inp)

        # the reconstructed temperatures must match the known profile
        with self.subTest("reconstructed temperature"):
            self.assertTrue(
                compare_tensors(
                    "reconstructed temperature", out[:, con.t_idx, ...], T, atol=1e-1, rtol=1e-3, verbose=False
                )
            )
        # geopotentials, aux (and humidity) pass through unchanged
        with self.subTest("geopotential passthrough"):
            self.assertTrue(
                compare_tensors(
                    "geopotential passthrough", out[:, con.z_idx, ...], Z, atol=1e-2, rtol=1e-5, verbose=False
                )
            )
        with self.subTest("aux passthrough"):
            self.assertTrue(
                compare_tensors(
                    "aux passthrough", out[:, con.aux_idx, ...], aux_vals, atol=1e-4, rtol=1e-5, verbose=False
                )
            )
        if use_moist_air_formula:
            with self.subTest("humidity passthrough"):
                self.assertTrue(
                    compare_tensors(
                        "humidity passthrough", out[:, con.q_idx, ...], q, atol=1e-4, rtol=1e-5, verbose=False
                    )
                )

    @parameterized.expand([("dry", False), ("moist", True)])
    def test_hydrostatic_balance_loss_on_balanced_profile(self, _name, use_moist_air_formula):
        """Independent check of the soft-constraint loss: a profile built to satisfy
        hydrostatic balance exactly yields ~0 loss, and perturbing a single geopotential
        level makes the loss clearly positive. Uses identity normalization so a synthetic
        physical-unit field is fed straight through (no dependence on the data sample)."""
        R = const.R_DRY_AIR
        qc = const.Q_CORRECTION_MOIST_AIR

        C = len(self.channel_names)
        # Use a full (uncropped) equiangular grid. The data sample is a 2-row band at
        # the pole (crop_offset=[719,0], crop_shape=[2,720]); the area-weighted spherical
        # quadrature gives that crop a normalized weight ~1e-5 (cos(lat) -> 0 at the pole),
        # which scales the *integrated* loss down by ~1e-5 and makes absolute thresholds
        # meaningless. On a full grid the quadrature is O(1).
        nlat, nlon = 32, 64
        img_shape = crop_shape = (nlat, nlon)
        crop_offset = (0, 0)
        H, W = nlat, nlon
        ident_bias = torch.zeros(1, C, 1, 1, dtype=torch.float32)
        ident_scale = torch.ones(1, C, 1, 1, dtype=torch.float32)

        hbloss = HydrostaticBalanceLoss(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=self.channel_names,
            grid_type="equiangular",
            bias=ident_bias,
            scale=ident_scale,
            p_min=50,
            p_max=900,
            use_moist_air_formula=use_moist_air_formula,
        ).to(self.device)
        pressures = hbloss.pressures
        n = len(pressures)
        B = 2

        # known temperature profile (physical units); aux channels arbitrary
        field = torch.randn(B, C, H, W, device=self.device)
        T = 200.0 + 60.0 * torch.rand(B, n, H, W, device=self.device)
        if use_moist_air_formula:
            q = 0.02 * torch.rand(B, n, H, W, device=self.device)
            Tv = T * (1.0 + qc * q)
            field[:, hbloss.q_idx, ...] = q
        else:
            Tv = T

        # integrate the hypsometric equation to get balanced geopotentials
        Z = torch.zeros(B, n, H, W, device=self.device)
        Z[:, 0, ...] = 1000.0
        for i in range(1, n):
            plog = float(np.log(pressures[i - 1] / pressures[i]))
            Z[:, i, ...] = Z[:, i - 1, ...] + R * 0.5 * (Tv[:, i, ...] + Tv[:, i - 1, ...]) * plog
        field[:, hbloss.t_idx, ...] = T
        field[:, hbloss.z_idx, ...] = Z

        # a balanced profile must give (numerically) zero loss
        loss_bal = torch.mean(torch.sum(hbloss(field, None), dim=1)).item()
        with self.subTest("balanced profile gives zero loss"):
            self.assertTrue(loss_bal <= 1e-4, f"balanced HB loss too large: {loss_bal}")

        # perturbing one geopotential level breaks balance -> loss clearly positive
        field_pert = field.clone()
        field_pert[:, hbloss.z_idx[1], ...] += 50.0
        loss_pert = torch.mean(torch.sum(hbloss(field_pert, None), dim=1)).item()
        with self.subTest("perturbation breaks balance"):
            self.assertTrue(loss_pert >= 1e-2, f"perturbed HB loss unexpectedly small: {loss_pert}")
            # and it must be vastly larger than the balanced residual
            self.assertGreater(loss_pert, 1e3 * loss_bal)

    def _water_names(self):
        from makani.utils.features import get_water_channels

        return [self.channel_names[c] for c in get_water_channels(self.channel_names)]

    @parameterized.expand([("silu",), ("softplus",), ("quadratic",)])
    def test_nonnegative_constraint_wrapper(self, mode):
        """The "nonnegative" constraint of the wrapper is NonNegativeConstraint on the water channels."""
        cwrap = ConstraintsWrapper(
            constraints=[{"type": "nonnegative", "options": {"mode": mode}}],
            channel_names=self.channel_names,
            bias=self.bias,
            scale=self.scale,
            model_handle=None,
        ).to(self.device)
        cwrap.eval()
        self.assertEqual(cwrap.N_in_channels, len(self.channel_names))

        water = self._water_names()
        self.assertGreater(len(water), 0)
        reference = NonNegativeConstraint(self.channel_names, water, bias=self.bias, scale=self.scale, mode=mode).to(
            self.device
        )
        reference.eval()

        inp = 3.0 * torch.randn_like(self.data)
        out = cwrap(inp)
        self.assertTrue(compare_tensors("wrapper vs constraint", out, reference(inp)))

        widx = [self.channel_names.index(n) for n in water]
        bias = self.bias[:, widx].to(self.device)
        physical = out[:, widx] * self.scale[:, widx].to(self.device) + bias
        # un-normalizing a value at the floor rounds to ~|bias| * float32 eps
        self.assertTrue((physical >= -1e-6 * (1.0 + bias.abs())).all())

    @parameterized.expand([("dry", False), ("moist", True)])
    def test_nonnegative_then_hydrostatic_balance_constraint_wrapper(self, _name, use_moist_air_formula):
        """Clamping before hydrostatic balance: the clamp acts on the reduced channels the model predicts,
        so the virtual temperature of the moist formula sees the clamped humidity. The expanded output
        is then balanced and its water channels are nonnegative."""
        hb_options = dict(p_min=50, p_max=900, use_moist_air_formula=use_moist_air_formula)
        constraints = [
            {"type": "nonnegative", "options": {"mode": "quadratic"}},
            {"type": "hydrostatic_balance", "options": hb_options},
        ]
        cwrap = ConstraintsWrapper(
            constraints=constraints, channel_names=self.channel_names, bias=self.bias, scale=self.scale
        ).to(self.device)
        hb_only = ConstraintsWrapper(
            constraints=constraints[1:], channel_names=self.channel_names, bias=self.bias, scale=self.scale
        ).to(self.device)
        self.assertEqual(cwrap.N_in_channels, hb_only.N_in_channels)

        # the clamp acts on the water channels of the reduced layout, with their normalization statistics
        hb = cwrap.constraint_list[1]
        clamp = cwrap.constraint_list[0]
        reduced_names = [self.channel_names[c] for c in hb.inp_channel_idx]
        expected_idx = [reduced_names.index(n) for n in self._water_names()]
        # the clamp follows the reduced layout, where the humidities are ordered by pressure, so compare as sets
        self.assertEqual(sorted(clamp.channel_indices.tolist()), sorted(expected_idx))

        B, C, H, W = self.data.shape
        data_short = torch.randn(B, cwrap.N_in_channels, H, W, dtype=torch.float32, device=self.device)
        out = cwrap(data_short)

        with self.subTest("composition"):
            expected = hb_only(clamp(data_short))
            self.assertTrue(compare_tensors("composition", out, expected))

        with self.subTest("hydrostatic balance loss"):
            hbloss = HydrostaticBalanceLoss(
                img_shape=self.img_shape,
                crop_shape=self.crop_shape,
                crop_offset=self.crop_offset,
                channel_names=self.channel_names,
                grid_type="equiangular",
                bias=self.bias,
                scale=self.scale,
                **hb_options,
            ).to(self.device)
            hb_loss_val = torch.mean(torch.sum(hbloss(out, None), dim=1)).item()
            self.assertTrue(hb_loss_val <= 1e-6)

        with self.subTest("water channels nonnegative"):
            widx = [self.channel_names.index(n) for n in self._water_names()]
            bias = self.bias[:, widx].to(self.device)
            physical = out[:, widx] * self.scale[:, widx].to(self.device) + bias
            self.assertTrue((physical >= -1e-6 * (1.0 + bias.abs())).all())

    def test_hydrostatic_balance_constraint_only_once(self):
        constraints = [
            {"type": "hydrostatic_balance", "options": dict(p_min=50, p_max=900)},
            {"type": "hydrostatic_balance", "options": dict(p_min=50, p_max=900)},
        ]
        with self.assertRaises(ValueError):
            ConstraintsWrapper(
                constraints=constraints, channel_names=self.channel_names, bias=self.bias, scale=self.scale
            )


@parameterized_class(("device",), _devices)
class TestNonNegativeConstraint(unittest.TestCase):
    """Tests for NonNegativeConstraint in makani/utils/constraints.py.

    Convention: x_norm = (x_raw - bias) / scale, so physical zero sits at
    x_norm = -bias/scale.  In eval mode the constraint hard-clamps to guarantee
    x_raw >= 0; in training mode it uses a smooth multiplicative approximation
    so gradients flow for slightly negative values.
    """

    # synthetic channel set used across all subtests
    ALL_CHANNELS = ["u10m", "q850", "t850", "q500", "t500"]
    CLAMP_NAMES = ["q850", "q500"]
    CLAMP_IDX = [1, 3]  # positions of CLAMP_NAMES in ALL_CHANNELS

    def setUp(self):
        disable_tf32()

        set_seed(333)

    def _make(self, names_to_clamp=None, means=None, stds=None, **kwargs):
        """Build a NonNegativeConstraint using channel names.

        means/stds are full-channel tensors (len(ALL_CHANNELS),); the
        constructor slices out the constrained channels itself, mirroring
        how _HydrostaticBalanceWrapper receives the full bias/scale.
        """
        names = names_to_clamp if names_to_clamp is not None else self.CLAMP_NAMES
        bias = means.view(1, -1, 1, 1) if means is not None else None
        scale = stds.view(1, -1, 1, 1) if stds is not None else None
        c = NonNegativeConstraint(self.ALL_CHANNELS, names, bias=bias, scale=scale, **kwargs)
        return c.to(self.device)

    # --- eval / hard clamp ---
    @parameterized.expand([("silu",), ("softplus",), ("quadratic",)])
    def test_eval_hard_clamp_no_normalization(self, mode):
        """Eval mode: constrained channels are >= 0; unconstrained channels unchanged."""
        B, C, H, W = 2, len(self.ALL_CHANNELS), 8, 8
        c = self._make(mode=mode)
        c.eval()
        x = torch.randn(B, C, H, W, device=self.device)
        y = c(x)
        self.assertTrue((y[:, self.CLAMP_IDX, :, :] >= 0).all().item())
        unconstrained = [i for i in range(C) if i not in self.CLAMP_IDX]
        self.assertTrue(compare_tensors("unconstrained channels", y[:, unconstrained, :, :], x[:, unconstrained, :, :]))

    @parameterized.expand([("silu",), ("softplus",), ("quadratic",)])
    def test_eval_hard_clamp_with_normalization(self, mode):
        """Eval mode: x_raw = y_norm * scale + bias >= 0 after clamping."""
        B, C, H, W = 2, len(self.ALL_CHANNELS), 6, 6
        means = torch.tensor([0.0, 5.0, 270.0, 3.0, 250.0])
        stds = torch.tensor([1.0, 2.0, 10.0, 1.5, 8.0])
        c = self._make(means=means, stds=stds, mode=mode)
        c.eval()
        x = torch.randn(B, C, H, W, device=self.device) * 3.0
        y = c(x)
        for i, ci in enumerate(self.CLAMP_IDX):
            x_raw = y[:, ci, :, :] * stds[ci].item() + means[ci].item()
            self.assertTrue(
                (x_raw >= -1e-6).all().item(), f"channel {self.ALL_CHANNELS[ci]} has negative physical values"
            )

    @parameterized.expand([("silu",), ("softplus",)])
    def test_eval_positive_input_unchanged(self, mode):
        """Eval mode: values already above physical zero are not modified."""
        B, C, H, W = 2, len(self.ALL_CHANNELS), 4, 4
        means = torch.tensor([0.0, 1.0, 270.0, 2.0, 250.0])
        stds = torch.ones(len(self.ALL_CHANNELS))
        c = self._make(means=means, stds=stds, mode=mode)
        c.eval()
        x = torch.ones(B, C, H, W, device=self.device) * 5.0
        y = c(x)
        self.assertTrue(compare_tensors("positive inputs unchanged", y, x))

    # --- training / soft clamp ---
    @parameterized.expand([("silu",), ("softplus",)])
    def test_train_slightly_negative_not_zeroed(self, mode):
        """Training mode: slightly negative values are not exactly zeroed (gradient path open)."""
        B, C, H, W = 1, len(self.ALL_CHANNELS), 4, 4
        c = self._make(names_to_clamp=["q850"], mode=mode)
        c.train()
        x = torch.full((B, C, H, W), -0.05, device=self.device)
        y = c(x)
        self.assertFalse((y[:, [1], :, :] == 0).all().item())

    @parameterized.expand([("silu",), ("softplus",), ("quadratic",)])
    def test_train_large_positive_identity(self, mode):
        """Training mode: on the bulk the clamp has unit slope, so it preserves spatial
        structure (only the l=0 DC mode may shift). silu is exact identity there; softplus
        is identity up to the constant (1-leak)*eps*ln2 introduced to pin the floor to physical
        zero -- a DC offset the decoder bias absorbs, so we check the slope, not the offset."""
        B, C, H, W = 2, len(self.ALL_CHANNELS), 4, 4
        c = self._make(eps=0.1, mode=mode)
        c.train()
        x1 = torch.ones(B, C, H, W, device=self.device) * 5.0
        y1 = c(x1)

        with self.subTest("bulk unit slope"):
            # two bulk inputs one unit apart: unit slope <=> y2 - y1 == x2 - x1 == 1
            y2 = c(x1 + 1.0)
            slope = (y2 - y1)[:, self.CLAMP_IDX, :, :]
            self.assertTrue(compare_tensors("bulk unit slope", slope, torch.ones_like(slope), atol=1e-3))

        if mode == "silu":
            with self.subTest("exact passthrough (silu has zero DC offset)"):
                self.assertTrue(
                    compare_tensors(
                        "large positive passthrough",
                        y1[:, self.CLAMP_IDX, :, :],
                        x1[:, self.CLAMP_IDX, :, :],
                        atol=1e-3,
                    )
                )

    @parameterized.expand([("silu",), ("softplus",), ("quadratic",)])
    def test_train_floor_fixed_point(self, mode):
        """The training soft clamp must map physical zero to physical zero (w(0)=0 in the
        shifted space). Otherwise the training floor sits above the eval hard-clamp floor,
        biasing genuinely-dry channels (e.g. q50) upward; the loss then drives raw predictions
        negative to cancel the bias and the inference hard clamp flattens them to 0."""
        c = self._make(names_to_clamp=["q850"], mode=mode)  # no normalization -> input is the shifted space
        c.train()
        C = len(self.ALL_CHANNELS)
        x = torch.full((1, C, 1, 1), 5.0, device=self.device)
        x[:, 1, 0, 0] = 0.0  # physical zero for the clamped channel
        y = c(x)
        self.assertAlmostEqual(
            y[0, 1, 0, 0].item(), 0.0, places=5, msg=f"{mode} training floor must be a fixed point at physical zero"
        )

    @parameterized.expand([("silu",), ("softplus",)])
    def test_train_gradient_flows(self, mode):
        """Training mode: gradient is nonzero for slightly negative inputs."""
        B, C, H, W = 1, len(self.ALL_CHANNELS), 4, 4
        c = self._make(mode=mode)
        c.train()
        x = torch.full((B, C, H, W), -0.2, device=self.device, requires_grad=True)
        c(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse((x.grad[:, self.CLAMP_IDX, :, :] == 0).all().item())

    @parameterized.expand([("silu",), ("softplus",), ("quadratic",)])
    def test_train_normalization_offset(self, mode):
        """Training mode: with normalization, the clamp boundary is at physical zero."""
        B, C, H, W = 1, len(self.ALL_CHANNELS), 4, 4
        means = torch.tensor([0.0, 4.0, 270.0, 6.0, 250.0])
        stds = torch.tensor([1.0, 2.0, 10.0, 3.0, 8.0])
        c = self._make(means=means, stds=stds, eps=0.01, mode=mode)
        c.train()
        # set constrained channels to physical zero in normalized space
        x = torch.zeros(B, C, H, W, device=self.device)
        for ci, mi, si in zip(self.CLAMP_IDX, means[self.CLAMP_IDX], stds[self.CLAMP_IDX]):
            x[:, ci, :, :] = -mi / si
        y = c(x)
        for ci, mi, si in zip(self.CLAMP_IDX, means[self.CLAMP_IDX], stds[self.CLAMP_IDX]):
            x_raw = y[:, ci, :, :] * si.item() + mi.item()
            self.assertTrue(
                compare_tensors(f"{self.ALL_CHANNELS[ci]} at boundary", x_raw, torch.zeros_like(x_raw), atol=0.1)
            )

    # --- mode switching ---
    @parameterized.expand([("silu",), ("softplus",)])
    def test_train_eval_switch(self, mode):
        """Switching train/eval changes hard vs soft clamping on the same instance."""
        B, C, H, W = 1, len(self.ALL_CHANNELS), 4, 4
        c = self._make(names_to_clamp=["q850"], mode=mode)
        x = torch.full((B, C, H, W), -1.0, device=self.device)
        c.train()
        y_train = c(x)
        c.eval()
        y_eval = c(x)
        self.assertFalse((y_train[:, [1], :, :] == 0).all().item())
        self.assertTrue(
            compare_tensors("hard clamp to zero", y_eval[:, [1], :, :], torch.zeros_like(y_eval[:, [1], :, :]))
        )

    # --- soft-clamp shape: the reason "softplus" mode exists ---
    @parameterized.expand([("silu",), ("softplus",)])
    def test_train_soft_clamp_shape(self, mode):
        """softplus is monotonic (grad > 0 everywhere), so a below-target channel is always
        pushed up. silu is non-monotonic: it has a negative-gradient dip that can drive a
        near-zero channel (e.g. q50) deeper negative until the gradient vanishes."""
        c = self._make(names_to_clamp=["q850"], mode=mode)  # no normalization -> input is the shifted space
        c.train()
        C = len(self.ALL_CHANNELS)
        # sweep the clamped channel (index 1) across negative values; keep the rest positive
        x = torch.full((1, C, 1, 40), 5.0, device=self.device)
        x[:, 1, 0, :] = torch.linspace(-1.0, 0.5, 40, device=self.device)
        x = x.requires_grad_(True)
        c(x).sum().backward()
        # d(sum output)/dx is the elementwise soft-clamp derivative on this channel
        g = x.grad[:, 1, 0, :]
        if mode == "softplus":
            self.assertTrue((g > 0).all().item(), "softplus soft clamp must be monotonic (grad > 0)")
        else:
            self.assertTrue((g < 0).any().item(), "silu soft clamp has a non-monotonic negative-gradient dip")

    # --- the FCN3 soft clamp, see TestClampWaterChannels in test_models.py for the comparison with FCN3 ---
    @parameterized.expand([("silu", 0.1), ("softplus", 0.1), ("quadratic", 0.5)])
    def test_default_eps(self, mode, eps):
        """eps defaults per mode, and an explicit value overrides it."""
        self.assertEqual(self._make(mode=mode).eps, eps)
        self.assertEqual(self._make(mode=mode, eps=0.3).eps, 0.3)

    def test_quadratic_continuous_slope(self):
        """The quadratic mode has continuous value and slope at physical zero and at eps."""
        eps = 0.2
        c = self._make(names_to_clamp=["q850"], eps=eps, mode="quadratic")  # no normalization
        c.train()
        C = len(self.ALL_CHANNELS)
        # steps of 1e-4 to either side change value and slope by O(1e-4 / eps), a jump would be O(1)
        for x0 in [0.0, eps]:
            pts = torch.tensor([x0 - 1e-4, x0 + 1e-4], device=self.device, dtype=torch.float64)
            x = torch.full((1, C, 1, 2), 5.0, device=self.device, dtype=torch.float64)
            x[:, 1, 0, :] = pts
            x.requires_grad_(True)
            y = c(x)
            y.sum().backward()
            with self.subTest(f"value at {x0}"):
                self.assertAlmostEqual(y[0, 1, 0, 0].item(), y[0, 1, 0, 1].item(), delta=1e-2)
            with self.subTest(f"slope at {x0}"):
                self.assertAlmostEqual(x.grad[0, 1, 0, 0].item(), x.grad[0, 1, 0, 1].item(), delta=1e-2)


@parameterized_class(("device",), _devices)
class TestHydrostaticBalanceProjection(unittest.TestCase):
    """Tests for HydrostaticBalanceProjection in makani/utils/constraints.py.

    The layer takes a freely-predicted (T, Z) state and applies the minimum-norm
    correction onto the hydrostatic-balance manifold

        Z_i - Z_{i-1} = c_i (Tv_i + Tv_{i-1}),   c_i = 0.5 * R_dry * ln(p_{i-1}/p_i),

    with Tv = T (dry) or Tv = T(1 + eps q) (moist, q held fixed). At strength 1
    the output satisfies balance exactly; the residual scales as (1 - strength).
    """

    LEVELS = [925, 850, 700, 500, 300, 100, 50]
    P_MIN, P_MAX = 0, 2000  # window that admits all of LEVELS

    def setUp(self):
        disable_tf32()
        set_seed(333)

    # --- helpers ---
    def _channels(self, moist):
        names = [f"t{p}" for p in self.LEVELS] + [f"z{p}" for p in self.LEVELS]
        if moist:
            names += [f"q{p}" for p in self.LEVELS]
        names += ["u10m", "v10m", "t2m"]  # aux passthrough channels
        return names

    def _stats(self, channel_names, identity):
        """Per-channel (bias, scale) with realistic magnitudes, or None for identity."""
        if identity:
            return None, None
        C = len(channel_names)
        bias = torch.zeros(1, C, 1, 1)
        scale = torch.ones(1, C, 1, 1)
        for i, nm in enumerate(channel_names):
            if nm.startswith("z"):
                bias[0, i, 0, 0], scale[0, i, 0, 0] = 5.0e4, 3.0e3
            elif nm.startswith("q"):
                bias[0, i, 0, 0], scale[0, i, 0, 0] = 5.0e-3, 5.0e-3
            elif nm.startswith("t"):
                bias[0, i, 0, 0], scale[0, i, 0, 0] = 250.0, 30.0
        return bias.to(self.device), scale.to(self.device)

    def _split(self, channel_names, moist, p_min=None, p_max=None):
        p_min = self.P_MIN if p_min is None else p_min
        p_max = self.P_MAX if p_max is None else p_max
        z_idx, t_idx, pressures = get_matching_channels_pl(channel_names, "z", "t", p_min, p_max)
        q_idx = None
        if moist:
            q_idx, _, _ = get_matching_channels_pl(channel_names, "q", "t", p_min, p_max)
        return t_idx, z_idx, q_idx, pressures

    def _residual(self, out, channel_names, moist, bias, scale, p_min=None, p_max=None):
        """Relative hydrostatic-balance residual per interior level, in physical units."""
        t_idx, z_idx, q_idx, pressures = self._split(channel_names, moist, p_min, p_max)

        def unorm(idx):
            v = out[:, idx, ...].float()
            if bias is not None:
                v = v * scale[:, idx, ...] + bias[:, idx, ...]
            return v

        T, Z = unorm(t_idx), unorm(z_idx)
        Tv = T * (1.0 + const.Q_CORRECTION_MOIST_AIR * unorm(q_idx)) if moist else T

        res, denom = [], []
        for i in range(1, len(pressures)):
            c = 0.5 * const.R_DRY_AIR * float(np.log(pressures[i - 1] / pressures[i]))
            res.append((Z[:, i] - Z[:, i - 1]) - c * (Tv[:, i] + Tv[:, i - 1]))
            denom.append((Z[:, i] - Z[:, i - 1]).abs() + c * (Tv[:, i].abs() + Tv[:, i - 1].abs()))
        return torch.stack(res, dim=1), torch.stack(denom, dim=1)

    def _dummy_climatology(self, n_levels):
        """Deterministic, non-uniform per-interior-level residual b_clim (physical
        units, m^2/s^2), length n_levels - 1, for affine-offset tests. Returned as a
        torch.Tensor, matching the constraint's expected input type (like bias/scale)."""
        return torch.from_numpy(np.linspace(-150.0, 150.0, n_levels - 1))

    def _balanced_field(self, channel_names, moist, B, H, W, offset=None):
        """A full normalized model-output field whose (T, Z[, q]) satisfy balance exactly
        (used for fixed-point tests). With an `offset` (length n-1, physical units) the
        geopotential is integrated so that the residual A x equals `offset` per interior
        level instead of zero -- i.e. the affine manifold A x = offset."""
        t_idx, z_idx, q_idx, pressures = self._split(channel_names, moist)
        n = len(pressures)
        T = 200.0 + 60.0 * torch.rand(B, n, H, W, device=self.device)
        if moist:
            q = 0.02 * torch.rand(B, n, H, W, device=self.device)
            Tv = T * (1.0 + const.Q_CORRECTION_MOIST_AIR * q)
        else:
            Tv = T
        Z = torch.zeros(B, n, H, W, device=self.device)
        Z[:, 0, ...] = 1000.0
        for i in range(1, n):
            plog = float(np.log(pressures[i - 1] / pressures[i]))
            Z[:, i, ...] = Z[:, i - 1, ...] + const.R_DRY_AIR * 0.5 * (Tv[:, i, ...] + Tv[:, i - 1, ...]) * plog
            if offset is not None:
                Z[:, i, ...] = Z[:, i, ...] + float(offset[i - 1])

        C = len(channel_names)
        field = torch.randn(B, C, H, W, device=self.device)  # aux channels arbitrary
        field[:, t_idx, ...] = T
        field[:, z_idx, ...] = Z
        if moist:
            field[:, q_idx, ...] = q
        return field

    # --- strict tests at strength = 1 ---
    @parameterized.expand(
        [
            ("dry_identity", False, True),
            ("dry_normalized", False, False),
            ("moist_identity", True, True),
            ("moist_normalized", True, False),
        ]
    )
    def test_projection_enforces_balance_strict(self, _name, moist, identity):
        """strength=1: an arbitrary (unbalanced) input is projected to satisfy balance
        to fp32 precision."""
        B, H, W = 2, 5, 6
        names = self._channels(moist)
        bias, scale = self._stats(names, identity)
        proj = HydrostaticBalanceProjection(
            names, bias=bias, scale=scale, p_min=self.P_MIN, p_max=self.P_MAX, strength=1.0, use_moist_air_formula=moist
        ).to(self.device)
        x = torch.randn(B, len(names), H, W, device=self.device)
        y = proj(x)
        res, denom = self._residual(y, names, moist, bias, scale)
        rel = (res.abs() / denom.clamp(min=1e-6)).max().item()
        self.assertLess(rel, 1e-4, f"residual not eliminated (rel={rel})")

    @parameterized.expand(
        [
            ("dry_identity", False, True),
            ("dry_normalized", False, False),
            ("moist_identity", True, True),
            ("moist_normalized", True, False),
        ]
    )
    def test_balanced_input_is_fixed_point(self, _name, moist, identity):
        """strength=1: a balanced input is returned essentially unchanged."""
        B, H, W = 2, 4, 5
        names = self._channels(moist)
        bias, scale = self._stats(names, identity)
        field = self._balanced_field(names, moist, B, H, W)
        x = field if identity else (field - bias) / scale
        proj = HydrostaticBalanceProjection(
            names, bias=bias, scale=scale, p_min=self.P_MIN, p_max=self.P_MAX, strength=1.0, use_moist_air_formula=moist
        ).to(self.device)
        y = proj(x)
        self.assertTrue(compare_tensors("balanced fixed point", y, x, atol=1e-2, rtol=1e-4, verbose=False))

    @parameterized.expand([("dry", False), ("moist", True)])
    def test_passthrough_channels_untouched(self, _name, moist):
        """strength=1: channels outside (T, Z) -- aux, and humidity in the moist case --
        are passed through bit-for-bit."""
        B, H, W = 2, 4, 4
        names = self._channels(moist)
        t_idx, z_idx, q_idx, _ = self._split(names, moist)
        touched = set(t_idx) | set(z_idx)
        aux_idx = [i for i in range(len(names)) if i not in touched]
        proj = HydrostaticBalanceProjection(
            names, bias=None, scale=None, p_min=self.P_MIN, p_max=self.P_MAX, strength=1.0, use_moist_air_formula=moist
        ).to(self.device)
        x = torch.randn(B, len(names), H, W, device=self.device)
        y = proj(x)
        # all non-(T,Z) channels, which includes humidity, are identical
        with self.subTest("aux/humidity passthrough"):
            self.assertTrue(compare_tensors("aux/humidity passthrough", y[:, aux_idx, ...], x[:, aux_idx, ...]))
        if moist:
            with self.subTest("humidity held fixed"):
                self.assertTrue(compare_tensors("humidity held fixed", y[:, q_idx, ...], x[:, q_idx, ...]))

    # --- soft tests: strength scales the correction linearly ---
    @parameterized.expand([("dry", False), ("moist", True)])
    def test_strength_scales_correction(self, _name, moist):
        """The correction (x - out) is linear in strength, and the leftover balance
        residual scales as (1 - strength). strength=0 is the identity."""
        B, H, W = 2, 5, 6
        names = self._channels(moist)
        bias, scale = self._stats(names, identity=False)
        proj = HydrostaticBalanceProjection(
            names, bias=bias, scale=scale, p_min=self.P_MIN, p_max=self.P_MAX, strength=1.0, use_moist_air_formula=moist
        ).to(self.device)
        x = torch.randn(B, len(names), H, W, device=self.device)

        proj.strength = 0.0
        self.assertTrue(compare_tensors("strength=0 identity", proj(x), x, atol=1e-5, rtol=1e-5))

        proj.strength = 1.0
        y1 = proj(x)
        proj.strength = 0.5
        yh = proj(x)

        # the two sub-checks below are facets of the same linearity property, so they
        # share one test under subTest rather than splitting into separate cases
        with self.subTest("deviation linear in strength"):
            # deviation from the input is exactly half of the full correction
            self.assertTrue(
                compare_tensors("half correction", (x - yh), 0.5 * (x - y1), atol=1e-4, rtol=1e-3, verbose=False)
            )
        with self.subTest("residual scales as (1 - strength)"):
            # the residual that remains at strength 0.5 is half the input's residual
            res_in, _ = self._residual(x, names, moist, bias, scale)
            res_half, _ = self._residual(yh, names, moist, bias, scale)
            self.assertTrue(
                compare_tensors("residual halved", res_half, 0.5 * res_in, atol=1e-1, rtol=1e-3, verbose=False)
            )

    # --- gradients ---
    @parameterized.expand([("dry", False), ("moist", True)])
    def test_gradients_flow(self, _name, moist):
        """Gradients reach T and Z (and q in the moist case, which modulates the split)."""
        B, H, W = 1, 4, 4
        names = self._channels(moist)
        bias, scale = self._stats(names, identity=False)
        t_idx, z_idx, q_idx, _ = self._split(names, moist)
        proj = HydrostaticBalanceProjection(
            names, bias=bias, scale=scale, p_min=self.P_MIN, p_max=self.P_MAX, strength=1.0, use_moist_air_formula=moist
        ).to(self.device)
        x = torch.randn(B, len(names), H, W, device=self.device, requires_grad=True)
        proj(x).pow(2).sum().backward()
        self.assertIsNotNone(x.grad)
        with self.subTest("temperature gradient"):
            self.assertFalse((x.grad[:, t_idx, ...] == 0).all().item())
        with self.subTest("geopotential gradient"):
            self.assertFalse((x.grad[:, z_idx, ...] == 0).all().item())
        if moist:
            with self.subTest("humidity gradient"):
                self.assertFalse((x.grad[:, q_idx, ...] == 0).all().item())

    # --- climatology affine offset ---
    @parameterized.expand([("dry", False), ("moist", True)])
    def test_climatology_pins_residual(self, _name, moist):
        """strength=1 with a climatology offset: the output residual equals b_clim
        exactly (the affine target), not zero, independent of the input."""
        B, H, W = 2, 5, 6
        names = self._channels(moist)
        bias, scale = self._stats(names, identity=False)
        _, _, _, pressures = self._split(names, moist)
        b_clim = self._dummy_climatology(len(pressures))
        proj = HydrostaticBalanceProjection(
            names,
            bias=bias,
            scale=scale,
            p_min=self.P_MIN,
            p_max=self.P_MAX,
            strength=1.0,
            use_moist_air_formula=moist,
            climatology_offset=b_clim,
        ).to(self.device)
        x = torch.randn(B, len(names), H, W, device=self.device)
        y = proj(x)
        res, _ = self._residual(y, names, moist, bias, scale)
        target = torch.as_tensor(b_clim, dtype=res.dtype, device=res.device).view(1, -1, 1, 1).expand_as(res)
        self.assertTrue(
            compare_tensors("residual pinned to climatology", res, target, atol=1e-1, rtol=1e-3, verbose=False)
        )

    @parameterized.expand([("dry", False), ("moist", True)])
    def test_climatology_affine_fixed_point(self, _name, moist):
        """strength=1: a state already on the affine manifold A x = b_clim is unchanged."""
        B, H, W = 2, 4, 5
        names = self._channels(moist)
        bias, scale = self._stats(names, identity=False)
        _, _, _, pressures = self._split(names, moist)
        b_clim = self._dummy_climatology(len(pressures))
        field = self._balanced_field(names, moist, B, H, W, offset=b_clim)
        x = (field - bias) / scale
        proj = HydrostaticBalanceProjection(
            names,
            bias=bias,
            scale=scale,
            p_min=self.P_MIN,
            p_max=self.P_MAX,
            strength=1.0,
            use_moist_air_formula=moist,
            climatology_offset=b_clim,
        ).to(self.device)
        y = proj(x)
        self.assertTrue(compare_tensors("affine fixed point", y, x, atol=1e-2, rtol=1e-4, verbose=False))

    @parameterized.expand([("dry", False), ("moist", True)])
    def test_climatology_strength_interpolates(self, _name, moist):
        """0<lambda<1 with an offset: the output residual is
        (1 - lambda) * input_residual + lambda * b_clim."""
        B, H, W = 2, 5, 6
        names = self._channels(moist)
        bias, scale = self._stats(names, identity=False)
        _, _, _, pressures = self._split(names, moist)
        b_clim = self._dummy_climatology(len(pressures))
        proj = HydrostaticBalanceProjection(
            names,
            bias=bias,
            scale=scale,
            p_min=self.P_MIN,
            p_max=self.P_MAX,
            strength=0.5,
            use_moist_air_formula=moist,
            climatology_offset=b_clim,
        ).to(self.device)
        x = torch.randn(B, len(names), H, W, device=self.device)
        y = proj(x)
        res_in, _ = self._residual(x, names, moist, bias, scale)
        res_out, _ = self._residual(y, names, moist, bias, scale)
        target = torch.as_tensor(b_clim, dtype=res_in.dtype, device=res_in.device).view(1, -1, 1, 1)
        expected = 0.5 * res_in + 0.5 * target
        self.assertTrue(
            compare_tensors(
                "residual interpolated toward climatology", res_out, expected, atol=1e-1, rtol=1e-3, verbose=False
            )
        )

    @parameterized.expand([("dry", False), ("moist", True)])
    def test_climatology_full_offset_sliced_to_window(self, _name, moist):
        """The offset is specified over ALL matching levels; a projection on a strict
        sub-window must verify the full length and pin only its own interior levels to
        the correct contiguous slice of b_clim."""
        B, H, W = 2, 4, 5
        names = self._channels(moist)
        bias, scale = self._stats(names, identity=False)
        # full level set and full-length climatology (one value per full interior level)
        _, _, _, pressures_all = self._split(names, moist)  # P_MIN/P_MAX admit all LEVELS
        b_clim_full = self._dummy_climatology(len(pressures_all))

        # narrow window: a strict, contiguous subset of the levels
        pw_min, pw_max = 200, 800
        _, _, _, pressures_win = self._split(names, moist, p_min=pw_min, p_max=pw_max)
        self.assertLess(len(pressures_win), len(pressures_all))  # genuinely narrower
        start = pressures_all.index(pressures_win[0])
        expected_slice = b_clim_full[start : start + (len(pressures_win) - 1)]

        proj = HydrostaticBalanceProjection(
            names,
            bias=bias,
            scale=scale,
            p_min=pw_min,
            p_max=pw_max,
            strength=1.0,
            use_moist_air_formula=moist,
            climatology_offset=b_clim_full,
        ).to(self.device)
        x = torch.randn(B, len(names), H, W, device=self.device)
        y = proj(x)
        # residual over the windowed levels must equal the corresponding b_clim slice
        res, _ = self._residual(y, names, moist, bias, scale, p_min=pw_min, p_max=pw_max)
        target = torch.as_tensor(expected_slice, dtype=res.dtype, device=res.device).view(1, -1, 1, 1).expand_as(res)
        self.assertTrue(
            compare_tensors(
                "windowed residual pinned to b_clim slice", res, target, atol=1e-1, rtol=1e-3, verbose=False
            )
        )

    def test_climatology_offset_wrong_length_raises(self):
        """A climatology offset whose length is not (#all matching levels - 1) is rejected."""
        names = self._channels(moist=False)
        _, _, _, pressures_all = self._split(names, moist=False)
        bad_offset = torch.zeros(len(pressures_all))  # one too long: should be len - 1
        with self.assertRaises(ValueError):
            HydrostaticBalanceProjection(names, p_min=self.P_MIN, p_max=self.P_MAX, climatology_offset=bad_offset)


if __name__ == "__main__":
    unittest.main()
