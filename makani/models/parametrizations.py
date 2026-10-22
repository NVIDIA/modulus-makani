# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp

import makani.utils.constants as const


class _HydrostaticBalanceWrapper(nn.Module):
    def __init__(self, channel_names, bias, scale, p_min=50, p_max=900, use_moist_air_formula=False):
        super().__init__()

        self.use_moist_air_formula = use_moist_air_formula

        # we need the matching routine:
        from makani.utils.constraints import get_matching_channels_pl

        # set up indices
        self.z_idx, self.t_idx, self.pressures = get_matching_channels_pl(channel_names, "z", "t", p_min, p_max)

        if self.use_moist_air_formula:
            self.q_idx, _, p_tmp = get_matching_channels_pl(channel_names, "q", "t", p_min, p_max)

        # sanity checks
        if len(self.pressures) == 0:
            raise ValueError(
                "Error, make sure that you have overlapping pressure levels for geopotential and temperature"
            )

        if self.use_moist_air_formula:
            for p1, p2 in zip(self.pressures, p_tmp):
                if p1 != p2:
                    raise ValueError("Error, make sure that you have the same pressure levels for t,z and q channels")

        # create a mapping for all channels except for z and t:
        self.aux_idx = [i for i in range(len(channel_names)) if (i not in self.t_idx + self.z_idx)]
        if self.use_moist_air_formula:
            self.aux_idx = [i for i in self.aux_idx if (i not in self.q_idx)]

        # channel layout of the reduced input, in terms of the full channels: T_0, the geopotentials,
        # the specific humidities if the moist formula is used, and the remaining channels
        self.inp_channel_idx = [self.t_idx[0]] + self.z_idx
        if self.use_moist_air_formula:
            self.inp_channel_idx += self.q_idx
        self.inp_channel_idx += self.aux_idx

        # compute prefact
        # prefactor, units [K * kg / J] = [K * kg * s^2 / (m^2 * kg)] = [K * s^2 / m^2]
        self.prefact = 1.0 / const.R_DRY_AIR

        if self.use_moist_air_formula:
            self.q_prefact = const.Q_CORRECTION_MOIST_AIR

        # assume that the first len(self.z_idx + 1) channels of the model output tensor are the geopotentials
        # and the reference temperature tensor (first component). Assume then that the remaining components correspond to the other output
        # channel.
        # create a matrix which generates a mapping from the original output to the actual output:
        # create conserved quantity matrix
        row_indices = []
        col_indices = []
        values = []

        # extract bias and scale
        if bias is not None:
            z_bias = bias[:, self.z_idx, ...]
            t_bias = bias[:, self.t_idx, ...]
            if self.use_moist_air_formula:
                q_bias = bias[:, self.q_idx, ...]
        else:
            z_bias = torch.zeros([1, len(self.z_idx), 1, 1], dtype=torch.float32)
            t_bias = torch.zeros([1, len(self.t_idx), 1, 1], dtype=torch.float32)
            if self.use_moist_air_formula:
                q_bias = torch.zeros([1, len(self.q_idx), 1, 1], dtype=torch.float32)

        if scale is not None:
            z_scale = scale[:, self.z_idx, ...]
            t_scale = scale[:, self.t_idx, ...]
            if self.use_moist_air_formula:
                q_scale = scale[:, self.q_idx, ...]
        else:
            z_scale = torch.ones([1, len(self.z_idx), 1, 1], dtype=torch.float32)
            t_scale = torch.ones([1, len(self.t_idx), 1, 1], dtype=torch.float32)
            if self.use_moist_air_formula:
                q_scale = torch.ones([1, len(self.q_idx), 1, 1], dtype=torch.float32)

        # initial t:
        row_indices.append(self.t_idx[0])
        col_indices.append(0)
        values.append(1.0)

        # do the z_indices first:
        for idx in range(1, len(self.z_idx) + 1):
            # z_idx
            row_indices.append(self.z_idx[idx - 1])
            col_indices.append(idx)
            values.append(1.0)

        # (Z_i - Z_{i-1}) / R = 0.5 * log(p_{i-1}/p_i) * (T_i + T_{i-1})
        # -> T_i = 2 * (Z_i - Z_{i-1}) / (R * log(p_{i-1} / p_i)) - T_{i-1}
        # re-insert T{i-1} recursively
        for oidx in range(1, len(self.z_idx)):
            for iidx in range(0, oidx):
                # sign
                sign = (-1.0) ** (oidx + iidx - 1)

                # t_idx
                plog = np.log(self.pressures[iidx] / self.pressures[iidx + 1])

                # 2. * Z_i / (R * log(p_{i-1} / p_i))
                row_indices.append(self.t_idx[oidx])
                col_indices.append(iidx + 2)
                values.append(sign * 2.0 * self.prefact / plog)

                # - 2. * Z_{i-1} / (R * log(p_{i-1} / p_i))
                row_indices.append(self.t_idx[oidx])
                col_indices.append(iidx + 1)
                values.append(-sign * 2.0 * self.prefact / plog)

            # - T_0
            row_indices.append(self.t_idx[oidx])
            col_indices.append(0)
            values.append((-1.0) ** oidx)

        # q-values when moist air is used:
        off_idx = len(self.z_idx) + 1
        if self.use_moist_air_formula:
            for idx in range(off_idx, off_idx + len(self.q_idx)):
                row_indices.append(self.q_idx[idx - off_idx])
                col_indices.append(idx)
                values.append(1.0)
            # shift the offset accordingly
            off_idx += len(self.q_idx)

        # now deal with the remaining channels:
        for idx in range(off_idx, off_idx + len(self.aux_idx)):
            row_indices.append(self.aux_idx[idx - off_idx])
            col_indices.append(idx)
            values.append(1.0)

        # create matrix:
        ncols = len(channel_names) - len(self.t_idx) + 1
        indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
        values = torch.tensor(values, dtype=torch.float32)
        mapping = torch.sparse_coo_tensor(indices, values, size=(len(channel_names), ncols)).coalesce().to_dense()

        # register
        self.register_buffer("mapping", mapping, persistent=False)

        # bias and scale terms:
        # input
        # scale
        inp_scale = torch.ones((1, mapping.shape[1], 1, 1), dtype=torch.float32)
        inp_scale[0, 0, 0, 0] = t_scale[0, 0, 0, 0]
        inp_scale[0, 1 : len(self.z_idx) + 1, 0, 0] = z_scale[0, :, 0, 0]
        # bias
        inp_bias = torch.zeros((1, mapping.shape[1], 1, 1), dtype=torch.float32)
        inp_bias[0, 0, 0, 0] = t_bias[0, 0, 0, 0]
        inp_bias[0, 1 : len(self.z_idx) + 1, 0, 0] = z_bias[0, :, 0, 0]
        # moist air
        if self.use_moist_air_formula:
            # assume the next entries are specific humidities:
            inp_bias[0, len(self.z_idx) + 1 : len(self.z_idx) + len(self.q_idx) + 1, 0, 0] = q_bias[0, :, 0, 0]
            inp_scale[0, len(self.z_idx) + 1 : len(self.z_idx) + len(self.q_idx) + 1, 0, 0] = q_scale[0, :, 0, 0]
        self.register_buffer("inp_scale", inp_scale, persistent=True)
        self.register_buffer("inp_bias", inp_bias, persistent=True)

        # output
        # scale
        out_scale = torch.ones((1, len(channel_names), 1, 1), dtype=torch.float32)
        out_scale[:, self.t_idx, ...] = t_scale[...]
        out_scale[:, self.z_idx, ...] = z_scale[...]
        # bias
        out_bias = torch.zeros((1, len(channel_names), 1, 1), dtype=torch.float32)
        out_bias[:, self.t_idx, ...] = t_bias[...]
        out_bias[:, self.z_idx, ...] = z_bias[...]
        # moist air
        if self.use_moist_air_formula:
            out_bias[:, self.q_idx, ...] = q_bias[...]
            out_scale[:, self.q_idx, ...] = q_scale[...]
        self.register_buffer("out_scale", out_scale, persistent=True)
        self.register_buffer("out_bias", out_bias, persistent=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        itype = inp.dtype

        # The hydrostatic recursion sums over ~11 pressure levels of large, un-normalized
        # geopotentials (O(1e5) m^2/s^2) with alternating signs. F.conv2d is autocast-eligible,
        # so under AMP it would quantize those geopotentials to bf16/fp16 and corrupt the
        # recovered temperatures by tens of K. Run the whole physical-units computation in
        # fp32, from the un-normalization onward.
        with amp.autocast(device_type=inp.device.type, enabled=False):
            inp = inp.to(torch.float32)

            # undo normalization
            inp_un = inp * self.inp_scale + self.inp_bias

            # convert temperature to virtual temperature
            if self.use_moist_air_formula:
                inp_un_scale = inp_un.clone()
                inp_un_scale[:, 0, ...] = inp_un[:, 0, ...] * (
                    1.0 + self.q_prefact * inp_un[:, len(self.z_idx) + 1, ...]
                )
            else:
                inp_un_scale = inp_un

            # expand:
            out_un = F.conv2d(inp_un_scale, self.mapping.unsqueeze(-1).unsqueeze(-1))

            # unscale temperatures if specific humidity is used
            if self.use_moist_air_formula:
                out_un[:, self.t_idx, ...] = out_un[:, self.t_idx, ...] / (
                    1.0 + self.q_prefact * out_un[:, self.q_idx, ...]
                )

            # undo normalization
            out = (out_un - self.out_bias) / self.out_scale

        # cast back to the input dtype so the layer is faithful to its input
        out = out.to(itype)

        return out


class ConstraintsWrapper(nn.Module):
    r"""
    Enforce physical constraints on a model's output.

    A purely data-driven prediction need not satisfy the balance relations the
    atmosphere obeys. This wrapper projects the output onto the constrained
    manifold after the forward pass, so the prediction is physically consistent
    by construction rather than only approximately so through the loss.

    Can either wrap a model (constraints applied to its output) or act
    standalone on a tensor, which is useful for post-processing predictions
    produced elsewhere.

    ``constraints`` is a list of dicts, applied in order, e.g.::

        constraints = [{"type": "nonnegative",
                        "options": {"mode": "quadratic"}},
                       {"type": "hydrostatic_balance",
                        "options": {...}}]

    Parameters
    ----------
    constraints : list of dict
        Constraint specifications. Each entry needs a ``"type"`` key and an
        optional ``"options"`` dict forwarded to that constraint. Supported are
        ``"hydrostatic_balance"`` (:class:`_HydrostaticBalanceWrapper`) and
        ``"nonnegative"`` (:class:`~makani.utils.constraints.NonNegativeConstraint`,
        whose ``names_to_clamp`` default to the water channels).
    channel_names : list of str
        Names of the output channels, used to locate the variables each
        constraint acts on.
    bias : torch.Tensor
        Normalization bias of the output channels. Needed because constraints
        are physical relations and must be applied in physical units, not
        normalized ones.
    scale : torch.Tensor
        Normalization scale of the output channels.
    model_handle : callable, optional
        Zero-argument factory returning a model to wrap. If omitted, the
        wrapper applies the constraints directly to its input.

    Raises
    ------
    NotImplementedError
        If an unsupported constraint type is requested.
    ValueError
        If ``"hydrostatic_balance"`` is requested more than once.

    Notes
    -----
    ``"hydrostatic_balance"`` is a reparametrization: the model predicts a
    reduced set of ``N_in_channels`` channels (``T_0``, the geopotentials, the
    humidities for the moist formula and the remaining channels), which it
    expands to the full channel set. Constraints listed before it act on the
    reduced channels, constraints after it on the full ones. The order matters:
    to clamp the humidities, ``"nonnegative"`` has to come first, so that the
    virtual temperature of the moist formula is computed from the clamped
    humidities and the output stays balanced.
    """

    def __init__(self, constraints, channel_names, bias, scale, model_handle=None):
        super().__init__()

        if model_handle is not None:
            self.model = model_handle()
        else:
            self.model = None

        types = [constraint["type"] for constraint in constraints]
        if types.count("hydrostatic_balance") > 1:
            raise ValueError("Error, the hydrostatic balance constraint can only be used once.")

        # without a reparametrization, the model predicts the full channel set
        self.N_in_channels = len(channel_names)
        inp_channel_names, inp_bias, inp_scale = channel_names, bias, scale

        # the hydrostatic balance reparametrization determines the reduced channels the model predicts,
        # which the constraints before it act on
        hb_pos = types.index("hydrostatic_balance") if "hydrostatic_balance" in types else len(constraints)
        if hb_pos < len(constraints):
            hb_constraint = _HydrostaticBalanceWrapper(
                **constraints[hb_pos].get("options", {}), channel_names=channel_names, bias=bias, scale=scale
            )
            inp_idx = hb_constraint.inp_channel_idx
            inp_channel_names = [channel_names[c] for c in inp_idx]
            inp_bias = bias[:, inp_idx, ...] if bias is not None else None
            inp_scale = scale[:, inp_idx, ...] if scale is not None else None
            self.N_in_channels = len(inp_idx)

        self.constraint_list = nn.ModuleList()
        for idx, constraint in enumerate(constraints):
            if idx == hb_pos:
                self.constraint_list.append(hb_constraint)
                continue

            if idx < hb_pos:
                names, cbias, cscale = inp_channel_names, inp_bias, inp_scale
            else:
                names, cbias, cscale = channel_names, bias, scale

            options = dict(constraint.get("options", {}))
            if constraint["type"] == "nonnegative":
                from makani.utils.constraints import NonNegativeConstraint
                from makani.utils.features import get_water_channels

                names_to_clamp = options.pop("names_to_clamp", [names[c] for c in get_water_channels(names)])
                self.constraint_list.append(
                    NonNegativeConstraint(names, names_to_clamp, bias=cbias, scale=cscale, **options)
                )
            else:
                raise NotImplementedError(f"Error, constraint type {constraint['type']} is not implemented.")

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        r"""
        Run the wrapped model, if any, and apply the constraints to the result.

        Parameters
        ----------
        inp : torch.Tensor
            Model input if a model is wrapped, otherwise the field to constrain
            directly.

        Returns
        -------
        torch.Tensor
            Constrained output. Channels the constraint does not act on pass
            through unchanged.
        """
        if self.model is not None:
            # get model output
            inp = self.model(inp)

        # apply the constraints in order
        out = inp
        for constraint in self.constraint_list:
            out = constraint(out)

        return out
