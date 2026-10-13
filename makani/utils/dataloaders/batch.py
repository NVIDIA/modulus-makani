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

"""
The batch every dataloader emits: a dict instead of a positional tuple.

A tuple leaves its meaning to the position of each tensor, and that position
shifted with the options a loader was built with, so every consumer had to
repeat the unpacking rule. The dict names every tensor instead:

==========================  ==========================================  =================================
key                         content                                     shape
==========================  ==========================================  =================================
``input``                   predicted state at the input steps          ``(B, n_history+1, C_in, *grid)``
``target``                  predicted state at the target steps         ``(B, n_future+1, C_out, *grid)``
``input_dynamic_features``  dict of time dependent features that are    ``(B, n_history+1, C_f, *grid)``
                            not predicted, e.g. ``zenith``              per feature
``target_dynamic_features`` the same at the target steps                ``(B, n_future+1, C_f, *grid)``
``input_timestamp``         time of each input step, seconds            ``(B, n_history+1)``
``target_timestamp``        time of each target step, seconds           ``(B, n_future+1)``
==========================  ==========================================  =================================

``grid`` is ``(H, W)`` for a lat/lon grid and ``(N_points,)`` for a mesh: the
batch, time and channel dimensions always lead, the grid always trails, and
the grid itself is described by the loader's ``DataShapes``, not by the shape.

Every key is present, except that a loader built without targets, as for
inference, omits the three target keys. The dynamic features are a dict even
when empty. ``static_features`` (``(C_static, *grid)``) is reserved: the static
fields are the same for every sample, so the preprocessor holds them on the
device rather than every batch carrying a copy.

Until the preprocessor and the models consume the dict, :func:`to_legacy_tuple`
turns it back into the tuple they expect.
"""

from typing import Any, Dict, Optional, Tuple

import torch

INPUT = "input"
TARGET = "target"
INPUT_DYNAMIC_FEATURES = "input_dynamic_features"
TARGET_DYNAMIC_FEATURES = "target_dynamic_features"
STATIC_FEATURES = "static_features"
INPUT_TIMESTAMP = "input_timestamp"
TARGET_TIMESTAMP = "target_timestamp"

# names of the dynamic features
ZENITH = "zenith"

Batch = Dict[str, Any]


def make_batch(
    inp: torch.Tensor,
    inp_timestamp: torch.Tensor,
    tar: Optional[torch.Tensor] = None,
    tar_timestamp: Optional[torch.Tensor] = None,
    inp_zenith: Optional[torch.Tensor] = None,
    tar_zenith: Optional[torch.Tensor] = None,
) -> Batch:
    """
    Assemble a batch, or a single sample, from its parts.

    Parameters
    ----------
    inp : torch.Tensor
        Input state.
    inp_timestamp : torch.Tensor
        Time of each input step.
    tar : torch.Tensor, optional
        Target state. Without it, the batch has no target keys.
    tar_timestamp : torch.Tensor, optional
        Time of each target step, required with ``tar``.
    inp_zenith, tar_zenith : torch.Tensor, optional
        Cosine of the solar zenith angle at the input and target steps.

    Returns
    -------
    dict
        The batch, see the module docstring.
    """

    if (tar is None) != (tar_timestamp is None):
        raise ValueError("A target and its timestamps go together")

    batch = {INPUT: inp, INPUT_DYNAMIC_FEATURES: {}, INPUT_TIMESTAMP: inp_timestamp}
    if inp_zenith is not None:
        batch[INPUT_DYNAMIC_FEATURES][ZENITH] = inp_zenith

    if tar is not None:
        batch[TARGET] = tar
        batch[TARGET_DYNAMIC_FEATURES] = {}
        batch[TARGET_TIMESTAMP] = tar_timestamp
        if tar_zenith is not None:
            batch[TARGET_DYNAMIC_FEATURES][ZENITH] = tar_zenith

    return batch


def batch_to_device(batch: Batch, device: torch.device, non_blocking: bool = False) -> Batch:
    """
    Move every tensor of a batch to ``device``, including those of nested dicts.

    Parameters
    ----------
    batch : dict
        The batch.
    device : torch.device
        Where to move it.
    non_blocking : bool, optional
        Passed on to ``Tensor.to``.

    Returns
    -------
    dict
        A new batch; the one passed in is left as it is.
    """

    def _move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=non_blocking)
        if isinstance(value, dict):
            return {key: _move(item) for key, item in value.items()}
        return value

    return _move(batch)


def to_legacy_tuple(batch: Batch, timestamps: bool = False) -> Tuple[torch.Tensor, ...]:
    """
    Convert a batch to the positional tuple the loaders used to emit.

    The order is input, target, input zenith, target zenith and, if requested,
    input and target timestamps, each item only if the batch has it. This is
    what the preprocessor and the trainers consume until they take the dict.

    Parameters
    ----------
    batch : dict
        The batch.
    timestamps : bool, optional
        Append the timestamps. The training loaders never returned them, the
        inference loaders always did.

    Returns
    -------
    tuple of torch.Tensor

    Raises
    ------
    ValueError
        If the batch is not on a lat/lon grid, or carries a dynamic feature
        the tuple has no place for.
    """

    inp = batch[INPUT]
    if inp.dim() != 5:
        raise ValueError(
            f"The legacy tuple needs a lat/lon grid, shape (B, T, C, H, W), but the input has shape {tuple(inp.shape)}"
        )

    has_target = TARGET in batch

    for key in (INPUT_DYNAMIC_FEATURES, TARGET_DYNAMIC_FEATURES):
        unknown = set(batch.get(key, {})) - {ZENITH}
        if unknown:
            raise ValueError(f"The legacy tuple has no place for the dynamic features {sorted(unknown)}")

    result = (inp,)
    if has_target:
        result += (batch[TARGET],)

    if ZENITH in batch[INPUT_DYNAMIC_FEATURES]:
        result += (batch[INPUT_DYNAMIC_FEATURES][ZENITH],)
        if has_target:
            result += (batch[TARGET_DYNAMIC_FEATURES][ZENITH],)

    if timestamps:
        result += (batch[INPUT_TIMESTAMP],)
        if has_target:
            result += (batch[TARGET_TIMESTAMP],)

    return result


def legacy_batch_to_device(batch: Batch, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """
    The legacy tuple of a batch, without timestamps, on ``device``.

    This is how the trainers take a batch until the preprocessor consumes the
    dict: the tensors the training step needs, in the order it expects.

    Parameters
    ----------
    batch : dict
        The batch.
    device : torch.device
        Where to move the tensors.

    Returns
    -------
    tuple of torch.Tensor
    """
    return tuple(item.to(device) for item in to_legacy_tuple(batch))
