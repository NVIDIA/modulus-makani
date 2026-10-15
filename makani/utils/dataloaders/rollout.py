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
The steps of an autoregressive rollout over one batch.

A batch holds the input window and the targets of several steps. A rollout
hands out one step at a time: the window the model predicts from, the dynamic
features that go with it, the target of that step and, if a noise source is
given, the noise for it. Since the next window depends on the prediction, the
caller hands each prediction back with :meth:`Rollout.advance`, which slides
the window, the features and the timestamps by one step::

    rollout = Rollout(batch, noise_source=noise)
    for step in rollout:
        pred = model_step(step)
        rollout.advance(pred)

All of this is bookkeeping over the keys of the batch, see
:mod:`makani.utils.dataloaders.batch`: nothing here knows about the model,
normalization or packing. The time axis stays explicit, and the grid trails,
so a rollout works for lat/lon and mesh batches alike.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch

from makani.utils.dataloaders.batch import (
    INPUT,
    TARGET,
    INPUT_DYNAMIC_FEATURES,
    TARGET_DYNAMIC_FEATURES,
    INPUT_TIMESTAMP,
    TARGET_TIMESTAMP,
    Batch,
)


@dataclass(frozen=True)
class RolloutStep:
    """
    What one step of a rollout predicts from, and what it predicts.

    Attributes
    ----------
    index : int
        The step, counted from 0.
    input : torch.Tensor
        The window to predict from, ``(B, n_history + 1, C, *grid)``, in
        physical units. From step 1 on, its latest steps are predictions.
    input_dynamic_features : dict
        The dynamic features over that window, name to ``(B, n_history + 1, C_f, *grid)``.
    input_timestamp : torch.Tensor
        Time of each step of the window, ``(B, n_history + 1)``.
    target : torch.Tensor or None
        The state to predict, ``(B, 1, C, *grid)``; ``None`` without targets.
    target_dynamic_features : dict
        The dynamic features at the predicted time, name to ``(B, 1, C_f, *grid)``.
    target_timestamp : torch.Tensor or None
        Time of the predicted step, ``(B, 1)``.
    noise : torch.Tensor or None
        The sample of the noise source for this step, ``None`` without one.
    """

    index: int
    input: torch.Tensor
    input_dynamic_features: Dict[str, torch.Tensor] = field(default_factory=dict)
    input_timestamp: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    target_dynamic_features: Dict[str, torch.Tensor] = field(default_factory=dict)
    target_timestamp: Optional[torch.Tensor] = None
    noise: Optional[torch.Tensor] = None


class Rollout:
    """
    Iterate over the steps of an autoregressive rollout of one batch.

    Each step has to be followed by :meth:`advance` with its prediction before
    the next one is handed out: the next window contains that prediction, so
    handing out the same step again, or the next one without it, would be wrong
    either way, and both raise.

    Parameters
    ----------
    batch : dict
        The batch, see :mod:`makani.utils.dataloaders.batch`.
    n_steps : int, optional
        Number of steps. Defaults to the number of target steps.
    noise_source : optional
        Source of the noise of each step, an external input like a forcing.
        Called as ``noise_source.update(replace_state=...)`` before each step,
        replacing the state at the first step and continuing it after that,
        and then as ``noise_source()`` for the sample of the step. The noise
        modules in :mod:`makani.models.noise` have that interface; the rollout
        does not own the source, the model does, since it may be trainable.
    continue_noise : bool, optional
        Continue the noise state of a previous rollout at the first step
        instead of replacing it, e.g. for an inference run split into several
        rollouts. By default ``False``.

    Raises
    ------
    ValueError
        If the batch has fewer target steps than ``n_steps``, or no targets and
        no ``n_steps``.
    NotImplementedError
        For more than one step without targets: the dynamic features and
        timestamps of later steps would have to come from elsewhere.
    """

    def __init__(
        self,
        batch: Batch,
        n_steps: Optional[int] = None,
        noise_source: Optional[Callable[[], torch.Tensor]] = None,
        continue_noise: bool = False,
    ):
        self._has_target = TARGET in batch
        if n_steps is None:
            if not self._has_target:
                raise ValueError("A rollout of a batch without targets needs n_steps")
            n_steps = batch[TARGET].shape[1]
        if self._has_target and n_steps > batch[TARGET].shape[1]:
            raise ValueError(f"{n_steps} rollout steps, but the batch has {batch[TARGET].shape[1]} target steps")
        if (not self._has_target) and n_steps > 1:
            raise NotImplementedError(
                "A rollout without targets over more than one step needs the dynamic features and "
                "timestamps of the later steps from elsewhere, which is not supported yet"
            )

        self._batch = batch
        self._n_steps = n_steps
        self._noise_source = noise_source
        self._continue_noise = continue_noise

        self._input = batch[INPUT]
        self._input_features = dict(batch[INPUT_DYNAMIC_FEATURES])
        self._input_timestamp = batch[INPUT_TIMESTAMP]

        self._index = 0
        self._pending = False

    @property
    def n_steps(self) -> int:
        """Number of steps of the rollout."""
        return self._n_steps

    def __len__(self) -> int:
        return self._n_steps

    def __iter__(self):
        return self

    def __next__(self) -> RolloutStep:
        if self._pending:
            raise RuntimeError(
                f"Step {self._index} of the rollout has not been advanced: call advance() with its "
                "prediction before asking for the next step"
            )
        if self._index >= self._n_steps:
            raise StopIteration

        k = self._index

        noise = None
        if self._noise_source is not None:
            replace = (k == 0) and not self._continue_noise
            self._noise_source.update(replace_state=replace)
            noise = self._noise_source()

        target = None
        target_features = {}
        target_timestamp = None
        if self._has_target:
            target = self._batch[TARGET][:, k : k + 1]
            target_features = {
                name: feature[:, k : k + 1] for name, feature in self._batch[TARGET_DYNAMIC_FEATURES].items()
            }
            target_timestamp = self._batch[TARGET_TIMESTAMP][:, k : k + 1]

        self._pending = True

        return RolloutStep(
            index=k,
            input=self._input,
            input_dynamic_features=dict(self._input_features),
            input_timestamp=self._input_timestamp,
            target=target,
            target_dynamic_features=target_features,
            target_timestamp=target_timestamp,
            noise=noise,
        )

    def advance(self, pred: torch.Tensor):
        """
        Slide the rollout by one step, with the prediction of the current one.

        The window drops its oldest step and appends ``pred``; the dynamic
        features and timestamps of the window move on to the predicted time.

        Parameters
        ----------
        pred : torch.Tensor
            The prediction of the current step, ``(B, 1, C, *grid)``, in
            physical units. It is used as it is, so detach it first for
            push-forward training.

        Raises
        ------
        RuntimeError
            If no step is waiting for its prediction.
        ValueError
            If the prediction does not have the shape of one step of the window.
        """
        if not self._pending:
            raise RuntimeError("advance() needs a step of the rollout to advance: ask for the next step first")

        expected = (self._input.shape[0], 1, *self._input.shape[2:])
        if tuple(pred.shape) != expected:
            raise ValueError(
                f"The prediction has shape {tuple(pred.shape)}, one step of the window has shape {expected}"
            )

        k = self._index
        self._input = torch.cat([self._input[:, 1:], pred], dim=1)
        if self._has_target:
            target_features = self._batch[TARGET_DYNAMIC_FEATURES]
            self._input_features = {
                name: torch.cat([window[:, 1:], target_features[name][:, k : k + 1]], dim=1)
                for name, window in self._input_features.items()
            }
            self._input_timestamp = torch.cat(
                [self._input_timestamp[:, 1:], self._batch[TARGET_TIMESTAMP][:, k : k + 1]], dim=1
            )

        self._index += 1
        self._pending = False


def run_rollout(
    model_step: Callable[[RolloutStep], torch.Tensor], rollout: Rollout, push_forward: bool = False
) -> torch.Tensor:
    """
    Run a rollout to its end and stack the predictions along the time axis.

    Parameters
    ----------
    model_step : callable
        Maps a :class:`RolloutStep` to its prediction, ``(B, 1, C, *grid)``.
    rollout : Rollout
        The rollout to run.
    push_forward : bool, optional
        Detach each prediction before it enters the next window, so gradients
        do not flow through the rollout. By default ``False``.

    Returns
    -------
    torch.Tensor
        The predictions, ``(B, n_steps, C, *grid)``, aligned with the targets.
    """
    preds = []
    for step in rollout:
        pred = model_step(step)
        preds.append(pred)
        rollout.advance(pred.detach() if push_forward else pred)

    return torch.cat(preds, dim=1)
