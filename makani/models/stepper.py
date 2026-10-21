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

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from makani.models.preprocessor import Preprocessor2D
from makani.utils.dataloaders.rollout import Rollout, RolloutStep


def _assert_checkpoint_safe(module: nn.Module):
    """Guard for rollout activation checkpointing.

    Checkpointing recomputes each step's model forward during backward. Global-RNG
    stochastic ops (e.g. DropPath) are covered by checkpoint's preserve_rng_state, but
    modules that advance their *own* torch.Generator (e.g. SeededDropout2d) are not:
    their recomputed mask would diverge from the original forward and corrupt gradients.
    Fail loudly rather than train silently-wrong.
    """
    offenders = sorted(
        {
            type(m).__name__
            for m in module.modules()
            if isinstance(getattr(m, "rng_cpu", None), torch.Generator)
            or isinstance(getattr(m, "rng_gpu", None), torch.Generator)
        }
    )
    if offenders:
        raise RuntimeError(
            f"multistep_checkpoint is incompatible with modules carrying private RNG "
            f"generators (found: {offenders}). Their dropout masks are not restored on "
            f"the checkpoint recompute, which would corrupt gradients. Disable "
            f"multistep_checkpoint or make these modules use the global RNG."
        )


class StepWrapper(nn.Module):
    r"""
    Bind a model to its preprocessor.

    Wraps the network so callers can hand it data as the dataloader emits it
    and get predictions back in the same space, without having to replicate
    the preprocessing pipeline at every call site. That is the data space: the
    fields normalized by the dataset statistics, which the dataloader already
    applies and this wrapper never undoes. Each step performs: appending the
    dynamic features and the noise, history normalization, packing the input
    for the network, the model forward, bias correction, and the history
    denormalization of the output.

    ``forward`` takes a :class:`~makani.utils.dataloaders.rollout.Rollout` and
    runs the model over all of its steps; :meth:`steps` hands out the
    predictions one at a time instead, so that an evaluation can release each
    one before the next, rather than keep all lead times. The rollout carries the input
    window, the dynamic features and the noise sample of each step, so the
    preprocessor keeps no state between steps, and the history statistics of a
    step are passed from its normalization to its denormalization. Outside a
    rollout, :meth:`predict` and :meth:`encode` take one window, its dynamic
    features and its noise sample directly, e.g. for the model package.

    Input and output keep the time axis explicit; only the network itself sees
    the time steps folded into its channels.

    Two options control cost and stability of training through a rollout of
    several steps. Both apply only in training mode, so evaluation and
    inference are unaffected by them:

    * **Push-forward mode.** Detaches the input at each step, so gradients do
      not flow through the whole rollout. This trains the model to be robust to
      its own errors without the cost -- or the instability -- of
      backpropagating through many steps.
    * **Rollout checkpointing.** Recomputes each step's forward during the
      backward pass instead of retaining its activations, turning the
      ``O(n_steps)`` activation multiplier of backprop-through-time into
      ``O(1)`` at the cost of one extra forward per step. Only used for
      rollouts of more than one step.

    Parameters
    ----------
    params : ParamsBase
        Configuration. Reads ``multistep.push_forward`` and
        ``multistep_checkpoint``; the rest is forwarded to
        :class:`~makani.models.preprocessor.Preprocessor2D`.
    model_handle : callable
        Zero-argument factory returning the network to wrap.
    """

    def __init__(self, params, model_handle):
        super().__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()
        multistep_parameters = params.get("multistep", {"push_forward": False})
        self.push_forward_mode = multistep_parameters["push_forward"]

        # rollout-level activation checkpointing, set via the --multistep_checkpoint CLI flag (top-level
        # param, like n_future). Off by default. Whether the model is safe to checkpoint is checked on
        # first use, so that loading a model for evaluation never fails on it.
        self.multistep_checkpoint = params.get("multistep_checkpoint", False)
        self._checkpoint_checked = False

    def _prepare_window(self, inp, dynamic_features, noise):
        # the network input of one window, and the history statistics its prediction is denormalized with
        pp = self.preprocessor

        x = pp.assemble_input(inp, dynamic_features, noise)
        stats = pp.compute_history_stats(x)
        inpans = pp.pack_model_input(pp.history_normalize(x, target=False, stats=stats))

        return inpans, stats

    def _predict_window(self, inp, dynamic_features, noise, use_checkpoint: bool = False):
        pp = self.preprocessor

        inpans, stats = self._prepare_window(inp, dynamic_features, noise)

        if use_checkpoint:
            # Only the pure network forward is recomputed in the backward pass; the input preparation
            # stays outside the checkpoint, so it runs once. Global-RNG ops inside the network are
            # handled by preserve_rng_state=True; private-generator modules are rejected up front by
            # _assert_checkpoint_safe.
            predn = checkpoint(self.model, inpans, use_reentrant=False, preserve_rng_state=True)
        else:
            predn = self.model(inpans)

        # the network predicts one step as a flat channel stack; give it its time axis back
        predn = pp.correct_bias(predn).unsqueeze(1)

        return pp.history_denormalize(predn, target=True, stats=stats)

    def compile_step(self, **compile_kwargs):
        r"""
        ``torch.compile`` the prediction of one rollout step.

        A rollout runs its loop, and draws its noise, in Python; what is worth
        compiling is the step: preprocessing, network and postprocessing. The
        compiled step does not depend on the number of steps, and the state
        dict keeps its keys.

        Parameters
        ----------
        **compile_kwargs
            Passed on to ``torch.compile``.
        """
        # layers whose forward breaks the graph, e.g. around distributed collectives, are compiled
        # as frames of their own, once per parameter shape. Models with many such layers of
        # different sizes exceed torch's default limit of 8 and fall back to eager for the rest
        torch._dynamo.config.recompile_limit = int(os.getenv("MAKANI_INDUCTOR_COMPILE_LIMIT", "32"))
        self._predict_window = torch.compile(self._predict_window, **compile_kwargs)

    def predict(self, inp, dynamic_features, noise=None):
        r"""
        Predict one step from a window, without a rollout.

        Parameters
        ----------
        inp : torch.Tensor
            The window, ``(B, n_history + 1, C, H, W)``, in data space.
        dynamic_features : dict
            The dynamic features over the window, name to ``(B, n_history + 1, C_f, H, W)``;
            empty if the model takes none.
        noise : torch.Tensor, optional
            The input noise sample, if the model takes noise, see
            :meth:`~makani.models.preprocessor.Preprocessor2D.noise_source`.

        Returns
        -------
        torch.Tensor
            The prediction, ``(B, 1, C_out, H, W)``, in data space.
        """
        return self._predict_window(inp, dynamic_features, noise)

    def encode(self, inp, dynamic_features, noise=None):
        r"""
        The latent state of the network for one window, before decoding.

        The input goes through the same preparation as in :meth:`predict`; only
        the decoder, the bias correction and the denormalization are skipped.

        Parameters
        ----------
        inp, dynamic_features, noise
            As in :meth:`predict`.

        Returns
        -------
        torch.Tensor
            The latent features, ``(B, embed_dim, h, w)`` on the internal grid of the network.

        Raises
        ------
        NotImplementedError
            If the network does not expose ``encode_process``.
        """
        if not hasattr(self.model, "encode_process"):
            raise NotImplementedError(f"{type(self.model).__name__} does not expose encode_process().")

        inpans, _ = self._prepare_window(inp, dynamic_features, noise)

        return self.model.encode_process(inpans)

    def predict_step(self, step: RolloutStep, use_checkpoint: bool = False):
        r"""
        Predict one step of a rollout.

        Parameters
        ----------
        step : RolloutStep
            The step, as a :class:`~makani.utils.dataloaders.rollout.Rollout` hands it out.
        use_checkpoint : bool, optional
            Recompute the network in the backward pass instead of keeping its
            activations. By default ``False``.

        Returns
        -------
        torch.Tensor
            The prediction, ``(B * E, 1, C_out, H, W)``, in data space, at
            the batch of the rollout with its ``E`` ensemble members folded in.
        """
        return self._predict_window(step.input, step.input_dynamic_features, step.noise, use_checkpoint)

    def forward(self, rollout: Rollout):
        r"""
        Run the model over all steps of a rollout.

        Each prediction is fed back into the window of the next step, see
        :class:`~makani.utils.dataloaders.rollout.Rollout`. In training mode,
        push-forward mode and rollout checkpointing apply if configured. For a
        single step outside a rollout, use :meth:`predict`; for the predictions
        one step at a time, use :meth:`steps`.

        Parameters
        ----------
        rollout : Rollout
            The rollout, not started yet.

        Returns
        -------
        torch.Tensor
            The predictions, ``(B * E, n_steps, C_out, H, W)``, in data
            space, at the batch of the rollout with its ``E`` ensemble members
            folded in; :meth:`~makani.utils.dataloaders.rollout.Rollout.unfold`
            splits them out. Step ``k`` predicts target step ``k``.

        Raises
        ------
        TypeError, ValueError, RuntimeError
            As in :meth:`steps`.
        """
        # (B * E, n_steps, C_out, H, W), step k predicts target step k
        return torch.cat(list(self.steps(rollout)), dim=1)

    def steps(self, rollout: Rollout):
        r"""
        Run the model over the steps of a rollout, one prediction at a time.

        The same rollout as :meth:`forward`, but the predictions are handed out
        as they are made, so that the caller can release each one before the
        next, e.g. in an evaluation over many lead times. Each prediction is
        fed back into the window of the next step before it is handed out, so
        the rollout has finished once the last one is. The rollout is checked
        right away, not at the first step.

        Parameters
        ----------
        rollout : Rollout
            The rollout, not started yet.

        Returns
        -------
        iterator of torch.Tensor
            The prediction of each step, ``(B * E, 1, C_out, H, W)``, in data
            space, at the batch of the rollout with its ``E`` ensemble members
            folded in.

        Raises
        ------
        TypeError
            If ``rollout`` is not a :class:`~makani.utils.dataloaders.rollout.Rollout`.
        ValueError
            If the rollout has no steps.
        RuntimeError
            If the rollout has been started already, or if rollout
            checkpointing is configured for a model that is not safe to
            checkpoint, see :func:`_assert_checkpoint_safe`.
        """
        if not isinstance(rollout, Rollout):
            raise TypeError(
                f"{type(self).__name__} takes a Rollout, got {type(rollout).__name__}; "
                "use predict(x, dynamic_features, noise) for a single step"
            )
        if rollout.n_steps == 0:
            raise ValueError("The rollout has no steps to predict")
        if rollout.started:
            raise RuntimeError("The rollout has been stepped already: a rollout can be run only once, make a new one")
        push_forward = self.training and self.push_forward_mode
        # push-forward already detaches between steps, so checkpointing would be redundant there, and a
        # single step has nothing to gain from it
        use_checkpoint = (
            self.training
            and self.multistep_checkpoint
            and torch.is_grad_enabled()
            and not push_forward
            and rollout.n_steps > 1
        )
        if use_checkpoint and not self._checkpoint_checked:
            _assert_checkpoint_safe(self.model)
            self._checkpoint_checked = True

        return self._steps(rollout, push_forward, use_checkpoint)

    def _steps(self, rollout, push_forward, use_checkpoint):
        for step in rollout:
            pred = self.predict_step(step, use_checkpoint=use_checkpoint)
            rollout.advance(pred.detach() if push_forward else pred)
            yield pred
