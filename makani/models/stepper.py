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


class _RolloutMixin:
    r"""
    The stateless rollout shared by the wrappers.

    Given a :class:`~makani.utils.dataloaders.rollout.Rollout`, the wrappers
    run the model over all of its steps. The rollout carries the input window,
    the dynamic features and the noise sample of each step, so the
    preprocessor keeps no state between steps: no feature caches, and the
    history statistics of a step are passed from its normalization to its
    denormalization rather than stored.

    Outside a rollout, :meth:`predict` and :meth:`encode` take one window, its
    dynamic features and its noise sample directly, e.g. for the model package.

    Expects ``self.preprocessor`` and ``self.model``. ``_rollout_push_forward``
    and ``_rollout_checkpoint`` are the options of a training rollout;
    :class:`MultiStepWrapper` sets them from the configuration.
    """

    _rollout_push_forward = False
    _rollout_checkpoint = False

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
            # see MultiStepWrapper._forward_train: only the network is recomputed
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
        self._predict_window = torch.compile(self._predict_window, **compile_kwargs)

    def predict(self, inp, dynamic_features, noise=None):
        r"""
        Predict one step from a window, without a rollout.

        Parameters
        ----------
        inp : torch.Tensor
            The window, ``(B, n_history + 1, C, H, W)``, in physical units.
        dynamic_features : dict
            The dynamic features over the window, name to ``(B, n_history + 1, C_f, H, W)``;
            empty if the model takes none.
        noise : torch.Tensor, optional
            The input noise sample, if the model takes noise, see
            :meth:`~makani.models.preprocessor.Preprocessor2D.noise_source`.

        Returns
        -------
        torch.Tensor
            The prediction, ``(B, 1, C_out, H, W)``, in physical units.
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
            The prediction, ``(B, 1, C_out, H, W)``, in physical units.
        """
        return self._predict_window(step.input, step.input_dynamic_features, step.noise, use_checkpoint)

    def _forward_rollout(self, rollout: Rollout):
        if rollout.n_steps == 0:
            raise ValueError("The rollout has no steps to predict")
        if rollout.started:
            raise RuntimeError("The rollout has been stepped already: a rollout can be run only once, make a new one")
        push_forward = self.training and self._rollout_push_forward
        # push-forward already detaches between steps, so checkpointing would be redundant there
        use_checkpoint = self.training and self._rollout_checkpoint and torch.is_grad_enabled() and not push_forward

        preds = []
        for step in rollout:
            pred = self.predict_step(step, use_checkpoint=use_checkpoint)
            preds.append(pred)
            rollout.advance(pred.detach() if push_forward else pred)

        # (B, n_steps, C_out, H, W), aligned with the targets
        return torch.cat(preds, dim=1)


class SingleStepWrapper(_RolloutMixin, nn.Module):
    r"""
    Bind a model to its preprocessor for a single forecast step.

    Wraps the network so callers can hand it raw data and get a prediction back
    in physical units, without having to replicate the preprocessing pipeline at
    every call site. One ``forward`` performs: state update, appending the
    dynamic features, history normalization, packing the input for the network,
    the model forward, bias correction, and denormalization of the output.

    Input and output keep the time axis explicit; only the network itself sees
    the time steps folded into its channels.

    Parameters
    ----------
    params : ParamsBase
        Configuration, forwarded to :class:`~makani.models.preprocessor.Preprocessor2D`.
    model_handle : callable
        Zero-argument factory returning the network to wrap.

    See Also
    --------
    MultiStepWrapper : the autoregressive multi-step counterpart.
    """

    def __init__(self, params, model_handle):
        super().__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()

    def _preprocess(self, inp, update_state=True, replace_state=True):
        """Run the shared input pipeline for one step, up to the packed network input.

        The stochastic state is sized to THIS call's batch. The noise module is
        constructed with ``params.batch_size`` -- a training setting, and 1 for a
        packaged model -- but inference legitimately drives it with other
        batches, e.g. an ensemble pushing B*E members through as a single
        batched forward. Without this, ``append_dynamic_features`` rejects
        the call with "batch mismatch between input_noise state (1) and input".
        ``_ensure_state`` reallocates only when the batch actually changes, so
        the steady-state cost is zero; the flip side is that changing the batch
        mid-rollout discards a stateful noise module's history.
        """
        # update internal state
        if update_state:
            self.preprocessor.update_internal_state(replace_state=replace_state, batch_size=inp.shape[0])

        return _prepare_step(self.preprocessor, inp)

    def forward(self, inp, update_state=True, replace_state=True):
        r"""
        Predict a single step from raw input.

        Parameters
        ----------
        inp : torch.Tensor or Rollout
            Input of shape ``(B, n_history + 1, C, H, W)`` in physical units.
            Given a :class:`~makani.utils.dataloaders.rollout.Rollout`, the
            model is run over all of its steps instead, see :class:`_RolloutMixin`;
            the state arguments do not apply then, the rollout drives the noise.
        update_state : bool, optional
            Advance the stochastic noise state before the forward pass, by
            default ``True``.
        replace_state : bool, optional
            Draw a fresh noise state rather than taking one autoregressive
            step, by default ``True``. Required if the batch size changed since
            the last call and the noise is stateful.

        Returns
        -------
        torch.Tensor
            Prediction of shape ``(B, 1, C_out, H, W)``, denormalized back to
            physical units; ``(B, n_steps, C_out, H, W)`` for a rollout.
        """
        if isinstance(inp, Rollout):
            return self._forward_rollout(inp)

        inpans = self._preprocess(inp, update_state=update_state, replace_state=replace_state)

        return _finish_step(self.preprocessor, self.model(inpans))

    def encode_process(self, inp, update_state=True, replace_state=True):
        r"""Prepare one input and return backbone features before decoding.

        Shares :meth:`_preprocess` with :meth:`forward`, so the preprocessing is
        identical by construction; only the decoder, bias correction, and target
        denormalization are skipped.

        Parameters
        ----------
        inp : torch.Tensor
            Input of shape ``(B, n_history + 1, C, H, W)`` in physical units.
        update_state : bool, optional
            Advance the stochastic noise state first, by default ``True``.
        replace_state : bool, optional
            Draw a fresh noise state rather than stepping, by default ``True``.

        Returns
        -------
        torch.Tensor
            Latent features of shape ``(B, embed_dim, h, w)`` on the model's
            internal grid.

        Raises
        ------
        NotImplementedError
            If the wrapped model does not expose ``encode_process``.
        """
        if not hasattr(self.model, "encode_process"):
            raise NotImplementedError(f"{type(self.model).__name__} does not expose encode_process().")

        inpans = self._preprocess(inp, update_state=update_state, replace_state=replace_state)

        return self.model.encode_process(inpans)


def _prepare_step(preprocessor, inp):
    """Input pipeline of one step: dynamic features, history normalization, packing.

    Takes ``(B, n_history + 1, C, H, W)`` and returns the flat channel stack
    the network takes.
    """
    torch._check(
        inp.dim() == 5,
        lambda: f"expected an input with an explicit time axis, (B, n_history + 1, C, H, W), got shape {tuple(inp.shape)}",
    )

    inpa = preprocessor.append_dynamic_features(inp)

    preprocessor.history_compute_stats(inpa)
    inpan = preprocessor.history_normalize(inpa, target=False)

    return preprocessor.pack_model_input(inpan)


def _finish_step(preprocessor, predn):
    """Output pipeline of one step: bias correction, time axis, denormalization.

    The network predicts one step as a flat channel stack, ``(B, C_out, H, W)``;
    it gets its time axis back here, ``(B, 1, C_out, H, W)``. Denormalize right
    away, before the statistics move on with the next step.
    """
    predn = preprocessor.correct_bias(predn).unsqueeze(1)

    return preprocessor.history_denormalize(predn, target=True)


class MultiStepWrapper(_RolloutMixin, nn.Module):
    r"""
    Bind a model to its preprocessor for an autoregressive rollout.

    Runs the wrapped model for ``n_future + 1`` steps, feeding each prediction
    back in as the next input and sliding the history window forward. Returns
    all steps stacked along the time axis, which is what multi-step training
    losses need.

    Two options control cost and stability:

    * **Push-forward mode.** Detaches the input at each step, so gradients do
      not flow through the whole rollout. This trains the model to be robust to
      its own errors without the cost -- or the instability -- of
      backpropagating through many steps.
    * **Rollout checkpointing.** Recomputes each step's forward during the
      backward pass instead of retaining its activations, turning the
      ``O(n_future)`` activation multiplier of backprop-through-time into
      ``O(1)`` at the cost of one extra forward per step.

    Parameters
    ----------
    params : ParamsBase
        Configuration. Reads ``n_future``, ``multistep.push_forward``, and
        ``multistep_checkpoint``; the rest is forwarded to the preprocessor.
    model_handle : callable
        Zero-argument factory returning the network to wrap.

    See Also
    --------
    SingleStepWrapper : the single-step counterpart.
    """

    def __init__(self, params, model_handle):
        super().__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()
        multistep_parameters = params.get("multistep", {"push_forward": False})
        self.push_forward_mode = multistep_parameters["push_forward"]

        # rollout-level activation checkpointing: recompute each step's model forward in
        # the backward pass instead of retaining its full activation graph. Turns the
        # O(n_future) activation multiplier of backprop-through-time back into O(1) at the
        # cost of one extra forward per step. Set via the --multistep_checkpoint CLI flag
        # (top-level param, like n_future). Off by default.
        self.multistep_checkpoint = params.get("multistep_checkpoint", False)
        if self.multistep_checkpoint:
            _assert_checkpoint_safe(self.model)

        # collect parameters for history
        self.n_future = params.n_future

        # the same options for a rollout
        self._rollout_push_forward = self.push_forward_mode
        self._rollout_checkpoint = self.multistep_checkpoint

    def _forward_train(self, inp, update_state=True, replace_state=True):
        result = []
        inpt = inp

        # initialize fresh buffer: decide whether we want to replace the state
        if update_state:
            self.preprocessor.update_internal_state(replace_state=replace_state)

        # do the rollout
        for step in range(self.n_future + 1):

            # in push-forward mode, we need to detach the tensor:
            if self.push_forward_mode:
                inpt = inpt.detach()

            inpans = _prepare_step(self.preprocessor, inpt)

            # prediction
            # Only the pure model forward is checkpointed; the stateful preprocessor calls
            # (history/noise/state updates) stay outside the checkpoint so they run once and
            # are not re-executed during the backward recompute. Global-RNG ops inside the
            # model are handled by preserve_rng_state=True; private-generator modules are
            # rejected up-front by _assert_checkpoint_safe. Push-forward mode already detaches
            # between steps, so checkpointing would be redundant there.
            if self.multistep_checkpoint and self.training and torch.is_grad_enabled() and not self.push_forward_mode:
                predn = checkpoint(self.model, inpans, use_reentrant=False, preserve_rng_state=True)
            else:
                predn = self.model(inpans)

            # denormalize here, while the statistics still belong to this step
            pred = _finish_step(self.preprocessor, predn)

            # append output
            result.append(pred)

            if step == self.n_future:
                break

            # update internal buffer
            self.preprocessor.update_internal_state(replace_state=False)

            # append history
            inpt = self.preprocessor.append_history(inpt, pred, step)

        # (B, n_future + 1, C_out, H, W), aligned with the target
        return torch.cat(result, dim=1)

    def _forward_eval(self, inp, update_state=True, replace_state=True):
        # update internal state; size the noise state to this call's batch so that
        # batched inference works regardless of params.batch_size. See
        # SingleStepWrapper._preprocess for the rationale. The training path
        # deliberately keeps the fixed-batch behaviour.
        if update_state:
            self.preprocessor.update_internal_state(replace_state=replace_state, batch_size=inp.shape[0])

        inpans = _prepare_step(self.preprocessor, inp)

        return _finish_step(self.preprocessor, self.model(inpans))

    def forward(self, inp, update_state=True, replace_state=True):
        r"""
        Roll the model forward autoregressively and return all predicted steps.

        Dispatches to the training or evaluation path based on the module's
        training mode; the training path additionally honors push-forward mode
        and rollout checkpointing, which do not apply under ``no_grad``. The
        evaluation path predicts one step; the caller drives the rollout.

        Given a :class:`~makani.utils.dataloaders.rollout.Rollout` instead of
        a tensor, the model is run over all steps of the rollout, in training
        and evaluation mode alike, see :class:`_RolloutMixin`. Push-forward
        mode and rollout checkpointing apply in training mode.

        Parameters
        ----------
        inp : torch.Tensor or Rollout
            Initial input of shape ``(B, n_history + 1, C, H, W)`` in physical
            units, or a rollout; the state arguments do not apply to a rollout,
            it drives the noise itself.
        update_state : bool, optional
            Advance the stochastic noise state before the rollout, by default
            ``True``.
        replace_state : bool, optional
            Draw a fresh noise state rather than continuing an existing
            trajectory, by default ``True``.

        Returns
        -------
        torch.Tensor
            Predictions of shape ``(B, n_future + 1, C_out, H, W)`` in training
            mode and ``(B, 1, C_out, H, W)`` in evaluation mode;
            ``(B, n_steps, C_out, H, W)`` for a rollout.
        """
        if isinstance(inp, Rollout):
            return self._forward_rollout(inp)

        # decide which routine to call
        if self.training:
            y = self._forward_train(inp, update_state=update_state, replace_state=replace_state)
        else:
            y = self._forward_eval(inp, update_state=update_state, replace_state=replace_state)

        return y
