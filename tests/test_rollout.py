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

import os
import shutil
import sys
import unittest

import torch
import torch.nn as nn
from parameterized import parameterized

from makani.models.stepper import StepWrapper
from makani.utils.dataloaders.batch import (
    INPUT,
    TARGET,
    INPUT_DYNAMIC_FEATURES,
    TARGET_DYNAMIC_FEATURES,
    INPUT_TIMESTAMP,
    TARGET_TIMESTAMP,
    ZENITH,
    make_batch,
)
from makani.utils.dataloaders.rollout import Rollout
from makani.utils.functions import expand_ensemble

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import get_default_parameters, set_seed, compare_tensors, disable_tf32

_B, _C, _H, _W = 2, 3, 4, 8


def _batch(n_history=1, n_future=2, with_target=True, channels=_C, grid=(_H, _W), dtype=torch.float32):
    t_in, t_out = n_history + 1, n_future + 1
    times_in = torch.arange(_B * t_in, dtype=torch.float64).reshape(_B, t_in)
    times_out = 100.0 + torch.arange(_B * t_out, dtype=torch.float64).reshape(_B, t_out)
    return make_batch(
        inp=torch.randn(_B, t_in, channels, *grid, dtype=dtype),
        inp_timestamp=times_in,
        tar=torch.randn(_B, t_out, channels, *grid, dtype=dtype) if with_target else None,
        tar_timestamp=times_out if with_target else None,
        inp_zenith=torch.randn(_B, t_in, 1, *grid, dtype=dtype),
        tar_zenith=torch.randn(_B, t_out, 1, *grid, dtype=dtype) if with_target else None,
    )


class _RecordingNoise:
    """A noise source that records how it is driven and returns a distinct sample per step."""

    def __init__(self):
        self.updates = []
        self.restored = []
        self.version = 0

    def update(self, replace_state=False):
        self.updates.append(replace_state)
        self.version += 1

    def __call__(self):
        return torch.full((_B, 1, 1, _H, _W), float(len(self.updates)))

    def get_state(self):
        return ("state", self.version)

    def set_state(self, state):
        self.restored.append(state)
        self.version += 1


class TestRollout(unittest.TestCase):
    """The bookkeeping of a rollout, without a model."""

    def setUp(self):
        set_seed(333)

    def test_windows_slide_with_the_predictions(self):
        batch = _batch(n_history=1, n_future=2)
        rollout = Rollout(batch)
        preds = [torch.randn(_B, 1, _C, _H, _W) for _ in range(3)]

        expected_input = batch[INPUT]
        expected_zenith = batch[INPUT_DYNAMIC_FEATURES][ZENITH]
        expected_time = batch[INPUT_TIMESTAMP]
        for k, step in enumerate(rollout):
            with self.subTest(step=k):
                self.assertEqual(step.index, k)
                self.assertTrue(torch.equal(step.input, expected_input))
                self.assertTrue(torch.equal(step.input_dynamic_features[ZENITH], expected_zenith))
                self.assertTrue(torch.equal(step.input_timestamp, expected_time))
                self.assertTrue(torch.equal(step.target, batch[TARGET][:, k : k + 1]))
                self.assertTrue(
                    torch.equal(
                        step.target_dynamic_features[ZENITH], batch[TARGET_DYNAMIC_FEATURES][ZENITH][:, k : k + 1]
                    )
                )
                self.assertTrue(torch.equal(step.target_timestamp, batch[TARGET_TIMESTAMP][:, k : k + 1]))
                self.assertIsNone(step.noise)

            rollout.advance(preds[k])

            # the window drops its oldest step and moves on to the predicted time
            expected_input = torch.cat([expected_input[:, 1:], preds[k]], dim=1)
            expected_zenith = torch.cat(
                [expected_zenith[:, 1:], batch[TARGET_DYNAMIC_FEATURES][ZENITH][:, k : k + 1]], dim=1
            )
            expected_time = torch.cat([expected_time[:, 1:], batch[TARGET_TIMESTAMP][:, k : k + 1]], dim=1)

        self.assertEqual(len(rollout), 3)

    def test_mesh_grid(self):
        # the grid only trails, so a mesh batch rolls out the same way
        batch = _batch(n_history=0, n_future=1, grid=(_H * _W,))
        rollout = Rollout(batch)
        next(rollout)
        pred = torch.randn(_B, 1, _C, _H * _W)
        rollout.advance(pred)
        self.assertTrue(torch.equal(next(rollout).input, pred))

    def test_next_without_advance_is_refused(self):
        rollout = Rollout(_batch())
        next(rollout)
        with self.assertRaises(RuntimeError):
            next(rollout)

    def test_advance_without_a_step_is_refused(self):
        rollout = Rollout(_batch())
        with self.assertRaises(RuntimeError):
            rollout.advance(torch.randn(_B, 1, _C, _H, _W))

    def test_prediction_of_the_wrong_shape_is_refused(self):
        rollout = Rollout(_batch())
        next(rollout)
        with self.assertRaises(ValueError):
            rollout.advance(torch.randn(_B, 2, _C, _H, _W))

    def test_only_the_last_prediction_may_have_other_channels(self):
        # a model with other outputs than inputs predicts a single step: nothing is fed back
        for with_target in [True, False]:
            with self.subTest(with_target=with_target):
                rollout = Rollout(_batch(n_future=0, with_target=with_target), n_steps=1)
                next(rollout)
                rollout.advance(torch.randn(_B, 1, _C + 2, _H, _W))
                with self.assertRaises(StopIteration):
                    next(rollout)

        # the rest of its shape is still checked
        rollout = Rollout(_batch(n_future=0))
        next(rollout)
        with self.assertRaises(ValueError):
            rollout.advance(torch.randn(_B, 1, _C + 2, _H, _W + 1))

        # a prediction that goes into the next window has to fit it
        rollout = Rollout(_batch(n_future=1))
        next(rollout)
        with self.assertRaises(ValueError):
            rollout.advance(torch.randn(_B, 1, _C + 2, _H, _W))

    def test_started(self):
        rollout = Rollout(_batch(n_future=1))
        self.assertFalse(rollout.started)
        next(rollout)
        self.assertTrue(rollout.started)
        rollout.advance(torch.randn(_B, 1, _C, _H, _W))
        self.assertTrue(rollout.started)

    def test_number_of_steps(self):
        self.assertEqual(len(Rollout(_batch(n_future=3))), 4)
        self.assertEqual(len(Rollout(_batch(n_future=3), n_steps=2)), 2)
        with self.assertRaises(ValueError):
            Rollout(_batch(n_future=1), n_steps=3)

    def test_without_targets(self):
        batch = _batch(with_target=False)
        with self.assertRaises(ValueError):
            Rollout(batch)

        step = next(Rollout(batch, n_steps=1))
        self.assertIsNone(step.target)
        self.assertEqual(step.target_dynamic_features, {})

    def test_without_targets_the_caller_supplies_the_next_step(self):
        # how the inferencer rolls out: the verification samples arrive one step at a time, and with
        # them the features and the time of the predicted step. That is the rollout of the batch with
        # those targets.
        E = 2
        batch = _batch(n_history=1, n_future=2)
        streamed = Rollout(_take(batch, None, without_target=True), n_steps=3, ensemble_size=E)
        reference = Rollout(batch, ensemble_size=E)

        for k, (step, ref) in enumerate(zip(streamed, reference)):
            with self.subTest(step=k):
                self.assertTrue(torch.equal(step.input, ref.input))
                self.assertTrue(torch.equal(step.input_dynamic_features[ZENITH], ref.input_dynamic_features[ZENITH]))
                self.assertTrue(torch.equal(step.input_timestamp, ref.input_timestamp))
                self.assertIsNone(step.target)

            pred = torch.randn(_B * E, 1, _C, _H, _W)
            reference.advance(pred)
            streamed.advance(
                pred,
                dynamic_features={ZENITH: batch[TARGET_DYNAMIC_FEATURES][ZENITH][:, k : k + 1]},
                timestamp=batch[TARGET_TIMESTAMP][:, k : k + 1],
            )

    def test_without_targets_the_next_step_is_checked(self):
        batch = _take(_batch(n_history=1, n_future=2), None, without_target=True)
        pred = torch.randn(_B, 1, _C, _H, _W)
        zenith = torch.randn(_B, 1, 1, _H, _W)
        time = torch.zeros(_B, 1, dtype=torch.float64)

        cases = {
            "missing": {},
            "no timestamp": {"dynamic_features": {ZENITH: zenith}},
            "other features": {"dynamic_features": {"other": zenith}, "timestamp": time},
            "two steps": {"dynamic_features": {ZENITH: torch.cat([zenith, zenith], 1)}, "timestamp": time},
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                rollout = Rollout(batch, n_steps=2)
                next(rollout)
                with self.assertRaises(ValueError):
                    rollout.advance(pred, **kwargs)

        # nothing is needed after the last step
        rollout = Rollout(batch, n_steps=1)
        next(rollout)
        rollout.advance(pred)

        # with targets, the next step comes from the batch
        rollout = Rollout(_batch())
        next(rollout)
        with self.assertRaises(ValueError):
            rollout.advance(pred, dynamic_features={ZENITH: zenith}, timestamp=time)

    def test_noise_is_replaced_first_and_continued_after(self):
        noise = _RecordingNoise()
        rollout = Rollout(_batch(n_future=2), noise_source=noise)
        samples = []
        for step in rollout:
            samples.append(step.noise[0, 0, 0, 0, 0].item())
            rollout.advance(torch.zeros(_B, 1, _C, _H, _W))

        # one update per step, the first one replacing the state, and the sample drawn after it
        self.assertEqual(noise.updates, [True, False, False])
        self.assertEqual(samples, [1.0, 2.0, 3.0])

    def test_noise_continues_from_a_saved_state(self):
        noise = _RecordingNoise()
        rollout = Rollout(_batch(n_future=1), noise_source=noise)
        for step in rollout:
            rollout.advance(torch.zeros(_B, 1, _C, _H, _W))
        state = rollout.noise_state()

        rollout = Rollout(_batch(n_future=1), noise_source=noise, noise_state=state)
        for step in rollout:
            rollout.advance(torch.zeros(_B, 1, _C, _H, _W))

        # the second rollout restores the state and continues it from its first step on
        self.assertEqual(noise.restored, [state])
        self.assertEqual(noise.updates, [True, False, False, False])

    def test_noise_state_needs_a_noise_source(self):
        with self.assertRaises(ValueError):
            Rollout(_batch(), noise_state=("state", 0))
        rollout = Rollout(_batch(n_future=0))
        next(rollout)
        rollout.advance(torch.zeros(_B, 1, _C, _H, _W))
        with self.assertRaises(ValueError):
            rollout.noise_state()

    def test_noise_state_only_of_a_finished_rollout(self):
        noise = _RecordingNoise()
        rollout = Rollout(_batch(n_future=1), noise_source=noise)
        with self.assertRaisesRegex(RuntimeError, "finished"):
            rollout.noise_state()
        next(rollout)
        with self.assertRaisesRegex(RuntimeError, "finished"):
            rollout.noise_state()
        rollout.advance(torch.zeros(_B, 1, _C, _H, _W))
        with self.assertRaisesRegex(RuntimeError, "finished"):
            rollout.noise_state()

    def test_noise_state_refuses_a_noise_advanced_since(self):
        noise = _RecordingNoise()
        rollout = Rollout(_batch(n_future=0), noise_source=noise)
        for step in rollout:
            rollout.advance(torch.zeros(_B, 1, _C, _H, _W))

        # another rollout draws from the same noise in between
        other = Rollout(_batch(n_future=0), noise_source=noise)
        next(other)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            rollout.noise_state()

    def test_ensemble_folds_what_the_model_sees(self):
        # with an ensemble, the rollout is that of the batch with every member a copy of its sample,
        # except for the targets, which stay at the batch of the data
        E = 3
        batch = _batch(n_history=1, n_future=2)
        rollout = Rollout(batch, ensemble_size=E)
        reference = Rollout(
            make_batch(
                inp=expand_ensemble(batch[INPUT], E),
                inp_timestamp=expand_ensemble(batch[INPUT_TIMESTAMP], E),
                tar=expand_ensemble(batch[TARGET], E),
                tar_timestamp=expand_ensemble(batch[TARGET_TIMESTAMP], E),
                inp_zenith=expand_ensemble(batch[INPUT_DYNAMIC_FEATURES][ZENITH], E),
                tar_zenith=expand_ensemble(batch[TARGET_DYNAMIC_FEATURES][ZENITH], E),
            )
        )

        self.assertEqual(rollout.ensemble_size, E)
        for k, (step, ref) in enumerate(zip(rollout, reference)):
            with self.subTest(step=k):
                self.assertEqual(step.input.shape[0], _B * E)
                self.assertTrue(torch.equal(step.input, ref.input))
                self.assertTrue(torch.equal(step.input_dynamic_features[ZENITH], ref.input_dynamic_features[ZENITH]))
                self.assertTrue(torch.equal(step.input_timestamp, ref.input_timestamp))
                self.assertTrue(torch.equal(step.target_dynamic_features[ZENITH], ref.target_dynamic_features[ZENITH]))
                self.assertTrue(torch.equal(step.target_timestamp, ref.target_timestamp))
                self.assertTrue(torch.equal(step.target, batch[TARGET][:, k : k + 1]))

            # predictions come at the batch of the model, and differ between the members
            pred = torch.randn(_B * E, 1, _C, _H, _W)
            rollout.advance(pred)
            reference.advance(pred)

    def test_unfold_splits_the_members_out(self):
        E = 3
        rollout = Rollout(_batch(), ensemble_size=E)
        x = torch.randn(_B, _C, _H, _W)
        unfolded = rollout.unfold(expand_ensemble(x, E))
        self.assertEqual(tuple(unfolded.shape), (_B, E, _C, _H, _W))
        for e in range(E):
            self.assertTrue(torch.equal(unfolded[:, e], x))

        with self.assertRaises(ValueError):
            rollout.unfold(torch.randn(_B * E + 1, _C))
        with self.assertRaises(ValueError):
            Rollout(_batch(), ensemble_size=0)

    def test_ensemble_predictions_of_the_wrong_batch_are_refused(self):
        rollout = Rollout(_batch(), ensemble_size=2)
        next(rollout)
        with self.assertRaises(ValueError):
            rollout.advance(torch.randn(_B, 1, _C, _H, _W))


# ---------------------------------------------------------------------------
# A rollout reproduces a loop over the steps written out by hand
# ---------------------------------------------------------------------------
#
# The reference spells out what a rollout does, without Rollout: it drives the
# noise module itself, replacing the state at the first step and continuing it
# after, slides the input window and the zenith angle by hand, and runs each step
# through the preprocessor and the network. The wrappers driven with a Rollout
# must agree with it in the predictions, the loss and every gradient, the noise
# parameters included when the noise is trainable.

_NOISE = {
    "none": None,
    "white": {"type": "white", "mode": "concatenate", "n_channels": 1},
    "white_learnable": {"type": "white", "mode": "concatenate", "n_channels": 1, "learnable": True},
    "diffusion": {"type": "diffusion", "mode": "concatenate", "n_channels": 1},
    # learnable diffusion noise supports a single time step only
    "diffusion_learnable": {"type": "diffusion", "mode": "concatenate", "n_channels": 1, "learnable": True},
}

_CASES = [
    (name, n_history, push_forward)
    for name, n_history in [
        ("none", 1),
        ("white", 1),
        ("white_learnable", 1),
        ("diffusion", 1),
        ("diffusion_learnable", 0),
    ]
    for push_forward in (False, True)
]

_N_FUTURE = 2
_STEP_WEIGHTS = torch.tensor([0.2, 0.3, 0.5])


class _TinyNet(nn.Module):
    """A small trainable network, nonlinear so that the rollout gradients do not factor trivially."""

    def __init__(self, in_chans, out_chans, hidden=8):
        super().__init__()
        self.fc1 = nn.Conv2d(in_chans, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, out_chans, 1)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def _params(n_history, push_forward, noise):
    params = get_default_parameters()
    params.img_shape_x = params.img_shape_x_resampled = _H * 4
    params.img_shape_y = params.img_shape_y_resampled = _W * 4
    params.n_history = n_history
    params.n_future = _N_FUTURE
    params.multistep = {"push_forward": push_forward}
    params.add_zenith = True
    params.history_normalization_mode = "mean"
    params.dhours = 6
    params.batch_size = _B
    params.in_channels = params.out_channels = list(range(_C))
    params.channel_names = params.channel_names[:_C]
    params.N_in_channels = params.N_out_channels = _C
    if noise is not None:
        params.input_noise = noise
    return params


def _make_wrapper(params, noise):
    n_noise = noise["n_channels"] if noise is not None else 0
    in_chans = (params.n_history + 1) * (_C + 1 + n_noise)
    # the same seed for every wrapper: the same weights, and the noise is seeded from its own fixed seed
    set_seed(333)
    wrapper = StepWrapper(params, lambda: _TinyNet(in_chans, _C))
    wrapper.train()
    return wrapper


def _rollout(wrapper, batch, **kwargs):
    """A rollout of the batch, with the input noise of the wrapper at the batch of the model."""
    batch_size = batch[INPUT].shape[0] * kwargs.get("ensemble_size", 1)
    return Rollout(batch, noise_source=wrapper.preprocessor.noise_source(batch_size=batch_size), **kwargs)


def _expand_batch(batch, ensemble_size):
    """The batch with every sample repeated for each member, batch-major, as the rollout folds it."""
    return {
        key: (_expand_batch(value, ensemble_size) if isinstance(value, dict) else expand_ensemble(value, ensemble_size))
        for key, value in batch.items()
    }


def _stepwise(wrapper, batch, push_forward=False):
    """The rollout of the batch, written out step by step; see the comment above."""
    pp = wrapper.preprocessor
    inp = batch[INPUT]
    zenith = batch[INPUT_DYNAMIC_FEATURES][ZENITH]
    target_zenith = batch[TARGET_DYNAMIC_FEATURES][ZENITH]

    preds = []
    for k in range(batch[TARGET].shape[1]):
        sample = None
        if hasattr(pp, "input_noise"):
            pp.update_internal_state(replace_state=(k == 0), batch_size=inp.shape[0])
            sample = pp.input_noise()

        x = pp.assemble_input(inp, {ZENITH: zenith}, sample)
        stats = pp.compute_history_stats(x)
        predn = wrapper.model(pp.pack_model_input(pp.history_normalize(x, stats)))
        pred = pp.history_denormalize(pp.correct_bias(predn).unsqueeze(1), stats, target=True)
        preds.append(pred)

        # the window drops its oldest step and takes the prediction, the zenith angle moves on with it
        inp = torch.cat([inp[:, 1:], pred.detach() if push_forward else pred], dim=1)
        zenith = torch.cat([zenith[:, 1:], target_zenith[:, k : k + 1]], dim=1)

    return torch.cat(preds, dim=1)


def _loss(pred, tar):
    # the per-step losses, weighted by step, as the multistep weighting does
    per_step = torch.square(pred - tar).mean(dim=(0, 2, 3, 4))
    return torch.sum(_STEP_WEIGHTS.to(per_step.device) * per_step)


def _gradients(module):
    return {name: p.grad.detach().clone() for name, p in module.named_parameters() if p.grad is not None}


def _scaled_rtol_match(msg, actual, expected):
    return compare_tensors(msg, actual, expected, atol=1e-6 * max(1.0, expected.abs().max().item()), rtol=1e-5)


class TestRolloutMatchesTheStepwiseLoop(unittest.TestCase):

    def setUp(self):
        disable_tf32()

    def _reference(self, params, noise, batch):
        wrapper = _make_wrapper(params, noise)
        pred = _stepwise(wrapper, batch, push_forward=params.multistep["push_forward"])
        loss = _loss(pred, batch[TARGET])
        loss.backward()
        return pred.detach(), loss.detach(), _gradients(wrapper)

    def _compare(self, label, pred, loss, grads, ref):
        ref_pred, ref_loss, ref_grads = ref
        with self.subTest(desc=f"{label}: predictions"):
            self.assertTrue(_scaled_rtol_match("predictions", pred, ref_pred))
        with self.subTest(desc=f"{label}: loss"):
            self.assertTrue(_scaled_rtol_match("loss", loss, ref_loss))
        with self.subTest(desc=f"{label}: same parameters"):
            self.assertEqual(set(grads), set(ref_grads))
        for name in ref_grads:
            with self.subTest(desc=f"{label}: gradient {name}"):
                self.assertTrue(_scaled_rtol_match(name, grads[name], ref_grads[name]))

    @parameterized.expand(_CASES)
    def test_loss_on_the_stacked_predictions(self, noise_name, n_history, push_forward):
        noise = _NOISE[noise_name]
        params = _params(n_history, push_forward, noise)
        set_seed(1)
        batch = _batch(n_history=n_history, n_future=_N_FUTURE, grid=(_H * 4, _W * 4))

        ref = self._reference(params, noise, batch)
        if noise is not None and noise.get("learnable", False):
            self.assertTrue(any(".input_noise." in name for name in ref[2]), "the noise should be trainable")

        wrapper = _make_wrapper(params, noise)
        pred = wrapper(_rollout(wrapper, batch))
        loss = _loss(pred, batch[TARGET])
        loss.backward()

        self._compare("stacked", pred.detach(), loss.detach(), _gradients(wrapper), ref)

    @parameterized.expand(_CASES)
    def test_loss_summed_over_the_steps(self, noise_name, n_history, push_forward):
        # the weighted per-step losses summed in the loop, with a single backward at the end
        noise = _NOISE[noise_name]
        params = _params(n_history, push_forward, noise)
        set_seed(1)
        batch = _batch(n_history=n_history, n_future=_N_FUTURE, grid=(_H * 4, _W * 4))

        ref = self._reference(params, noise, batch)

        wrapper = _make_wrapper(params, noise)
        rollout = _rollout(wrapper, batch)
        preds = []
        loss = torch.zeros(())
        for step in rollout:
            pred = wrapper.predict_step(step)
            preds.append(pred)
            loss = loss + _STEP_WEIGHTS[step.index] * torch.square(pred - step.target).mean()
            rollout.advance(pred.detach() if push_forward else pred)
        loss.backward()

        self._compare("summed", torch.cat(preds, dim=1).detach(), loss.detach(), _gradients(wrapper), ref)


def _take(batch, n, without_target=False):
    """The first n samples of a batch, all of them for ``None``; without the target side if asked."""
    target_keys = (TARGET, TARGET_DYNAMIC_FEATURES, TARGET_TIMESTAMP) if without_target else ()
    return {
        key: (_take(value, n) if isinstance(value, dict) else value[:n])
        for key, value in batch.items()
        if key not in target_keys
    }


class TestWrapperRollout(unittest.TestCase):
    """The rollout path of the wrappers: evaluation, single steps, checkpointing, ensembles and the noise."""

    def setUp(self):
        disable_tf32()

    def _batch(self, n_history=1, n_future=_N_FUTURE):
        set_seed(1)
        return _batch(n_history=n_history, n_future=n_future, grid=(_H * 4, _W * 4))

    def test_evaluation_matches_the_stepwise_loop(self):
        params = _params(1, False, None)
        batch = self._batch()
        wrapper = _make_wrapper(params, None)
        wrapper.eval()

        with torch.no_grad():
            ref = _stepwise(wrapper, batch)
            pred = wrapper(_rollout(wrapper, batch))

        self.assertEqual(tuple(pred.shape), tuple(batch[TARGET].shape))
        self.assertTrue(_scaled_rtol_match("predictions", pred, ref))

    def test_a_single_step_is_what_predict_computes(self):
        # with n_future = 0 the rollout has one step, which is what predict computes
        noise = _NOISE["white_learnable"]
        params = _params(1, False, noise)
        params.n_future = 0
        batch = self._batch(n_future=0)

        ref_wrapper = _make_wrapper(params, noise)
        ref_pp = ref_wrapper.preprocessor
        ref_pp.update_internal_state(replace_state=True, batch_size=_B)
        ref = ref_wrapper.predict(batch[INPUT], batch[INPUT_DYNAMIC_FEATURES], ref_pp.input_noise())
        torch.square(ref - batch[TARGET]).mean().backward()

        wrapper = _make_wrapper(params, noise)
        pred = wrapper(_rollout(wrapper, batch))
        torch.square(pred - batch[TARGET]).mean().backward()

        self.assertTrue(_scaled_rtol_match("predictions", pred.detach(), ref.detach()))
        ref_grads, grads = _gradients(ref_wrapper), _gradients(wrapper)
        self.assertEqual(set(grads), set(ref_grads))
        for name in ref_grads:
            with self.subTest(gradient=name):
                self.assertTrue(_scaled_rtol_match(name, grads[name], ref_grads[name]))

    def test_checkpointing_keeps_the_gradients(self):
        batch = self._batch()
        grads = {}
        for multistep_checkpoint in (False, True):
            params = _params(1, False, None)
            params.multistep_checkpoint = multistep_checkpoint
            wrapper = _make_wrapper(params, None)
            _loss(wrapper(_rollout(wrapper, batch)), batch[TARGET]).backward()
            grads[multistep_checkpoint] = _gradients(wrapper)

        self.assertEqual(set(grads[True]), set(grads[False]))
        for name in grads[False]:
            with self.subTest(gradient=name):
                self.assertTrue(_scaled_rtol_match(name, grads[True][name], grads[False][name]))

    def test_checkpointing_refuses_private_generators_only_when_used(self):
        # a model with its own generator cannot be checkpointed; that is refused when the rollout would
        # checkpoint, not when the wrapper is built or runs without checkpointing
        params = _params(1, False, None)
        params.multistep_checkpoint = True
        wrapper = _make_wrapper(params, None)
        wrapper.model.rng_cpu = torch.Generator()

        # a single step is not checkpointed, and neither is evaluation
        wrapper(_rollout(wrapper, self._batch(n_future=0)))
        wrapper.eval()
        wrapper(_rollout(wrapper, self._batch()))

        wrapper.train()
        with self.assertRaisesRegex(RuntimeError, "multistep_checkpoint"):
            wrapper(_rollout(wrapper, self._batch()))

    @unittest.skipUnless(shutil.which("c++") or shutil.which("g++"), "inductor needs a C++ compiler on the CPU")
    def test_compiled_step_matches_eager(self):
        # compile_step compiles the prediction of one step; the rollout around it, and the noise it draws,
        # stay in Python. The compiled rollout has to agree with the eager one, in the gradients too
        noise = _NOISE["white"]
        params = _params(1, False, noise)
        batch = self._batch()

        results = {}
        try:
            for compiled in (False, True):
                wrapper = _make_wrapper(params, noise)
                if compiled:
                    wrapper.compile_step()
                wrapper.preprocessor.set_rng(reset=True, seed=333)
                pred = wrapper(_rollout(wrapper, batch))
                _loss(pred, batch[TARGET]).backward()
                results[compiled] = (pred.detach(), _gradients(wrapper))
        finally:
            torch._dynamo.reset()

        (ref, ref_grads), (pred, grads) = results[False], results[True]
        self.assertTrue(_scaled_rtol_match("predictions", pred, ref))
        self.assertEqual(set(grads), set(ref_grads))
        for name in ref_grads:
            with self.subTest(gradient=name):
                self.assertTrue(_scaled_rtol_match(name, grads[name], ref_grads[name]))

    def test_steps_hand_out_what_forward_stacks(self):
        # the same rollout, the predictions one at a time; the noise continues over the steps either way
        noise = _NOISE["diffusion"]
        params = _params(1, False, noise)
        batch = self._batch()

        with torch.no_grad():
            wrapper = _make_wrapper(params, noise)
            wrapper.eval()
            ref = wrapper(_rollout(wrapper, batch))

            wrapper = _make_wrapper(params, noise)
            wrapper.eval()
            rollout = _rollout(wrapper, batch)
            preds = list(wrapper.steps(rollout))

        self.assertEqual(len(preds), rollout.n_steps)
        self.assertTrue(torch.equal(torch.cat(preds, dim=1), ref))
        # the last prediction has gone into the rollout too, so it has finished
        rollout.noise_state()

        # the rollout is checked when the steps are asked for, not at the first one
        with self.assertRaisesRegex(RuntimeError, "only once"):
            wrapper.steps(rollout)
        with self.assertRaises(TypeError):
            wrapper.steps(batch)

    def test_a_rollout_runs_only_once(self):
        wrapper = _make_wrapper(_params(1, False, None), None)
        rollout = _rollout(wrapper, self._batch())
        wrapper(rollout)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            wrapper(rollout)

    def test_ensemble_members_without_noise_match_the_plain_rollout(self):
        # without noise nothing distinguishes the members, so each one predicts what the rollout of the
        # data batch predicts
        E = 3
        wrapper = _make_wrapper(_params(1, False, None), None)
        wrapper.eval()
        batch = self._batch()
        with torch.no_grad():
            ref = wrapper(_rollout(wrapper, batch))
            rollout = Rollout(batch, ensemble_size=E)
            pred = rollout.unfold(wrapper(rollout))

        self.assertEqual(tuple(pred.shape), (_B, E, *ref.shape[1:]))
        for e in range(E):
            with self.subTest(member=e):
                self.assertTrue(_scaled_rtol_match("member", pred[:, e], ref))

    @parameterized.expand([("white",), ("diffusion",)])
    def test_ensemble_with_noise_is_the_rollout_of_the_members(self, noise_name):
        # the folded ensemble is the rollout of the batch with every sample repeated for each member, the noise
        # included: it is drawn at the batch of the members, so each member gets its own
        E = 3
        noise = _NOISE[noise_name]
        wrapper = _make_wrapper(_params(1, False, noise), noise)
        wrapper.eval()
        pp = wrapper.preprocessor
        batch = self._batch()

        with torch.no_grad():
            pp.set_rng(reset=True, seed=333)
            rollout = _rollout(wrapper, batch, ensemble_size=E)
            pred = rollout.unfold(wrapper(rollout))

            pp.set_rng(reset=True, seed=333)
            ref = wrapper(_rollout(wrapper, _expand_batch(batch, E)))

        self.assertEqual(tuple(pred.shape), (_B, E, *batch[TARGET].shape[1:]))
        self.assertTrue(torch.equal(pred, ref.reshape(_B, E, *ref.shape[1:])))
        for e in range(1, E):
            with self.subTest(member=e):
                self.assertFalse(torch.equal(pred[:, e], pred[:, 0]))

    def test_noise_follows_the_batch_of_the_rollout(self):
        # validation batches can differ from the training ones: the state is resized when the rollout
        # replaces it at its first step
        noise = _NOISE["diffusion"]
        wrapper = _make_wrapper(_params(1, False, noise), noise)
        batch = self._batch()
        for n in (_B, 1, _B):
            with self.subTest(batch=n):
                pred = wrapper(_rollout(wrapper, _take(batch, n)))
                self.assertEqual(pred.shape[0], n)

        # continuing the sequence at another batch is refused
        rollout = _rollout(wrapper, batch)
        wrapper(rollout)
        with self.assertRaises(ValueError):
            wrapper(_rollout(wrapper, _take(batch, 1), noise_state=rollout.noise_state()))

    @parameterized.expand([("white", 1, 1), ("diffusion", 1, 1), ("diffusion_learnable", 0, 1), ("diffusion", 1, 3)])
    def test_split_rollout_draws_the_noise_of_one_rollout(self, noise_name, n_history, E):
        # two rollouts, the second continuing the noise state of the first, draw the noise of one rollout
        # over all their steps, also with another rollout drawing from the same noise in between; with an
        # ensemble, the state is that of all members
        noise = _NOISE[noise_name]
        wrapper = _make_wrapper(_params(n_history, False, noise), noise)
        pp = wrapper.preprocessor
        batch = self._batch(n_history=n_history, n_future=3)

        def draw(rollout):
            samples = []
            for step in rollout:
                samples.append(step.noise)
                rollout.advance(torch.zeros_like(step.input[:, -1:]))
            return samples

        with torch.no_grad():
            pp.set_rng(reset=True, seed=333)
            ref = draw(_rollout(wrapper, batch, ensemble_size=E))

            pp.set_rng(reset=True, seed=333)
            first = _rollout(wrapper, batch, n_steps=2, ensemble_size=E)
            samples = draw(first)
            state = first.noise_state()
            draw(_rollout(wrapper, batch, n_steps=2))
            samples += draw(_rollout(wrapper, batch, n_steps=2, noise_state=state, ensemble_size=E))

        self.assertEqual(len(samples), len(ref))
        for k, (sample, expected) in enumerate(zip(samples, ref)):
            with self.subTest(step=k):
                self.assertTrue(torch.equal(sample, expected))

    def test_a_continued_rollout_trains_on_its_own_graph(self):
        # learnable diffusion noise keeps its state in the graph during a rollout; a continued rollout starts
        # from the detached snapshot, so it does not reach into the graph of the previous rollout, whose
        # backward has freed it
        noise = _NOISE["diffusion_learnable"]
        wrapper = _make_wrapper(_params(0, False, noise), noise)
        batch = self._batch(n_history=0)

        first = _rollout(wrapper, batch)
        _loss(wrapper(first), batch[TARGET]).backward()
        state = first.noise_state()
        self.assertFalse(state.tensor.requires_grad)

        wrapper.zero_grad()
        _loss(wrapper(_rollout(wrapper, batch, noise_state=state)), batch[TARGET]).backward()
        grads = _gradients(wrapper)
        self.assertTrue(any(name.startswith("preprocessor.") for name in grads))


class TestStatelessPreprocessor(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(1)

    def test_assemble_input_checks_its_inputs(self):
        batch = self._batch()
        inp, features = batch[INPUT], batch[INPUT_DYNAMIC_FEATURES]

        noise = _NOISE["white"]
        pp = _make_wrapper(_params(1, False, noise), noise).preprocessor
        sample = torch.zeros(_B, 2, 1, _H * 4, _W * 4)
        with self.subTest("missing feature"), self.assertRaises(ValueError):
            pp.assemble_input(inp, {}, sample)
        with self.subTest("extra feature"), self.assertRaises(ValueError):
            pp.assemble_input(inp, {**features, "other": features[ZENITH]}, sample)
        with self.subTest("missing noise"), self.assertRaises(ValueError):
            pp.assemble_input(inp, features, None)

        pp = _make_wrapper(_params(1, False, None), None).preprocessor
        with self.subTest("unexpected noise"), self.assertRaises(ValueError):
            pp.assemble_input(inp, features, sample)

    def _batch(self):
        return _batch(n_history=1, n_future=_N_FUTURE, grid=(_H * 4, _W * 4))


if __name__ == "__main__":
    unittest.main()
