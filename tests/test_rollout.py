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
import sys
import unittest

import torch
import torch.nn as nn
from parameterized import parameterized

from makani.models.stepper import MultiStepWrapper, SingleStepWrapper
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
from makani.utils.dataloaders.rollout import Rollout, run_rollout
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

    def update(self, replace_state=False):
        self.updates.append(replace_state)

    def __call__(self):
        return torch.full((_B, 1, 1, _H, _W), float(len(self.updates)))


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

    def test_noise_can_continue_a_previous_rollout(self):
        noise = _RecordingNoise()
        rollout = Rollout(_batch(n_future=1), noise_source=noise, continue_noise=True)
        for step in rollout:
            rollout.advance(torch.zeros(_B, 1, _C, _H, _W))
        self.assertEqual(noise.updates, [False, False])

    def test_run_rollout_stacks_and_push_forward_detaches(self):
        batch = _batch(n_history=0, n_future=2)
        scale = torch.tensor(2.0, requires_grad=True)

        def model_step(step):
            return scale * step.input[:, -1:]

        for push_forward in (False, True):
            with self.subTest(push_forward=push_forward):
                out = run_rollout(model_step, Rollout(batch), push_forward=push_forward)
                self.assertEqual(tuple(out.shape), (_B, 3, _C, _H, _W))
                # each step doubles the previous prediction
                for k in range(3):
                    self.assertTrue(torch.allclose(out[:, k], 2.0 ** (k + 1) * batch[INPUT][:, -1]))

                # out_2 = scale^3 x through the rollout, d/dscale = 3 scale^2 x = 12 x; with push-forward
                # the earlier steps are constants, out_2 = scale (4 x), d/dscale = 4 x
                (grad,) = torch.autograd.grad(out[:, 2].sum(), scale)
                expected = (4.0 if push_forward else 12.0) * batch[INPUT][:, -1].sum()
                self.assertTrue(torch.allclose(grad, expected, rtol=1e-5))

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
# A rollout reproduces MultiStepWrapper
# ---------------------------------------------------------------------------
#
# The reference is the MultiStepWrapper driven with a tensor, the stateful path: the
# preprocessor caches the dynamic features of the batch, draws the noise itself and
# slides its caches with append_history. Driven with a Rollout, the wrappers take
# the features and the noise from each RolloutStep instead, and the preprocessor
# keeps no state. Both must agree in the predictions, the loss and every gradient,
# the noise parameters included when the noise is trainable.

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


def _make_wrapper(params, noise, wrapper_cls=MultiStepWrapper):
    n_noise = noise["n_channels"] if noise is not None else 0
    in_chans = (params.n_history + 1) * (_C + 1 + n_noise)
    # the same seed for every wrapper: the same weights, and the noise is seeded from its own fixed seed
    set_seed(333)
    wrapper = wrapper_cls(params, lambda: _TinyNet(in_chans, _C))
    wrapper.train()
    return wrapper


def _rollout(wrapper, batch, **kwargs):
    """A rollout of the batch, with the input noise of the wrapper."""
    return Rollout(batch, noise_source=wrapper.preprocessor.noise_source(batch_size=batch[INPUT].shape[0]), **kwargs)


def _loss(pred, tar):
    # the per-step losses, weighted by step, as the multistep weighting does
    per_step = torch.square(pred - tar).mean(dim=(0, 2, 3, 4))
    return torch.sum(_STEP_WEIGHTS.to(per_step.device) * per_step)


def _gradients(module):
    return {name: p.grad.detach().clone() for name, p in module.named_parameters() if p.grad is not None}


def _scaled_rtol_match(msg, actual, expected):
    return compare_tensors(msg, actual, expected, atol=1e-6 * max(1.0, expected.abs().max().item()), rtol=1e-5)


class TestRolloutMatchesMultiStepWrapper(unittest.TestCase):

    def setUp(self):
        disable_tf32()

    def _reference(self, params, noise, batch):
        wrapper = _make_wrapper(params, noise)
        wrapper.preprocessor.cache_dynamic_features(batch)
        pred = wrapper(batch[INPUT])
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
    """The rollout path of the wrappers: evaluation, single steps, checkpointing, and no state left behind."""

    def setUp(self):
        disable_tf32()

    def _batch(self, n_history=1, n_future=_N_FUTURE):
        set_seed(1)
        return _batch(n_history=n_history, n_future=n_future, grid=(_H * 4, _W * 4))

    def test_evaluation_matches_the_stepwise_loop(self):
        # how the deterministic trainer validated before: one forward per step in eval mode, the
        # window slid by append_history
        params = _params(1, False, None)
        batch = self._batch()
        wrapper = _make_wrapper(params, None)
        wrapper.eval()

        with torch.no_grad():
            wrapper.preprocessor.cache_dynamic_features(batch)
            inpt = batch[INPUT]
            ref = []
            for idt in range(batch[TARGET].shape[1]):
                pred = wrapper(inpt)
                ref.append(pred)
                inpt = wrapper.preprocessor.append_history(inpt, pred, idt)
            ref = torch.cat(ref, dim=1)

            pred = wrapper(_rollout(wrapper, batch))

        self.assertEqual(tuple(pred.shape), tuple(batch[TARGET].shape))
        self.assertTrue(_scaled_rtol_match("predictions", pred, ref))

    def test_single_step_wrapper(self):
        # with n_future = 0 the trainer gets a SingleStepWrapper: one step, as its tensor forward predicts it
        noise = _NOISE["white_learnable"]
        params = _params(1, False, noise)
        params.n_future = 0
        batch = self._batch(n_future=0)

        ref_wrapper = _make_wrapper(params, noise, SingleStepWrapper)
        ref_wrapper.preprocessor.cache_dynamic_features(batch)
        ref = ref_wrapper(batch[INPUT])
        torch.square(ref - batch[TARGET]).mean().backward()

        wrapper = _make_wrapper(params, noise, SingleStepWrapper)
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

    def test_a_rollout_runs_only_once(self):
        wrapper = _make_wrapper(_params(1, False, None), None)
        rollout = _rollout(wrapper, self._batch())
        wrapper(rollout)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            wrapper(rollout)

    def test_preprocessor_keeps_no_state(self):
        noise = _NOISE["white"]
        wrapper = _make_wrapper(_params(1, False, noise), noise)
        wrapper(_rollout(wrapper, self._batch()))

        pp = wrapper.preprocessor
        self.assertIsNone(pp.history_mean)
        self.assertIsNone(pp.history_std)
        self.assertEqual(pp.get_dynamic_features(), ({}, {}))

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

        # continuing the sequence at another batch would restart the stateful noise from zero
        with self.assertRaises(RuntimeError):
            wrapper(_rollout(wrapper, _take(batch, 1), continue_noise=True))


class TestStatelessPreprocessor(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(1)

    def test_statistics_match_the_stored_ones(self):
        pp = _make_wrapper(_params(1, False, None), None).preprocessor
        x = torch.randn(_B, 2, _C, _H * 4, _W * 4)
        y = torch.randn(_B, 1, _C - 1, _H * 4, _W * 4)

        stats = pp.compute_history_stats(x)
        self.assertIsNone(pp.history_mean)
        pp.history_compute_stats(x)

        self.assertTrue(torch.equal(stats.mean, pp.history_mean))
        self.assertTrue(torch.equal(stats.std, pp.history_std))
        self.assertTrue(torch.equal(pp.history_normalize(x, stats=stats), pp.history_normalize(x)))
        self.assertTrue(
            torch.equal(pp.history_denormalize(y, target=True, stats=stats), pp.history_denormalize(y, target=True))
        )

    def test_assemble_input_matches_append_dynamic_features(self):
        noise = _NOISE["white"]
        pp = _make_wrapper(_params(1, False, noise), noise).preprocessor
        batch = self._batch()
        pp.cache_dynamic_features(batch)
        pp.update_internal_state(replace_state=True)

        ref = pp.append_dynamic_features(batch[INPUT])
        sample = pp.input_noise()
        self.assertTrue(torch.equal(pp.assemble_input(batch[INPUT], batch[INPUT_DYNAMIC_FEATURES], sample), ref))

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
