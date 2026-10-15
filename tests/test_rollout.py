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

from makani.models.stepper import MultiStepWrapper
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

    def test_number_of_steps(self):
        self.assertEqual(len(Rollout(_batch(n_future=3))), 4)
        self.assertEqual(len(Rollout(_batch(n_future=3), n_steps=2)), 2)
        with self.assertRaises(ValueError):
            Rollout(_batch(n_future=1), n_steps=3)

    def test_without_targets(self):
        batch = _batch(with_target=False)
        with self.assertRaises(ValueError):
            Rollout(batch)
        with self.assertRaises(NotImplementedError):
            Rollout(batch, n_steps=2)

        step = next(Rollout(batch, n_steps=1))
        self.assertIsNone(step.target)
        self.assertEqual(step.target_dynamic_features, {})

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


# ---------------------------------------------------------------------------
# A rollout reproduces MultiStepWrapper
# ---------------------------------------------------------------------------
#
# The reference is the MultiStepWrapper with its Preprocessor2D as it is now: the
# preprocessor caches the dynamic features of the batch, draws the noise itself and
# slides its caches with append_history. The new path takes the features and the
# noise from the RolloutStep instead, and runs the same preprocessor operations on
# them. Both must agree in the predictions, the loss and every gradient, the noise
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
    wrapper = MultiStepWrapper(params, lambda: _TinyNet(in_chans, _C))
    wrapper.train()
    return wrapper


def _model_step(preprocessor, model):
    """One step as the stateless preprocessor will run it: features and noise come from the step."""

    def step_fn(step):
        parts = [step.input] + [step.input_dynamic_features[name] for name in preprocessor.dynamic_features]
        if step.noise is not None:
            parts.append(step.noise)
        x = torch.cat(parts, dim=2)
        preprocessor.history_compute_stats(x)
        xn = preprocessor.history_normalize(x, target=False)
        predn = model(preprocessor.pack_model_input(xn))
        return preprocessor.history_denormalize(preprocessor.correct_bias(predn).unsqueeze(1), target=True)

    return step_fn


def _noise_source(preprocessor):
    return getattr(preprocessor, "input_noise", None)


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
        pp = wrapper.preprocessor
        rollout = Rollout(batch, noise_source=_noise_source(pp))
        pred = run_rollout(_model_step(pp, wrapper.model), rollout, push_forward=push_forward)
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
        pp = wrapper.preprocessor
        step_fn = _model_step(pp, wrapper.model)
        rollout = Rollout(batch, noise_source=_noise_source(pp))
        preds = []
        loss = torch.zeros(())
        for step in rollout:
            pred = step_fn(step)
            preds.append(pred)
            loss = loss + _STEP_WEIGHTS[step.index] * torch.square(pred - step.target).mean()
            rollout.advance(pred.detach() if push_forward else pred)
        loss.backward()

        self._compare("summed", torch.cat(preds, dim=1).detach(), loss.detach(), _gradients(wrapper), ref)


if __name__ == "__main__":
    unittest.main()
