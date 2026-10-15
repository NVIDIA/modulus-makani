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
A rollout under DDP.

The trainers wrap the model in DDP, and a rollout runs the model once per step
before a single backward. Two ways of doing that are compared against the same
rollout without DDP:

* the loop inside one DDP forward: DDP wraps a module whose forward runs the
  whole rollout, as it wraps MultiStepWrapper now;
* one DDP forward per step, with ``no_sync`` for all but the last.

The model is serial and replicated on every rank, and every rank rolls out the
same batch with the same noise, so averaging the gradients over the ranks has to
reproduce the gradients of the rollout without DDP exactly. What could break is
DDP's bookkeeping -- the parameters are used once per step, the noise parameters
outside the network, and makani configures ``static_graph`` -- and two iterations
are run, since ``static_graph`` records the graph in the first and relies on it in
the second.
"""

import contextlib
import os
import sys
import unittest

import torch
import torch.nn as nn
from parameterized import parameterized

from makani.utils import comm
from makani.mpu.mappings import init_gradient_reduction_hooks
from makani.utils.dataloaders.batch import TARGET, batch_to_device
from makani.utils.dataloaders.rollout import Rollout, run_rollout

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from .distributed_helpers import _init_grid, reduce_success, teardown_comms
from ..testutils import compare_tensors, disable_tf32, set_seed
from ..test_rollout import (
    _H,
    _N_FUTURE,
    _NOISE,
    _W,
    _batch,
    _gradients,
    _loss,
    _make_wrapper,
    _model_step,
    _noise_source,
    _params,
)


class _RolloutModule(nn.Module):
    """The unit DDP wraps: its forward runs the whole rollout."""

    def __init__(self, wrapper, push_forward):
        super().__init__()
        self.preprocessor = wrapper.preprocessor
        self.model = wrapper.model
        self.push_forward = push_forward

    def forward(self, rollout):
        return run_rollout(_model_step(self.preprocessor, self.model), rollout, push_forward=self.push_forward)


class _StepModule(nn.Module):
    """The unit DDP wraps per step."""

    def __init__(self, wrapper):
        super().__init__()
        self.preprocessor = wrapper.preprocessor
        self.model = wrapper.model

    def forward(self, step):
        return _model_step(self.preprocessor, self.model)(step)


_CASES = [
    (name, n_history, push_forward)
    for name, n_history in [("none", 1), ("white_learnable", 1), ("diffusion_learnable", 0)]
    for push_forward in (False, True)
]


class TestDistributedRollout(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)

    @classmethod
    def tearDownClass(cls):
        teardown_comms()

    def setUp(self):
        disable_tf32()

    def _wrapper(self, noise_name, n_history, push_forward):
        noise = _NOISE[noise_name]
        # serial: the noise is seeded by rank otherwise, and every rank has to draw the same
        with comm.serial():
            return _make_wrapper(_params(n_history, push_forward, noise), noise).to(self.device)

    def _batch(self, n_history):
        set_seed(1)
        return batch_to_device(_batch(n_history=n_history, n_future=_N_FUTURE, grid=(_H * 4, _W * 4)), self.device)

    def _reference(self, wrapper, push_forward, batch):
        # the same wrapper in every iteration, so that its noise moves on like the one under DDP
        wrapper.zero_grad(set_to_none=True)
        with comm.serial():
            rollout = Rollout(batch, noise_source=_noise_source(wrapper.preprocessor))
            pred = run_rollout(_model_step(wrapper.preprocessor, wrapper.model), rollout, push_forward=push_forward)
            _loss(pred, batch[TARGET]).backward()
        return _gradients(wrapper)

    def _ddp(self, module):
        # as the trainers wrap the model; with a serial model this is plain DDP over all ranks
        with comm.serial():
            return init_gradient_reduction_hooks(
                module,
                device=self.device,
                reduction_buffer_count=1,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
                verbose=False,
            )

    def _check(self, label, grads, ref):
        with self.subTest(desc=f"{label}: same parameters"):
            self.assertTrue(reduce_success(set(grads) == set(ref), self.device))
        for name in ref:
            with self.subTest(desc=f"{label}: gradient {name}"):
                atol = 1e-6 * max(1.0, ref[name].abs().max().item())
                self.assertTrue(reduce_success(compare_tensors(name, grads[name], ref[name], atol, 1e-5), self.device))

    @parameterized.expand(_CASES)
    def test_rollout_inside_one_ddp_forward(self, noise_name, n_history, push_forward):
        batch = self._batch(n_history)
        module = _RolloutModule(self._wrapper(noise_name, n_history, push_forward), push_forward)
        ddp = self._ddp(module)
        reference = self._wrapper(noise_name, n_history, push_forward)

        for iteration in range(2):
            ref = self._reference(reference, push_forward, batch)
            module.zero_grad(set_to_none=True)
            with comm.serial():
                rollout = Rollout(batch, noise_source=_noise_source(module.preprocessor))
                _loss(ddp(rollout), batch[TARGET]).backward()
            # the module names its children like the wrapper does, so the gradients compare by name
            self._check(f"iteration {iteration}", _gradients(module), ref)

    @parameterized.expand(_CASES)
    def test_one_ddp_forward_per_step_with_no_sync(self, noise_name, n_history, push_forward):
        batch = self._batch(n_history)
        module = _StepModule(self._wrapper(noise_name, n_history, push_forward))
        ddp = self._ddp(module)
        reference = self._wrapper(noise_name, n_history, push_forward)

        for iteration in range(2):
            ref = self._reference(reference, push_forward, batch)
            module.zero_grad(set_to_none=True)
            with comm.serial():
                rollout = Rollout(batch, noise_source=_noise_source(module.preprocessor))
                preds = []
                for step in rollout:
                    last = step.index == rollout.n_steps - 1
                    with contextlib.nullcontext() if last else ddp.no_sync():
                        pred = ddp(step)
                    preds.append(pred)
                    rollout.advance(pred.detach() if push_forward else pred)
                _loss(torch.cat(preds, dim=1), batch[TARGET]).backward()
            self._check(f"iteration {iteration}", _gradients(module), ref)


if __name__ == "__main__":
    unittest.main()
