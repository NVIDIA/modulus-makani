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

import math
import unittest

import torch
from torch.optim import lr_scheduler

from makani.utils.driver import Driver
from makani.utils.YParams import ParamsBase

_LR = 1.0


def _params(scheduler, **kwargs):
    params = ParamsBase()
    params["scheduler"] = scheduler
    params["lr"] = _LR
    params["lr_warmup_steps"] = 0
    for key, value in kwargs.items():
        params[key] = value
    return params


class TestGetScheduler(unittest.TestCase):
    """
    The schedulers ``Driver.get_scheduler`` builds from the parameters, stepped as the trainers do:
    once per epoch, and with the validation loss for ``ReduceLROnPlateau``.
    """

    def setUp(self):
        self.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=_LR)
        # get_scheduler does not use the state of the driver, so the expensive constructor is skipped
        self.driver = Driver.__new__(Driver)

    def _lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def _lrs(self, scheduler, num_steps):
        """The learning rate before the first step and after each step."""
        lrs = [self._lr()]
        for _ in range(num_steps):
            self.optimizer.step()
            scheduler.step()
            lrs.append(self._lr())
        return lrs

    def test_step_lr(self):
        scheduler = self.driver.get_scheduler(
            self.optimizer, _params("StepLR", scheduler_step_size=2, scheduler_gamma=0.5)
        )
        self.assertIsInstance(scheduler, lr_scheduler.StepLR)
        for lr, expected in zip(self._lrs(scheduler, 4), [1.0, 1.0, 0.5, 0.5, 0.25]):
            self.assertAlmostEqual(lr, expected)

    def test_cosine_annealing_lr(self):
        scheduler = self.driver.get_scheduler(
            self.optimizer, _params("CosineAnnealingLR", scheduler_T_max=4, scheduler_min_lr=0.1)
        )
        self.assertIsInstance(scheduler, lr_scheduler.CosineAnnealingLR)
        lrs = self._lrs(scheduler, 4)
        for step, lr in enumerate(lrs):
            self.assertAlmostEqual(lr, 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / 4)))

    def test_cosine_annealing_lr_defaults_to_zero_min_lr(self):
        params = _params("CosineAnnealingLR", scheduler_T_max=2)
        scheduler = self.driver.get_scheduler(self.optimizer, params)
        self.assertEqual(params.scheduler_min_lr, 0.0)
        self.assertAlmostEqual(self._lrs(scheduler, 2)[-1], 0.0)

    def test_one_cycle_lr(self):
        scheduler = self.driver.get_scheduler(self.optimizer, _params("OneCycleLR", scheduler_T_max=10))
        self.assertIsInstance(scheduler, lr_scheduler.OneCycleLR)
        lrs = self._lrs(scheduler, 9)
        # it starts at the configured lr divided by the default div_factor of 25 and peaks at the
        # configured lr at the end of the default pct_start of 30% of the steps, which torch places
        # at step 0.3 * 10 - 1 = 2
        self.assertAlmostEqual(lrs[0], _LR / 25.0)
        self.assertAlmostEqual(max(lrs), _LR)
        self.assertEqual(lrs.index(max(lrs)), 2)
        self.assertLess(lrs[-1], lrs[0])

    def test_cosine_annealing_warm_restarts(self):
        scheduler = self.driver.get_scheduler(
            self.optimizer, _params("CosineAnnealingWarmRestarts", scheduler_T_0=2, scheduler_min_lr=0.0)
        )
        self.assertIsInstance(scheduler, lr_scheduler.CosineAnnealingWarmRestarts)
        for lr, expected in zip(self._lrs(scheduler, 4), [1.0, 0.5, 1.0, 0.5, 1.0]):
            self.assertAlmostEqual(lr, expected)

    def test_reduce_lr_on_plateau(self):
        params = _params("ReduceLROnPlateau", scheduler_factor=0.5, scheduler_patience=0)
        scheduler = self.driver.get_scheduler(self.optimizer, params)
        self.assertIsInstance(scheduler, lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(params.scheduler_mode, "min")
        # an improving loss keeps the lr, one that does not improve halves it
        for loss, expected in [(1.0, 1.0), (0.5, 1.0), (0.5, 0.5), (0.4, 0.5), (0.4, 0.25)]:
            scheduler.step(loss)
            self.assertAlmostEqual(self._lr(), expected)

    def test_reduce_lr_on_plateau_needs_validation(self):
        params = _params("ReduceLROnPlateau", scheduler_factor=0.5, scheduler_patience=0, skip_validation=True)
        with self.assertRaises(ValueError):
            self.driver.get_scheduler(self.optimizer, params)

    def test_reduce_lr_on_plateau_has_no_warmup(self):
        params = _params("ReduceLROnPlateau", scheduler_factor=0.5, scheduler_patience=0, lr_warmup_steps=2)
        with self.assertRaises(NotImplementedError):
            self.driver.get_scheduler(self.optimizer, params)

    def test_warmup(self):
        params = _params("StepLR", scheduler_step_size=1, scheduler_gamma=0.5, lr_warmup_steps=2, lr_start=0.1)
        scheduler = self.driver.get_scheduler(self.optimizer, params)
        self.assertIsInstance(scheduler, lr_scheduler.SequentialLR)
        # linear from lr_start * lr to lr over the warmup steps, then the StepLR from the configured lr
        for lr, expected in zip(self._lrs(scheduler, 4), [0.1, 0.55, 1.0, 0.5, 0.25]):
            self.assertAlmostEqual(lr, expected)

    def test_no_scheduler(self):
        # None is what YParams makes of "None" in the yaml files
        for name in [None, "None", "none"]:
            with self.subTest(scheduler=name):
                self.assertIsNone(self.driver.get_scheduler(self.optimizer, _params(name)))

    def test_unset_scheduler(self):
        # a params object without the key at all, with and without warmup
        for warmup in [0, 2]:
            with self.subTest(lr_warmup_steps=warmup):
                params = ParamsBase()
                params["lr"] = _LR
                params["lr_warmup_steps"] = warmup
                scheduler = self.driver.get_scheduler(self.optimizer, params)
                if warmup == 0:
                    self.assertIsNone(scheduler)
                else:
                    self.assertIsInstance(scheduler, lr_scheduler.LinearLR)
                # the trainers read params.scheduler after setting up the scheduler
                self.assertIsNone(params.scheduler)

    def test_warmup_without_scheduler(self):
        params = _params(None, lr_warmup_steps=2, lr_start=0.1)
        scheduler = self.driver.get_scheduler(self.optimizer, params)
        self.assertIsInstance(scheduler, lr_scheduler.LinearLR)
        # linear from lr_start * lr to lr over the warmup steps, then constant
        for lr, expected in zip(self._lrs(scheduler, 4), [0.1, 0.55, 1.0, 1.0, 1.0]):
            self.assertAlmostEqual(lr, expected)

    def test_unknown_scheduler_raises(self):
        # a typo must not silently train without a scheduler
        with self.assertRaises(ValueError):
            self.driver.get_scheduler(self.optimizer, _params("CosineAnealingLR"))


if __name__ == "__main__":
    unittest.main()
