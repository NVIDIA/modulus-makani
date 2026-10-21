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
from torch.utils.data import default_collate

from makani.utils.dataloaders.batch import (
    INPUT,
    TARGET,
    INPUT_DYNAMIC_FEATURES,
    TARGET_DYNAMIC_FEATURES,
    INPUT_TIMESTAMP,
    TARGET_TIMESTAMP,
    ZENITH,
    make_batch,
    batch_to_device,
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

_B, _H, _W = 2, 4, 8


def _field(steps, channels, *grid):
    grid = grid or (_H, _W)
    return torch.randn(_B, steps, channels, *grid)


def _times(steps):
    return torch.arange(_B * steps, dtype=torch.float64).reshape(_B, steps)


def _batch(with_target=True, with_zenith=True, grid=()):
    return make_batch(
        inp=_field(2, 3, *grid),
        inp_timestamp=_times(2),
        tar=_field(1, 2, *grid) if with_target else None,
        tar_timestamp=_times(1) if with_target else None,
        inp_zenith=_field(2, 1, *grid) if with_zenith else None,
        tar_zenith=_field(1, 1, *grid) if (with_zenith and with_target) else None,
    )


class TestMakeBatch(unittest.TestCase):

    def test_keys_with_target(self):
        batch = _batch(with_zenith=False)
        self.assertEqual(
            set(batch),
            {INPUT, TARGET, INPUT_DYNAMIC_FEATURES, TARGET_DYNAMIC_FEATURES, INPUT_TIMESTAMP, TARGET_TIMESTAMP},
        )
        # present even without any dynamic feature, so a consumer never has to ask
        self.assertEqual(batch[INPUT_DYNAMIC_FEATURES], {})
        self.assertEqual(batch[TARGET_DYNAMIC_FEATURES], {})

    def test_keys_without_target(self):
        batch = _batch(with_target=False)
        self.assertEqual(set(batch), {INPUT, INPUT_DYNAMIC_FEATURES, INPUT_TIMESTAMP})
        self.assertEqual(set(batch[INPUT_DYNAMIC_FEATURES]), {ZENITH})

    def test_target_without_timestamps_is_rejected(self):
        with self.assertRaises(ValueError):
            make_batch(inp=_field(1, 3), inp_timestamp=_times(1), tar=_field(1, 3))

    def test_samples_collate_into_a_batch(self):
        # the torch DataLoader stacks the samples of a map-style dataset key by key,
        # nested dicts included
        samples = [
            make_batch(
                inp=torch.zeros(1, 3, _H, _W),
                inp_timestamp=torch.zeros(1, dtype=torch.float64),
                tar=torch.ones(2, 3, _H, _W),
                tar_timestamp=torch.ones(2, dtype=torch.float64),
                inp_zenith=torch.zeros(1, 1, _H, _W),
                tar_zenith=torch.ones(2, 1, _H, _W),
            )
            for _ in range(_B)
        ]
        batch = default_collate(samples)
        self.assertEqual(tuple(batch[INPUT].shape), (_B, 1, 3, _H, _W))
        self.assertEqual(tuple(batch[TARGET_DYNAMIC_FEATURES][ZENITH].shape), (_B, 2, 1, _H, _W))
        self.assertEqual(tuple(batch[TARGET_TIMESTAMP].shape), (_B, 2))
        self.assertEqual(batch[TARGET_TIMESTAMP].dtype, torch.float64)


class TestBatchToDevice(unittest.TestCase):

    def test_moves_nested_tensors_and_leaves_the_original(self):
        batch = _batch()
        moved = batch_to_device(batch, torch.device("meta"))

        self.assertEqual(moved[INPUT].device.type, "meta")
        self.assertEqual(moved[INPUT_DYNAMIC_FEATURES][ZENITH].device.type, "meta")
        self.assertEqual(moved[TARGET_TIMESTAMP].device.type, "meta")
        self.assertEqual(moved[TARGET_TIMESTAMP].dtype, torch.float64)

        self.assertEqual(batch[INPUT].device.type, "cpu")
        self.assertEqual(batch[INPUT_DYNAMIC_FEATURES][ZENITH].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
