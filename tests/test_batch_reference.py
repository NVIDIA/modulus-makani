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

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .batch_reference import REFERENCE_PATH, compute_reference
from .testutils import compare_arrays

# zenith angles and the static grid go through sin/cos, whose last bit may differ
# between CPUs; a misplaced channel is off by far more than this
_ATOL = 1e-6


class TestBatchReference(unittest.TestCase):
    """The data pipeline still hands the model what it did before the batches became dicts."""

    @classmethod
    def setUpClass(cls):
        with np.load(REFERENCE_PATH) as stored:
            cls.reference = {key: stored[key] for key in stored.files}
        cls.current = compute_reference()

    def _check_group(self, group):
        keys = sorted(key for key in self.reference if key.startswith(f"{group}/"))
        self.assertTrue(keys, f"the reference has no entries for {group}")

        for key in keys:
            with self.subTest(key=key):
                self.assertIn(key, self.current, f"{key} is no longer produced")
                expected = self.reference[key]
                actual = self.current[key]
                self.assertEqual(actual.dtype, expected.dtype, f"dtype of {key}")
                self.assertTrue(
                    compare_arrays(key, actual, expected, atol=_ATOL, rtol=0.0, verbose=True, shape_check=True),
                    f"{key} differs from the reference",
                )

        # nothing new may sneak into a batch either
        extra = sorted(key for key in self.current if key.startswith(f"{group}/") and key not in self.reference)
        self.assertEqual(extra, [], f"{group} produces entries the reference does not have")

    def test_multifiles_batches(self):
        self._check_group("multifiles")

    def test_dummy_batches(self):
        self._check_group("dummy")

    def test_dummy_inference_batches(self):
        self._check_group("dummy_inference")

    def test_stepper_packing(self):
        self._check_group("stepper")


if __name__ == "__main__":
    unittest.main()
