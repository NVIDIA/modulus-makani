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

import unittest
from unittest import mock

from makani.utils import comm


class _FakeManager:
    """Stands in for the DistributedManager of rank 3 in a world of 8 with h=2, w=2."""

    world_size = 8
    rank = 3
    local_rank = 1
    group_names = ["world", "model", "spatial", "h", "w", "data", "batch", "__orthogonal_to_h"]

    def group_size(self, name):
        return 2

    def group_rank(self, name):
        return 1

    def group(self, name):
        return f"group:{name}"


class TestSerial(unittest.TestCase):

    def setUp(self):
        patcher_dm = mock.patch.object(comm, "_DM", _FakeManager())
        patcher_roots = mock.patch.object(comm, "_COMM_ROOTS", {"h": 2})
        patcher_dm.start()
        patcher_roots.start()
        self.addCleanup(patcher_dm.stop)
        self.addCleanup(patcher_roots.stop)

    def _assert_distributed(self):
        self.assertFalse(comm.is_serial())
        self.assertEqual(comm.get_size("h"), 2)
        self.assertEqual(comm.get_rank("h"), 1)
        self.assertEqual(comm.get_group("h"), "group:h")
        self.assertEqual(comm.get_root("h"), 2)
        self.assertTrue(comm.is_distributed("spatial"))
        self.assertIn("h", comm.get_model_comm_names())

    def _assert_serial(self):
        self.assertTrue(comm.is_serial())
        self.assertEqual(comm.get_size("h"), 1)
        self.assertEqual(comm.get_rank("h"), 0)
        self.assertIsNone(comm.get_group("h"))
        self.assertEqual(comm.get_root("h"), 0)
        self.assertFalse(comm.is_distributed("spatial"))
        self.assertEqual(comm.get_comm_names(), [])
        self.assertEqual(comm.get_model_comm_names(), [])

    def test_group_queries_report_a_single_process(self):
        self._assert_distributed()
        with comm.serial():
            self._assert_serial()
        self._assert_distributed()

    def test_world_queries_are_not_affected(self):
        with comm.serial():
            self.assertEqual(comm.get_world_size(), 8)
            self.assertEqual(comm.get_world_rank(), 3)
            self.assertEqual(comm.get_local_rank(), 1)

    def test_nested_contexts(self):
        with comm.serial():
            with comm.serial():
                self._assert_serial()
            # still inside the outer context
            self._assert_serial()
        self._assert_distributed()

    def test_restored_after_an_exception(self):
        with self.assertRaises(KeyError):
            with comm.serial():
                raise KeyError("boom")
        self._assert_distributed()

    def test_init_is_refused_inside(self):
        with comm.serial():
            with self.assertRaises(RuntimeError):
                comm.init()
        self._assert_distributed()


class TestSerialWithoutComms(unittest.TestCase):

    def test_same_answers_as_without_the_context(self):
        # without initialized comms the queries already report a single process
        with mock.patch.object(comm, "_DM", None):
            outside = (comm.get_size("h"), comm.get_rank("h"), comm.get_group("h"), comm.is_distributed("h"))
            with comm.serial():
                inside = (comm.get_size("h"), comm.get_rank("h"), comm.get_group("h"), comm.is_distributed("h"))
        self.assertEqual(inside, outside)


if __name__ == "__main__":
    unittest.main()
