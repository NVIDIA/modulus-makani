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

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from parameterized import parameterized
from importlib.metadata import entry_points, EntryPoint

class TestEntryPoints(unittest.TestCase):

    def setUp(self):
        self.model_entry_points = {entry_point.name: entry_point for entry_point in entry_points(group="modulus.models") if not entry_point.value.startswith("modulus.experimental.models")}

    @parameterized.expand(["SFNO"])
    def test_model_entry_points(self, model_name):
        """Test model entry points"""

        # Check the model entry point.
        model_ep = self.model_entry_points.get(model_name)
        self.assertIsNotNone(model_ep)

        # Try loading the model type.
        model_type = model_ep.load()
        self.assertIsNotNone(model_type)

        # Create the model.
        model = model_type()
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
