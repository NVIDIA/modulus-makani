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
import json
import logging
import subprocess
import tempfile
import unittest

from ruamel.yaml import YAML

from makani.models import model_registry
from makani.utils.YParams import ParamsBase, YParams

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# keys every training configuration has to provide, since the driver and the trainers read them unconditionally
_REQUIRED_KEYS = [
    "nettype",
    "channel_names",
    "dt",
    "n_history",
    "batch_size",
    "max_epochs",
    "lr",
    "lr_warmup_steps",
    "optimizer_type",
    "optimizer_beta1",
    "optimizer_beta2",
    "scheduler",
]

# optimizers and schedulers Driver.get_optimizer and Driver.get_scheduler know, and the keys each scheduler reads
_OPTIMIZERS = ["Adam", "AdamW", "SGD", "SIRFShampoo"]
_SCHEDULER_KEYS = {
    None: [],
    "ReduceLROnPlateau": ["scheduler_factor", "scheduler_patience"],
    "StepLR": ["scheduler_step_size", "scheduler_gamma"],
    "CosineAnnealingLR": ["scheduler_T_max"],
    "OneCycleLR": ["scheduler_T_max"],
    "CosineAnnealingWarmRestarts": [],
}


def _tracked_configs():
    """The config files under version control; config/ also holds untracked personal configs, which are not tested."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "config/*.yaml"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [os.path.join(_REPO_ROOT, path) for path in out.split()]


def _sections(path):
    with open(path) as f:
        return list(YAML(typ="safe").load(f).keys())


class TestConfigs(unittest.TestCase):
    """Every section of every configuration shipped in config/ loads and is complete."""

    @classmethod
    def setUpClass(cls):
        cls.configs = _tracked_configs()
        if not cls.configs:
            raise unittest.SkipTest("config files are only found in a git checkout")
        cls.registered_models = model_registry.list_models()

    def test_configs_load_and_are_complete(self):
        for path in self.configs:
            for section in _sections(path):
                name = f"{os.path.basename(path)}:{section}"
                with self.subTest(config=name):
                    params = YParams(path, section)

                    missing = [key for key in _REQUIRED_KEYS if key not in params]
                    self.assertEqual(missing, [], f"{name} lacks required keys")

                    # a nettype is either a registered model or a path:class reference to a model file
                    if self.registered_models and ":" not in params.nettype:
                        self.assertIn(params.nettype, self.registered_models, f"{name} has an unknown nettype")

                    self.assertIn(params.optimizer_type, _OPTIMIZERS, f"{name} has an unknown optimizer")

                    self.assertIn(params.scheduler, _SCHEDULER_KEYS, f"{name} has an unknown scheduler")
                    missing = [key for key in _SCHEDULER_KEYS[params.scheduler] if key not in params]
                    self.assertEqual(missing, [], f"{name} lacks keys of scheduler {params.scheduler}")

    def test_yaml_round_trip(self):
        path = self.configs[0]
        section = _sections(path)[0]
        params = YParams(path, section, print_params=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "params.yaml")
            params.to_yaml(out)

            with self.subTest(desc="no overwrite by default"):
                with self.assertRaises(FileExistsError):
                    params.to_yaml(out)
            params.to_yaml(out, overwrite=True)

            # the written file holds the parameters at the top level, without a section
            restored = YParams(out)
            for key, val in params.params.items():
                with self.subTest(key=key):
                    self.assertEqual(restored[key], val)

    def test_log_names_file_and_section(self):
        path = self.configs[0]
        section = _sections(path)[0]
        params = YParams(path, section)

        logger = logging.getLogger("makani.test_configs")
        with self.assertLogs(logger, level="INFO") as logs:
            params.log(logger)
        output = "\n".join(logs.output)
        self.assertIn(path, output)
        self.assertIn(section, output)
        self.assertIn(f"nettype {params.nettype}", output)

    def test_from_json(self):
        config = {"nettype": "SFNO", "lr": 1e-3, "scheduler": "None", "channel_names": ["u10m", "v10m"]}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "params.json")
            with open(path, "w") as f:
                json.dump(config, f)
            params = ParamsBase.from_json(path)

        self.assertEqual(params.nettype, "SFNO")
        self.assertEqual(params["lr"], 1e-3)
        self.assertEqual(params.channel_names, ["u10m", "v10m"])
        # the string "None" becomes None, as in the yaml files
        self.assertIsNone(params.scheduler)


if __name__ == "__main__":
    unittest.main()
