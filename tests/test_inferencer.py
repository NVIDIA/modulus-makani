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
import tempfile
import unittest
from typing import Optional

import torch
from torch.utils.data import default_collate
from parameterized import parameterized

from makani.utils.dataloader import get_dataloader
from makani.utils.dataloaders.batch import INPUT, INPUT_TIMESTAMP, TARGET, TARGET_TIMESTAMP
from makani.utils.dataloaders.rollout import Rollout
from makani.utils.functions import expand_ensemble
from makani.utils.inference.inferencer import Inferencer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, init_hdf5_dataset
from .test_trainers import init_params

# the initial conditions, run in two batches, so that the second one starts a new rollout
_INDICES = [3, 7, 20, 41]
_BATCH_SIZE = 2
_ROLLOUT_STEPS = 3
_NOISE_SEED = 1234

_CASES = [
    # name, ensemble size, n_history, input noise, zenith
    ("plain", 1, 0, None, False),
    ("ensemble_noise_history_zenith", 2, 1, {"type": "diffusion", "mode": "concatenate", "n_channels": 1}, True),
]


class _RecordingBuffer:
    """Stands in for the rollout buffer, and records what the inferencer hands it."""

    def __init__(self):
        self.updates = []

    def zero_buffers(self):
        pass

    def finalize(self):
        pass

    def update(self, pred, tstamps, idt):
        self.updates.append((pred.clone(), tstamps.clone(), idt))


class _RecordingMetrics:
    """Stands in for the metrics handler, and records what the inferencer hands it."""

    def __init__(self):
        self.updates = []

    def zero_buffers(self):
        pass

    def finalize(self):
        return {}

    def update(self, pred, targ, loss, idt, mask=None):
        self.updates.append((pred.clone(), targ.clone(), idt))


class TestInferencerRollout(unittest.TestCase):
    """
    The inference loop against a rollout of the same initial conditions with targets.

    The inferencer streams the verification: it loads one sample per lead time
    without targets, takes the latest step of each as the verification and as
    the dynamic features and time the window moves on to, and rolls the model
    out on a rollout without targets. The reference loads the same initial
    conditions with their targets in one sample each, from the dataset's own
    indexing, and runs them through the model with a rollout over those targets.
    Both draw their noise from the same seed, so the predictions have to agree
    bit for bit, and so do the verification and its times.
    """

    @classmethod
    def setUpClass(cls, path: Optional[str] = "/tmp"):
        cls.datadir = tempfile.TemporaryDirectory(dir=path)
        (
            cls.train_path,
            cls.n_train_samples,
            cls.valid_path,
            cls.n_eval_samples,
            cls.stats_path,
            cls.metadata_path,
            _,
        ) = init_hdf5_dataset(cls.datadir.name)

    @classmethod
    def tearDownClass(cls):
        cls.datadir.cleanup()

    def setUp(self, path: Optional[str] = "/tmp"):
        disable_tf32()
        set_seed(333)
        self.tmpdir = tempfile.TemporaryDirectory(dir=path)
        self.exp_path = os.path.join(self.tmpdir.name, "experiment_dir")
        os.makedirs(os.path.join(self.exp_path, "training_checkpoints"))

    def tearDown(self):
        self.tmpdir.cleanup()

    def _params(self, ensemble_size, n_history, noise, zenith):
        params = init_params(
            self.exp_path,
            self.train_path,
            self.valid_path,
            self.stats_path,
            batch_size=_BATCH_SIZE,
            ensemble_size=ensemble_size,
            stochastic_size=1,
            n_history=n_history,
            n_future=0,
            normalization="zscore",
            # the inference loop asks for a prefetch factor, which needs a worker
            num_data_workers=1,
        )
        params.inf_data_path = self.valid_path
        params.n_train_samples = self.n_train_samples
        params.n_eval_samples = self.n_eval_samples
        params.add_zenith = zenith
        if noise is not None:
            params.input_noise = noise
        # the reference uses the same model, so its weights do not matter
        params.allow_random_weights = True
        return params

    def _reference(self, inferencer, ensemble_size):
        """The rollouts of the initial-condition batches with their targets, in the inferencer's order."""
        params = inferencer.params
        params.valid_autoreg_steps = _ROLLOUT_STEPS - 1
        dataset = get_dataloader(params, self.valid_path, mode="eval", device=inferencer.device)[0].dataset

        model = inferencer.model
        model.eval()
        pp = model.preprocessor
        pp.set_rng(reset=True, seed=_NOISE_SEED)

        references = []
        with torch.no_grad():
            for start in range(0, len(_INDICES), _BATCH_SIZE):
                samples = [dataset.get_sample_at_index(idx) for idx in _INDICES[start : start + _BATCH_SIZE]]
                batch = default_collate(samples)
                noise_source = pp.noise_source(batch_size=batch[INPUT].shape[0] * ensemble_size)
                rollout = Rollout(batch, noise_source=noise_source, ensemble_size=ensemble_size)
                # (B, E, n_steps, C, H, W)
                preds = rollout.unfold(model(rollout))
                references.append((batch, preds))

        return references

    @parameterized.expand(_CASES)
    def test_predictions_match_a_rollout_with_targets(self, name, ensemble_size, n_history, noise, zenith):
        params = self._params(ensemble_size, n_history, noise, zenith)
        inferencer = Inferencer(params, world_rank=0, device="cpu")
        pp = inferencer.preprocessor

        buffer, metrics = _RecordingBuffer(), _RecordingMetrics()
        pp.set_rng(reset=True, seed=_NOISE_SEED)
        inferencer._inference_indexlist(_INDICES, _ROLLOUT_STEPS, _BATCH_SIZE, metrics=metrics, rollout_buffer=buffer)

        references = self._reference(inferencer, ensemble_size)

        # one initial condition and the predictions per batch to the buffer, the predictions to the metrics
        n_batches = len(references)
        self.assertEqual([idt for *_, idt in buffer.updates], list(range(_ROLLOUT_STEPS + 1)) * n_batches)
        self.assertEqual([idt for *_, idt in metrics.updates], list(range(_ROLLOUT_STEPS)) * n_batches)

        for ib, (batch, preds) in enumerate(references):
            buffered = buffer.updates[ib * (_ROLLOUT_STEPS + 1) : (ib + 1) * (_ROLLOUT_STEPS + 1)]
            scored = metrics.updates[ib * _ROLLOUT_STEPS : (ib + 1) * _ROLLOUT_STEPS]

            with self.subTest(batch=ib, step="initial condition"):
                # the latest step of the window, (B, E, C, H, W), every member a copy; not its history
                inp, tstamps, _ = buffered[0]
                expected = expand_ensemble(batch[INPUT][:, -1], ensemble_size)
                self.assertTrue(torch.equal(inp, expected.reshape(-1, ensemble_size, *expected.shape[1:])))
                self.assertTrue(torch.equal(tstamps, batch[INPUT_TIMESTAMP][:, -1]), "initial condition time")

            for k in range(_ROLLOUT_STEPS):
                with self.subTest(batch=ib, step=k):
                    pred, tstamps, _ = buffered[k + 1]
                    metric_pred, targ, _ = scored[k]

                    self.assertTrue(torch.equal(pred, preds[:, :, k]), "predictions")
                    self.assertTrue(torch.equal(metric_pred, pred), "predictions to the metrics")
                    self.assertTrue(torch.equal(targ, batch[TARGET][:, k]), "verification")
                    self.assertTrue(torch.equal(tstamps, batch[TARGET_TIMESTAMP][:, k]), "verification times")

                    if (noise is not None) and (ensemble_size > 1):
                        self.assertFalse(torch.equal(pred[:, 0], pred[:, 1]), "the members draw different noise")


if __name__ == "__main__":
    unittest.main()
