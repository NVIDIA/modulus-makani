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

import datetime as dt
import json
import os
import sys
import tempfile
import unittest
from typing import Optional

import h5py as h5
import numpy as np
import torch
import torch_harmonics as th
from torch.utils.data import default_collate
from parameterized import parameterized

from makani.utils.dataloader import get_dataloader
from makani.utils.dataloaders.data_helpers import get_date_from_string
from makani.utils.dataloaders.batch import INPUT, INPUT_TIMESTAMP, TARGET, TARGET_TIMESTAMP
from makani.utils.dataloaders.rollout import Rollout
from makani.utils.functions import expand_ensemble
from makani.utils.inference.inferencer import Inferencer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, init_hdf5_dataset, H5_PATH
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

# score_model over a date range of the 2019 validation year: the initial conditions on Jan 4, 6, 8 and 10,
# in two batches, rolled out over three lead times
_SCORE_START = "2019-01-04T00:00:00"
_SCORE_END = "2019-01-12T00:00:00"
_SCORE_STEP_HOURS = 48
_SCORE_AUTOREG_STEPS = 2


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


class _InferencerTestCase(unittest.TestCase):
    """The data, the experiment directory and the parameters the inferencer tests share."""

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
        # the output buffers place their data on the dataset's coordinates, which parse_dataset_metadata
        # reads from the metadata file in a real run
        with open(os.path.join(self.metadata_path, "data.json"), "r") as f:
            coords = json.load(f)["coords"]
        params.lat = coords["lat"]
        params.lon = coords["lon"]
        params.n_train_samples = self.n_train_samples
        params.n_eval_samples = self.n_eval_samples
        params.add_zenith = zenith
        if noise is not None:
            params.input_noise = noise
        # the reference uses the same model, so its weights do not matter
        params.allow_random_weights = True
        return params


class TestInferencerRollout(_InferencerTestCase):
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


def _order_multiplicity(nlon, mmax):
    """How often each order m < mmax occurs in the spectrum of a real field on nlon longitudes."""
    # the negative orders count as much as the positive ones, except for m = 0 and, for an even nlon, the
    # Nyquist order nlon / 2, which are their own conjugates
    weights = torch.full((mmax,), 2.0)
    weights[0] = 1.0
    if (nlon % 2 == 0) and (nlon // 2 < mmax):
        weights[nlon // 2] = 1.0
    return weights


def _power_spectrum(sht, fields):
    """Total power per degree, (..., H, W) -> (..., L), as the spectrum buffer computes it."""
    power = torch.square(torch.abs(sht(torch.from_numpy(fields))))
    power = power * _order_multiplicity(fields.shape[-1], power.shape[-1])
    return torch.sum(power, dim=-1).to(torch.float64)


def _zonal_power_spectrum(fields, lat):
    """Latitude-weighted zonal power per order, (..., H, W) -> (..., H, M), as the zonal spectrum buffer computes it."""
    coeffs = torch.fft.rfft(torch.from_numpy(fields), dim=-1, norm="forward")
    power = torch.cos(torch.deg2rad(torch.from_numpy(lat))).reshape(-1, 1) * torch.square(torch.abs(coeffs))
    power = power * _order_multiplicity(fields.shape[-1], power.shape[-1]).to(power.dtype)
    return power.to(torch.float64)


class TestInferencerScoring(_InferencerTestCase):
    """
    ``score_model``, the entry point of ``makani/inference.py``, over a short date range with every output it
    can write. The outputs are checked against each other and against the raw data: the initial conditions
    in the rollout file against the data at their times, the spectra against the spectra of the rollout file's
    predictions and of the data at the lead times, and the logged scores against the metrics file.
    """

    def _score(self, params, outputs):
        params.valid_autoreg_steps = _SCORE_AUTOREG_STEPS
        inferencer = Inferencer(params, world_rank=0, device="cpu")

        # keep the logs score_model hands to log_score, which it does not return
        captured = {}
        log_score = inferencer.log_score

        def _log_score(scoring_logs, scoring_time):
            captured["logs"] = scoring_logs
            log_score(scoring_logs, scoring_time)

        inferencer.log_score = _log_score

        # output_file, metrics_file, bias_file, spectrum_file and zonal_spectrum_file
        paths = {name: os.path.join(self.tmpdir.name, f"{name}.h5") for name in outputs}
        inferencer.score_model(
            start_date=_SCORE_START,
            end_date=_SCORE_END,
            date_step=_SCORE_STEP_HOURS,
            **{f"{name}_file": path for name, path in paths.items()},
        )

        return inferencer, captured["logs"], paths

    def _raw_indices(self, inferencer):
        """The positions of the initial conditions in the raw data file, in the order score_model runs them."""
        dataset = inferencer.valid_dataset
        start = dataset.get_index_at_time(get_date_from_string(_SCORE_START))
        end = dataset.get_index_at_time(get_date_from_string(_SCORE_END))
        indices = range(start, end, _SCORE_STEP_HOURS // dataset.dhours)

        # a sample covers the window idx, ..., idx + n_history * dt, and its latest step is the initial condition
        history_offset = dataset.n_history * dataset.dt

        year_start = dt.datetime(year=2019, month=1, day=1, tzinfo=dt.timezone.utc)
        sample_hours = dt.timedelta(hours=dataset.dhours)
        return [int((dataset.get_time_at_index(idx) - year_start) / sample_hours) + history_offset for idx in indices]

    def _raw_data(self):
        with h5.File(os.path.join(self.valid_path, "2019.h5"), "r") as hf:
            return hf[H5_PATH][...]

    def _assert_close(self, actual, expected, msg, rtol=1e-4):
        actual = torch.as_tensor(np.asarray(actual), dtype=torch.float64)
        expected = torch.as_tensor(expected, dtype=torch.float64)
        atol = 1e-6 * max(expected.abs().max().item(), 1.0)
        self.assertTrue(torch.allclose(actual, expected, rtol=rtol, atol=atol), msg)

    @parameterized.expand(_CASES)
    def test_score_model_outputs_are_consistent(self, name, ensemble_size, n_history, noise, zenith):
        params = self._params(ensemble_size, n_history, noise, zenith)
        inferencer, logs, paths = self._score(params, ["metrics", "output", "spectrum", "zonal_spectrum"])

        rollout_steps = _SCORE_AUTOREG_STEPS + 1
        channels = list(inferencer.params.channel_names)
        raw_ics = self._raw_indices(inferencer)
        raw = self._raw_data()
        dt_steps = inferencer.params.dt
        num_ics = len(raw_ics)

        # the rollout file: the initial condition and the predictions per lead time, in physical units
        with h5.File(paths["output"], "r") as hf:
            fields = hf["fields"][...]
            timestamps = hf["timestamp"][...]
            rollout_lead_times = hf["lead_time"][...]
        self.assertEqual(fields.shape, (num_ics, rollout_steps + 1, ensemble_size, len(channels), *raw.shape[-2:]))
        self.assertTrue(np.all(np.diff(timestamps) == _SCORE_STEP_HOURS * 3600), "initial condition times")
        for member in range(ensemble_size):
            self._assert_close(fields[:, 0, member], raw[raw_ics], f"initial conditions of member {member}")
        self.assertTrue(np.all(np.isfinite(fields)), "predictions")

        # the verification at each lead time, from the raw data
        targets = np.stack([raw[[idx + (k + 1) * dt_steps for idx in raw_ics]] for k in range(rollout_steps)], axis=1)

        # the metrics file and the logs agree, and the lead times with the rollout file
        with h5.File(paths["metrics"], "r") as hf:
            for metric in ["L1", "RMSE", "ACC"]:
                with self.subTest(metric=metric):
                    data = hf[metric]["metric_data"][...]
                    self.assertEqual(data.shape, (rollout_steps, len(channels)))
                    self.assertTrue(np.all(np.isfinite(data)))
                    self.assertTrue(np.allclose(hf[metric]["lead_time"][...], rollout_lead_times[1:]), "lead times")

                    last_lead_time = int(rollout_lead_times[-1])
                    for ic, channel in enumerate(channels):
                        logged = logs["metrics"][f"validation {metric} {channel}({last_lead_time})"]
                        self.assertAlmostEqual(float(logged), float(data[-1, ic]), places=5)
        self.assertTrue(np.isfinite(logs["base"]["validation loss"]), "validation loss")

        # the spectra: the verification in the first ensemble slot, the members after it
        sht = th.RealSHT(*raw.shape[-2:], grid=inferencer.params.model_grid_type)
        with h5.File(paths["spectrum"], "r") as hf:
            spectrum_mean, spectrum_std = hf["mean"][...], hf["std"][...]
            spectrum_lead_times = hf["lead_time"][...]
        with h5.File(paths["zonal_spectrum"], "r") as hf:
            zonal_mean, zonal_std = hf["mean"][...], hf["std"][...]
            zonal_lead_times = hf["lead_time"][...]
            lat = hf["lat"][...]
        # every output file gives the lead times in hours, like the rollout file
        self.assertTrue(np.allclose(spectrum_lead_times, rollout_lead_times[1:]), "spectrum lead times")
        self.assertTrue(np.allclose(zonal_lead_times, rollout_lead_times[1:]), "zonal spectrum lead times")
        self.assertEqual(spectrum_mean.shape, (rollout_steps, len(channels), ensemble_size + 1, sht.lmax))
        nlat, nlon = raw.shape[-2:]
        self.assertEqual(zonal_mean.shape, (rollout_steps, len(channels), ensemble_size + 1, nlat, nlon // 2 + 1))

        for k in range(rollout_steps):
            with self.subTest(lead_time=k):
                # (N, E + 1, C, H, W)
                states = np.concatenate([targets[:, k, None], fields[:, k + 1]], axis=1)

                # (N, C, E + 1, L), averaged over the initial conditions
                spect = _power_spectrum(sht, states).permute(0, 2, 1, 3)
                self._assert_close(spectrum_mean[k], spect.mean(dim=0), "spectrum mean")
                self._assert_close(spectrum_std[k], spect.std(dim=0), "spectrum std", rtol=1e-3)

                # (N, C, E + 1, H, M)
                zonal = _zonal_power_spectrum(states, lat).permute(0, 2, 1, 3, 4)
                self._assert_close(zonal_mean[k], zonal.mean(dim=0), "zonal spectrum mean")
                self._assert_close(zonal_std[k], zonal.std(dim=0), "zonal spectrum std", rtol=1e-3)

    def test_score_model_bias(self):
        params = self._params(1, 0, None, False)
        inferencer, _, paths = self._score(params, ["output", "bias"])

        rollout_steps = _SCORE_AUTOREG_STEPS + 1
        raw_ics = self._raw_indices(inferencer)
        raw = self._raw_data()

        with h5.File(paths["output"], "r") as hf:
            fields = hf["fields"][...]
            rollout_lead_times = hf["lead_time"][...]
        with h5.File(paths["bias"], "r") as hf:
            bias_mean = hf["mean"][...]
            bias_lead_times = hf["lead_time"][...]
        self.assertEqual(bias_mean.shape, (rollout_steps, raw.shape[1], *raw.shape[-2:]))
        # lead times in hours, like the rollout file
        self.assertTrue(np.allclose(bias_lead_times, rollout_lead_times[1:]), "bias lead times")

        # the mean error over the initial conditions per lead time, from the rollout file and the raw data
        for k in range(rollout_steps):
            with self.subTest(lead_time=k):
                targets = raw[[idx + (k + 1) * inferencer.params.dt for idx in raw_ics]]
                error = fields[:, k + 1, 0] - targets
                self._assert_close(bias_mean[k], torch.from_numpy(error).mean(dim=0), "mean error")


if __name__ == "__main__":
    unittest.main()
