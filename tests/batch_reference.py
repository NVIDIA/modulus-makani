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
Golden reference for the batches the data pipeline hands to the model.

The batches are moving from positional tuples to dicts, and the preprocessor
will consume those dicts directly. Neither change may alter what the model
sees, so this module records it, using the code as it was before the change:

* ``multifiles/...``: batches of ``MultifilesDataset`` over a small, seeded
  HDF5 dataset, in train, eval and inference mode, with and without history,
  future steps and the zenith angle. The input and output channels are
  unsorted and differ, so a mix-up of the two shows.
* ``dummy/...`` and ``dummy_inference/...``: the synthetic loaders, seeded.
* ``stepper/...``: what ``MultiStepWrapper`` packs for the model in the first
  and second step of a rollout (zenith angle, concatenated noise, static grid
  features), plus the rollout output.

Each entry is stored as the legacy tuple, one array per item. The reference
was recorded while the loaders still returned that tuple; now that they emit
dicts, :func:`compute_reference` converts them back with ``to_legacy_tuple``,
so the comparison holds the new code to the old layout.

``tests/test_batch_reference.py`` compares against the stored file. Regenerate
it only on purpose, when the batch content is meant to change::

    python -m tests.generate_batch_reference
"""

import os
import copy
import tempfile

import numpy as np
import torch
from torch.utils.data import default_collate

from makani.utils.dataloader import get_dataloader
from makani.utils.dataloaders.batch import to_legacy_tuple, INPUT, TARGET
from makani.models.noise import DummyNoiseS2
from makani.models.stepper import MultiStepWrapper

from .testutils import get_default_parameters, init_hdf5_dataset, H5_PATH

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "data", "batch_reference.npz")

# small enough to keep the file at a few hundred kB, big enough for a real grid
NUM_SAMPLES_PER_YEAR = 24
NUM_CHANNELS = 5
IMG_SHAPE = (8, 16)

# unsorted on purpose, and different between input and output
IN_CHANNELS = [3, 0, 2]
OUT_CHANNELS = [2, 3]

# the second sample's window crosses from the first file into the second
SAMPLE_INDICES = [0, NUM_SAMPLES_PER_YEAR - 2]

# (mode, n_history, n_future, add_zenith); n_future does not apply to inference
LOADER_CASES = [
    (mode, n_history, n_future, add_zenith)
    for mode in ("train", "eval", "inference")
    for n_history in (0, 1)
    for n_future in ((0, 1) if mode != "inference" else (0,))
    for add_zenith in (False, True)
]

# (n_history, n_future, add_zenith)
DUMMY_CASES = [(0, 0, False), (1, 1, True)]

# (n_history, add_zenith); the rollout always has two steps
STEPPER_CASES = [(n_history, add_zenith) for n_history in (0, 1) for add_zenith in (False, True)]


def case_name(*parts):
    return "_".join(str(int(p)) if isinstance(p, bool) else str(p) for p in parts)


def _as_items(batch, timestamps=False):
    """The batch as the legacy tuple, which had timestamps only where inference asked for them."""
    return to_legacy_tuple(batch, timestamps=timestamps)


def _store(result, prefix, items):
    result[f"{prefix}/num_items"] = np.asarray(len(items))
    for idx, item in enumerate(items):
        result[f"{prefix}/{idx}"] = item.detach().cpu().numpy()


def _init_dataset(root):
    train_path, _, valid_path, _, stats_path, _, _ = init_hdf5_dataset(
        root,
        num_samples_per_year=NUM_SAMPLES_PER_YEAR,
        num_channels=NUM_CHANNELS,
        img_size_h=IMG_SHAPE[0],
        img_size_w=IMG_SHAPE[1],
    )

    # the fixture writes identity statistics, which would hide a mix-up of the
    # input and the output normalization
    channels = np.arange(NUM_CHANNELS, dtype=np.float64).reshape(1, -1, 1, 1)
    np.save(os.path.join(stats_path, "global_means.npy"), 0.1 * channels)
    np.save(os.path.join(stats_path, "global_stds.npy"), 1.0 + 0.5 * channels)

    return train_path, valid_path, stats_path


def _params(paths, n_history, n_future, add_zenith, out_channels=OUT_CHANNELS):
    train_path, valid_path, stats_path = paths

    params = get_default_parameters()
    params.train_data_path = train_path
    params.valid_data_path = valid_path
    params.global_means_path = os.path.join(stats_path, "global_means.npy")
    params.global_stds_path = os.path.join(stats_path, "global_stds.npy")
    params.normalization = "zscore"
    params.h5_path = H5_PATH
    params.dhours = (365 * 24) // NUM_SAMPLES_PER_YEAR

    params.in_channels = list(IN_CHANNELS)
    params.out_channels = list(out_channels)
    params.N_in_channels = len(params.in_channels)
    params.N_out_channels = len(params.out_channels)

    params.n_history = n_history
    params.n_future = n_future
    params.valid_autoreg_steps = n_future
    params.add_zenith = add_zenith

    params.batch_size = len(SAMPLE_INDICES)
    params.num_data_workers = 0

    return params


def _multifiles_batch(params, mode):
    loader, _, _ = get_dataloader(params, params.train_data_path, device=torch.device("cpu"), mode=mode)
    return default_collate([loader.dataset[idx] for idx in SAMPLE_INDICES])


def _capture_multifiles(paths, result):
    for mode, n_history, n_future, add_zenith in LOADER_CASES:
        params = _params(paths, n_history, n_future, add_zenith)
        batch = _multifiles_batch(params, mode)
        items = _as_items(batch, timestamps=(mode == "inference"))
        _store(result, f"multifiles/{case_name(mode, n_history, n_future, add_zenith)}", items)


def _dummy_kwargs(n_history, n_future, add_zenith):
    return dict(
        location="/nonexistent",
        device=torch.device("cpu"),
        batch_size=len(SAMPLE_INDICES),
        dt=1,
        dhours=6,
        in_channels=list(IN_CHANNELS),
        out_channels=list(OUT_CHANNELS),
        img_shape=IMG_SHAPE,
        n_samples_per_epoch=32,
        n_history=n_history,
        n_future=n_future,
        add_zenith=add_zenith,
    )


def _capture_dummy(result):
    from makani.utils.dataloaders.data_loader_dummy import DummyLoader, DummyInferenceDataset

    for n_history, n_future, add_zenith in DUMMY_CASES:
        name = case_name(n_history, n_future, add_zenith)

        # the payload is drawn from the global generator at construction
        torch.manual_seed(333)
        loader = DummyLoader(**_dummy_kwargs(n_history, n_future, add_zenith))
        _store(result, f"dummy/{name}", _as_items(next(iter(loader))))

        # the inferencer drives this one, without a target
        torch.manual_seed(333)
        dataset = DummyInferenceDataset(**_dummy_kwargs(n_history, n_future, add_zenith), return_target=False)
        batch = default_collate([dataset[idx] for idx in SAMPLE_INDICES])
        _store(result, f"dummy_inference/{name}", _as_items(batch, timestamps=True))


class _PatternNoise(DummyNoiseS2):
    """Noise with a fixed pattern that changes with every update.

    Seeded random noise would tie the reference to the random number generator
    of one torch version. A pattern built from integers is exact everywhere and
    still shows where the noise ends up and when it is refreshed.
    """

    def __init__(self, *args, **kwargs):
        # set first, in case the base class updates during construction
        self.num_updates = 0
        super().__init__(*args, **kwargs)

    def update(self, replace_state=False, batch_size=None):
        if batch_size is not None:
            self._ensure_state(batch_size)

        self.num_updates += 1
        with torch.no_grad():
            pattern = torch.arange(self.state.numel(), dtype=torch.float32) % 7
            self.state.copy_(pattern.reshape(self.state.shape) * (0.125 * self.num_updates))


class _Probe(torch.nn.Module):
    """Stands in for the network: records its input and predicts a simple function of it."""

    def __init__(self, num_out_channels):
        super().__init__()
        self.num_out_channels = num_out_channels
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return 0.5 * x[:, : self.num_out_channels] + 0.25


def _capture_stepper(paths, result):
    for n_history, add_zenith in STEPPER_CASES:
        # the rollout feeds predictions back as input, which needs matching channels
        params = _params(paths, n_history, n_future=1, add_zenith=add_zenith, out_channels=IN_CHANNELS)
        batch = _multifiles_batch(copy.deepcopy(params), "train")

        params.img_shape_x, params.img_shape_y = IMG_SHAPE
        params.img_shape_x_resampled, params.img_shape_y_resampled = IMG_SHAPE
        params.add_grid = True
        params.input_noise = {"type": "dummy", "mode": "concatenate", "n_channels": 1}

        probe = _Probe(len(IN_CHANNELS))
        wrapper = MultiStepWrapper(params, lambda: probe)
        wrapper.preprocessor.input_noise = _PatternNoise(
            img_shape=list(IMG_SHAPE),
            batch_size=params.batch_size,
            num_channels=1,
            num_time_steps=n_history + 1,
        )
        wrapper.train()

        # what the trainers do with a batch
        wrapper.preprocessor.cache_dynamic_features(batch)
        with torch.no_grad():
            out = wrapper(batch[INPUT])

        # recorded with the time steps folded into the channels, as the losses take them
        flatten = wrapper.preprocessor.flatten_history
        _store(
            result, f"stepper/{case_name(n_history, add_zenith)}", (*probe.inputs, flatten(out), flatten(batch[TARGET]))
        )


def compute_reference():
    """Capture every entry of the reference from the code as it is now.

    Returns
    -------
    dict
        Arrays keyed ``<group>/<case>/<item>``, plus ``<group>/<case>/num_items``.
    """
    result = {}
    with tempfile.TemporaryDirectory() as root:
        paths = _init_dataset(root)
        _capture_multifiles(paths, result)
        _capture_dummy(result)
        _capture_stepper(paths, result)

    return result
