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

"""
Unit tests for ``SampleSource``, the sample assembly above the storage backends.

One suite, ``_BaseESTests``, is run against every layout a dataset can be stored
in: per-year HDF5, a single concatenated HDF5 file, per-year zarr, and a
WeatherBench2 store. What is under test is the assembly -- windows, cropping,
subsampling, channel ordering, shuffling, epoch cycling, zenith angles -- and
running it against all four is what keeps the backends honest, since every one
of them has to produce the same answers.

``SampleSource`` implements the DALI external-source protocol but has no DALI
dependency: it is a plain callable taking a ``SampleInfo``. The tests mock that
namedtuple and call ``__call__`` directly, so they need neither DALI nor a GPU.
The backends themselves are unit tested separately, in ``test_data_backends.py``.

Dataset fixture
---------------
TestMakaniHDF5 reuses init_hdf5_dataset() from testutils, which creates two training
years (2017, 2018) of 365 samples each (dhours=24) under a temp directory.

TestMakaniConcat uses the same dimensions but builds a single concatenated
HDF5 file.

Boundary exclusion test
-----------------------
With dhours=24 and the default n_history=0/n_future=0/dt=1 config:
  lookback_hours = 24 * (n_future+1) = 24
  boundary "2017-01-25T00:00:00+00:00" → exclusion range [2017-01-24, 2017-01-25)
  sample 23 of 2017 (= 2017-01-01 + 23*24h = 2017-01-24) falls inside → excluded.
"""

import math
import os
import sys
import contextlib
import datetime as dt
import io
import tempfile
import time
import unittest
from unittest import mock

import h5py
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import (
    H5_PATH,
    NUM_CHANNELS,
    IMG_SIZE_H,
    IMG_SIZE_W,
    CHANNEL_NAMES,
    NUM_SAMPLES_PER_YEAR,
    TRAIN_YEARS,
    DHOURS,
    init_hdf5_dataset,
    init_zarr_dataset,
    init_wb2_zarr_dataset,
    compare_arrays,
    local_timezone,
)

# ---------------------------------------------------------------------------
# Dataset parameters — derived from testutils constants, not hardcoded
# ---------------------------------------------------------------------------
_N_CH = NUM_CHANNELS
_IMG_H = IMG_SIZE_H
_IMG_W = IMG_SIZE_W
_N_PER_YEAR = NUM_SAMPLES_PER_YEAR
_DHOURS = DHOURS
_YEARS = TRAIN_YEARS
_N_TOTAL = len(_YEARS) * _N_PER_YEAR

# Boundary timestamp = exactly sample 24 of the first training year.
# lookback_hours = DHOURS*(n_future+1) = DHOURS*1, so the exclusion window
# is [sample_24_time - DHOURS, sample_24_time), which contains sample 23.
_BOUNDARY_DT = dt.datetime(_YEARS[0], 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(hours=24 * _DHOURS)  # sample 24
_BOUNDARY_TS = _BOUNDARY_DT.isoformat()


# ---------------------------------------------------------------------------
# Mock DALI SampleInfo
# ---------------------------------------------------------------------------
class _SampleInfo:
    def __init__(self, idx_in_epoch=0, epoch_idx=0, iteration=0):
        self.idx_in_epoch = idx_in_epoch
        self.epoch_idx = epoch_idx
        self.iteration = iteration


# ---------------------------------------------------------------------------
# Distinctive-channel datasets (for channel-reordering test only)
# Channel c has the constant value (c+1.0) so reordering can be verified.
# ---------------------------------------------------------------------------


def _distinctive_year_arrays(year):
    """Return (data, timestamps) arrays for one year of distinctive-channel data."""
    year_start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc).timestamp()
    timestamps = year_start + np.arange(_N_PER_YEAR, dtype=np.float64) * _DHOURS * 3600
    data = np.zeros((_N_PER_YEAR, _N_CH, _IMG_H, _IMG_W), dtype=np.float32)
    for c in range(_N_CH):
        data[:, c, :, :] = float(c + 1)
    return data, timestamps


def _make_distinctive_dir(root):
    """Per-year directory with distinctive-channel data (for SampleSource)."""
    os.makedirs(root, exist_ok=True)
    for year in _YEARS:
        data, timestamps = _distinctive_year_arrays(year)
        with h5py.File(os.path.join(root, f"{year}.h5"), "w") as f:
            ds = f.create_dataset(H5_PATH, data=data)
            ts = f.create_dataset("timestamp", data=timestamps)
            ts.make_scale("timestamp")
            ds.dims[0].attach_scale(ts)
    return root


def _make_distinctive_zarr_dir(root):
    """Per-year zarr directory with distinctive-channel data (makani flat format)."""
    import zarr

    os.makedirs(root, exist_ok=True)
    for year in _YEARS:
        data, timestamps = _distinctive_year_arrays(year)
        store_path = os.path.join(root, f"{year}.zarr")
        g = zarr.open_group(store_path, mode="w")
        g.create_array(H5_PATH, data=data, chunks=(1, _N_CH, _IMG_H, _IMG_W))
        g.create_array("time", data=timestamps)
        zarr.consolidate_metadata(store_path)
    return root


def _make_distinctive_zarr_wb2_dir(root):
    """Per-year WB2-layout zarr directory with distinctive-channel data."""
    import zarr
    import re
    from makani.utils.dataloaders.wb2_helpers import atmospheric_wb2_name, surface_wb2_name

    os.makedirs(root, exist_ok=True)

    # parse CHANNEL_NAMES to determine variable layout
    atm_vars, surf_vars = {}, {}
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        m = re.search(r"[0-9]{1,4}$", ch_name)
        if m and ch_name != "d2":
            wb2n = atmospheric_wb2_name(ch_name[: m.start()])
            atm_vars.setdefault(wb2n, {})[int(m.group())] = ch_idx
        else:
            surf_vars[surface_wb2_name(ch_name)] = ch_idx

    all_levels = sorted({lv for lvs in atm_vars.values() for lv in lvs})
    levels = np.array(all_levels, dtype=np.int64)
    level_to_idx = {lv: i for i, lv in enumerate(levels)}

    for year in _YEARS:
        data, timestamps = _distinctive_year_arrays(year)
        store_path = os.path.join(root, f"{year}.zarr")
        g = zarr.open_group(store_path, mode="w")

        times_ns = np.array([np.datetime64(int(ts * 1e9), "ns") for ts in timestamps])
        g.create_array("time", data=times_ns)
        g.create_array("latitude", data=np.linspace(90, -90, _IMG_H, endpoint=True, dtype=np.float32))
        g.create_array("longitude", data=np.linspace(0, 360, _IMG_W, endpoint=False, dtype=np.float32))
        g.create_array("level", data=levels)

        for wb2n, ch_idx in surf_vars.items():
            g.create_array(wb2n, data=data[:, ch_idx, :, :], chunks=(1, _IMG_H, _IMG_W))

        for wb2n, level_map in atm_vars.items():
            arr = np.zeros((_N_PER_YEAR, len(levels), _IMG_H, _IMG_W), dtype=np.float32)
            for lv, ch_idx in level_map.items():
                arr[:, level_to_idx[lv], :, :] = data[:, ch_idx, :, :]
            g.create_array(wb2n, data=arr, chunks=(1, len(levels), _IMG_H, _IMG_W))

        zarr.consolidate_metadata(store_path)
    return root


def _make_distinctive_concat(root):
    """Single concatenated file with distinctive-channel data (for SampleSource)."""
    os.makedirs(root, exist_ok=True)
    all_data, all_ts = [], []
    for year in _YEARS:
        data, timestamps = _distinctive_year_arrays(year)
        all_data.append(data)
        all_ts.append(timestamps)
    path = os.path.join(root, "train_concat.h5")
    with h5py.File(path, "w") as f:
        ds = f.create_dataset(H5_PATH, data=np.concatenate(all_data, axis=0))
        ts = f.create_dataset("timestamp", data=np.concatenate(all_ts))
        ts.make_scale("timestamp")
        ds.dims[0].attach_scale(ts)
    return path


# ---------------------------------------------------------------------------
# ES factory helpers
# ---------------------------------------------------------------------------


def _default_kwargs(location):
    return dict(
        location=location,
        max_samples=None,
        samples_per_epoch=None,
        train=False,  # no shuffle → deterministic
        batch_size=1,
        dt=1,
        dhours=_DHOURS,
        n_history=0,
        n_future=0,
        in_channels=np.arange(_N_CH),
        out_channels=np.arange(_N_CH),
        crop_size=[None, None],
        crop_anchor=[0, 0],
        subsampling_factor=1,
        num_shards=1,
        shard_id=0,
        io_grid=[1, 1, 1],
        io_rank=[0, 0, 0],
        device_id=0,
        truncate_old=False,
        enable_logging=False,
        zenith_angle=False,
        return_timestamp=False,
        is_parallel=False,
        timestamp_boundary_list=[],
    )


def _make_source(location, **overrides):
    """Build a SampleSource for a location, whatever layout is stored there.

    Nothing layout-specific is needed here any more: the backend is detected
    from the location and opens its own handles on first read, so a single
    factory serves every suite.
    """
    from makani.utils.dataloaders.sample_source import SampleSource

    kw = _default_kwargs(location)
    kw.update(overrides)
    return SampleSource(**kw)


def _ts_to_posix(t):
    """Normalise a timestamp to a POSIX float.

    SampleSource._compute_timestamps() returns numpy float64 (seconds since epoch).
    SampleSource._compute_timestamps_and_zenith_angle() returns datetime objects
    (slices of self.timestamps).  This helper handles both.
    """
    try:
        return float(t.timestamp())  # datetime path
    except AttributeError:
        return float(t)  # float / np.float64 path


# ===========================================================================
# Shared test logic (mixin)
# ===========================================================================


class _BaseESTests:
    """
    Tests shared between TestMakaniHDF5 and TestMakaniConcat.

    Subclasses must implement:
        _make(**overrides)             → ES backed by standard random dataset
        _make_distinctive(**overrides) → ES backed by distinctive-channel dataset
        _expected_n_valid(dt, n_history, n_future)
                                       → number of usable samples for the
                                         given window config.  Differs between
                                         backends because SampleSource drops
                                         windows that cross year boundaries
                                         while SampleSource does not.
    """

    def _expected_n_valid(self, dt, n_history, n_future):
        raise NotImplementedError

    @property
    def expected_n_valid(self):
        return self._expected_n_valid(dt=1, n_history=0, n_future=0)

    @property
    def expected_n_valid_with_boundary(self):
        # the boundary-list fixture excludes exactly one extra sample
        return self.expected_n_valid - 1

    def _call0(self, es):
        """Call es at the very first sample of epoch 0."""
        return es(_SampleInfo(idx_in_epoch=0, epoch_idx=0, iteration=0))

    # ---- 1. basic shapes ------------------------------------------------

    def test_basic_output_shapes(self):
        es = self._make()
        inp, tar = self._call0(es)
        self.assertEqual(inp.shape, (1, _N_CH, _IMG_H, _IMG_W))
        self.assertEqual(tar.shape, (1, _N_CH, _IMG_H, _IMG_W))

    # ---- 2. n_history ---------------------------------------------------

    def test_n_history_adds_input_timesteps(self):
        es = self._make(n_history=2)
        inp, _tar = self._call0(es)
        self.assertEqual(inp.shape[0], 3)  # n_history + 1

    # ---- 3. n_future ----------------------------------------------------

    def test_n_future_adds_target_timesteps(self):
        es = self._make(n_future=2)
        _inp, tar = self._call0(es)
        self.assertEqual(tar.shape[0], 3)  # n_future + 1

    # ---- 4. subsampling -------------------------------------------------

    def test_subsampling_halves_spatial_dims(self):
        es = self._make(subsampling_factor=2)
        inp, tar = self._call0(es)
        self.assertEqual(inp.shape, (1, _N_CH, math.ceil(_IMG_H / 2), math.ceil(_IMG_W / 2)))
        self.assertEqual(tar.shape, (1, _N_CH, math.ceil(_IMG_H / 2), math.ceil(_IMG_W / 2)))

    def test_subsampling_quarter_spatial_dims(self):
        es = self._make(subsampling_factor=4)
        inp, tar = self._call0(es)
        self.assertEqual(inp.shape, (1, _N_CH, math.ceil(_IMG_H / 4), math.ceil(_IMG_W / 4)))
        self.assertEqual(tar.shape, (1, _N_CH, math.ceil(_IMG_H / 4), math.ceil(_IMG_W / 4)))

    def test_subsampling_matches_manual_subsample(self, verbose=False):
        """ES subsampling == slicing the full-res sample every S pixels."""
        S = 2
        es_full = self._make()
        es_sub = self._make(subsampling_factor=S)
        inp_full, tar_full = self._call0(es_full)
        inp_sub, tar_sub = self._call0(es_sub)
        self.assertTrue(compare_arrays("subsampled inp", inp_sub, inp_full[:, :, ::S, ::S], verbose=verbose))
        self.assertTrue(compare_arrays("subsampled tar", tar_sub, tar_full[:, :, ::S, ::S], verbose=verbose))

    # ---- 5. spatial crop ------------------------------------------------

    def test_spatial_crop_limits_dims(self):
        crop_h, crop_w = _IMG_H // 2, _IMG_W // 2
        es = self._make(crop_size=[crop_h, crop_w], crop_anchor=[0, 0])
        inp, tar = self._call0(es)
        self.assertEqual(inp.shape, (1, _N_CH, crop_h, crop_w))
        self.assertEqual(tar.shape, (1, _N_CH, crop_h, crop_w))

    def test_spatial_crop_with_nonzero_anchor(self):
        crop_h, crop_w = _IMG_H // 2, _IMG_W // 2
        anchor_h, anchor_w = 4, 8
        es = self._make(crop_size=[crop_h, crop_w], crop_anchor=[anchor_h, anchor_w])
        inp, _tar = self._call0(es)
        self.assertEqual(inp.shape, (1, _N_CH, crop_h, crop_w))

    def test_crop_matches_manual_crop(self, verbose=False):
        """ES crop (zero anchor) == slicing the full-res sample."""
        crop_h, crop_w = _IMG_H // 2, _IMG_W // 2
        es_full = self._make()
        es_crop = self._make(crop_size=[crop_h, crop_w], crop_anchor=[0, 0])
        inp_full, tar_full = self._call0(es_full)
        inp_crop, tar_crop = self._call0(es_crop)
        self.assertTrue(compare_arrays("cropped inp", inp_crop, inp_full[:, :, :crop_h, :crop_w], verbose=verbose))
        self.assertTrue(compare_arrays("cropped tar", tar_crop, tar_full[:, :, :crop_h, :crop_w], verbose=verbose))

    def test_crop_with_anchor_matches_manual_crop(self, verbose=False):
        """ES crop with nonzero anchor == slicing full-res at the anchor offset."""
        crop_h, crop_w = _IMG_H // 2, _IMG_W // 2
        anchor_h, anchor_w = 4, 8
        es_full = self._make()
        es_crop = self._make(crop_size=[crop_h, crop_w], crop_anchor=[anchor_h, anchor_w])
        inp_full, tar_full = self._call0(es_full)
        inp_crop, tar_crop = self._call0(es_crop)
        self.assertTrue(
            compare_arrays(
                "anchored crop inp",
                inp_crop,
                inp_full[:, :, anchor_h : anchor_h + crop_h, anchor_w : anchor_w + crop_w],
                verbose=verbose,
            )
        )
        self.assertTrue(
            compare_arrays(
                "anchored crop tar",
                tar_crop,
                tar_full[:, :, anchor_h : anchor_h + crop_h, anchor_w : anchor_w + crop_w],
                verbose=verbose,
            )
        )

    # ---- 6. temporal window consistency ---------------------------------

    def test_window_matches_individual_samples(self, verbose=False):
        """
        Load one windowed sample at index i with n_history=1, n_future=3.
        The resulting (inp, tar) should equal 6 individually loaded frames
        stitched together:
          inp[0..1]  ←  single samples at global indices i-1, i
          tar[0..3]  ←  single samples at global indices i+1, i+2, i+3, i+4

        With train=False and truncate_old=False, the windowed ES starts at
        samples_start = dt * n_history = 1, so i = 1 and the window spans
        global indices 0..5.  The single-sample ES (n_history=0, n_future=0)
        starts at 0, so idx_in_epoch=k deterministically selects sample k.
        """
        N_HIST, N_FUT = 1, 3

        es_window = self._make(n_history=N_HIST, n_future=N_FUT)
        es_single = self._make()  # n_history=0, n_future=0

        inp_win, tar_win = self._call0(es_window)

        # collect (N_HIST+1) + (N_FUT+1) = 6 individual frames
        n_frames = (N_HIST + 1) + (N_FUT + 1)
        singles = []
        for k in range(n_frames):
            inp_k, _ = es_single(_SampleInfo(idx_in_epoch=k, epoch_idx=0, iteration=0))
            singles.append(inp_k[0])  # shape (n_ch, H, W) — copy returned by _reorder_channels

        for t in range(N_HIST + 1):
            self.assertTrue(
                compare_arrays(
                    f"inp t={t}",
                    inp_win[t],
                    singles[t],
                    verbose=verbose,
                )
            )
        for t in range(N_FUT + 1):
            self.assertTrue(
                compare_arrays(
                    f"tar t={t}",
                    tar_win[t],
                    singles[N_HIST + 1 + t],
                    verbose=verbose,
                )
            )

    # ---- 7. channel subset ----------------------------------------------

    def test_channel_subset_shapes(self):
        es = self._make(
            in_channels=np.array([0, 1, 2]),
            out_channels=np.array([0, 1]),
        )
        inp, tar = self._call0(es)
        self.assertEqual(inp.shape[1], 3)
        self.assertEqual(tar.shape[1], 2)

    # ---- 7. unsorted channels reordered ---------------------------------

    def test_unsorted_channels_reordered(self, verbose=False):
        """
        With in_channels=[3,1,0,2] and channel c having constant value (c+1):
          output position 0 → channel 3 → value 4
          output position 1 → channel 1 → value 2
          output position 2 → channel 0 → value 1
          output position 3 → channel 2 → value 3
        """
        sel = [3, 1, 0, 2]
        es = self._make_distinctive(in_channels=np.array(sel))
        inp, _tar = self._call0(es)
        # expected shape matches the number of selected channels (4), not _N_CH (5)
        vals = [4.0, 2.0, 1.0, 3.0]
        expected = np.empty((len(sel), _IMG_H, _IMG_W), dtype=np.float32)
        for pos, val in enumerate(vals):
            expected[pos] = val
        self.assertTrue(compare_arrays("unsorted channel reorder", inp[0], expected, verbose=verbose))

    # ---- 7b. unsorted out_channels reordered ----------------------------

    def test_unsorted_out_channels_reordered(self, verbose=False):
        """
        With out_channels=[3,1,0,2] and channel c having constant value (c+1):
          tar output position 0 → channel 3 → value 4
          tar output position 1 → channel 1 → value 2
          tar output position 2 → channel 0 → value 1
          tar output position 3 → channel 2 → value 3
        """
        sel = [3, 1, 0, 2]
        es = self._make_distinctive(out_channels=np.array(sel))
        _inp, tar = self._call0(es)
        vals = [4.0, 2.0, 1.0, 3.0]
        expected = np.empty((len(sel), _IMG_H, _IMG_W), dtype=np.float32)
        for pos, val in enumerate(vals):
            expected[pos] = val
        self.assertTrue(compare_arrays("unsorted out_channel reorder", tar[0], expected, verbose=verbose))

    # ---- 8. zenith angle ------------------------------------------------

    def test_zenith_angle_appended(self):
        es = self._make(zenith_angle=True)
        result = self._call0(es)
        self.assertEqual(len(result), 4)  # inp, tar, zen_inp, zen_tar
        zen_inp, zen_tar = result[2], result[3]
        self.assertEqual(zen_inp.shape, (1, 1, _IMG_H, _IMG_W))
        self.assertEqual(zen_tar.shape, (1, 1, _IMG_H, _IMG_W))
        # cosine zenith angle is bounded in [-1, 1]
        self.assertTrue(np.all(zen_inp >= -1.0) and np.all(zen_inp <= 1.0))
        self.assertTrue(np.all(zen_tar >= -1.0) and np.all(zen_tar <= 1.0))

    def test_zenith_with_subsampling(self):
        es = self._make(zenith_angle=True, subsampling_factor=2)
        result = self._call0(es)
        self.assertEqual(result[2].shape, (1, 1, math.ceil(_IMG_H / 2), math.ceil(_IMG_W / 2)))

    # ---- 9. return timestamp --------------------------------------------

    def test_return_timestamp_appended(self):
        es = self._make(return_timestamp=True)
        result = self._call0(es)
        self.assertEqual(len(result), 4)  # inp, tar, inp_time, tar_time
        self.assertEqual(len(result[2]), 1)  # n_history + 1
        self.assertEqual(len(result[3]), 1)  # n_future + 1

    def test_return_timestamp_n_history(self):
        es = self._make(return_timestamp=True, n_history=2)
        result = self._call0(es)
        self.assertEqual(len(result[2]), 3)  # n_history + 1 = 3

    # ---- 10. zenith + timestamp combined --------------------------------

    def test_zenith_and_timestamp_both_appended(self):
        es = self._make(zenith_angle=True, return_timestamp=True)
        result = self._call0(es)
        self.assertEqual(len(result), 6)  # inp, tar, zen_inp, zen_tar, inp_t, tar_t

    def test_output_layout_matches_outputs(self):
        # the DALI pipeline takes the number of outputs and their layouts from this
        for zenith_angle in [False, True]:
            for return_timestamp in [False, True]:
                with self.subTest(zenith_angle=zenith_angle, return_timestamp=return_timestamp):
                    es = self._make(zenith_angle=zenith_angle, return_timestamp=return_timestamp)
                    result = self._call0(es)
                    layout = es.output_layout()
                    self.assertEqual(len(layout), len(result))
                    for output, axes in zip(result, layout):
                        self.assertEqual(np.ndim(output), len(axes))

    # ---- 11. pickle round-trip ------------------------------------------

    def test_pickle_roundtrip(self, verbose=False):
        """
        Simulates the DALI main-process → worker-process transfer.

        Main-process state: __init__ has run but no file handles are open and
        no buffers are allocated (is_parallel=True).  After pickle.loads the
        worker calls __setstate__, which opens the file, installs the data
        handle, and allocates buffers.  The restored ES must produce the same
        output as a freshly constructed (worker-state) ES.
        """
        import pickle

        es_ref = self._make()  # worker-process state (our simulation)
        es_restored = pickle.loads(pickle.dumps(self._make_for_pickle()))

        inp_ref, tar_ref = self._call0(es_ref)
        inp_res, tar_res = self._call0(es_restored)

        with self.subTest(desc="inp"):
            self.assertTrue(compare_arrays("pickled inp", inp_res, inp_ref, verbose=verbose))
        with self.subTest(desc="tar"):
            self.assertTrue(compare_arrays("pickled tar", tar_res, tar_ref, verbose=verbose))

    # ---- 12. StopIteration at epoch end ---------------------------------

    def test_stop_iteration_at_epoch_end(self):
        es = self._make()
        with self.assertRaises(StopIteration):
            es(_SampleInfo(iteration=es.num_steps_per_epoch))

    # ---- 12. n_samples_total / max_samples ------------------------------

    def test_max_samples_limits_dataset(self):
        """max_samples caps the usable sample count."""
        max_s = 50
        es_full = self._make()
        es_capped = self._make(max_samples=max_s)
        self.assertLess(es_capped.n_samples_total, es_full.n_samples_total)
        self.assertLessEqual(es_capped.n_samples_total, max_s)

    def test_n_samples_total(self):
        es = self._make()
        self.assertEqual(es.n_samples_total, self.expected_n_valid)

    def test_shuffle_epoch_cycle(self):
        """
        Exercises train=True (shuffled) with a short epoch and max_samples cap.

        Setup
        -----
        max_samples = N_SAMPLES + 1  →  n_samples_total = N_SAMPLES = 50
        samples_per_epoch = EPOCH_LEN = 10, batch_size = 1
        → num_steps_per_epoch = 10, num_steps_per_cycle = 50

        DALI epoch model
        ----------------
        iteration resets to 0 at the start of each epoch.
        idx_in_epoch and epoch_idx together form global_sample_idx, which
        walks through the shuffle permutation across epochs.
        5 epochs × 10 steps = 50 total steps = 1 full cycle  →  every sample
        visited exactly once.

        Assertions
        ----------
        1. StopIteration is raised after exactly EPOCH_LEN steps in each epoch.
        2. Across all epochs, exactly N_SAMPLES samples are loaded.
        3. Every sample is visited exactly once (no repeats, no omissions):
           the collected timestamp set equals es.timestamps[es.indices_select].
        """
        N_SAMPLES = 50
        EPOCH_LEN = 10
        N_EPOCHS = N_SAMPLES // EPOCH_LEN  # 5 full epochs = 1 cycle

        es = self._make(
            train=True,
            max_samples=N_SAMPLES + 1,  # +1: samples_end = (N+1) - 1 = N
            samples_per_epoch=EPOCH_LEN,
            batch_size=1,
            return_timestamp=True,
        )
        self.assertEqual(es.n_samples_total, N_SAMPLES)
        self.assertEqual(es.num_steps_per_epoch, EPOCH_LEN)

        loaded_ts = []

        for epoch_idx in range(N_EPOCHS):
            # --- normal steps for this epoch ---
            for step in range(EPOCH_LEN):
                result = es(
                    _SampleInfo(
                        idx_in_epoch=step,
                        epoch_idx=epoch_idx,
                        iteration=step,  # iteration resets to 0 each epoch
                    )
                )
                inp_time = result[2]  # (inp, tar, inp_time, tar_time)
                loaded_ts.append(_ts_to_posix(inp_time[0]))

            # --- StopIteration fires when iteration reaches EPOCH_LEN ---
            with self.subTest(desc=f"StopIteration epoch {epoch_idx}"):
                with self.assertRaises(StopIteration):
                    es(
                        _SampleInfo(
                            idx_in_epoch=0,
                            epoch_idx=epoch_idx,
                            iteration=EPOCH_LEN,
                        )
                    )

        # 1. Correct total count
        with self.subTest(desc="total samples loaded"):
            self.assertEqual(len(loaded_ts), N_SAMPLES)

        # 2. No sample repeated
        with self.subTest(desc="no repeated samples"):
            self.assertEqual(len(set(loaded_ts)), N_SAMPLES)

        # 3. Loaded set == complete valid-sample set (all indices_select covered)
        expected_ts = {_ts_to_posix(ts) for ts in es.timestamps[es.indices_select]}
        with self.subTest(desc="complete coverage"):
            self.assertEqual(set(loaded_ts), expected_ts)

    def test_shuffle_multi_cycle_coverage_and_ordering(self):
        """
        Two consecutive cycles must
          1. each independently cover every valid sample exactly once, and
          2. use different permutations (cycle seed is base_seed + cycle_idx).
        """
        N_SAMPLES = 50
        EPOCH_LEN = 10
        N_EPOCHS_PER_CYCLE = N_SAMPLES // EPOCH_LEN  # 5
        N_CYCLES = 2

        es = self._make(
            train=True,
            max_samples=N_SAMPLES + 1,  # samples_end = (N+1) - 1 = N
            samples_per_epoch=EPOCH_LEN,
            batch_size=1,
            return_timestamp=True,
        )

        per_cycle_ts = []
        for cycle in range(N_CYCLES):
            cycle_ts = []
            for epoch_in_cycle in range(N_EPOCHS_PER_CYCLE):
                epoch_idx = cycle * N_EPOCHS_PER_CYCLE + epoch_in_cycle
                for step in range(EPOCH_LEN):
                    result = es(
                        _SampleInfo(
                            idx_in_epoch=step,
                            epoch_idx=epoch_idx,
                            iteration=step,
                        )
                    )
                    cycle_ts.append(_ts_to_posix(result[2][0]))
            per_cycle_ts.append(cycle_ts)

        expected = {_ts_to_posix(ts) for ts in es.timestamps[es.indices_select]}

        for c, ts_list in enumerate(per_cycle_ts):
            with self.subTest(desc=f"cycle {c} full coverage"):
                self.assertEqual(set(ts_list), expected)
            with self.subTest(desc=f"cycle {c} no intra-cycle repeats"):
                self.assertEqual(len(ts_list), len(set(ts_list)))

        with self.subTest(desc="cycles use different permutations"):
            # same sample set, different ordering
            self.assertNotEqual(per_cycle_ts[0], per_cycle_ts[1])

    def test_shuffle_epoch_not_divisible_by_cycle(self):
        """
        Virtual-epoch size does not divide cycle size.  An epoch then
        spans a cycle boundary mid-epoch; the permutation regenerates
        partway through and cycle_sample_idx wraps to 0 on the next call.

        Walk 8 epochs of 7 steps = 56 samples, against a 50-sample cycle:
          - the first 50 samples form cycle 0 (full coverage, no repeats),
          - the remaining 6 are the start of cycle 1 (distinct permutation).
        """
        N_SAMPLES = 50
        EPOCH_LEN = 7  # 50 % 7 != 0
        N_EPOCHS = 8  # 56 total steps

        es = self._make(
            train=True,
            max_samples=N_SAMPLES + 1,
            samples_per_epoch=EPOCH_LEN,
            batch_size=1,
            return_timestamp=True,
        )
        self.assertEqual(es.n_samples_total, N_SAMPLES)
        self.assertEqual(es.num_steps_per_epoch, EPOCH_LEN)

        all_ts = []
        for epoch_idx in range(N_EPOCHS):
            for step in range(EPOCH_LEN):
                result = es(
                    _SampleInfo(
                        idx_in_epoch=step,
                        epoch_idx=epoch_idx,
                        iteration=step,
                    )
                )
                all_ts.append(_ts_to_posix(result[2][0]))

        expected = {_ts_to_posix(ts) for ts in es.timestamps[es.indices_select]}

        # cycle 0 = first N_SAMPLES calls → full coverage, no repeats
        cycle0 = all_ts[:N_SAMPLES]
        with self.subTest(desc="cycle 0 full coverage"):
            self.assertEqual(set(cycle0), expected)
        with self.subTest(desc="cycle 0 no repeats"):
            self.assertEqual(len(cycle0), len(set(cycle0)))

        # cycle 1 prefix = remaining calls → subset of expected, no repeats
        cycle1_prefix = all_ts[N_SAMPLES:]
        expected_prefix_len = N_EPOCHS * EPOCH_LEN - N_SAMPLES
        with self.subTest(desc="cycle 1 prefix length"):
            self.assertEqual(len(cycle1_prefix), expected_prefix_len)
        with self.subTest(desc="cycle 1 prefix is a valid subset"):
            self.assertTrue(set(cycle1_prefix).issubset(expected))
        with self.subTest(desc="cycle 1 prefix no repeats"):
            self.assertEqual(len(cycle1_prefix), len(set(cycle1_prefix)))

        # the two permutations must differ: cycle 1's start cannot match
        # cycle 0's start element-by-element
        with self.subTest(desc="distinct permutations across boundary"):
            self.assertNotEqual(cycle1_prefix, cycle0[:expected_prefix_len])

    # ---- 13. timestamp boundary list ------------------------------------

    def test_timestamp_boundary_excludes_one_sample(self):
        """
        Boundary at 2017-01-25T00:00:00Z puts sample-23 of 2017 inside the
        lookback window [2017-01-24, 2017-01-25), excluding exactly 1 index.
        """
        es_plain = self._make()
        es_boundary = self._make(timestamp_boundary_list=[_BOUNDARY_TS])
        self.assertEqual(es_plain.n_samples_total, self.expected_n_valid)
        self.assertEqual(es_boundary.n_samples_total, self.expected_n_valid_with_boundary)

    def test_empty_boundary_list_unchanged(self):
        es = self._make(timestamp_boundary_list=[])
        self.assertEqual(es.n_samples_total, self.expected_n_valid)

    # ---- 14. window exclusion across (dt, n_history, n_future) ----------

    def test_window_exclusion_various_settings(self):
        """
        Vary dt, n_history, and n_future.  For every combination
          1. n_samples_total must match the backend-specific formula
             (dir-based drops the window per year; concat drops it once),
          2. walking the full non-shuffled epoch must yield unique samples
             (no silent per-year clamp).
        """
        configs = [
            # (dt, n_history, n_future)
            (1, 0, 0),
            (1, 0, 3),
            (1, 2, 0),
            (1, 1, 2),
            (2, 0, 0),
            (2, 1, 2),
            (3, 0, 5),
        ]
        for dt_, nh, nf in configs:
            with self.subTest(dt=dt_, n_history=nh, n_future=nf):
                es = self._make(dt=dt_, n_history=nh, n_future=nf)

                self.assertEqual(
                    es.n_samples_total,
                    self._expected_n_valid(dt_, nh, nf),
                )

                fps = set()
                for step in range(es.num_steps_per_epoch):
                    inp, _tar = es(
                        _SampleInfo(
                            idx_in_epoch=step,
                            epoch_idx=0,
                            iteration=step,
                        )
                    )
                    # inp has shape (n_history+1, C, H, W); the last (t=0)
                    # frame is unique per sample_idx and is cheaper to hash
                    # than the full window.
                    fps.add(inp[nh].tobytes())
                self.assertEqual(len(fps), es.num_steps_per_epoch)


# ===========================================================================
# Concrete test classes
# ===========================================================================


class TestMakaniHDF5(_BaseESTests, unittest.TestCase):
    """Tests for SampleSource (directory of per-year HDF5 files).

    The main dataset is created by init_hdf5_dataset() from testutils, which
    produces the same two-year training layout used by the other dataloader tests.
    """

    def _expected_n_valid(self, dt, n_history, n_future):
        # window must sit inside one year
        window = dt * (n_history + n_future + 1)
        return sum(max(0, _N_PER_YEAR - window) for _ in range(len(_YEARS)))

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._train_path, *_ = init_hdf5_dataset(self._tmpdir.name)
        # small distinctive-channel dataset for the reordering test
        self._dc_path = _make_distinctive_dir(os.path.join(self._tmpdir.name, "dc"))

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make(self, **overrides):
        return _make_source(self._train_path, **overrides)

    def _make_distinctive(self, **overrides):
        return _make_source(self._dc_path, **overrides)

    def _make_for_pickle(self, **overrides):
        """Main-process state: __init__ done, no file handles, no buffers."""
        from makani.utils.dataloaders.sample_source import SampleSource

        kw = _default_kwargs(self._train_path)
        kw["is_parallel"] = True  # buffers not yet allocated; __setstate__ will do it
        kw.update(overrides)
        return SampleSource(**kw)

    def test_missing_directory_raises(self):
        with self.assertRaises(IOError):
            _make_source(os.path.join(self._tmpdir.name, "does_not_exist"))


class TestMakaniConcat(_BaseESTests, unittest.TestCase):
    """Tests for SampleSource (single concatenated HDF5 file)."""

    def _expected_n_valid(self, dt, n_history, n_future):
        # single contiguous file: samples_start=dt*n_history, samples_end
        # trims dt*(n_future+1) off the global tail
        window = dt * (n_history + n_future + 1)
        return max(0, _N_TOTAL - window)

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        # init_hdf5_dataset with create_concat=True writes train_concat.h5 alongside
        # the per-year files and returns its path as the 7th element.
        *_, self._file = init_hdf5_dataset(self._tmpdir.name, create_concat=True)
        # distinctive-channel concat file for the reordering test
        dc_root = os.path.join(self._tmpdir.name, "dc")
        self._file_dc = _make_distinctive_concat(dc_root)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make(self, **overrides):
        return _make_source(self._file, **overrides)

    def _make_distinctive(self, **overrides):
        return _make_source(self._file_dc, **overrides)

    def _make_for_pickle(self, **overrides):
        """Main-process state: __init__ done, no file handles, no buffers."""
        from makani.utils.dataloaders.sample_source import SampleSource

        kw = _default_kwargs(self._file)
        kw["is_parallel"] = True  # buffers not yet allocated; __setstate__ will do it
        kw.update(overrides)
        return SampleSource(**kw)

    def test_missing_file_raises(self):
        with self.assertRaises(IOError):
            _make_source(os.path.join(self._tmpdir.name, "no_such.h5"))

    def test_s3_raises_not_implemented(self):
        from makani.utils.dataloaders.sample_source import SampleSource

        kw = _default_kwargs(self._file)
        kw["enable_s3"] = True
        with self.assertRaises(NotImplementedError):
            SampleSource(**kw)


class TestMakaniZarr(_BaseESTests, unittest.TestCase):
    """Tests for SampleSource backed by zarr files in makani flat format.

    Uses the same ``_BaseESTests`` suite as ``TestMakaniHDF5``.  The zarr stores
    mirror the HDF5 layout: a single ``fields`` array of shape
    ``(time, channels, lat, lon)`` per year file.
    """

    def _expected_n_valid(self, dt, n_history, n_future):
        window = dt * (n_history + n_future + 1)
        return sum(max(0, _N_PER_YEAR - window) for _ in range(len(_YEARS)))

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._train_path, *_ = init_zarr_dataset(self._tmpdir.name)
        self._dc_path = _make_distinctive_zarr_dir(os.path.join(self._tmpdir.name, "dc_zarr"))

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make(self, **overrides):
        return _make_source(self._train_path, **overrides)

    def _make_distinctive(self, **overrides):
        return _make_source(self._dc_path, **overrides)

    def _make_for_pickle(self, **overrides):
        from makani.utils.dataloaders.sample_source import SampleSource

        kw = _default_kwargs(self._train_path)
        kw["is_parallel"] = True
        kw.update(overrides)
        return SampleSource(**kw)


class TestArcoWB2(_BaseESTests, unittest.TestCase):
    """Tests for SampleSource backed by zarr files in WB2 per-variable format.

    Each variable is stored as a separate zarr array (surface: ``(time, lat, lon)``,
    atmospheric: ``(time, level, lat, lon)``).  The ES auto-detects the WB2 format
    at init time and builds the variable→level conversion table from
    ``channel_names``.
    """

    def _expected_n_valid(self, dt, n_history, n_future):
        window = dt * (n_history + n_future + 1)
        return sum(max(0, _N_PER_YEAR - window) for _ in range(len(_YEARS)))

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._train_path, *_ = init_wb2_zarr_dataset(self._tmpdir.name)
        self._dc_path = _make_distinctive_zarr_wb2_dir(os.path.join(self._tmpdir.name, "dc_zarr_wb2"))

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make(self, **overrides):
        kw = dict(channel_names=list(CHANNEL_NAMES))
        kw.update(overrides)
        return _make_source(self._train_path, **kw)

    def _make_distinctive(self, **overrides):
        kw = dict(channel_names=list(CHANNEL_NAMES))
        kw.update(overrides)
        return _make_source(self._dc_path, **kw)

    def _make_for_pickle(self, **overrides):
        from makani.utils.dataloaders.sample_source import SampleSource

        kw = _default_kwargs(self._train_path)
        kw["is_parallel"] = True
        kw["channel_names"] = list(CHANNEL_NAMES)
        kw.update(overrides)
        return SampleSource(**kw)


class TestOpenRetries(unittest.TestCase):
    """Opening a file is retried before the run is given up on.

    A shared filesystem under a few thousand ranks fails transiently: a handle
    that cannot be had now can usually be had a moment later, and killing a
    multi-day run over one of those is expensive. Equally, retrying forever
    turns a genuinely missing file into a hang, so the attempts are bounded and
    the last failure is raised.

    Neither half is reachable from a normal run, so these substitute a backend
    open that fails on demand, and patch out the sleep so the bounded case does
    not take the wall-clock time the real one would.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._train_path, *_ = init_hdf5_dataset(self._tmpdir.name)
        self.source = _make_source(self._train_path)

    def tearDown(self):
        self._tmpdir.cleanup()

    @contextlib.contextmanager
    def _failing_open(self, failures):
        """Replace the backend open with one that fails its first N calls."""
        attempts = []
        real_open = self.source.backend.open

        def flaky(file_idx):
            attempts.append(file_idx)
            if len(attempts) <= failures:
                raise OSError("Resource temporarily unavailable")
            return real_open(file_idx)

        # the retry notice goes to stdout, which is noise in a passing test
        with mock.patch.object(self.source.backend, "open", flaky):
            with mock.patch("makani.utils.dataloaders.sample_source.time.sleep"):
                with contextlib.redirect_stdout(io.StringIO()):
                    yield attempts

    def test_a_transient_failure_is_survived(self):
        with self._failing_open(failures=2) as attempts:
            self.source._open_with_retries(0)

        self.assertEqual(len(attempts), 3)

    def test_the_last_attempt_still_counts(self):
        # the boundary: failing every time but the final attempt has to succeed,
        # otherwise the retry budget is off by one
        with self._failing_open(failures=self.source.num_retries - 1) as attempts:
            self.source._open_with_retries(0)

        self.assertEqual(len(attempts), self.source.num_retries)

    def test_a_persistent_failure_is_given_up_on(self):
        with self._failing_open(failures=self.source.num_retries) as attempts:
            with self.assertRaises(OSError) as ctx:
                self.source._open_with_retries(0)

        self.assertEqual(len(attempts), self.source.num_retries)
        self.assertIn("after {} attempts".format(self.source.num_retries), str(ctx.exception))

    def test_the_happy_path_does_not_sleep(self):
        # every sample of every epoch goes through here, so a success must not
        # pay for the retry machinery
        with mock.patch("makani.utils.dataloaders.sample_source.time.sleep") as sleep:
            self.source._open_with_retries(0)

        sleep.assert_not_called()


class TestPartialYearTimestamps(unittest.TestCase):
    """Times returned for a file that does not start on January 1st.

    The converters write partial years when a date range starts or ends
    mid-year. The times have to come from the file's timestamp scale: deriving
    them from the year in the file name and the row index places every sample
    of a mid-year start months too early, which goes unnoticed in the data but
    breaks zenith angles and anything keyed on the returned times.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        root = os.path.join(self._tmpdir.name, "partial")
        os.makedirs(root)

        # the layout the converters write for a range starting on July 1st: a
        # partial first year running to its end, then a full year
        year = _YEARS[0]
        starts = [dt.datetime(year, 7, 1, tzinfo=dt.timezone.utc), dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc)]
        ends = [
            dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc),
            dt.datetime(year + 2, 1, 1, tzinfo=dt.timezone.utc),
        ]
        self.file_timestamps = []
        for year, start, end in zip([year, year + 1], starts, ends):
            timestamps = np.arange(start.timestamp(), end.timestamp(), _DHOURS * 3600, dtype=np.float64)
            data = np.zeros((len(timestamps), _N_CH, _IMG_H, _IMG_W), dtype=np.float32)
            with h5py.File(os.path.join(root, f"{year}.h5"), "w") as f:
                ds = f.create_dataset(H5_PATH, data=data)
                ts = f.create_dataset("timestamp", data=timestamps)
                ts.make_scale("timestamp")
                ds.dims[0].attach_scale(ts)
            self.file_timestamps.append(timestamps)
        self.root = root

    def tearDown(self):
        self._tmpdir.cleanup()

    def _expected(self, file_idx, local_idx, n_history, n_future):
        times = self.file_timestamps[file_idx]
        inp = times[local_idx - n_history : local_idx + 1]
        tar = times[local_idx + 1 : local_idx + n_future + 2]
        return inp, tar

    def test_files_do_not_overlap(self):
        # guards the fixture itself: the partial year has to end before the next file starts
        self.assertLess(self.file_timestamps[0][-1], self.file_timestamps[1][0])
        last = dt.datetime.fromtimestamp(self.file_timestamps[0][-1], tz=dt.timezone.utc)
        self.assertEqual((last.month, last.day), (12, 31))

    def test_times_come_from_the_file(self):
        es = _make_source(self.root, return_timestamp=True, n_history=1, n_future=1)
        for file_idx in range(2):
            # the first and the last window that fit inside the file
            length = len(self.file_timestamps[file_idx])
            for local_idx in [1, length - 3]:
                inp_time, tar_time = es._compute_timestamps(local_idx, file_idx)
                exp_inp, exp_tar = self._expected(file_idx, local_idx, n_history=1, n_future=1)
                self.assertTrue(compare_arrays("input times", inp_time, exp_inp, atol=0.0, rtol=0.0, shape_check=True))
                self.assertTrue(compare_arrays("target times", tar_time, exp_tar, atol=0.0, rtol=0.0, shape_check=True))

    def test_returned_times_match_the_first_sample(self):
        es = _make_source(self.root, return_timestamp=True, n_history=1, n_future=1)
        _inp, _tar, inp_time, tar_time = es(_SampleInfo(idx_in_epoch=0, epoch_idx=0, iteration=0))

        # with shuffling off, the first sample is the first valid window of the July file
        exp_inp, exp_tar = self._expected(0, es.indices_select[0], n_history=1, n_future=1)
        self.assertTrue(compare_arrays("input times", inp_time, exp_inp, atol=0.0, rtol=0.0, shape_check=True))
        self.assertTrue(compare_arrays("target times", tar_time, exp_tar, atol=0.0, rtol=0.0, shape_check=True))
        self.assertEqual(dt.datetime.fromtimestamp(inp_time[0], tz=dt.timezone.utc).month, 7)

    @unittest.skipUnless(hasattr(time, "tzset"), "TZ manipulation is POSIX only")
    def test_naive_times_are_taken_as_utc(self):
        # the backends hand out aware datetimes, but a naive one must not pick up
        # the offset of whichever machine the training runs on
        from makani.utils.dataloaders.data_helpers import get_seconds_from_times

        naive = [dt.datetime(_YEARS[0], 7, 1, 6)]
        expected = dt.datetime(_YEARS[0], 7, 1, 6, tzinfo=dt.timezone.utc).timestamp()
        for zone in ("UTC", "Europe/Berlin", "America/Los_Angeles"):
            with local_timezone(zone):
                with self.subTest(zone=zone):
                    self.assertTrue(
                        compare_arrays(
                            "naive epoch seconds",
                            get_seconds_from_times(naive),
                            np.array([expected]),
                            atol=0.0,
                            rtol=0.0,
                            shape_check=True,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
