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

import importlib.util
import sys
import os
import json
import tempfile
import unittest
import argparse as ap
import datetime as dt
from unittest import mock
import numpy as np
import h5py as h5
from parameterized import parameterized

import torch

from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, init_hdf5_dataset, H5_PATH, IMG_SIZE_H, IMG_SIZE_W, compare_arrays
from data_process.date_range import add_date_range_arguments, date_range_from_args, parse_date, yearly_sample_times
from data_process.convert_era5_to_makani_input import _batched, _create_output_file, _write_missing
from data_process.sources import InvariantSource, Source


def _utc(year, month, day, hour=0):
    return dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc)


class TestAnnotateDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create temporary directory
        cls.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = cls.tmpdir.name

        # Create unannotated dataset
        path = os.path.join(tmp_path, "data")
        os.makedirs(path, exist_ok=True)
        cls.train_path, cls.num_train, cls.test_path, cls.num_test, _, cls.metadata_path, _ = init_hdf5_dataset(
            path, nan_fraction=0.0, annotate=False
        )

        # Create reference dataset with annotations
        ref_path = os.path.join(tmp_path, "ref_data")
        os.makedirs(ref_path, exist_ok=True)
        cls.ref_train_path, cls.ref_num_train, cls.ref_test_path, cls.ref_num_test, _, _, _ = init_hdf5_dataset(
            ref_path, annotate=True
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def setUp(self):
        disable_tf32()

    def test_annotate_dataset(self, verbose=False):
        # import necessary modules
        from data_process.annotate_dataset import annotate

        # Load metadata
        with open(os.path.join(self.metadata_path, "data.json"), "r") as f:
            metadata = json.load(f)

        # Get list of files to annotate
        train_files = sorted(
            [os.path.join(self.train_path, f) for f in os.listdir(self.train_path) if f.endswith(".h5")]
        )
        test_files = sorted([os.path.join(self.test_path, f) for f in os.listdir(self.test_path) if f.endswith(".h5")])
        all_files = train_files + test_files
        years = [2017, 2018, 2019]  # Corresponding years for the files

        # Run annotation
        annotate(metadata, all_files, years, verbose=False)

        # reference files:
        train_files_ref = sorted(
            [os.path.join(self.ref_train_path, f) for f in os.listdir(self.ref_train_path) if f.endswith(".h5")]
        )
        test_files_ref = sorted(
            [os.path.join(self.ref_test_path, f) for f in os.listdir(self.ref_test_path) if f.endswith(".h5")]
        )
        all_files_ref = train_files_ref + test_files_ref

        # Compare with reference dataset
        for file_path, ref_file_path in zip(all_files, all_files_ref):
            with h5.File(file_path, "r") as f, h5.File(ref_file_path, "r") as ref_f:
                # Check data content
                with self.subTest(desc="data"):
                    self.assertTrue(compare_arrays("data", f[H5_PATH][...], ref_f[H5_PATH][...], verbose=verbose))

                # Check annotations
                with self.subTest(desc="timestamp"):
                    self.assertTrue(
                        compare_arrays("timestamp", f["timestamp"][...], ref_f["timestamp"][...], verbose=verbose)
                    )
                with self.subTest(desc="lat"):
                    self.assertTrue(compare_arrays("lat", f["lat"][...], ref_f["lat"][...], verbose=verbose))
                with self.subTest(desc="lon"):
                    self.assertTrue(compare_arrays("lon", f["lon"][...], ref_f["lon"][...], verbose=verbose))
                with self.subTest(desc="channel"):
                    self.assertEqual(f["channel"][...].tolist(), ref_f["channel"][...].tolist())

                # Check dimension labels
                with self.subTest(desc="timestamp label"):
                    self.assertEqual(f[H5_PATH].dims[0].label, "Timestamp in UTC time zone")
                with self.subTest(desc="channel label"):
                    self.assertEqual(f[H5_PATH].dims[1].label, "Channel name")
                with self.subTest(desc="latitude label"):
                    self.assertEqual(f[H5_PATH].dims[2].label, "Latitude in degrees")
                with self.subTest(desc="longitude label"):
                    self.assertEqual(f[H5_PATH].dims[3].label, "Longitude in degrees")

                # Check scales
                with self.subTest(desc="timestamp scale"):
                    self.assertTrue(
                        compare_arrays(
                            "timestamp scale",
                            f[H5_PATH].dims[0]["timestamp"][...],
                            ref_f[H5_PATH].dims[0]["timestamp"][...],
                            verbose=verbose,
                        )
                    )
                with self.subTest(desc="channel scale"):
                    self.assertTrue(
                        compare_arrays(
                            "channel scale",
                            f[H5_PATH].dims[2]["lat"][...],
                            ref_f[H5_PATH].dims[2]["lat"][...],
                            verbose=verbose,
                        )
                    )
                with self.subTest(desc="longitude scale"):
                    self.assertTrue(
                        compare_arrays(
                            "longitude scale",
                            f[H5_PATH].dims[3]["lon"][...],
                            ref_f[H5_PATH].dims[3]["lon"][...],
                            verbose=verbose,
                        )
                    )


class TestConcatenateDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create temporary directory
        cls.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = cls.tmpdir.name

        # Create dataset
        path = os.path.join(tmp_path, "data")
        os.makedirs(path, exist_ok=True)
        cls.train_path, cls.num_train, cls.test_path, cls.num_test, _, cls.metadata_path, _ = init_hdf5_dataset(
            path, annotate=True
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def setUp(self):
        disable_tf32()

    @parameterized.expand(
        [1, 5],
        skip_on_empty=False,
    )
    def test_concatenate_dataset(self, dhoursrel):
        # import necessary modules
        from data_process.concatenate_dataset import concatenate

        # Load metadata
        with open(os.path.join(self.metadata_path, "data.json"), "r") as f:
            metadata = json.load(f)
        channel_names = metadata["coords"]["channel"]

        # Get list of files to concatenate
        train_files = sorted([f for f in os.listdir(self.train_path) if f.endswith(".h5")])
        years = [2017, 2018]  # Corresponding years for the files

        # Create output directory
        input_dirs = [self.train_path]

        # Run concatenation
        output_file = os.path.join(input_dirs[0], "concatenated.h5v")
        concatenate(
            input_dirs, output_file, metadata, [channel_names], train_files, years, dhoursrel=dhoursrel, verbose=False
        )

        # Compare concatenated file with original files
        with h5.File(output_file, "r") as f_conc:
            # Get total number of samples
            total_samples = f_conc[H5_PATH].shape[0]

            # Track current position in concatenated file
            current_pos = 0

            # Compare each original file's data with corresponding section in concatenated file
            for file_path in train_files:
                ifile_path = os.path.join(self.train_path, file_path)
                with h5.File(ifile_path, "r") as f_orig:
                    num_samples = f_orig[H5_PATH].shape[0] // dhoursrel

                    # Compare data
                    self.assertTrue(
                        compare_arrays(
                            "concat data",
                            f_conc[H5_PATH][current_pos : current_pos + num_samples, ...],
                            f_orig[H5_PATH][::dhoursrel, ...],
                        )
                    )

                    # Compare timestamps
                    self.assertTrue(
                        compare_arrays(
                            "concat timestamp",
                            f_conc["timestamp"][current_pos : current_pos + num_samples, ...],
                            f_orig["timestamp"][::dhoursrel, ...],
                        )
                    )

                    # Update position
                    current_pos += num_samples

            # Verify total number of samples
            with self.subTest(desc="total number of samples"):
                self.assertEqual(current_pos, total_samples)

            # Verify metadata
            with self.subTest(desc="lat"):
                self.assertTrue(compare_arrays("lat", f_conc["lat"][...], np.asarray(metadata["coords"]["lat"])))
            with self.subTest(desc="lon"):
                self.assertTrue(compare_arrays("lon", f_conc["lon"][...], np.asarray(metadata["coords"]["lon"])))
            with self.subTest(desc="channel"):
                self.assertEqual([c.decode() for c in f_conc["channel"][...].tolist()], metadata["coords"]["channel"])

            # Verify dimension labels
            with self.subTest(desc="timestamp label"):
                self.assertEqual(f_conc[H5_PATH].dims[0].label, "Timestamp in UTC time zone")
            with self.subTest(desc="channel label"):
                self.assertEqual(f_conc[H5_PATH].dims[1].label, "Channel name")
            with self.subTest(desc="latitude label"):
                self.assertEqual(f_conc[H5_PATH].dims[2].label, "Latitude in degrees")
            with self.subTest(desc="longitude label"):
                self.assertEqual(f_conc[H5_PATH].dims[3].label, "Longitude in degrees")

    def test_concatenate_leap_year_indivisible_dhoursrel(self):
        """
        Regression: when a year's sample count is not a multiple of dhoursrel
        (the realistic case for a leap year at daily cadence: 366 % 5 = 1),
        concatenate() must produce a virtual layout of size floor(ne/dhoursrel)
        per year — and the source slice fed to the layout must contain exactly
        that many elements.

        Pre-fix, the source slice was ``vsource[::dhoursrel]`` whose length is
        ceil(ne/dhoursrel) — one more than the layout slot for any year whose
        sample count is not divisible by dhoursrel. h5py rejected this with
        ``ValueError: Invalid mapping selections``.

        Post-fix, the source slice is ``vsource[:ne_red*dhoursrel:dhoursrel]``,
        which is exactly ``ne_red`` elements regardless of divisibility.
        """
        from data_process.concatenate_dataset import concatenate

        # Year 2017: 365 samples (non-leap, 365 % 5 == 0)  → no bug exposure.
        # Year 2020: 366 samples (leap year, 366 % 5 == 1) → triggers the bug.
        years_local = [2017, 2020]
        sample_counts = {2017: 365, 2020: 366}
        dhoursrel = 5
        dhours_local = 24
        img_h = 16
        img_w = 32
        num_chans = 3
        channel_names = [f"chan_{i}" for i in range(num_chans)]
        latitudes = np.linspace(90, -90, img_h, endpoint=True).astype(np.float32)
        longitudes = np.linspace(0, 360, img_w, endpoint=False).astype(np.float32)

        rng = np.random.default_rng(seed=2020)

        with tempfile.TemporaryDirectory() as work_dir, tempfile.TemporaryDirectory() as out_dir:

            # Write source files manually so we control the per-year sample count.
            # init_hdf5_dataset's default is 365 across all files — we need 366 for one.
            data_per_year = {}
            ts_per_year = {}
            chanlen = max(len(c) for c in channel_names)
            for year in years_local:
                n = sample_counts[year]
                data = rng.random((n, num_chans, img_h, img_w), dtype=np.float32)
                data_per_year[year] = data

                jan_01 = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
                ts = np.array(
                    [(jan_01 + dt.timedelta(hours=h * dhours_local)).timestamp() for h in range(n)],
                    dtype=np.float64,
                )
                ts_per_year[year] = ts

                fpath = os.path.join(work_dir, f"{year}.h5")
                with h5.File(fpath, "w") as f:
                    f.create_dataset(H5_PATH, data=data)
                    f.create_dataset("timestamp", data=ts)
                    f.create_dataset("lat", data=latitudes)
                    f.create_dataset("lon", data=longitudes)
                    f.create_dataset("channel", num_chans, dtype=h5.string_dtype(length=chanlen))
                    f["channel"][...] = channel_names
                    f["timestamp"].make_scale("timestamp")
                    f["channel"].make_scale("channel")
                    f["lat"].make_scale("lat")
                    f["lon"].make_scale("lon")
                    f[H5_PATH].dims[0].attach_scale(f["timestamp"])
                    f[H5_PATH].dims[1].attach_scale(f["channel"])
                    f[H5_PATH].dims[2].attach_scale(f["lat"])
                    f[H5_PATH].dims[3].attach_scale(f["lon"])

            metadata = dict(
                dataset_name="testing_leap",
                h5_path=H5_PATH,
                dims=["time", "channel", "lat", "lon"],
                dhours=dhours_local,
                coords=dict(
                    grid_type="equiangular",
                    lat=latitudes.tolist(),
                    lon=longitudes.tolist(),
                    channel=channel_names,
                ),
            )

            output_file = os.path.join(out_dir, "concatenated_leap.h5v")
            concatenate(
                input_dirs=[work_dir],
                output_file=output_file,
                metadata=metadata,
                channel_names=[channel_names],
                file_names_to_concatenate=[f"{y}.h5" for y in years_local],
                years=years_local,
                dhoursrel=dhoursrel,
                verbose=False,
            )

            # Build expected output: each year is sliced to floor(n/dhoursrel)
            # contiguous strided samples, taking exactly that many (NOT the ceil).
            ne_red_per_year = {y: sample_counts[y] // dhoursrel for y in years_local}
            total_red = sum(ne_red_per_year.values())  # 73 + 73 = 146

            expected_data = []
            expected_ts = []
            for year in years_local:
                ne_red = ne_red_per_year[year]
                slc = slice(None, ne_red * dhoursrel, dhoursrel)
                expected_data.append(data_per_year[year][slc])
                expected_ts.append(ts_per_year[year][slc])
            expected_data = np.concatenate(expected_data, axis=0)
            expected_ts = np.concatenate(expected_ts, axis=0)

            with h5.File(output_file, "r") as f_conc:
                with self.subTest(desc="layout shape uses floor(ne/dhoursrel) per year"):
                    self.assertEqual(
                        f_conc[H5_PATH].shape,
                        (total_red, num_chans, img_h, img_w),
                    )
                    # Sanity: 365//5 + 366//5 = 73 + 73 = 146
                    self.assertEqual(total_red, 146)

                with self.subTest(desc="data round-trips correctly"):
                    self.assertTrue(compare_arrays("leap data", f_conc[H5_PATH][...], expected_data))

                with self.subTest(desc="timestamps match expected stride"):
                    self.assertTrue(compare_arrays("leap ts", f_conc["timestamp"][...], expected_ts))

                # Pin the leap-year boundary specifically: the last sample stored
                # for year 2020 must come from source index (ne_red-1)*dhoursrel = 360.
                # The buggy ceil-based slice would have written index 365 instead.
                with self.subTest(desc="leap-year last sample is the correct stride index"):
                    ne_red_2020 = ne_red_per_year[2020]
                    t_last = ne_red_per_year[2017] + ne_red_2020 - 1
                    expected_last = data_per_year[2020][(ne_red_2020 - 1) * dhoursrel, :, :, :]
                    self.assertTrue(compare_arrays("leap last sample", f_conc[H5_PATH][t_last], expected_last))


class TestConcatenateDatasetChannelsAndTime(unittest.TestCase):
    """
    Tests concatenation across both channel AND time axes simultaneously.

    Setup creates two parallel datasets ('a' and 'b') with disjoint channel sets
    but matching timestamps. concatenate() is then asked to combine them via the
    pattern: files in different input_dirs are stitched along the channel axis,
    and files within an input_dir are stitched along the time axis.

    Crucially, dir_a, dir_b, and the output file each live in separate temp
    directories — this exercises the path handling that previously broke when
    the concatenated file's source references crossed FS boundaries.

    Verification reads the resulting virtual dataset along several axis-slicing
    patterns (time-only, channel-only across the a/b boundary, and combined)
    and compares against arrays loaded directly from the original serial files.
    """

    @classmethod
    def setUpClass(cls):
        # Three independent temp directories: one per input dataset, one for the
        # virtual output file. Forces the concatenated file's source references
        # to span paths that share no common parent directory.
        cls.tmpdir_a = tempfile.TemporaryDirectory()
        cls.tmpdir_b = tempfile.TemporaryDirectory()
        cls.tmpdir_out = tempfile.TemporaryDirectory()
        cls.dir_a = cls.tmpdir_a.name
        cls.dir_b = cls.tmpdir_b.name
        cls.dir_out = cls.tmpdir_out.name

        cls.years = [2017, 2018]
        cls.num_samples_per_year = 365
        cls.img_h = 8
        cls.img_w = 16
        cls.dhours = (365 * 24) // cls.num_samples_per_year

        # Two disjoint channel sets — 'a' has 3 channels, 'b' has 2.
        # Asymmetric counts catch off-by-one errors in the channel offset logic.
        cls.channels_a = ["a_chan_0", "a_chan_1", "a_chan_2"]
        cls.channels_b = ["b_chan_0", "b_chan_1"]
        cls.channels_combined = cls.channels_a + cls.channels_b
        cls.num_chans_a = len(cls.channels_a)
        cls.num_chans_b = len(cls.channels_b)
        cls.num_chans_total = cls.num_chans_a + cls.num_chans_b

        rng = np.random.default_rng(seed=333)

        cls.data_a = {}
        cls.data_b = {}
        cls.timestamps = {}

        latitudes = np.linspace(90, -90, cls.img_h, endpoint=True).astype(np.float32)
        longitudes = np.linspace(0, 360, cls.img_w, endpoint=False).astype(np.float32)
        cls.latitudes = latitudes
        cls.longitudes = longitudes

        for year in cls.years:
            data_a = rng.random((cls.num_samples_per_year, cls.num_chans_a, cls.img_h, cls.img_w), dtype=np.float32)
            data_b = rng.random((cls.num_samples_per_year, cls.num_chans_b, cls.img_h, cls.img_w), dtype=np.float32)
            cls.data_a[year] = data_a
            cls.data_b[year] = data_b

            jan_01 = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
            ts = np.array(
                [(jan_01 + dt.timedelta(hours=h * cls.dhours)).timestamp() for h in range(cls.num_samples_per_year)],
                dtype=np.float64,
            )
            cls.timestamps[year] = ts

            cls._write_h5(os.path.join(cls.dir_a, f"{year}.h5"), data_a, ts, cls.channels_a, latitudes, longitudes)
            cls._write_h5(os.path.join(cls.dir_b, f"{year}.h5"), data_b, ts, cls.channels_b, latitudes, longitudes)

    @staticmethod
    def _write_h5(path, data, ts, channel_names, lats, lons, entry_key=H5_PATH, annotate=True):
        """Write a single makani-format HDF5 file. When ``annotate=False`` the
        coordinate datasets and dim scales are omitted, which exercises the
        timestamp-derivation fallback in ``concatenate()``."""
        chanlen = max(len(c) for c in channel_names)
        with h5.File(path, "w") as f:
            f.create_dataset(entry_key, data=data)

            if not annotate:
                return

            f.create_dataset("timestamp", data=ts)
            f.create_dataset("lat", data=lats)
            f.create_dataset("lon", data=lons)
            f.create_dataset("channel", len(channel_names), dtype=h5.string_dtype(length=chanlen))
            f["channel"][...] = channel_names

            f["timestamp"].make_scale("timestamp")
            f["channel"].make_scale("channel")
            f["lat"].make_scale("lat")
            f["lon"].make_scale("lon")

            f[entry_key].dims[0].attach_scale(f["timestamp"])
            f[entry_key].dims[1].attach_scale(f["channel"])
            f[entry_key].dims[2].attach_scale(f["lat"])
            f[entry_key].dims[3].attach_scale(f["lon"])

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir_a.cleanup()
        cls.tmpdir_b.cleanup()
        cls.tmpdir_out.cleanup()

    def setUp(self):
        disable_tf32()

    @parameterized.expand([1, 5], skip_on_empty=False)
    def test_concatenate_time_and_channels(self, dhoursrel):
        from data_process.concatenate_dataset import concatenate

        metadata = dict(
            dataset_name="testing_concat",
            h5_path=H5_PATH,
            dims=["time", "channel", "lat", "lon"],
            dhours=self.dhours,
            coords=dict(
                grid_type="equiangular",
                lat=self.latitudes.tolist(),
                lon=self.longitudes.tolist(),
                channel=self.channels_combined,
            ),
        )

        output_file = os.path.join(self.dir_out, f"concatenated_dhrel{dhoursrel}.h5v")
        file_names = [f"{y}.h5" for y in self.years]

        concatenate(
            input_dirs=[self.dir_a, self.dir_b],
            output_file=output_file,
            metadata=metadata,
            channel_names=[self.channels_a, self.channels_b],
            file_names_to_concatenate=file_names,
            years=self.years,
            dhoursrel=dhoursrel,
            verbose=False,
        )

        # Build the expected concatenated array and timestamp vector for comparison.
        # 'a' channels precede 'b' channels along the channel axis.
        # Within each year, samples are subsampled by dhoursrel. Use a bounded
        # stride slice (``[:n_red*dhoursrel:dhoursrel]``) rather than bare
        # ``[::dhoursrel]`` so the count matches the VDS layout slot
        # (floor(n/dhoursrel)) regardless of whether n divides evenly.
        n_per_year_red = self.num_samples_per_year // dhoursrel
        slc = slice(None, n_per_year_red * dhoursrel, dhoursrel)
        expected_full = []
        expected_ts = []
        for year in self.years:
            year_combined = np.concatenate([self.data_a[year][slc], self.data_b[year][slc]], axis=1)
            expected_full.append(year_combined)
            expected_ts.append(self.timestamps[year][slc])
        expected_full = np.concatenate(expected_full, axis=0)
        expected_ts = np.concatenate(expected_ts, axis=0)

        boundary_t = n_per_year_red  # first index of year-2018 in concatenated time axis
        boundary_c = self.num_chans_a  # first index of 'b' channels in channel axis

        with h5.File(output_file, "r") as f_conc:
            # Full-array sanity: shape, dtype, total content
            with self.subTest(desc="full shape"):
                self.assertEqual(
                    f_conc[H5_PATH].shape,
                    (2 * n_per_year_red, self.num_chans_total, self.img_h, self.img_w),
                )
            with self.subTest(desc="full data"):
                self.assertTrue(compare_arrays("full", f_conc[H5_PATH][...], expected_full))

            with self.subTest(desc="timestamp"):
                self.assertTrue(compare_arrays("timestamp", f_conc["timestamp"][...], expected_ts))

            with self.subTest(desc="channel order"):
                self.assertEqual(
                    [c.decode() for c in f_conc["channel"][...].tolist()],
                    self.channels_combined,
                )

            # Slicing across the time-axis boundary (last sample of 2017 + first of 2018)
            with self.subTest(desc="time slice across year boundary"):
                t_slice = slice(boundary_t - 1, boundary_t + 1)
                got = f_conc[H5_PATH][t_slice, :, :, :]
                self.assertTrue(compare_arrays("time slice", got, expected_full[t_slice, :, :, :]))

            # Slicing across the channel-axis boundary (last 'a' channel + first 'b' channel)
            with self.subTest(desc="channel slice across a/b boundary"):
                c_slice = slice(boundary_c - 1, boundary_c + 1)
                got = f_conc[H5_PATH][:, c_slice, :, :]
                self.assertTrue(compare_arrays("channel slice", got, expected_full[:, c_slice, :, :]))

            # Slicing both axes simultaneously across both boundaries
            with self.subTest(desc="combined slice across both boundaries"):
                t_slice = slice(boundary_t - 1, boundary_t + 1)
                c_slice = slice(boundary_c - 1, boundary_c + 1)
                got = f_conc[H5_PATH][t_slice, c_slice, :, :]
                self.assertTrue(compare_arrays("combined", got, expected_full[t_slice, c_slice, :, :]))

            # ----------------------------------------------------------------------
            # Same slicing patterns via read_direct(). This exercises HDF5's
            # hyperslab selection path (with a caller-allocated destination buffer)
            # rather than fancy indexing through __getitem__, which can hit
            # different VDS code paths — particularly when the virtual file's
            # source references span multiple parent directories.
            # ----------------------------------------------------------------------
            dset = f_conc[H5_PATH]

            with self.subTest(desc="read_direct: time slice across year boundary"):
                t_slice = slice(boundary_t - 1, boundary_t + 1)
                expected = expected_full[t_slice, :, :, :]
                buf = np.empty(expected.shape, dtype=expected.dtype)
                dset.read_direct(buf, source_sel=np.s_[t_slice, :, :, :])
                self.assertTrue(compare_arrays("read_direct time", buf, expected))

            with self.subTest(desc="read_direct: channel slice across a/b boundary"):
                c_slice = slice(boundary_c - 1, boundary_c + 1)
                expected = expected_full[:, c_slice, :, :]
                buf = np.empty(expected.shape, dtype=expected.dtype)
                dset.read_direct(buf, source_sel=np.s_[:, c_slice, :, :])
                self.assertTrue(compare_arrays("read_direct channel", buf, expected))

            with self.subTest(desc="read_direct: combined slice across both boundaries"):
                t_slice = slice(boundary_t - 1, boundary_t + 1)
                c_slice = slice(boundary_c - 1, boundary_c + 1)
                expected = expected_full[t_slice, c_slice, :, :]
                buf = np.empty(expected.shape, dtype=expected.dtype)
                dset.read_direct(buf, source_sel=np.s_[t_slice, c_slice, :, :])
                self.assertTrue(compare_arrays("read_direct combined", buf, expected))

            # Strided source selection: every 4th time sample across both years.
            # Forces read_direct to materialize a non-contiguous hyperslab that
            # straddles the year-boundary VDS source split.
            with self.subTest(desc="read_direct: strided time read"):
                t_slice = slice(0, 2 * n_per_year_red, 4)
                expected = expected_full[t_slice, :, :, :]
                buf = np.empty(expected.shape, dtype=expected.dtype)
                dset.read_direct(buf, source_sel=np.s_[t_slice, :, :, :])
                self.assertTrue(compare_arrays("read_direct strided time", buf, expected))

            # Strided on both axes simultaneously — hits both the time-VDS and
            # channel-VDS source-mapping paths through a single hyperslab.
            with self.subTest(desc="read_direct: strided time and channel read"):
                t_slice = slice(0, 2 * n_per_year_red, 3)
                c_slice = slice(0, self.num_chans_total, 2)
                expected = expected_full[t_slice, c_slice, :, :]
                buf = np.empty(expected.shape, dtype=expected.dtype)
                dset.read_direct(buf, source_sel=np.s_[t_slice, c_slice, :, :])
                self.assertTrue(compare_arrays("read_direct strided combined", buf, expected))

            # ----------------------------------------------------------------------
            # Dimension scales survived the virtualization. Each axis of the VDS
            # should expose its associated 1-D coordinate dataset via ``dims[i]``.
            # ----------------------------------------------------------------------
            with self.subTest(desc="dim 0 scale: timestamp"):
                self.assertTrue(compare_arrays("dim0 timestamp", dset.dims[0]["timestamp"][...], expected_ts))
            with self.subTest(desc="dim 1 scale: channel"):
                names_via_scale = [c.decode() for c in dset.dims[1]["channel"][...].tolist()]
                self.assertEqual(names_via_scale, self.channels_combined)
            with self.subTest(desc="dim 2 scale: lat"):
                self.assertTrue(compare_arrays("dim2 lat", dset.dims[2]["lat"][...], self.latitudes))
            with self.subTest(desc="dim 3 scale: lon"):
                self.assertTrue(compare_arrays("dim3 lon", dset.dims[3]["lon"][...], self.longitudes))

            # Dtype passes through unchanged from the source files.
            with self.subTest(desc="dtype preservation"):
                self.assertEqual(dset.dtype, np.float32)

            # ----------------------------------------------------------------------
            # Loader-realistic access patterns. Mirrors `_get_data_h5` in
            # makani/utils/dataloaders/backends/makani_concat.py.
            # ----------------------------------------------------------------------

            # Pattern 1: cropped spatial region with subsampling on both spatial dims.
            # The loader uses `start_x:end_x:subsampling_factor` and the equivalent
            # for longitude — non-zero starts and a non-1 stride simultaneously.
            with self.subTest(desc="loader-style: cropped spatial region with subsampling"):
                sub = 2
                h_crop = slice(self.img_h // 4, 3 * self.img_h // 4, sub)
                w_crop = slice(self.img_w // 4, 3 * self.img_w // 4, sub)
                expected = expected_full[:, :, h_crop, w_crop]
                buf = np.empty(expected.shape, dtype=expected.dtype)
                dset.read_direct(buf, source_sel=np.s_[:, :, h_crop, w_crop])
                self.assertTrue(compare_arrays("spatial crop", buf, expected))

            # Pattern 2: multiple read_direct calls into a SINGLE pre-allocated
            # buffer, each targeting a different `dest_sel` channel range. This is
            # how the loader assembles non-contiguous input-channel groups (e.g. it
            # may want channel indices [0, 2, 3] which split into slice(0,1) and
            # slice(2,4)). The second source slice here intentionally straddles the
            # a/b VDS source boundary at channel index 3, so the read crosses the
            # channel-axis virtualization split inside a single read_direct call.
            with self.subTest(desc="loader-style: split channel reads with strides and dest_sel"):
                dt_step = 2
                sub = 2
                in_channel_slices = [slice(0, 1), slice(2, 4)]
                n_t = 2 * n_per_year_red

                expected_parts = [expected_full[0:n_t:dt_step, c_slice, ::sub, ::sub] for c_slice in in_channel_slices]
                expected = np.concatenate(expected_parts, axis=1)
                buf = np.empty(expected.shape, dtype=expected.dtype)

                off = 0
                for c_slice in in_channel_slices:
                    cnt = c_slice.stop - c_slice.start
                    dset.read_direct(
                        buf,
                        source_sel=np.s_[0:n_t:dt_step, c_slice, ::sub, ::sub],
                        dest_sel=np.s_[:, off : off + cnt, ...],
                    )
                    off += cnt

                self.assertTrue(compare_arrays("loader-style split read", buf, expected))

            # Pattern 3: simultaneous stride on three axes (time, lat, lon) — the
            # loader does this whenever dt > 1 and subsampling_factor > 1. Catches
            # any edge cases where a stride on one axis interacts badly with the
            # VDS source-mapping on another.
            with self.subTest(desc="loader-style: stride on time + lat + lon"):
                dt_step = 2
                sub = 2
                t_slice = slice(0, 2 * n_per_year_red, dt_step)
                h_slice = slice(0, self.img_h, sub)
                w_slice = slice(0, self.img_w, sub)
                expected = expected_full[t_slice, :, h_slice, w_slice]
                buf = np.empty(expected.shape, dtype=expected.dtype)
                dset.read_direct(buf, source_sel=np.s_[t_slice, :, h_slice, w_slice])
                self.assertTrue(compare_arrays("3-axis stride", buf, expected))

            # Cross-check against the original serial files (no expected_full intermediate):
            # for each year, pick a sample and verify the 'a'/'b' channel halves match the
            # respective source files directly.
            for year_idx, year in enumerate(self.years):
                t_in_year = 0  # first sample of this year (after subsampling)
                t_concat = year_idx * n_per_year_red + t_in_year
                got = f_conc[H5_PATH][t_concat, :, :, :]

                with self.subTest(desc=f"year {year} a-channels match source"):
                    self.assertTrue(
                        compare_arrays(
                            f"year {year} a",
                            got[: self.num_chans_a, :, :],
                            self.data_a[year][t_in_year * dhoursrel, :, :, :],
                        )
                    )
                with self.subTest(desc=f"year {year} b-channels match source"):
                    self.assertTrue(
                        compare_arrays(
                            f"year {year} b",
                            got[self.num_chans_a :, :, :],
                            self.data_b[year][t_in_year * dhoursrel, :, :, :],
                        )
                    )

    def test_concatenate_custom_entry_key(self):
        """
        Verify the ``entry_key`` parameter is honored end-to-end: source files
        store data under a non-default key, and the virtual output exposes the
        same key (and only that key — the default ``"fields"`` should NOT exist).
        """
        from data_process.concatenate_dataset import concatenate

        custom_key = "custom_data"

        # Per-test scratch dirs — independent from the class-level fixtures so
        # we don't pollute the shared dir_a/dir_b with a second set of files.
        with (
            tempfile.TemporaryDirectory() as sub_a,
            tempfile.TemporaryDirectory() as sub_b,
            tempfile.TemporaryDirectory() as sub_out,
        ):

            n = 32
            dhours_local = (365 * 24) // n
            rng = np.random.default_rng(seed=999)

            data_a = {}
            data_b = {}
            timestamps = {}
            for year in self.years:
                da = rng.random((n, self.num_chans_a, self.img_h, self.img_w), dtype=np.float32)
                db = rng.random((n, self.num_chans_b, self.img_h, self.img_w), dtype=np.float32)
                data_a[year] = da
                data_b[year] = db

                jan_01 = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
                ts = np.array(
                    [(jan_01 + dt.timedelta(hours=h * dhours_local)).timestamp() for h in range(n)],
                    dtype=np.float64,
                )
                timestamps[year] = ts

                # Write files using the custom entry_key
                self._write_h5(
                    os.path.join(sub_a, f"{year}.h5"),
                    da,
                    ts,
                    self.channels_a,
                    self.latitudes,
                    self.longitudes,
                    entry_key=custom_key,
                )
                self._write_h5(
                    os.path.join(sub_b, f"{year}.h5"),
                    db,
                    ts,
                    self.channels_b,
                    self.latitudes,
                    self.longitudes,
                    entry_key=custom_key,
                )

            metadata = dict(
                dataset_name="testing_custom_key",
                h5_path=custom_key,
                dims=["time", "channel", "lat", "lon"],
                dhours=dhours_local,
                coords=dict(
                    grid_type="equiangular",
                    lat=self.latitudes.tolist(),
                    lon=self.longitudes.tolist(),
                    channel=self.channels_combined,
                ),
            )

            output_file = os.path.join(sub_out, "concatenated_custom_key.h5v")
            concatenate(
                input_dirs=[sub_a, sub_b],
                output_file=output_file,
                metadata=metadata,
                channel_names=[self.channels_a, self.channels_b],
                file_names_to_concatenate=[f"{y}.h5" for y in self.years],
                years=self.years,
                dhoursrel=1,
                entry_key=custom_key,
                verbose=False,
            )

            expected_full = np.concatenate(
                [np.concatenate([data_a[y], data_b[y]], axis=1) for y in self.years],
                axis=0,
            )

            with h5.File(output_file, "r") as f_conc:
                with self.subTest(desc="custom entry_key dataset present"):
                    self.assertIn(custom_key, f_conc)
                with self.subTest(desc="default entry_key absent"):
                    self.assertNotIn(H5_PATH, f_conc)
                with self.subTest(desc="custom entry_key data matches"):
                    self.assertTrue(compare_arrays("custom data", f_conc[custom_key][...], expected_full))
                with self.subTest(desc="custom entry_key dim labels"):
                    self.assertEqual(f_conc[custom_key].dims[0].label, "Timestamp in UTC time zone")
                    self.assertEqual(f_conc[custom_key].dims[1].label, "Channel name")

    def test_concatenate_with_nan_channel(self):
        """
        Verify concatenate() correctly preserves NaN entries that exist in only
        a subset of samples on a single channel — i.e. partial masking.

        Setup: file ``2018.h5`` in dir_a has NaN values injected at a few time
        indices on a single channel (``a_chan_1``). All other samples and
        channels are finite. After concatenation the NaN positions must appear
        at the correct (time, channel) coordinates in the virtual output, and
        finite values must still match bit-for-bit.

        Comparison uses ``np.array_equal(equal_nan=True)`` instead of the
        ``compare_arrays`` helper, since ``np.allclose`` treats NaN != NaN.
        """
        from data_process.concatenate_dataset import concatenate

        with (
            tempfile.TemporaryDirectory() as sub_a,
            tempfile.TemporaryDirectory() as sub_b,
            tempfile.TemporaryDirectory() as sub_out,
        ):

            n = 32
            dhours_local = (365 * 24) // n
            rng = np.random.default_rng(seed=1234)

            data_a = {}
            data_b = {}
            timestamps = {}
            for year in self.years:
                da = rng.random((n, self.num_chans_a, self.img_h, self.img_w), dtype=np.float32)
                db = rng.random((n, self.num_chans_b, self.img_h, self.img_w), dtype=np.float32)
                data_a[year] = da
                data_b[year] = db

                jan_01 = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
                ts = np.array(
                    [(jan_01 + dt.timedelta(hours=h * dhours_local)).timestamp() for h in range(n)],
                    dtype=np.float64,
                )
                timestamps[year] = ts

            # Inject NaN into a_chan_1 of year-2018 at three specific time indices
            # and a small spatial region — partial masking, not whole-channel NaN.
            nan_chan = 1  # 'a_chan_1'
            nan_year = 2018
            nan_t_indices = [3, 11, 25]
            nan_h_slice = slice(2, 5)
            nan_w_slice = slice(7, 12)
            for t in nan_t_indices:
                data_a[nan_year][t, nan_chan, nan_h_slice, nan_w_slice] = np.nan

            for year in self.years:
                self._write_h5(
                    os.path.join(sub_a, f"{year}.h5"),
                    data_a[year],
                    timestamps[year],
                    self.channels_a,
                    self.latitudes,
                    self.longitudes,
                )
                self._write_h5(
                    os.path.join(sub_b, f"{year}.h5"),
                    data_b[year],
                    timestamps[year],
                    self.channels_b,
                    self.latitudes,
                    self.longitudes,
                )

            metadata = dict(
                dataset_name="testing_nan",
                h5_path=H5_PATH,
                dims=["time", "channel", "lat", "lon"],
                dhours=dhours_local,
                coords=dict(
                    grid_type="equiangular",
                    lat=self.latitudes.tolist(),
                    lon=self.longitudes.tolist(),
                    channel=self.channels_combined,
                ),
            )

            output_file = os.path.join(sub_out, "concatenated_nan.h5v")
            concatenate(
                input_dirs=[sub_a, sub_b],
                output_file=output_file,
                metadata=metadata,
                channel_names=[self.channels_a, self.channels_b],
                file_names_to_concatenate=[f"{y}.h5" for y in self.years],
                years=self.years,
                dhoursrel=1,
                verbose=False,
            )

            expected_full = np.concatenate(
                [np.concatenate([data_a[y], data_b[y]], axis=1) for y in self.years],
                axis=0,
            )

            year_idx_2018 = self.years.index(nan_year)
            t_offsets_in_concat = [year_idx_2018 * n + t for t in nan_t_indices]

            with h5.File(output_file, "r") as f_conc:
                dset = f_conc[H5_PATH]

                # Full-array comparison with NaN-aware equality
                with self.subTest(desc="full data with NaN"):
                    self.assertTrue(np.array_equal(dset[...], expected_full, equal_nan=True))

                # NaN positions land at the right (time, channel) coordinates
                with self.subTest(desc="NaN mask matches expected positions"):
                    actual_mask = np.isnan(dset[...])
                    expected_mask = np.isnan(expected_full)
                    self.assertTrue(np.array_equal(actual_mask, expected_mask))

                # No NaN should leak outside the (year=2018, channel=a_chan_1) region
                with self.subTest(desc="NaN is confined to a_chan_1 of year 2018"):
                    finite_region_mask = np.isnan(expected_full)
                    finite_region_mask[year_idx_2018 * n :, nan_chan, nan_h_slice, nan_w_slice] = False
                    # Everything outside the injected region must be finite
                    self.assertFalse(finite_region_mask.any())

                # Slice across the year boundary — must still reproduce NaNs that fall
                # right after the boundary
                with self.subTest(desc="time slice across year boundary preserves NaN"):
                    t_slice = slice(n - 1, n + 5)  # last sample of 2017 + first 5 of 2018
                    got = dset[t_slice, :, :, :]
                    self.assertTrue(np.array_equal(got, expected_full[t_slice, :, :, :], equal_nan=True))

                # Slice the masked channel only across both years — confirms cross-channel
                # virtualization preserves the NaN pattern when the NaN lives in the 'a' bank
                with self.subTest(desc="masked channel slice across years"):
                    got = dset[:, nan_chan, :, :]
                    self.assertTrue(np.array_equal(got, expected_full[:, nan_chan, :, :], equal_nan=True))

                # read_direct path must also surface NaNs correctly
                with self.subTest(desc="read_direct: NaN-containing region"):
                    t_slice = slice(t_offsets_in_concat[0] - 1, t_offsets_in_concat[-1] + 2)
                    expected = expected_full[t_slice, :, :, :]
                    buf = np.empty(expected.shape, dtype=expected.dtype)
                    dset.read_direct(buf, source_sel=np.s_[t_slice, :, :, :])
                    self.assertTrue(np.array_equal(buf, expected, equal_nan=True))

    @parameterized.expand([1, 5], skip_on_empty=False)
    def test_concatenate_unannotated_source_files(self, dhoursrel):
        """
        Verify the timestamp-derivation fallback when source files lack dim scales.

        ``concatenate()`` has a try/except block that derives timestamps from
        ``years[idx]`` and ``dhours * dhoursrel`` when
        ``f[entry_key].dims[0]["timestamp"]`` is unavailable. We write source
        files with no scales attached and confirm:
          1. the derived timestamp vector matches the documented formula,
          2. data still round-trips correctly,
          3. dim labels still get applied to the virtual dataset.

        Parameterized over ``dhoursrel`` to catch off-by-one bugs in the
        ``h * dhours * dhoursrel`` derivation specifically.
        """
        from data_process.concatenate_dataset import concatenate

        with (
            tempfile.TemporaryDirectory() as sub_a,
            tempfile.TemporaryDirectory() as sub_b,
            tempfile.TemporaryDirectory() as sub_out,
        ):

            n = 32
            dhours_local = (365 * 24) // n
            rng = np.random.default_rng(seed=4242)

            data_a = {}
            data_b = {}
            for year in self.years:
                da = rng.random((n, self.num_chans_a, self.img_h, self.img_w), dtype=np.float32)
                db = rng.random((n, self.num_chans_b, self.img_h, self.img_w), dtype=np.float32)
                data_a[year] = da
                data_b[year] = db

                # annotate=False → no scales, no timestamp/lat/lon/channel datasets.
                # Forces concatenate() into the derive-timestamps fallback.
                self._write_h5(
                    os.path.join(sub_a, f"{year}.h5"),
                    da,
                    None,
                    self.channels_a,
                    self.latitudes,
                    self.longitudes,
                    annotate=False,
                )
                self._write_h5(
                    os.path.join(sub_b, f"{year}.h5"),
                    db,
                    None,
                    self.channels_b,
                    self.latitudes,
                    self.longitudes,
                    annotate=False,
                )

            metadata = dict(
                dataset_name="testing_unannotated",
                h5_path=H5_PATH,
                dims=["time", "channel", "lat", "lon"],
                dhours=dhours_local,
                coords=dict(
                    grid_type="equiangular",
                    lat=self.latitudes.tolist(),
                    lon=self.longitudes.tolist(),
                    channel=self.channels_combined,
                ),
            )

            output_file = os.path.join(sub_out, f"concatenated_unannotated_dhrel{dhoursrel}.h5v")
            concatenate(
                input_dirs=[sub_a, sub_b],
                output_file=output_file,
                metadata=metadata,
                channel_names=[self.channels_a, self.channels_b],
                file_names_to_concatenate=[f"{y}.h5" for y in self.years],
                years=self.years,
                dhoursrel=dhoursrel,
                verbose=False,
            )

            n_red = n // dhoursrel

            # Reproduce the formula from concatenate's fallback path:
            # ts = jan_01_epoch + h * dhours * dhoursrel for h in range(ne_red)
            # If concatenate's derivation drifts (off-by-one in dhoursrel multiplier,
            # wrong base date, etc.), the comparison fails.
            expected_ts = []
            for year in self.years:
                jan_01 = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
                ts = np.array(
                    [(jan_01 + dt.timedelta(hours=h * dhours_local * dhoursrel)).timestamp() for h in range(n_red)],
                    dtype=np.float64,
                )
                expected_ts.append(ts)
            expected_ts = np.concatenate(expected_ts, axis=0)

            # Use floor-count slicing to match the layout slot exactly. Bare
            # ``[::dhoursrel]`` yields ceil(n/dhoursrel) which over-counts the
            # year by one when n is not a multiple of dhoursrel.
            slc = slice(None, n_red * dhoursrel, dhoursrel)
            expected_data = np.concatenate(
                [np.concatenate([data_a[y][slc], data_b[y][slc]], axis=1) for y in self.years],
                axis=0,
            )

            with h5.File(output_file, "r") as f_conc:
                with self.subTest(desc="derived timestamps match formula"):
                    self.assertTrue(compare_arrays("derived ts", f_conc["timestamp"][...], expected_ts))
                with self.subTest(desc="data round-trips with unannotated sources"):
                    self.assertTrue(compare_arrays("unannotated data", f_conc[H5_PATH][...], expected_data))
                with self.subTest(desc="dim labels still applied"):
                    self.assertEqual(f_conc[H5_PATH].dims[0].label, "Timestamp in UTC time zone")
                    self.assertEqual(f_conc[H5_PATH].dims[1].label, "Channel name")


class TestH5Convert(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create temporary directory
        cls.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = cls.tmpdir.name

        # Create annotated dataset
        path = os.path.join(tmp_path, "data")
        os.makedirs(path, exist_ok=True)
        cls.train_path, _, _, _, _, _, _ = init_hdf5_dataset(path, num_samples_per_year=8, annotate=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    @parameterized.expand(
        [
            ("auto", False, None),
            ("auto", True, None),
            # one sample is 5 x 64 x 128 x 4 bytes = 160 KiB, so 1 MiB holds 6 of the 8 samples
            ("1MB", False, (6, 5, 64, 128)),
            ("1MB", True, (6, 64, 128, 5)),
        ]
    )
    def test_h5_convert(self, chunksize, transpose, expected_chunks):
        from data_process.h5_convert import h5_convert

        output_path = os.path.join(self.tmpdir.name, f"converted_{chunksize}_{transpose}")
        os.makedirs(output_path)

        # batchsize 3 over 8 samples exercises the partial last batch
        h5_convert(
            self.train_path,
            output_path,
            chunksize=chunksize,
            compression_mode="gzip",
            compression_parameter=4,
            batchsize=3,
            transpose=transpose,
        )

        files = sorted([f for f in os.listdir(self.train_path) if f.endswith(".h5")])
        self.assertEqual(sorted(os.listdir(output_path)), files)

        perm = (0, 2, 3, 1) if transpose else (0, 1, 2, 3)
        keys = ["timestamp", "channel", "lat", "lon"]

        for fname in files:
            with (
                h5.File(os.path.join(self.train_path, fname), "r") as fin,
                h5.File(os.path.join(output_path, fname), "r") as fout,
            ):
                with self.subTest(file=fname, desc="data"):
                    self.assertEqual(fout[H5_PATH].compression, "gzip")
                    if expected_chunks is not None:
                        self.assertEqual(fout[H5_PATH].chunks, expected_chunks)
                    self.assertTrue(compare_arrays("data", fout[H5_PATH][...], np.transpose(fin[H5_PATH][...], perm)))
                for key in keys:
                    with self.subTest(file=fname, desc=key):
                        self.assertEqual(fout[key][...].tolist(), fin[key][...].tolist())
                for odim, idim in enumerate(perm):
                    key = keys[idim]
                    with self.subTest(file=fname, desc=f"dim {odim}"):
                        self.assertEqual(fout[H5_PATH].dims[odim].label, fin[H5_PATH].dims[idim].label)
                        self.assertEqual(fout[H5_PATH].dims[odim][key][...].tolist(), fin[key][...].tolist())

    @parameterized.expand(
        [
            # everything fits into a single chunk
            ((8, 5, 64, 128), 4, 2**30, (8, 5, 64, 128)),
            # a single sample does not fit, split channels next
            ((8, 5, 64, 128), 4, 100 * 1024, (1, 3, 64, 128)),
            # a single channel does not fit, split lat next
            ((8, 5, 64, 128), 4, 10 * 1024, (1, 1, 20, 128)),
            # budget below a single element still yields a valid chunk
            ((8, 5, 64, 128), 4, 1, (1, 1, 1, 1)),
        ]
    )
    def test_chunk_shape_from_bytes(self, shape, itemsize, target_bytes, expected):
        from data_process.h5_convert import _chunk_shape_from_bytes

        chunks = _chunk_shape_from_bytes(shape, itemsize, target_bytes)
        self.assertEqual(chunks, expected)
        self.assertLessEqual(np.prod(chunks) * itemsize, max(target_bytes, itemsize))


class TestGetStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary directory
        cls.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = cls.tmpdir.name

        # Create dataset
        path = os.path.join(tmp_path, "data")
        os.makedirs(path, exist_ok=True)
        cls.train_path, cls.num_train, cls.test_path, cls.num_test, _, cls.metadata_path, _ = init_hdf5_dataset(
            path, annotate=True
        )

        # Create dataset with annotations and NaNs:
        nan_path = os.path.join(tmp_path, "nan_data")
        os.makedirs(nan_path, exist_ok=True)
        cls.nan_train_path, cls.nan_num_train, cls.nan_test_path, cls.nan_num_test, _, _, _ = init_hdf5_dataset(
            nan_path, nan_fraction=0.1, annotate=True
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    @parameterized.expand(
        [
            (8, False),
            (16, False),
            (8, True),
            (16, True),
        ],
        skip_on_empty=False,
    )
    @unittest.skipUnless(importlib.util.find_spec("mpi4py") is not None, "mpi4py needs to be installed for this test")
    def test_get_stats(self, batch_size, allow_nan, verbose=False):
        # import necessary modules
        from data_process.get_stats import welford_combine, get_file_stats, mask_data

        # Get list of files to process
        if allow_nan:
            train_files = sorted(
                [os.path.join(self.nan_train_path, f) for f in os.listdir(self.nan_train_path) if f.endswith(".h5")]
            )
        else:
            train_files = sorted(
                [os.path.join(self.train_path, f) for f in os.listdir(self.train_path) if f.endswith(".h5")]
            )

        # Create quadrature rule
        quadrature_rule = grid_to_quadrature_rule("equiangular")
        quadrature = GridQuadrature(quadrature_rule, (IMG_SIZE_H, IMG_SIZE_W), normalize=False)

        # Get stats using get_file_stats
        stats = None
        for file_path in train_files:
            file_stats = get_file_stats(
                filename=file_path,
                file_slice=slice(0, None),  # Process entire file
                wind_indices=None,  # No wind indices
                quadrature=quadrature,
                fail_on_nan=not allow_nan,
                dt=1,
                batch_size=batch_size,
            )
            if stats is None:
                stats = file_stats
            else:
                stats = welford_combine(stats, file_stats)

        # Compute stats naively by loading entire dataset
        all_data = []
        for file_path in train_files:
            with h5.File(file_path, "r") as f:
                data = f[H5_PATH][...].astype(np.float64)
                all_data.append(data)
        all_data = np.concatenate(all_data, axis=0)

        # Convert to torch tensor for quadrature
        tdata = torch.as_tensor(all_data)
        tdata_masked, valid_mask = mask_data(tdata)
        valid_count = torch.sum(quadrature(valid_mask), dim=0).reshape(1, -1, 1, 1)

        # Compute means and variances using quadrature
        tmean = (
            torch.sum(quadrature(tdata_masked * valid_mask), keepdims=False, dim=0).reshape(1, -1, 1, 1) / valid_count
        )
        tm2 = torch.sum(quadrature(torch.square(tdata_masked - tmean) * valid_mask), keepdims=False, dim=0).reshape(
            1, -1, 1, 1
        )

        # Compute time differences
        tdiff = tdata[1:] - tdata[:-1]
        tdiff_masked, tdiff_valid_mask = mask_data(tdiff)
        tdiff_valid_count = torch.sum(quadrature(tdiff_valid_mask), dim=0).reshape(1, -1, 1, 1)
        tdiffmean = (
            torch.sum(quadrature(tdiff_masked * tdiff_valid_mask), keepdims=False, dim=0).reshape(1, -1, 1, 1)
            / tdiff_valid_count
        )
        tdiffvar = (
            torch.sum(
                quadrature(torch.square(tdiff_masked - tdiffmean) * tdiff_valid_mask), keepdims=False, dim=0
            ).reshape(1, -1, 1, 1)
            / tdiff_valid_count
        )

        # Compare results
        with self.subTest(desc="mean"):
            self.assertTrue(
                compare_arrays("mean", stats["global_meanvar"]["values"][0].numpy(), tmean.numpy(), verbose=verbose)
            )
        with self.subTest(desc="m2"):
            self.assertTrue(
                compare_arrays("m2", stats["global_meanvar"]["values"][1].numpy(), tm2.numpy(), verbose=verbose)
            )

        # Compare min/max
        with self.subTest(desc="max"):
            self.assertTrue(
                compare_arrays(
                    "max",
                    stats["maxs"]["values"].numpy(),
                    np.nanmax(all_data, keepdims=True, axis=(0, 2, 3)),
                    verbose=verbose,
                )
            )
        with self.subTest(desc="min"):
            self.assertTrue(
                compare_arrays(
                    "min",
                    stats["mins"]["values"].numpy(),
                    np.nanmin(all_data, keepdims=True, axis=(0, 2, 3)),
                    verbose=verbose,
                )
            )


# conversion scripts, with the extras each needs to be importable at all
_CONVERTERS = [
    ("data_process.convert_era5_to_makani_input", ["h5py"]),
    ("data_process.sources.ncar", ["h5py"]),
    ("data_process.sources.wb2", ["xarray", "h5py"]),
    ("data_process.convert_era5_invariants_to_makani_input", ["h5py"]),
    ("data_process.convert_makani_output_to_wb2", ["mpi4py", "xarray", "dask", "h5py"]),
    ("data_process.generate_wb2_climatology", ["mpi4py", "xarray", "h5py"]),
]


class TestConverterImports(unittest.TestCase):
    """
    Import each conversion script in ``data_process``.

    The converters are driven by MPI against multi terabyte archives and are not
    otherwise exercised by any test, so this is the only thing standing between a
    rename here and a failure on a production run.

    What it covers is the *import surface*: module level ``from ... import``
    statements and any code that runs at import time. It says nothing about what
    happens inside the functions -- a helper called with the wrong arguments in
    the middle of a conversion loop is still invisible here.

    Each is skipped when its dependencies are missing, since ``mpi4py``,
    ``xarray`` and ``dask`` live in the ``data_process`` extra rather than the
    base install; that means these normally skip in CI and run in the container.
    """

    @parameterized.expand(_CONVERTERS)
    def test_converter_imports(self, module_name, requirements):
        missing = [name for name in requirements if importlib.util.find_spec(name) is None]
        if missing:
            self.skipTest(f"{module_name} needs {', '.join(missing)}")
        importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Date ranges: data_process/date_range.py
# ---------------------------------------------------------------------------
#
# Unit tests for ``data_process.date_range``, which decides which samples go into
# each yearly file written by the Weatherbench and NCAR converters.
#
# Pure datetime arithmetic, so no MPI and no data are involved. What happens to
# an existing file on disk is up to the converters and is not covered here.


class TestParseDate(unittest.TestCase):
    def test_bare_start_date_is_midnight(self):
        self.assertEqual(parse_date("2026-06-30"), _utc(2026, 6, 30, 0))

    def test_bare_end_date_includes_whole_day(self):
        self.assertEqual(parse_date("2026-06-30", end_of_day=True), _utc(2026, 6, 30, 23))

    def test_explicit_hour_is_kept_for_end_date(self):
        # only a bare date is widened to the end of the day
        self.assertEqual(parse_date("2026-06-30T06", end_of_day=True), _utc(2026, 6, 30, 6))

    def test_naive_values_are_utc(self):
        self.assertEqual(parse_date("2026-01-01T06:00"), _utc(2026, 1, 1, 6))

    def test_aware_values_are_converted_to_utc(self):
        self.assertEqual(parse_date("2026-01-01T06:00+02:00"), _utc(2026, 1, 1, 4))

    def test_command_line_round_trip(self):
        parser = ap.ArgumentParser()
        add_date_range_arguments(parser)
        args = parser.parse_args(["--start_date", "2018-01-01", "--end_date", "2019-12-31"])
        self.assertEqual(date_range_from_args(args), (_utc(2018, 1, 1, 0), _utc(2019, 12, 31, 23)))


class TestYearlySampleTimes(unittest.TestCase):
    def test_full_years_match_the_yearly_grid(self):
        # 2023 is a common year, 2024 a leap year
        result = yearly_sample_times(_utc(2023, 1, 1), _utc(2024, 12, 31, 23), 6)
        self.assertEqual([year for year, _ in result], [2023, 2024])
        self.assertEqual([len(times) for _, times in result], [365 * 4, 366 * 4])
        self.assertEqual(result[0][1][0], _utc(2023, 1, 1, 0))
        self.assertEqual(result[1][1][-1], _utc(2024, 12, 31, 18))

    def test_partial_year_ends_inclusively(self):
        [(year, times)] = yearly_sample_times(_utc(2026, 1, 1), _utc(2026, 6, 30, 23), 6)
        self.assertEqual(year, 2026)
        self.assertEqual(times[0], _utc(2026, 1, 1, 0))
        self.assertEqual(times[-1], _utc(2026, 6, 30, 18))
        self.assertEqual(len(times), 181 * 4)

    def test_end_on_a_sample_is_included(self):
        [(_, times)] = yearly_sample_times(_utc(2026, 1, 1), _utc(2026, 1, 2, 12), 6)
        self.assertEqual(times[-1], _utc(2026, 1, 2, 12))

    def test_off_grid_start_rounds_up(self):
        [(_, times)] = yearly_sample_times(_utc(2026, 3, 1, 7), _utc(2026, 3, 1, 23), 6)
        self.assertEqual(times, [_utc(2026, 3, 1, 12), _utc(2026, 3, 1, 18)])

    def test_grid_is_anchored_at_the_start_of_each_year(self):
        # a start inside the year must not shift the grid of the following year
        result = yearly_sample_times(_utc(2025, 12, 31, 7), _utc(2026, 1, 1, 23), 6)
        self.assertEqual(result[0], (2025, [_utc(2025, 12, 31, 12), _utc(2025, 12, 31, 18)]))
        self.assertEqual(result[1][1][0], _utc(2026, 1, 1, 0))

    def test_range_across_years_is_split(self):
        result = yearly_sample_times(_utc(2025, 7, 1), _utc(2027, 3, 31, 23), 24)
        self.assertEqual([year for year, _ in result], [2025, 2026, 2027])
        for year, times in result:
            self.assertTrue(all(t.year == year for t in times))
        self.assertEqual(result[0][1][0], _utc(2025, 7, 1))
        self.assertEqual(len(result[1][1]), 365)
        self.assertEqual(result[2][1][-1], _utc(2027, 3, 31))

    def test_year_without_samples_is_dropped(self):
        # nothing in 2025 lies on the 6h grid after 19Z on December 31st
        result = yearly_sample_times(_utc(2025, 12, 31, 19), _utc(2026, 1, 1, 0), 6)
        self.assertEqual(result, [(2026, [_utc(2026, 1, 1, 0)])])

    def test_range_without_samples_raises(self):
        with self.assertRaises(ValueError):
            yearly_sample_times(_utc(2026, 1, 1, 1), _utc(2026, 1, 1, 5), 6)

    def test_end_before_start_raises(self):
        with self.assertRaises(ValueError):
            yearly_sample_times(_utc(2026, 2, 1), _utc(2026, 1, 1), 6)


# ---------------------------------------------------------------------------
# ERA5 converter and its sources: data_process/convert_era5_to_makani_input.py, data_process/sources/
# ---------------------------------------------------------------------------
#
# Tests for ``data_process/convert_era5_to_makani_input.py`` and its sources.
#
# Everything but the last suite runs without MPI and without network access: the
# output file helpers write serial files, the NCAR source reads from a stand-in
# for its S3 store and the WB2 source from a small local zarr store. The full
# conversion needs mpi4py and parallel HDF5 and is skipped where they are missing.


_LAT = [90.0, 0.0, -90.0]
_LON = [0.0, 120.0, 240.0]
_DHOURS = 6


def _metadata(channels):
    return {"dhours": _DHOURS, "coords": {"channel": channels, "lat": _LAT, "lon": _LON}}


def _samples(start, num):
    return [(idx, start + dt.timedelta(hours=_DHOURS * idx)) for idx in range(num)]


def _have_mpi_hdf5():
    return importlib.util.find_spec("mpi4py") is not None and h5.get_config().mpi


class _OutputFileCase(unittest.TestCase):
    """Provides a serial output file laid out like the converter's."""

    channels = ["a", "b", "c"]

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.times = [t for _, t in _samples(_utc(2023, 6, 15), 8)]
        timestamps = np.array([t.timestamp() for t in self.times], dtype=np.float64)
        self.out = _create_output_file(
            os.path.join(self._tmpdir.name, "2023.h5"), None, "fields", timestamps, self.channels, _LAT, _LON
        )

    def tearDown(self):
        self.out.close()
        self._tmpdir.cleanup()


class TestBatched(unittest.TestCase):
    def test_consecutive_batches_with_remainder(self):
        self.assertEqual(list(_batched(range(7), 3)), [(0, 1, 2), (3, 4, 5), (6,)])

    def test_empty_input_gives_no_batches(self):
        self.assertEqual(list(_batched([], 3)), [])


class TestOutputFile(_OutputFileCase):
    def test_fill_value_is_declared_but_never_written(self):
        dset = self.out["fields"]
        self.assertTrue(np.isnan(dset.fillvalue))
        self.assertEqual(dset.id.get_create_plist().get_fill_time(), h5.h5d.FILL_TIME_NEVER)

    def test_layout(self):
        self.assertEqual(self.out["fields"].shape, (8, 3, len(_LAT), len(_LON)))
        self.assertTrue(np.all(self.out["valid_data"][...] == 1))
        self.assertEqual([c.decode() for c in self.out["channel"][...]], self.channels)
        expected = np.array([t.timestamp() for t in self.times])
        self.assertTrue(
            compare_arrays("timestamps", self.out["timestamp"][...], expected, atol=0.0, rtol=0.0, shape_check=True)
        )

    def test_write_missing_touches_only_the_given_channels_and_samples(self):
        self.out["fields"][...] = 1.0
        _write_missing(self.out, "fields", _samples(_utc(2023, 6, 15), 8)[2:5], [0, 2])

        fields = self.out["fields"][...]
        valid = self.out["valid_data"][...]
        for cidx in [0, 2]:
            self.assertTrue(np.all(np.isnan(fields[2:5, cidx])))
            self.assertTrue(np.all(valid[2:5, cidx] == 0))
            self.assertTrue(np.all(fields[[0, 1, 5, 6, 7], cidx] == 1.0))
            self.assertTrue(np.all(valid[[0, 1, 5, 6, 7], cidx] == 1))
        self.assertTrue(np.all(fields[:, 1] == 1.0))
        self.assertTrue(np.all(valid[:, 1] == 1))


class _FakeNcarStore(object):
    """Stand-in for NcarStore, serving local files for the keys it knows and raising for all others."""

    def __init__(self, files):
        self.files = files
        self.handles = {}
        self.released = []

    def _path(self, key):
        for marker, path in self.files.items():
            if marker in key:
                return path
        raise FileNotFoundError(key)

    def set_read_plan(self, keys):
        pass

    def open(self, key):
        if key not in self.handles:
            self.handles[key] = h5.File(self._path(key), "r")
        return self.handles[key]

    def coord(self, key, name):
        return self.open(key)[name][:]

    def release(self, key):
        self.released.append(key)
        handle = self.handles.pop(key, None)
        if handle is not None:
            handle.close()

    def close(self):
        for handle in self.handles.values():
            handle.close()


class TestNcarSource(_OutputFileCase):
    """Filling and imputation of the NCAR source against local stand-ins for the bucket objects."""

    channels = ["z500", "t2m", "tp", "not_a_variable"]

    def setUp(self):
        super().setUp()
        from makani.utils.dataloaders.ncar_helpers import to_ncar_hours

        # a surface file for t2m that only holds the first of the two days, so
        # that the second day exercises the missing timestep path
        self.t2m = np.arange(4 * len(_LAT) * len(_LON), dtype=np.float32).reshape(4, len(_LAT), len(_LON))
        path = os.path.join(self._tmpdir.name, "t2m.nc")
        with h5.File(path, "w") as f:
            f["latitude"] = np.array(_LAT)
            f["longitude"] = np.array(_LON)
            f["time"] = np.array([to_ncar_hours(t) for t in self.times[:4]])
            f["VAR_2T"] = self.t2m
        self.store = _FakeNcarStore({"128_167_2t": path})

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def _make(self, **options):
        from data_process.sources.ncar import NcarSource

        with mock.patch("data_process.sources.ncar.NcarStore", lambda *args, **kwargs: self.store):
            return NcarSource(_metadata(self.channels), 0, skip_missing_channels=True, **options)

    def _fill(self, source):
        units = source.split_units(_samples(_utc(2023, 6, 15), 8))
        self.assertEqual([len(unit) for unit in units], [4, 4])
        source.begin_year(units)
        source.fill(self.out, "fields", units)

    def test_unsupported_channels_are_reported_as_skipped(self):
        self.assertEqual(self._make().skipped_channel_indices(), [3])

    def test_missing_data_fails_without_imputation(self):
        with self.assertRaises(FileNotFoundError):
            self._fill(self._make())

    def test_missing_data_is_imputed(self):
        self._fill(self._make(impute_missing_timestamps=True))
        fields = self.out["fields"][...]
        valid = self.out["valid_data"][...]

        # t2m is read for the first day, and imputed for the second, missing from its file
        self.assertTrue(compare_arrays("t2m", fields[:4, 1], self.t2m, atol=0.0, rtol=0.0, shape_check=True))
        self.assertTrue(np.all(valid[:4, 1] == 1))
        self.assertTrue(np.all(np.isnan(fields[4:, 1])))
        self.assertTrue(np.all(valid[4:, 1] == 0))

        # z500 and tp have no objects at all
        for cidx in [0, 2]:
            self.assertTrue(np.all(np.isnan(fields[:, cidx])))
            self.assertTrue(np.all(valid[:, cidx] == 0))

        # the skipped channel is the converter's business, not the source's
        self.assertTrue(np.all(valid[:, 3] == 1))

    def test_pressure_level_file_is_released_after_imputation(self):
        self._fill(self._make(impute_missing_timestamps=True))
        self.assertEqual(sum("128_129_z" in key for key in self.store.released), 2)


@unittest.skipUnless(
    importlib.util.find_spec("xarray") is not None and importlib.util.find_spec("zarr") is not None,
    "needs xarray and zarr",
)
class TestWb2Source(_OutputFileCase):
    """Filling, skipping and imputation of the WB2 source against a small local zarr store."""

    channels = ["z500", "t2m", "u10m"]

    def setUp(self):
        super().setUp()
        import xarray as xr

        # u10m is absent from the store, and the last sample is missing from its time axis
        times = np.array([np.datetime64(t.replace(tzinfo=None), "ns") for t in self.times[:-1]])
        shape = (len(times), len(_LAT), len(_LON))
        self.t2m = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        self.z = -np.arange(2 * np.prod(shape), dtype=np.float32).reshape(len(times), 2, len(_LAT), len(_LON))
        data = xr.Dataset(
            {
                "2m_temperature": (("time", "latitude", "longitude"), self.t2m),
                "geopotential": (("time", "level", "latitude", "longitude"), self.z),
            },
            coords={"time": times, "level": [500, 850], "latitude": _LAT, "longitude": _LON},
        )
        self.store = os.path.join(self._tmpdir.name, "wb2.zarr")
        data.to_zarr(self.store)

    def _make(self, **options):
        from data_process.sources.wb2 import Wb2Source

        return Wb2Source(_metadata(self.channels), 0, input_file=self.store, **options)

    def test_nonpositive_batch_size_is_rejected(self):
        for batch_size in [0, -1]:
            with self.subTest(batch_size=batch_size):
                with self.assertRaises(ValueError):
                    self._make(skip_missing_channels=True, batch_size=batch_size)

    def test_missing_variable_fails_without_skipping(self):
        with self.assertRaises(IndexError):
            self._make()

    def test_missing_variables_are_reported_as_skipped(self):
        self.assertEqual(self._make(skip_missing_channels=True).skipped_channel_indices(), [2])

    def test_missing_timestamp_fails_without_imputation(self):
        source = self._make(skip_missing_channels=True, batch_size=8)
        with self.assertRaises(IndexError):
            source.fill(self.out, "fields", source.split_units(_samples(_utc(2023, 6, 15), 8)))

    def test_fill_and_impute(self):
        source = self._make(skip_missing_channels=True, impute_missing_timestamps=True, batch_size=3)
        units = source.split_units(_samples(_utc(2023, 6, 15), 8))
        for batch in _batched(units, source.units_per_fill):
            source.fill(self.out, "fields", list(batch))
        fields = self.out["fields"][...]
        valid = self.out["valid_data"][...]

        self.assertTrue(compare_arrays("t2m", fields[:7, 1], self.t2m, atol=0.0, rtol=0.0, shape_check=True))
        self.assertTrue(compare_arrays("z500", fields[:7, 0], self.z[:, 0], atol=0.0, rtol=0.0, shape_check=True))
        self.assertTrue(np.all(valid[:7, :2] == 1))
        # the last sample is not in the store
        self.assertTrue(np.all(np.isnan(fields[7, :2])))
        self.assertTrue(np.all(valid[7, :2] == 0))


class _FakeSource(Source):
    """Writes the sample index into every channel it provides and skips the last one."""

    units_per_fill = 2

    def fill(self, out, entry_key, units):
        for unit in units:
            for sample_index, _ in unit:
                for cidx in range(len(self.channel_names) - 1):
                    out[entry_key][sample_index, cidx, ...] = np.full(
                        (len(self.lat), len(self.lon)), sample_index, dtype=np.float32
                    )

    def skipped_channel_indices(self):
        return [len(self.channel_names) - 1]


@unittest.skipUnless(_have_mpi_hdf5(), "needs mpi4py and h5py built with MPI")
class TestConvert(unittest.TestCase):
    """A full conversion on a single rank across a year boundary, with a source that skips a channel."""

    def setUp(self):
        import json

        self._tmpdir = tempfile.TemporaryDirectory()
        self.metadata_file = os.path.join(self._tmpdir.name, "metadata.json")
        with open(self.metadata_file, "w") as f:
            json.dump(_metadata(["a", "b", "c"]), f)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_yearly_files(self):
        from data_process.convert_era5_to_makani_input import convert

        convert(_FakeSource, self._tmpdir.name, self.metadata_file, _utc(2022, 12, 31, 6), _utc(2023, 1, 1, 12))

        expected = {
            "2022.h5": [_utc(2022, 12, 31, 6), _utc(2022, 12, 31, 12), _utc(2022, 12, 31, 18)],
            "2023.h5": [_utc(2023, 1, 1, 0), _utc(2023, 1, 1, 6), _utc(2023, 1, 1, 12)],
        }
        for name, times in expected.items():
            with self.subTest(file=name), h5.File(os.path.join(self._tmpdir.name, name), "r") as f:
                stamps = np.array([t.timestamp() for t in times])
                self.assertTrue(
                    compare_arrays("timestamps", f["timestamp"][...], stamps, atol=0.0, rtol=0.0, shape_check=True)
                )
                fields = f["fields"][...]
                valid = f["valid_data"][...]
                for idx in range(len(times)):
                    self.assertTrue(np.all(fields[idx, :2] == idx))
                self.assertTrue(np.all(valid[:, :2] == 1))
                # the skipped channel is written as missing rather than left unwritten
                self.assertTrue(np.all(np.isnan(fields[:, 2])))
                self.assertTrue(np.all(valid[:, 2] == 0))


# ---------------------------------------------------------------------------
# Invariant converter and its sources: data_process/convert_era5_invariants_to_makani_input.py
# ---------------------------------------------------------------------------
#
# The time invariant counterpart of the suites above. It is serial throughout, so
# the full conversion runs without MPI as well.


def _invariant_metadata(channels):
    return {"coords": {"channel": channels, "lat": _LAT, "lon": _LON}}


def _invariant_field(offset):
    return (offset + np.arange(len(_LAT) * len(_LON), dtype=np.float32)).reshape(len(_LAT), len(_LON))


class _InvariantOutputFileCase(unittest.TestCase):
    """Provides a serial invariant output file laid out like the converter's."""

    channels = ["z", "lsm", "not_a_variable"]

    def setUp(self):
        from data_process.convert_era5_invariants_to_makani_input import _create_output_file

        self._tmpdir = tempfile.TemporaryDirectory()
        self.out = _create_output_file(
            os.path.join(self._tmpdir.name, "invariants.h5"), "fields", self.channels, _LAT, _LON
        )

    def tearDown(self):
        self.out.close()
        self._tmpdir.cleanup()


class TestInvariantOutputFile(_InvariantOutputFileCase):
    def test_layout(self):
        self.assertEqual(self.out["fields"].shape, (3, len(_LAT), len(_LON)))
        self.assertTrue(np.all(self.out["valid_data"][...] == 1))
        self.assertEqual([c.decode() for c in self.out["channel"][...]], self.channels)
        self.assertNotIn("timestamp", self.out)

    def test_unwritten_channels_are_nan(self):
        self.assertTrue(np.all(np.isnan(self.out["fields"][...])))

    def test_write_missing_touches_only_the_given_channels(self):
        from data_process.convert_era5_invariants_to_makani_input import _write_missing

        self.out["fields"][...] = 1.0
        _write_missing(self.out, "fields", [0, 2])

        fields = self.out["fields"][...]
        valid = self.out["valid_data"][...]
        self.assertTrue(np.all(np.isnan(fields[[0, 2]])))
        self.assertTrue(np.all(valid[[0, 2]] == 0))
        self.assertTrue(np.all(fields[1] == 1.0))
        self.assertEqual(valid[1], 1)


class TestNcarInvariantSource(_InvariantOutputFileCase):
    """Filling of the NCAR invariant source against local stand-ins for the bucket objects."""

    def setUp(self):
        super().setUp()

        # lsm carries the netCDF fill value at one point, which has to come out as NaN
        self.z = _invariant_field(0.0)
        self.lsm = _invariant_field(100.0)
        self.lsm[1, 1] = 9.999e20
        files = {}
        for marker, name, data in [("128_129_z", "Z", self.z), ("128_172_lsm", "LSM", self.lsm)]:
            path = os.path.join(self._tmpdir.name, f"{marker}.nc")
            with h5.File(path, "w") as f:
                f["latitude"] = np.array(_LAT)
                f["longitude"] = np.array(_LON)
                f.create_dataset(name, data=data[np.newaxis])
                f[name].attrs["_FillValue"] = np.float32(9.999e20)
            files[marker] = path
        self.store = _FakeNcarStore(files)

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def _make(self, channels=None, **options):
        from data_process.sources.ncar import NcarInvariantSource

        channels = self.channels if channels is None else channels
        with mock.patch("data_process.sources.ncar.NcarStore", lambda *args, **kwargs: self.store):
            return NcarInvariantSource(_invariant_metadata(channels), **options)

    def test_unknown_channel_fails_without_skipping(self):
        with self.assertRaises(ValueError):
            self._make()

    def test_unknown_channels_are_reported_as_skipped(self):
        self.assertEqual(self._make(skip_missing_channels=True).skipped_channel_indices(), [2])

    def test_fill(self):
        self._make(skip_missing_channels=True).fill(self.out, "fields")
        fields = self.out["fields"][...]

        self.assertTrue(compare_arrays("z", fields[0], self.z, atol=0.0, rtol=0.0, shape_check=True))
        self.assertTrue(np.isnan(fields[1, 1, 1]))
        lsm = self.lsm.copy()
        lsm[1, 1] = np.nan
        self.assertTrue(np.array_equal(fields[1], lsm, equal_nan=True))
        # every object is released once read
        self.assertEqual(len(self.store.handles), 0)

    def test_field_absent_from_the_bucket_fails(self):
        source = self._make(channels=["z", "slt"])
        with self.assertRaises(FileNotFoundError):
            source.fill(self.out, "fields")


@unittest.skipUnless(
    importlib.util.find_spec("xarray") is not None and importlib.util.find_spec("zarr") is not None,
    "needs xarray and zarr",
)
class TestWb2InvariantSource(_InvariantOutputFileCase):
    """Filling and skipping of the WB2 invariant source against a small local zarr store."""

    channels = ["z", "lsm", "slt", "not_a_variable"]

    def setUp(self):
        super().setUp()
        import xarray as xr

        # z without a time axis as in WB2, lsm repeated along one as in ARCO-ERA5,
        # and slt absent from the store; latitude ascending, to exercise the matching
        self.z = _invariant_field(0.0)
        self.lsm = _invariant_field(100.0)
        times = np.array([np.datetime64("2023-06-15T00", "ns"), np.datetime64("2023-06-15T06", "ns")])
        data = xr.Dataset(
            {
                "geopotential_at_surface": (("latitude", "longitude"), self.z[::-1]),
                "land_sea_mask": (("time", "latitude", "longitude"), np.stack([self.lsm[::-1], -self.lsm[::-1]])),
            },
            coords={"time": times, "latitude": _LAT[::-1], "longitude": _LON},
        )
        self.store = os.path.join(self._tmpdir.name, "wb2.zarr")
        data.to_zarr(self.store)

    def _make(self, **options):
        from data_process.sources.wb2 import Wb2InvariantSource

        return Wb2InvariantSource(_invariant_metadata(self.channels), input_file=self.store, **options)

    def test_missing_variable_fails_without_skipping(self):
        with self.assertRaises(IndexError):
            self._make()

    def test_missing_variables_are_reported_as_skipped(self):
        self.assertEqual(self._make(skip_missing_channels=True).skipped_channel_indices(), [2, 3])

    def test_fill(self):
        self._make(skip_missing_channels=True).fill(self.out, "fields")
        fields = self.out["fields"][...]

        self.assertTrue(compare_arrays("z", fields[0], self.z, atol=0.0, rtol=0.0, shape_check=True))
        # read at the first timestep
        self.assertTrue(compare_arrays("lsm", fields[1], self.lsm, atol=0.0, rtol=0.0, shape_check=True))


class _FakeInvariantSource(InvariantSource):
    """Writes the channel index into every channel it provides and skips the last one."""

    def fill(self, out, entry_key):
        for cidx in range(len(self.channel_names) - 1):
            out[entry_key][cidx, ...] = np.full((len(self.lat), len(self.lon)), cidx, dtype=np.float32)

    def skipped_channel_indices(self):
        return [len(self.channel_names) - 1]


class _FailingInvariantSource(_FakeInvariantSource):
    """Fails after writing the first channel, as an interrupted download would."""

    closed = False

    def fill(self, out, entry_key):
        out[entry_key][0, ...] = np.zeros((len(self.lat), len(self.lon)), dtype=np.float32)
        raise FileNotFoundError("lost connection")

    def close(self):
        _FailingInvariantSource.closed = True


class TestConvertInvariants(unittest.TestCase):
    """A full invariant conversion with a source that skips a channel."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        # the dataset metadata, whose channels are not the invariant ones
        self.metadata_file = os.path.join(self._tmpdir.name, "metadata.json")
        with open(self.metadata_file, "w") as f:
            json.dump(_metadata(["z500", "t2m"]), f)
        self.output_file = os.path.join(self._tmpdir.name, "invariants.h5")

    def tearDown(self):
        self._tmpdir.cleanup()

    def _convert(self, **options):
        from data_process.convert_era5_invariants_to_makani_input import convert

        convert(_FakeInvariantSource, self.output_file, self.metadata_file, ["z", "lsm", "slt"], **options)

    def test_invariant_file(self):
        self._convert()
        with h5.File(self.output_file, "r") as f:
            self.assertEqual([c.decode() for c in f["channel"][...]], ["z", "lsm", "slt"])
            self.assertEqual(list(f["lat"][...]), _LAT)
            fields = f["fields"][...]
            valid = f["valid_data"][...]
        self.assertTrue(np.all(fields[0] == 0) and np.all(fields[1] == 1))
        self.assertTrue(np.all(valid[:2] == 1))
        # the skipped channel is written as missing
        self.assertTrue(np.all(np.isnan(fields[2])))
        self.assertEqual(valid[2], 0)

    def test_existing_file_is_kept_unless_forced(self):
        with open(self.output_file, "w") as f:
            f.write("sentinel")
        self._convert()
        with open(self.output_file, "r") as f:
            self.assertEqual(f.read(), "sentinel")

        self._convert(force_overwrite=True)
        with h5.File(self.output_file, "r") as f:
            self.assertEqual(f["fields"].shape, (3, len(_LAT), len(_LON)))

    def test_failed_conversion_leaves_no_file_behind(self):
        from data_process.convert_era5_invariants_to_makani_input import convert

        _FailingInvariantSource.closed = False
        with self.assertRaises(FileNotFoundError):
            convert(_FailingInvariantSource, self.output_file, self.metadata_file, ["z", "lsm", "slt"])

        # neither the output nor its temporary file, so that a rerun converts afresh
        self.assertEqual(os.listdir(self._tmpdir.name), ["metadata.json"])
        self.assertTrue(_FailingInvariantSource.closed)

        self._convert()
        with h5.File(self.output_file, "r") as f:
            self.assertTrue(np.all(f["fields"][1] == 1))

    def test_command_line_defaults_to_all_invariants(self):
        from data_process.convert_era5_invariants_to_makani_input import build_parser
        from makani.utils.dataloaders.ncar_helpers import invariant_variables

        args = build_parser().parse_args(["ncar", "--output_file", "out.h5", "--metadata_file", "m.json"])
        self.assertEqual(args.channels, list(invariant_variables))


# ---------------------------------------------------------------------------
# NCAR against ARCO-ERA5, read from the public archives
# ---------------------------------------------------------------------------
#
# Cross check of the NCAR source against ARCO-ERA5, read through the WB2 source.
#
# Both archives hold the same ECMWF reanalysis on the same 0.25 degree grid, but
# were produced independently and are stored at different precisions, so they
# do not agree bitwise. Each channel therefore has to match ARCO closely, and
# far better than a deliberately wrong counterpart does: ARCO six hours later,
# and ARCO at the neighbouring pressure level. That is what the bugs this guards
# against look like -- a timestep picked off by one, a level mixed up, channels
# swapped.
#
# The test reads about a gigabyte from the public S3 and GCS buckets, so it only
# runs when ``MAKANI_TEST_REMOTE_DATA=1`` is set. It prints the metrics the
# thresholds below were calibrated from.
#
# Total precipitation is left out: ARCO stores it hourly, while the WB2 source
# expects the 6 hourly accumulation of the WeatherBench2 datasets.


_ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# one day at 6 hours plus the next 00Z, which crosses NCAR's daily pressure level files
_ARCO_TIMES = [dt.datetime(2023, 6, 15, tzinfo=dt.timezone.utc) + dt.timedelta(hours=6 * idx) for idx in range(5)]

# the channels compared, each with the neighbouring level used as wrong counterpart
_ARCO_COMPARED = ["z500", "t850", "t2m", "u10m", "msl"]
_ARCO_NEIGHBOURS = {"z500": "z550", "t850": "t875"}

# ARCO also provides the neighbours; the WB2 source reads every pressure level
# variable on every level requested, so they come as the full product
_ARCO_CHANNELS = [f"{var}{level}" for var in ["z", "t"] for level in [500, 550, 850, 875]] + ["t2m", "u10m", "msl"]

# the 0.25 degree grid both archives are on, latitude descending
_ARCO_LAT = np.linspace(90.0, -90.0, 721).tolist()
_ARCO_LON = (np.arange(1440) * 0.25).tolist()

# Calibrated on the dates above: the errors against ARCO, normalized by ARCO's
# spread, are 2e-5 to 5e-5 for all channels, and the wrong counterparts are at
# least about 4000 times worse (t850 and z500 against six hours later). The
# thresholds leave an order of magnitude of headroom on both.
# a channel has to match ARCO this well ...
_ARCO_MAX_ERROR = 5e-4
# ... and this many times better than its wrong counterparts
_ARCO_MIN_MARGIN = 400.0


def _have_network_dependencies():
    return all(importlib.util.find_spec(name) is not None for name in ["s3fs", "gcsfs", "xarray", "zarr"])


def _arco_convert(source, channels, workdir, name):
    """Fill ``_ARCO_TIMES`` from ``source`` into a serial output file and return fields and validity."""
    stamps = np.array([t.timestamp() for t in _ARCO_TIMES], dtype=np.float64)
    path = os.path.join(workdir, f"{name}.h5")
    with _create_output_file(path, None, "fields", stamps, channels, _ARCO_LAT, _ARCO_LON) as out:
        units = source.split_units(list(enumerate(_ARCO_TIMES)))
        source.begin_year(units)
        for batch in _batched(units, source.units_per_fill):
            source.fill(out, "fields", list(batch))
        fields, valid = out["fields"][...], out["valid_data"][...]
    source.close()
    return fields, valid


def _normalized_error(values, reference):
    """Area weighted RMSE of ``values`` against ``reference``, relative to the spread of ``reference``."""
    weights = np.broadcast_to(np.cos(np.deg2rad(np.asarray(_ARCO_LAT)))[:, None], reference.shape[-2:])
    weights = np.broadcast_to(weights, reference.shape)

    def mean(x):
        return np.sum(weights * x) / np.sum(weights)

    rmse = np.sqrt(mean((values - reference) ** 2))
    spread = np.sqrt(mean((reference - mean(reference)) ** 2))
    return float(rmse / spread)


@unittest.skipUnless(
    os.environ.get("MAKANI_TEST_REMOTE_DATA") == "1", "reads from public buckets, set MAKANI_TEST_REMOTE_DATA=1"
)
@unittest.skipUnless(_have_network_dependencies(), "needs s3fs, gcsfs, xarray and zarr")
class TestNcarVsArco(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from data_process.sources.ncar import NcarSource
        from data_process.sources.wb2 import Wb2Source

        cls._tmpdir = tempfile.TemporaryDirectory()

        ncar = NcarSource(cls._metadata(_ARCO_COMPARED), 0)
        arco = Wb2Source(cls._metadata(_ARCO_CHANNELS), 0, input_file=_ARCO_STORE, batch_size=len(_ARCO_TIMES))

        cls.ncar, cls.ncar_valid = _arco_convert(ncar, _ARCO_COMPARED, cls._tmpdir.name, "ncar")
        cls.arco, cls.arco_valid = _arco_convert(arco, _ARCO_CHANNELS, cls._tmpdir.name, "arco")

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    @staticmethod
    def _metadata(channels):
        return {"dhours": 6, "coords": {"channel": channels, "lat": _ARCO_LAT, "lon": _ARCO_LON}}

    def _arco(self, channel):
        return self.arco[:, _ARCO_CHANNELS.index(channel)]

    def test_everything_was_read(self):
        self.assertTrue(np.all(self.ncar_valid == 1))
        self.assertTrue(np.all(self.arco_valid == 1))
        self.assertFalse(np.any(np.isnan(self.ncar)))
        self.assertFalse(np.any(np.isnan(self.arco)))

    def test_channels_match_arco_better_than_wrong_counterparts(self):
        for cidx, channel in enumerate(_ARCO_COMPARED):
            with self.subTest(channel=channel):
                ncar = self.ncar[:, cidx]
                arco = self._arco(channel)

                error = _normalized_error(ncar, arco)
                # NCAR at t against ARCO at t + 6h
                wrong = {"6h later": _normalized_error(ncar[:-1], arco[1:])}
                if channel in _ARCO_NEIGHBOURS:
                    wrong[_ARCO_NEIGHBOURS[channel]] = _normalized_error(ncar, self._arco(_ARCO_NEIGHBOURS[channel]))

                print(f"{channel}: error {error:.2e}, " + ", ".join(f"{k} {v:.2e}" for k, v in wrong.items()))

                self.assertLess(error, _ARCO_MAX_ERROR)
                for desc, value in wrong.items():
                    self.assertGreater(value, _ARCO_MIN_MARGIN * error, f"{channel} is not distinguishable from {desc}")


if __name__ == "__main__":
    unittest.main()
