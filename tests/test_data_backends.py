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
Unit tests for the storage backends.

``test_sample_source.py`` drives the raster backends end to end through
``SampleSource``, which covers the read path thoroughly. What it cannot see are
the properties that hold *between* the pieces, and those are what this file
pins -- along with the whole of the ICON backend, which no end to end suite
reaches yet because nothing downstream consumes an unstructured grid:

* **coordinates match the data.** A backend promises that ``chunk.lat`` and
  ``chunk.lon`` give the position of every value it emits. Nothing downstream
  checks that, and getting it wrong stays invisible until something resamples
  with those coordinates and produces a subtly displaced field.
* **layout detection.** New logic replacing checks that used to be scattered
  across the loader; the end to end tests exercise one happy path each and none
  of the failure modes.
* **contiguous_slices.** A broken version still returns correct data, just in
  many small reads instead of a few large ones, so no behavioural test can see
  it -- only a profile can.
* **pickling.** ``test_pickle_roundtrip`` covers handles, but only for local
  files; the connection a remote backend holds is dropped by different
  machinery.
* **times are absolute.** Ordering and a non-null timezone are cheap to satisfy
  while still being wrong by an epoch, an offset or a unit, so the first sample
  of every layout is checked against the time it was written at.
* **which cells a rank reads,** for the mesh: that the runs cover the selection,
  that a target point's neighbours are on the same rank, and that the values and
  the coordinates emitted describe the same cells. All three are silent when
  wrong -- the arrays stay self consistent and the field is scrambled.

These need neither DALI nor a GPU, so unlike the end to end suite they run in
CI.
"""

import datetime as dt
import os
import pickle
import sys
import tempfile
import unittest

import h5py as h5
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.utils.dataloaders.backends import (
    ArcoWB2Backend,
    IconBackend,
    BackendMetadata,
    MakaniConcatBackend,
    MakaniHDF5Backend,
    MakaniZarrBackend,
    get_backend,
)
from makani.utils.dataloaders.backends.base import GridSpec
from makani.utils.dataloaders.backends.factory import detect_backend
from makani.utils.dataloaders.backends.makani_hdf5 import contiguous_slices
from makani.utils.dataloaders.backends.icon import date_of_file, survey_files, variable_of_file
from makani.utils.dataloaders.backends.mesh import block_of_rank, coalesce_runs
from makani.utils.grid_types import expected_latitudes

from .testutils import (
    CHANNEL_NAMES,
    ICON_FILL,
    ICON_FILL_CELLS,
    ICON_N_CELLS,
    ICON_PLEV_HPA,
    compare_arrays,
    icon_expected_field,
    init_icon_dataset,
    DHOURS,
    NUM_SAMPLES_PER_YEAR,
    TRAIN_YEARS,
    H5_PATH,
    IMG_SIZE_H,
    IMG_SIZE_W,
    NUM_CHANNELS,
    init_hdf5_dataset,
    init_wb2_zarr_dataset,
    init_zarr_dataset,
)


class TestContiguousSlices(unittest.TestCase):
    """
    Channel selections are read as runs of adjacent indices rather than one
    element at a time. A regression here is silent: the data stays correct and
    only the number of reads changes, so it is asserted directly.
    """

    def test_single_run(self):
        self.assertEqual(list(contiguous_slices([0, 1, 2])), [slice(0, 3)])

    def test_two_runs(self):
        self.assertEqual(list(contiguous_slices([0, 1, 2, 7, 8])), [slice(0, 3), slice(7, 9)])

    def test_isolated_indices_do_not_merge(self):
        self.assertEqual(list(contiguous_slices([0, 2, 4])), [slice(0, 1), slice(2, 3), slice(4, 5)])

    def test_single_index(self):
        self.assertEqual(list(contiguous_slices([5])), [slice(5, 6)])

    def test_empty(self):
        self.assertEqual(list(contiguous_slices([])), [])

    def test_slices_cover_exactly_the_input(self):
        indices = [1, 2, 3, 9, 10, 40]
        covered = [i for s in contiguous_slices(indices) for i in range(s.start, s.stop)]
        self.assertEqual(covered, indices)


class TestCoalesceRuns(unittest.TestCase):
    """Turning selected cells into the reads that fetch them.

    Like ``contiguous_slices``, a broken version still returns the right data:
    the cells are all there, just fetched in more reads, or in one read far
    larger than it needed to be. Only the slices themselves show it, so they are
    asserted directly.
    """

    def test_adjacent_cells_are_one_run(self):
        self.assertEqual(coalesce_runs(np.array([4, 5, 6])), [slice(4, 7)])

    def test_a_gap_splits_the_run(self):
        self.assertEqual(coalesce_runs(np.array([0, 1, 2, 7, 8])), [slice(0, 3), slice(7, 9)])

    def test_a_short_gap_is_read_through(self):
        # one read of nine cells beats two reads of three and two, once the
        # fixed cost of a read exceeds the four cells bridged
        self.assertEqual(coalesce_runs(np.array([0, 1, 2, 7, 8]), merge_gap=4), [slice(0, 9)])

    def test_the_gap_is_a_limit_not_a_hint(self):
        # the gap here is five cells, one more than allowed, so it still splits
        self.assertEqual(coalesce_runs(np.array([0, 1, 2, 8, 9]), merge_gap=4), [slice(0, 3), slice(8, 10)])

    def test_empty(self):
        self.assertEqual(coalesce_runs(np.array([], dtype=int)), [])

    def test_single_cell(self):
        self.assertEqual(coalesce_runs(np.array([5])), [slice(5, 6)])

    def test_runs_cover_every_selected_cell(self):
        """The property that makes bridging safe: nothing is dropped.

        A cell that falls outside every run is read as whatever the buffer held,
        which is silent and looks like data.
        """
        selection = np.sort(np.random.default_rng(0).choice(10_000, 500, replace=False))
        for merge_gap in (0, 8, 64, 1024):
            with self.subTest(merge_gap=merge_gap):
                covered = np.concatenate(
                    [np.arange(run.start, run.stop) for run in coalesce_runs(selection, merge_gap)]
                )
                self.assertTrue(np.all(np.isin(selection, covered)))

    def test_bridging_trades_reads_for_bytes(self):
        """Fewer reads with a larger gap, and every surviving gap is a real one."""
        selection = np.sort(np.random.default_rng(0).choice(10_000, 500, replace=False))

        tight = coalesce_runs(selection, 0)
        loose = coalesce_runs(selection, 1024)

        self.assertLess(len(loose), len(tight))

        # whatever the setting, a gap that was left unbridged has to be wider
        # than the one we were willing to read through
        for merge_gap, runs in ((0, tight), (1024, loose)):
            with self.subTest(merge_gap=merge_gap):
                gaps = [later.start - earlier.stop for earlier, later in zip(runs, runs[1:])]
                self.assertTrue(all(gap > merge_gap for gap in gaps))


class TestGridSpec(unittest.TestCase):

    def test_raster_is_structured(self):
        grid = GridSpec("equiangular", (4, 8), np.zeros(4), np.zeros(8))
        self.assertTrue(grid.is_structured)

    def test_mesh_is_not_structured(self):
        # what an unstructured backend would report: one spatial dimension, and
        # a coordinate per cell rather than per axis
        grid = GridSpec("unstructured", (32,), np.zeros(32), np.zeros(32))
        self.assertFalse(grid.is_structured)


class _BackendFixture(unittest.TestCase):
    """A small on-disk dataset shared by the tests below."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        (
            cls.train_path,
            cls.num_train,
            cls.valid_path,
            cls.num_valid,
            cls.stats_path,
            cls.metadata_path,
            cls.concat_path,
        ) = init_hdf5_dataset(cls.tmpdir.name, create_concat=True)

        cls.zarr_dir = tempfile.TemporaryDirectory()
        cls.zarr_train_path = init_zarr_dataset(cls.zarr_dir.name)[0]

        cls.wb2_dir = tempfile.TemporaryDirectory()
        cls.wb2_train_path = init_wb2_zarr_dataset(cls.wb2_dir.name)[0]

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()
        cls.zarr_dir.cleanup()
        cls.wb2_dir.cleanup()


class TestDetectBackend(_BackendFixture):
    """
    Which layout is at a path used to be decided in two places, by an
    ``os.path.isfile`` in the loader and by the order of two globs in the
    reader. It is one function now, so its decisions are worth stating.
    """

    def test_directory_of_h5_files(self):
        self.assertEqual(detect_backend(self.train_path), "makani_hdf5")

    def test_single_file_is_concatenated(self):
        self.assertEqual(detect_backend(self.concat_path), "makani_concat")

    def test_directory_of_zarr_stores(self):
        self.assertEqual(detect_backend(self.zarr_train_path), "makani_zarr")

    def test_s3_implies_hdf5(self):
        # nothing local to stat, and the per-year HDF5 layout is the only one
        # reachable over the ROS3 driver
        self.assertEqual(detect_backend("s3://bucket/prefix", enable_s3=True), "makani_hdf5")

    def test_hdf5_wins_over_zarr_in_a_mixed_directory(self):
        # a directory holding both is ambiguous; the order is a decision, so it
        # is pinned rather than left to whichever glob happens to run first
        mixed = os.path.join(self.tmpdir.name, "mixed")
        os.makedirs(mixed, exist_ok=True)
        for name in ("2017.h5", "2018.h5"):
            with h5.File(os.path.join(mixed, name), "w") as handle:
                handle.create_dataset(H5_PATH, data=np.zeros((2, NUM_CHANNELS, 4, 4), dtype=np.float32))
        os.makedirs(os.path.join(mixed, "2019.zarr"), exist_ok=True)

        self.assertEqual(detect_backend(mixed), "makani_hdf5")

    def test_wb2_store_is_told_apart_from_the_makani_one(self):
        # the two share a container and a naming convention, so this is the one
        # branch that has to open a store to decide: a WB2 store has no single
        # fields array, its variables are named individually
        self.assertEqual(detect_backend(self.wb2_train_path), "arco_wb2")

    def test_wb2_backend_is_built_for_a_wb2_store(self):
        backend = get_backend(self.wb2_train_path, channel_names=list(CHANNEL_NAMES))
        self.assertIsInstance(backend, ArcoWB2Backend)

    def test_wb2_store_without_channel_names_raises(self):
        # detection succeeds and construction then fails, which is the right
        # order: the message names what is missing rather than what was not found
        with self.assertRaises(ValueError) as ctx:
            get_backend(self.wb2_train_path)
        self.assertIn("channel_names", str(ctx.exception))

    def test_hdf5_only_arguments_are_not_passed_to_other_backends(self):
        # O_DIRECT and S3 are HDF5 driver settings; a zarr backend has no use for
        # them and would reject them as unexpected keywords
        backend = get_backend(self.zarr_train_path, enable_odirect=True, odirect_alignment=4096, enable_s3=False)
        self.assertIsInstance(backend, MakaniZarrBackend)

    def test_odirect_and_s3_together_are_rejected(self):
        with self.assertRaises(NotImplementedError):
            get_backend(self.train_path, enable_odirect=True, enable_s3=True)

    def test_empty_directory_raises(self):
        empty = os.path.join(self.tmpdir.name, "empty")
        os.makedirs(empty, exist_ok=True)
        with self.assertRaises(IOError):
            detect_backend(empty)

    def test_explicit_backend_skips_detection(self):
        backend = get_backend(self.train_path, backend="makani_hdf5")
        self.assertIsInstance(backend, MakaniHDF5Backend)

    def test_unknown_backend_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_backend(self.train_path, backend="not_a_backend")
        self.assertIn("not_a_backend", str(ctx.exception))

    def test_factory_detects_each_layout(self):
        for location, expected in [
            (self.train_path, MakaniHDF5Backend),
            (self.concat_path, MakaniConcatBackend),
            (self.zarr_train_path, MakaniZarrBackend),
        ]:
            with self.subTest(location=os.path.basename(location)):
                self.assertIsInstance(get_backend(location), expected)


class TestDiscovery(_BackendFixture):

    def test_metadata_describes_the_dataset(self):
        backend = get_backend(self.train_path)
        metadata = backend.discover()

        self.assertIsInstance(metadata, BackendMetadata)
        self.assertEqual(len(metadata.files), len(metadata.samples_per_file))
        self.assertEqual(len(metadata.files), len(metadata.labels))
        self.assertEqual(sum(metadata.samples_per_file), len(metadata.timestamps))
        self.assertEqual(metadata.grid.shape, (IMG_SIZE_H, IMG_SIZE_W))
        self.assertEqual(metadata.total_channels, NUM_CHANNELS)

    def test_timestamps_are_utc_and_ordered(self):
        metadata = get_backend(self.train_path).discover()

        self.assertIsNotNone(metadata.timestamps[0].tzinfo)
        self.assertTrue(all(a < b for a, b in zip(metadata.timestamps, metadata.timestamps[1:])))

    def test_annotated_timestamps_are_the_times_the_file_carries(self):
        """The times a file was annotated with, read back as they were written.

        Ordering and a non-null timezone are cheap to satisfy while still being
        wrong by an epoch, an offset or a unit. The fixture writes the first
        sample at midnight UTC on new year, so that is what has to come back.
        """
        metadata = get_backend(self.train_path).discover()

        self.assertEqual(metadata.timestamps[0], dt.datetime(TRAIN_YEARS[0], 1, 1, tzinfo=dt.timezone.utc))
        self.assertEqual(metadata.timestamps[1] - metadata.timestamps[0], dt.timedelta(hours=DHOURS))
        self.assertEqual(metadata.timestamps[0].utcoffset(), dt.timedelta(0))

    def test_labels_are_the_years(self):
        metadata = get_backend(self.train_path).discover()

        self.assertEqual(metadata.labels, sorted(metadata.labels))

        # timestamps run over every sample, not every file, so a label has to be
        # checked against the first sample of its own file
        offset = 0
        for label, count in zip(metadata.labels, metadata.samples_per_file):
            with self.subTest(label=label):
                self.assertEqual(label, metadata.timestamps[offset].year)
                self.assertEqual(label, metadata.timestamps[offset + count - 1].year)
            offset += count

    def test_concatenated_dataset_reports_one_file(self):
        metadata = get_backend(self.concat_path).discover()

        self.assertEqual(len(metadata.files), 1)
        self.assertEqual(metadata.samples_per_file[0], len(metadata.timestamps))


class TestChunkCoordinates(_BackendFixture):
    """
    The promise the whole contract rests on: a backend says where every value it
    emits belongs. A resampler consumes those coordinates, so an off-by-one
    between them and the data would not fail here, it would displace a field
    later.
    """

    def _backend(self, **kwargs):
        backend = get_backend(self.train_path, **kwargs)
        backend.discover()
        return backend

    def test_full_grid_chunk_matches_the_dataset(self):
        backend = self._backend()

        self.assertEqual(backend.chunk.shape, (IMG_SIZE_H, IMG_SIZE_W))
        self.assertTrue(compare_arrays("backend.chunk.lat", backend.chunk.lat, backend.metadata.grid.lat))
        self.assertTrue(compare_arrays("backend.chunk.lon", backend.chunk.lon, backend.metadata.grid.lon))

    def test_coordinate_lengths_match_the_chunk_shape(self):
        for io_grid, io_rank in [([1, 1], [0, 0]), ([2, 1], [1, 0]), ([2, 2], [1, 1]), ([3, 1], [2, 0])]:
            with self.subTest(io_grid=io_grid, io_rank=io_rank):
                backend = self._backend(io_grid=io_grid, io_rank=io_rank)

                self.assertEqual(backend.chunk.lat.shape[0], backend.chunk.shape[0])
                self.assertEqual(backend.chunk.lon.shape[0], backend.chunk.shape[1])

    def test_coordinates_are_the_slice_of_the_global_ones(self):
        # the emitted block has to carry exactly the coordinates of the region
        # it was cut from, which is what makes them usable for resampling
        backend = self._backend(io_grid=[2, 2], io_rank=[1, 1])
        grid = backend.metadata.grid

        lat_start = backend.read_anchor[0]
        lon_start = backend.read_anchor[1]
        self.assertTrue(
            compare_arrays(
                "backend.chunk.lat", backend.chunk.lat, grid.lat[lat_start : lat_start + backend.read_shape[0]]
            )
        )
        self.assertTrue(
            compare_arrays(
                "backend.chunk.lon", backend.chunk.lon, grid.lon[lon_start : lon_start + backend.read_shape[1]]
            )
        )

    def test_chunks_tile_the_grid_without_gaps_or_overlap(self):
        # every latitude of the global grid is claimed by exactly one rank
        claimed = []
        for rank in range(3):
            backend = self._backend(io_grid=[3, 1], io_rank=[rank, 0])
            claimed.append(backend.chunk.lat)

        self.assertTrue(compare_arrays("np.concatenate(claimed)", np.concatenate(claimed), backend.metadata.grid.lat))

    def test_data_at_a_coordinate_is_the_data_at_that_coordinate(self):
        """Read a decomposed chunk and check it against the same region read whole.

        This is the assertion the contract needs: the values a rank emits, and
        the coordinates it reports for them, describe the same points as the
        global grid does.
        """
        whole = self._backend()
        part = self._backend(io_grid=[2, 2], io_rank=[1, 0])

        channels = np.arange(NUM_CHANNELS)
        reference = whole.read(0, slice(0, 1), channels)
        chunk = part.read(0, slice(0, 1), channels)

        # locate the chunk's coordinates within the global axes, then compare
        lat_offset = int(np.argmin(np.abs(whole.chunk.lat - part.chunk.lat[0])))
        lon_offset = int(np.argmin(np.abs(whole.chunk.lon - part.chunk.lon[0])))
        expected = reference[
            :,
            :,
            lat_offset : lat_offset + part.chunk.shape[0],
            lon_offset : lon_offset + part.chunk.shape[1],
        ]

        self.assertTrue(compare_arrays("chunk", chunk, expected))

    def test_subsampling_takes_every_nth_coordinate(self):
        backend = self._backend(subsampling_factor=2)

        self.assertTrue(compare_arrays("backend.chunk.lat", backend.chunk.lat, backend.metadata.grid.lat[::2]))
        self.assertTrue(compare_arrays("backend.chunk.lon", backend.chunk.lon, backend.metadata.grid.lon[::2]))

    def test_crop_decomposition_and_subsampling_compose(self):
        """All three narrow the chunk, and they are applied in one place.

        Tested separately elsewhere; here they act together, which is where an
        interaction bug would live -- an anchor that ignores the subsampling
        step, say, or a split taken before the crop rather than after.
        """
        crop_anchor, crop_size = [2, 4], [32, 32]
        backend = self._backend(
            crop_anchor=crop_anchor,
            crop_size=crop_size,
            io_grid=[2, 1],
            io_rank=[1, 0],
            subsampling_factor=2,
        )
        grid = backend.metadata.grid

        # the crop is split over two ranks and this is the second, so it starts
        # half a crop in; subsampling then takes every other row of that
        expected_start = crop_anchor[0] + crop_size[0] // 2
        expected_lat = grid.lat[expected_start : crop_anchor[0] + crop_size[0] : 2]
        expected_lon = grid.lon[crop_anchor[1] : crop_anchor[1] + crop_size[1] : 2]

        self.assertTrue(compare_arrays("backend.chunk.lat", backend.chunk.lat, expected_lat))
        self.assertTrue(compare_arrays("backend.chunk.lon", backend.chunk.lon, expected_lon))
        self.assertEqual(backend.chunk.shape, (len(expected_lat), len(expected_lon)))

    def test_crop_beyond_the_grid_raises(self):
        with self.assertRaises(ValueError):
            self._backend(crop_anchor=[0, 0], crop_size=[IMG_SIZE_H + 1, IMG_SIZE_W])


class TestLifecycle(_BackendFixture):

    def test_num_files_before_discovery_raises(self):
        # asking what is in a dataset before looking is a caller error, and a
        # clear one beats an AttributeError on None
        backend = get_backend(self.train_path)
        with self.assertRaises(RuntimeError) as ctx:
            backend.num_files
        self.assertIn("discover", str(ctx.exception))

    def test_num_files_after_discovery(self):
        backend = get_backend(self.train_path)
        metadata = backend.discover()
        self.assertEqual(backend.num_files, len(metadata.files))

    def test_close_releases_handles(self):
        backend = get_backend(self.train_path)
        backend.discover()
        backend.open(0)
        self.assertIsNotNone(backend.files[0])

        backend.close()

        self.assertTrue(all(handle is None for handle in backend.files))
        self.assertTrue(all(handle is None for handle in backend.dsets))

    def test_close_is_idempotent(self):
        # __del__ calls it, so a second call during interpreter shutdown must
        # not raise
        backend = get_backend(self.train_path)
        backend.discover()
        backend.close()
        backend.close()

    def test_reopening_after_close_works(self):
        backend = get_backend(self.train_path)
        backend.discover()
        channels = np.arange(NUM_CHANNELS)

        before = backend.read(0, slice(0, 1), channels)
        backend.close()
        after = backend.read(0, slice(0, 1), channels)

        self.assertTrue(compare_arrays("before", before, after))


class TestConcatBackend(_BackendFixture):

    def test_unannotated_file_is_rejected(self):
        """A concatenated file has to carry its timestamps.

        The per-year layout can fall back on the year in the filename; one file
        holding every year cannot, so the times have to be read. Failing at
        discovery with a message naming the fix beats a KeyError from h5py deep
        in a run.
        """
        path = os.path.join(self.tmpdir.name, "unannotated.h5")
        with h5.File(path, "w") as handle:
            handle.create_dataset(H5_PATH, data=np.zeros((4, NUM_CHANNELS, 8, 16), dtype=np.float32))

        backend = get_backend(path)
        with self.assertRaises(ValueError) as ctx:
            backend.discover()
        self.assertIn("annotate", str(ctx.exception))


class TestPickling(_BackendFixture):
    """
    A backend crosses into a worker process by being pickled, so anything it
    holds that cannot cross has to be dropped and rebuilt.
    """

    def test_handles_are_dropped(self):
        backend = get_backend(self.train_path)
        backend.discover()
        backend.open(0)
        self.assertIsNotNone(backend.files[0])

        state = backend.__getstate__()

        self.assertTrue(all(handle is None for handle in state["files"]))
        self.assertTrue(all(handle is None for handle in state["dsets"]))

    def test_transient_attributes_are_dropped(self):
        # the S3 connector holds a session; unlike a file handle it is not in a
        # list, so it needs its own declaration and would otherwise be pickled
        backend = get_backend(self.train_path)
        backend.discover()
        backend.aws_connector = object()

        self.assertIsNone(backend.__getstate__()["aws_connector"])

    def test_roundtrip_reads_the_same_data(self):
        backend = get_backend(self.train_path)
        backend.discover()
        channels = np.arange(NUM_CHANNELS)

        before = backend.read(0, slice(0, 2), channels)
        restored = pickle.loads(pickle.dumps(backend))
        after = restored.read(0, slice(0, 2), channels)

        self.assertTrue(compare_arrays("before", before, after))

    def test_roundtrip_preserves_the_chunk(self):
        backend = get_backend(self.train_path, io_grid=[2, 2], io_rank=[0, 1])
        backend.discover()

        restored = pickle.loads(pickle.dumps(backend))

        self.assertEqual(restored.chunk.shape, backend.chunk.shape)
        self.assertTrue(compare_arrays("restored.chunk.lat", restored.chunk.lat, backend.chunk.lat))
        self.assertTrue(compare_arrays("restored.chunk.lon", restored.chunk.lon, backend.chunk.lon))


class TestReading(_BackendFixture):

    def test_out_is_filled_in_place(self):
        backend = get_backend(self.train_path)
        backend.discover()
        channels = np.arange(NUM_CHANNELS)

        out = np.zeros((2, NUM_CHANNELS, *backend.chunk.shape), dtype=np.float32)
        returned = backend.read(0, slice(0, 2), channels, out=out)

        self.assertIs(returned, out)
        self.assertTrue(np.any(out != 0.0))

    def test_allocating_and_filling_agree(self):
        backend = get_backend(self.train_path)
        backend.discover()
        channels = np.arange(NUM_CHANNELS)

        allocated = backend.read(0, slice(0, 2), channels)
        out = np.zeros_like(allocated)
        backend.read(0, slice(0, 2), channels, out=out)

        self.assertTrue(compare_arrays("allocated", allocated, out))

    def test_channel_subset_is_read_in_order(self):
        backend = get_backend(self.train_path)
        backend.discover()

        every = backend.read(0, slice(0, 1), np.arange(NUM_CHANNELS))
        subset = backend.read(0, slice(0, 1), np.array([0, 2]))

        self.assertTrue(compare_arrays("subset[:, 0]", subset[:, 0], every[:, 0]))
        self.assertTrue(compare_arrays("subset[:, 1]", subset[:, 1], every[:, 2]))

    def test_strided_time_slice(self):
        backend = get_backend(self.train_path)
        backend.discover()
        channels = np.arange(NUM_CHANNELS)

        strided = backend.read(0, slice(0, 4, 2), channels)
        first = backend.read(0, slice(0, 1), channels)
        third = backend.read(0, slice(2, 3), channels)

        self.assertEqual(strided.shape[0], 2)
        self.assertTrue(compare_arrays("strided[0]", strided[0], first[0]))
        self.assertTrue(compare_arrays("strided[1]", strided[1], third[0]))


class TestTimestampSynthesis(unittest.TestCase):
    """Timestamps for a dataset that does not carry any.

    An unannotated file has only the year in its name to go on, so the sample
    times are derived from that year plus the ``dhours`` cadence. This is the
    one place in the backends that does date arithmetic, it runs before anything
    else can notice it went wrong, and a mistake here misdates every sample --
    which matters, because the timestamps drive both the zenith angle and the
    boundary exclusions.

    The strongest statement available is that synthesis and annotation agree:
    the same dataset written both ways has to produce the same times.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()

        cls.h5_annotated = init_hdf5_dataset(os.path.join(cls.tmpdir.name, "h5_annotated"))[0]
        cls.h5_bare = init_hdf5_dataset(os.path.join(cls.tmpdir.name, "h5_bare"), annotate=False)[0]
        cls.zarr_annotated = init_zarr_dataset(os.path.join(cls.tmpdir.name, "zarr_annotated"))[0]
        cls.zarr_bare = init_zarr_dataset(os.path.join(cls.tmpdir.name, "zarr_bare"), annotate=False)[0]

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _discover(self, path):
        return get_backend(path, dhours=DHOURS).discover()

    def test_hdf5_synthesizes_one_timestamp_per_sample(self):
        metadata = self._discover(self.h5_bare)
        self.assertEqual(len(metadata.timestamps), sum(metadata.samples_per_file))

    def test_zarr_synthesizes_one_timestamp_per_sample(self):
        metadata = self._discover(self.zarr_bare)
        self.assertEqual(len(metadata.timestamps), sum(metadata.samples_per_file))

    def test_hdf5_synthesized_times_match_the_annotated_ones(self):
        expected = self._discover(self.h5_annotated).timestamps
        synthesized = self._discover(self.h5_bare).timestamps

        self.assertEqual(len(synthesized), len(expected))
        for idx, (got, want) in enumerate(zip(synthesized, expected)):
            with self.subTest(sample=idx):
                self.assertEqual(got, want)

    def test_zarr_synthesized_times_match_the_annotated_ones(self):
        expected = self._discover(self.zarr_annotated).timestamps
        synthesized = self._discover(self.zarr_bare).timestamps

        self.assertEqual(len(synthesized), len(expected))
        for idx, (got, want) in enumerate(zip(synthesized, expected)):
            with self.subTest(sample=idx):
                self.assertEqual(got, want)

    def test_synthesis_starts_at_new_year_utc(self):
        # the year in the filename means midnight on the first of January, in
        # UTC and not in whatever zone the machine happens to sit in
        metadata = self._discover(self.h5_bare)
        first = metadata.timestamps[0]

        self.assertEqual((first.year, first.month, first.day), (TRAIN_YEARS[0], 1, 1))
        self.assertEqual((first.hour, first.minute, first.second), (0, 0, 0))
        self.assertEqual(first.utcoffset(), dt.timedelta(0))

    def test_synthesis_advances_by_dhours(self):
        metadata = self._discover(self.h5_bare)
        step = dt.timedelta(hours=DHOURS)

        for idx in range(1, 5):
            with self.subTest(sample=idx):
                self.assertEqual(metadata.timestamps[idx] - metadata.timestamps[idx - 1], step)

    def test_synthesis_restarts_each_file(self):
        # every file is dated from its own year, so the cadence does not carry
        # over a file boundary -- the next file starts at its own new year
        metadata = self._discover(self.h5_bare)
        boundary = metadata.timestamps[NUM_SAMPLES_PER_YEAR]

        self.assertEqual((boundary.year, boundary.month, boundary.day), (TRAIN_YEARS[1], 1, 1))
        self.assertEqual(boundary.hour, 0)


class TestUnconsolidatedZarr(unittest.TestCase):
    """A store whose metadata was never consolidated.

    Consolidation is an optimisation, not a requirement, and a store written
    without it has no ``zarr.json`` at the root to open. The backend falls back
    to opening the group directly; without that, a perfectly valid store is
    unreadable.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.consolidated = init_zarr_dataset(os.path.join(cls.tmpdir.name, "consolidated"))[0]
        cls.plain = init_zarr_dataset(os.path.join(cls.tmpdir.name, "plain"), consolidate=False)[0]

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_discovery_agrees_with_the_consolidated_store(self):
        expected = get_backend(self.consolidated).discover()
        metadata = get_backend(self.plain).discover()

        self.assertEqual(metadata.samples_per_file, expected.samples_per_file)
        self.assertEqual(metadata.labels, expected.labels)
        self.assertEqual(metadata.total_channels, expected.total_channels)
        self.assertEqual(metadata.grid.shape, expected.grid.shape)

    def test_reads_agree_with_the_consolidated_store(self):
        channels = np.arange(NUM_CHANNELS)

        reference = get_backend(self.consolidated)
        reference.discover()
        backend = get_backend(self.plain)
        backend.discover()

        self.assertTrue(
            compare_arrays(
                "backend.read(0, slice(0, 3), channels)",
                backend.read(0, slice(0, 3), channels),
                reference.read(0, slice(0, 3), channels),
            )
        )


def _write_h5(path, data, timestamps, latitude=None, longitude=None):
    """A minimal makani-layout file, annotated as much as asked for."""
    with h5.File(path, "w") as handle:
        dset = handle.create_dataset(H5_PATH, data=data)
        if timestamps is not None:
            scale = handle.create_dataset("timestamp", data=np.asarray(timestamps, dtype=np.float64))
            scale.make_scale("timestamp")
            dset.dims[0].attach_scale(scale)
        if latitude is not None:
            handle.create_dataset("lat", data=np.asarray(latitude, dtype=np.float32))
            handle["lat"].make_scale("lat")
            dset.dims[2].attach_scale(handle["lat"])
            handle.create_dataset("lon", data=np.asarray(longitude, dtype=np.float32))
            handle["lon"].make_scale("lon")
            dset.dims[3].attach_scale(handle["lon"])
    return path


class TestTimeOrderedDiscovery(unittest.TestCase):
    """Files are ordered by the data, not by the name.

    The makani layouts name a file after its year, so sorting by name orders
    them by time as well. That is a property of those layouts, not of datasets
    in general -- an inference dataset can be a directory of arbitrarily named
    files -- so the order comes from the first sample of each file. Getting it
    wrong silently misdates every sample after the first file.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_part(self, name, start_day, n_samples=4):
        base = dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=start_day)
        timestamps = [(base + dt.timedelta(days=idx)).timestamp() for idx in range(n_samples)]
        data = np.full((n_samples, NUM_CHANNELS, 4, 8), float(start_day), dtype=np.float32)
        return _write_h5(os.path.join(self.tmpdir.name, name), data, timestamps)

    def test_files_are_ordered_by_time_not_by_name(self):
        # names deliberately sort the other way round from the timestamps
        self._write_part("charlie.h5", start_day=0)
        self._write_part("bravo.h5", start_day=4)
        self._write_part("alpha.h5", start_day=8)

        metadata = get_backend(self.tmpdir.name, file_pattern="*").discover()

        self.assertEqual([os.path.basename(path) for path in metadata.files], ["charlie.h5", "bravo.h5", "alpha.h5"])
        self.assertTrue(
            all(metadata.timestamps[i] < metadata.timestamps[i + 1] for i in range(len(metadata.timestamps) - 1))
        )

    def test_samples_per_file_follows_the_same_order(self):
        # the sample index walks files in this order, so the counts have to be
        # permuted alongside them or every index past the first file is wrong
        self._write_part("charlie.h5", start_day=0, n_samples=2)
        self._write_part("alpha.h5", start_day=8, n_samples=5)

        metadata = get_backend(self.tmpdir.name, file_pattern="*").discover()

        self.assertEqual(metadata.samples_per_file, [2, 5])

    def test_overlapping_files_are_refused(self):
        # one file's range inside another's makes a global sample index
        # ambiguous, so it is refused rather than silently resolved
        self._write_part("first.h5", start_day=0, n_samples=10)
        self._write_part("inside.h5", start_day=2, n_samples=2)

        with self.assertRaises(RuntimeError) as ctx:
            get_backend(self.tmpdir.name, file_pattern="*").discover()
        self.assertIn("overlapping", str(ctx.exception))

    def test_a_file_that_is_not_a_year_needs_timestamps(self):
        # synthesis derives the times from the year in the name; without one
        # there is nothing to derive them from
        data = np.zeros((4, NUM_CHANNELS, 4, 8), dtype=np.float32)
        _write_h5(os.path.join(self.tmpdir.name, "part_one.h5"), data, timestamps=None)

        with self.assertRaises(ValueError) as ctx:
            get_backend(self.tmpdir.name, file_pattern="*").discover()
        self.assertIn("not a year", str(ctx.exception))

    def test_the_year_layout_is_unaffected(self):
        # name order and time order agree here, which is why the training path
        # sees no change from ordering by time
        for year in (2018, 2017):
            base = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
            timestamps = [(base + dt.timedelta(days=idx)).timestamp() for idx in range(3)]
            data = np.zeros((3, NUM_CHANNELS, 4, 8), dtype=np.float32)
            _write_h5(os.path.join(self.tmpdir.name, f"{year}.h5"), data, timestamps)

        metadata = get_backend(self.tmpdir.name).discover()

        self.assertEqual([os.path.basename(path) for path in metadata.files], ["2017.h5", "2018.h5"])
        self.assertEqual(metadata.labels, [2017, 2018])


class TestFileCoordinates(unittest.TestCase):
    """Where the grid comes from.

    A dataset that is not equiangular still has to describe itself correctly:
    the coordinates go to the resampler and to the solar zenith angle, and both
    give quietly wrong answers on a grid they were not told about. So the file
    is asked first, and the equiangular assumption is only the fallback for a
    file that carries no coordinates at all.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.shape = (8, 16)
        # a real grid, and not the one the fallback would have invented: Gauss
        # nodes sit nowhere near the equiangular ones, so a fabricated grid
        # cannot pass for this file's -- and unlike arbitrary numbers, these are
        # a grid the dataset can honestly declare itself to be on
        self.grid_type = "legendre-gauss"
        self.latitude = np.flip(expected_latitudes(self.grid_type, self.shape[0]))
        self.longitude = np.linspace(0.0, 350.0, self.shape[1])

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write(self, name="2017.h5", with_coordinates=True):
        base = dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc)
        timestamps = [(base + dt.timedelta(days=idx)).timestamp() for idx in range(3)]
        data = np.zeros((3, NUM_CHANNELS, *self.shape), dtype=np.float32)
        return _write_h5(
            os.path.join(self.tmpdir.name, name),
            data,
            timestamps,
            latitude=self.latitude if with_coordinates else None,
            longitude=self.longitude if with_coordinates else None,
        )

    def test_coordinates_come_from_the_file(self):
        self._write()
        metadata = get_backend(self.tmpdir.name, grid_type=self.grid_type).discover()

        self.assertTrue(compare_arrays("metadata.grid.lat", metadata.grid.lat, self.latitude, rtol=1e-6))
        self.assertTrue(compare_arrays("metadata.grid.lon", metadata.grid.lon, self.longitude, rtol=1e-6))

    def test_a_file_without_coordinates_falls_back(self):
        self._write(with_coordinates=False)
        metadata = get_backend(self.tmpdir.name).discover()

        self.assertTrue(
            compare_arrays("metadata.grid.lat", metadata.grid.lat, np.linspace(90, -90, self.shape[0], endpoint=True))
        )
        self.assertTrue(
            compare_arrays("metadata.grid.lon", metadata.grid.lon, np.linspace(0, 360, self.shape[1], endpoint=False))
        )

    def test_an_explicit_grid_wins_over_the_file(self):
        # the caller knows something the file does not, e.g. a dataset written
        # with the wrong coordinates
        self._write()
        override = (np.linspace(1, 8, self.shape[0]), np.linspace(1, 16, self.shape[1]))
        metadata = get_backend(self.tmpdir.name, lat_lon=override).discover()

        self.assertTrue(compare_arrays("metadata.grid.lat", metadata.grid.lat, override[0]))

    def test_the_chunk_carries_the_file_coordinates(self):
        # the chunk is what a consumer actually reads coordinates from, so the
        # decomposition has to slice the file's grid, not a fabricated one
        self._write()
        backend = get_backend(self.tmpdir.name, grid_type=self.grid_type, io_grid=[2, 1], io_rank=[1, 0])
        backend.discover()

        self.assertTrue(
            compare_arrays("backend.chunk.lat", backend.chunk.lat, self.latitude[self.shape[0] // 2 :], rtol=1e-6)
        )


class _IconFixture(unittest.TestCase):
    """A scaled down ICON dataset, shared by the tests below."""

    def assertFieldEqual(self, got, want, msg="field"):
        """Compare fields that carry NaN where the fixture wrote a fill.

        Where the NaNs sit is asserted separately, and deliberately: NaN never
        compares equal, so folding it into the value check would either hide a
        misplaced fill or fail for the wrong reason.
        """
        got, want = np.asarray(got), np.asarray(want)
        self.assertTrue(np.array_equal(np.isnan(got), np.isnan(want)), f"{msg}: fills are in different places")

        finite = ~np.isnan(want)
        self.assertTrue(compare_arrays(msg, got[finite], want[finite], atol=1e-5, rtol=1e-5, verbose=False))

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.data_path, cls.grid_path, cls.n_samples, cls.channel_names = init_icon_dataset(cls.tmpdir.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _backend(self, **overrides):
        kwargs = dict(grid_file=self.grid_path, channel_names=list(self.channel_names))
        kwargs.update(overrides)
        return get_backend(self.data_path, **kwargs)


class TestIconDiscovery(_IconFixture):
    """What the backend makes of a directory of ICON output.

    The layout differs from every other one here in three ways at once -- a
    sample spans files, the variables disagree about what times exist, and a
    variable name does not identify a variable -- so what discovery settles is
    worth stating rather than inferring from a read.
    """

    def test_the_mesh_is_reported_as_unstructured(self):
        metadata = self._backend().discover()

        self.assertEqual(metadata.grid.kind, "unstructured")
        self.assertEqual(metadata.grid.shape, (ICON_N_CELLS,))
        self.assertFalse(metadata.grid.is_structured)

    def test_coordinates_are_degrees_with_longitude_in_zero_to_360(self):
        # the files store radians with longitude in [-pi, pi]; makani works in
        # degrees with longitude in [0, 360)
        grid = self._backend().discover().grid

        self.assertGreaterEqual(grid.lon.min(), 0.0)
        self.assertLess(grid.lon.max(), 360.0)
        self.assertGreaterEqual(grid.lat.min(), -90.0)
        self.assertLessEqual(grid.lat.max(), 90.0)
        self.assertEqual(len(grid.lat), ICON_N_CELLS)

    def test_samples_are_the_times_every_variable_has(self):
        """The sample axis is an intersection, not a union.

        Pressure level variables are three hourly and the surface one is
        hourly, so an hourly sample would have no temperature to go with it.
        Taking the union instead would produce samples that cannot be read.
        """
        metadata = self._backend().discover()

        self.assertEqual(len(metadata.timestamps), self.n_samples)
        steps = {
            (metadata.timestamps[idx + 1] - metadata.timestamps[idx]).total_seconds() / 3600
            for idx in range(len(metadata.timestamps) - 1)
        }
        self.assertEqual(steps, {3.0})

    def test_units_break_where_a_file_changes(self):
        # the level variables are daily and the surface variable spans both
        # days, so the units are the days: within one, no variable changes file
        metadata = self._backend().discover()

        self.assertEqual(metadata.samples_per_file, [8, 8])
        self.assertEqual(sum(metadata.samples_per_file), len(metadata.timestamps))

    def test_timestamps_are_absolute_utc_times(self):
        """The sample times, not merely their spacing.

        The files say ``minutes since 2020-1-1 00:00:00`` and start on the
        first of June 2021. A misparsed epoch, a wrong unit or a naive local
        time would all still produce an evenly spaced, ordered axis.
        """
        metadata = self._backend().discover()

        self.assertEqual(metadata.timestamps[0], dt.datetime(2021, 6, 1, tzinfo=dt.timezone.utc))
        self.assertEqual(metadata.timestamps[1], dt.datetime(2021, 6, 1, 3, tzinfo=dt.timezone.utc))
        self.assertEqual(metadata.timestamps[0].utcoffset(), dt.timedelta(0))

    def test_coordinates_keep_the_order_the_grid_file_has(self):
        """Cell n of the grid describes cell n of the data.

        The only thing tying a value to a place on the sphere is that both are
        indexed by the same cell number. Sorting or otherwise reordering the
        coordinates on the way in would leave every array self consistent, every
        test of shapes and ranges passing, and every field scrambled -- so the
        coordinates are checked against the grid file itself rather than against
        anything else the backend produced.
        """
        metadata = self._backend().discover()

        with h5.File(self.grid_path, "r") as handle:
            clon, clat = handle["clon"][...], handle["clat"][...]

        self.assertTrue(compare_arrays("latitudes", metadata.grid.lat, np.degrees(clat), atol=1e-6))
        self.assertTrue(compare_arrays("longitudes", metadata.grid.lon, np.mod(np.degrees(clon), 360.0), atol=1e-6))

    def test_total_channels_is_what_was_asked_for(self):
        # ICON has no channel axis to count, so the number is the request
        metadata = self._backend().discover()
        self.assertEqual(metadata.total_channels, len(self.channel_names))

    def test_detected_from_the_extension(self):
        self.assertEqual(detect_backend(self.data_path), "icon")
        self.assertIsInstance(self._backend(), IconBackend)


class TestVariableOfFile(unittest.TestCase):
    """Reading a variable name off an ICON filename.

    ICON names a file after the variable and then decorates it, which is enough
    to tell -- before opening anything -- whether a run has any use for it. What
    makes it delicate is that the names nest.
    """

    CANDIDATES = ["u", "v", "temp", "t_2m", "u_10m", "v_10m", "qv", "qg"]

    def test_the_leading_name_is_the_variable(self):
        self.assertEqual(variable_of_file("temp_EXCLAIM_atm_1_3_20210601T000000Z.nc", self.CANDIDATES), "temp")

    def test_the_longest_match_wins(self):
        """``u_10m`` is not ``u``.

        Ten metre wind and upper air wind are both plausible fields of the same
        name, so a prefix match that stops at the first candidate reads one as
        the other and nothing downstream can tell.
        """
        self.assertEqual(variable_of_file("u_10m_dyamond_atm_7_202106.nc", self.CANDIDATES), "u_10m")
        self.assertEqual(variable_of_file("u_EXCLAIM_atm_1_4_20210601T000000Z.nc", self.CANDIDATES), "u")

    def test_a_name_with_underscores_is_matched_whole(self):
        # splitting on the separator would give "t", which is a different table entry
        self.assertEqual(variable_of_file("t_2m_dyamond_atm_3_202106.nc", self.CANDIDATES), "t_2m")

    def test_a_prefix_has_to_end_at_a_separator(self):
        # "temperature" is not "temp": matching mid-word would claim files that
        # belong to a variable nobody asked about
        self.assertIsNone(variable_of_file("temperature_of_something.nc", ["temp"]))

    def test_an_unknown_name_says_nothing(self):
        self.assertIsNone(variable_of_file("icon_grid_0056_R02B10_G.nc", self.CANDIDATES))

    def test_the_survey_groups_by_variable(self):
        survey = survey_files(["u_a_20210601.nc", "u_a_20210602.nc", "u_10m_b.nc", "mystery.nc"], self.CANDIDATES)

        self.assertEqual(survey["u"], ["u_a_20210601.nc", "u_a_20210602.nc"])
        self.assertEqual(survey["u_10m"], ["u_10m_b.nc"])
        self.assertEqual(survey[None], ["mystery.nc"])


class TestDateOfFile(unittest.TestCase):
    """Reading the period a file covers off its name."""

    def test_a_daily_file_carries_a_full_timestamp(self):
        self.assertEqual(
            date_of_file("temp_EXCLAIM_coupled_dyamond_atm_1_3_20210601T000000Z.nc"),
            dt.datetime(2021, 6, 1, tzinfo=dt.timezone.utc),
        )

    def test_an_aggregated_file_carries_a_year_and_month(self):
        self.assertEqual(date_of_file("t_2m_dyamond_atm_3_202106.nc"), dt.datetime(2021, 6, 1, tzinfo=dt.timezone.utc))

    def test_the_date_is_utc(self):
        # the times inside the files are UTC, so a naive date here would shift
        # every derived axis by the reader's own timezone
        self.assertEqual(date_of_file("u_atm_20210602T000000Z.nc").utcoffset(), dt.timedelta(0))

    def test_a_name_without_a_date_says_nothing(self):
        self.assertIsNone(date_of_file("mystery.nc"))
        self.assertIsNone(date_of_file("icon_grid_0056_R02B10_G.nc"))


class TestIconDerivedTimes(unittest.TestCase):
    """Describing a variable's files without opening all of them.

    The level coordinate, the cadence and the length belong to the variable
    rather than to the file, so one member describes them all; only the sample
    times differ, and the name says those. What makes it safe rather than a
    guess is that the ends are always read, an odd sized file is always read,
    and what the names predict for the last file is checked against what that
    file actually says.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        # four days, so that two files are neither the first nor the last
        cls.data_path, cls.grid_path, cls.n_samples, cls.channel_names = init_icon_dataset(cls.tmpdir.name, n_days=4)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _backend(self, **overrides):
        kwargs = dict(grid_file=self.grid_path, channel_names=list(self.channel_names))
        kwargs.update(overrides)
        return get_backend(self.data_path, **kwargs)

    def _scramble(self, name, keep_size=True):
        """Make a file unreadable, optionally without changing its size."""
        path = os.path.join(self.data_path, name)
        with open(path, "rb") as handle:
            original = handle.read()

        def restore():
            with open(path, "wb") as handle:
                handle.write(original)

        self.addCleanup(restore)

        garbage = b"x" * (len(original) if keep_size else 128)
        with open(path, "wb") as handle:
            handle.write(garbage)
        return path

    def test_the_times_are_the_ones_the_files_hold(self):
        # the shortcut has to produce exactly what reading every file would
        metadata = self._backend().discover()

        self.assertEqual(len(metadata.timestamps), self.n_samples)
        self.assertEqual(metadata.timestamps[0], dt.datetime(2021, 6, 1, tzinfo=dt.timezone.utc))
        self.assertEqual(metadata.timestamps[-1], dt.datetime(2021, 6, 4, 21, tzinfo=dt.timezone.utc))
        steps = {
            (metadata.timestamps[i + 1] - metadata.timestamps[i]).total_seconds() / 3600
            for i in range(len(metadata.timestamps) - 1)
        }
        self.assertEqual(steps, {3.0})

    def test_a_middle_file_is_not_opened(self):
        """The assertion is that unreadable content goes unnoticed.

        Day three is neither end and is the usual size, so its times come from
        its name. Scrambling its contents while keeping its size is invisible to
        a reader that never opens it -- and fatal to one that does.
        """
        self._scramble("temp_dyamond_atm_1_3_20210603T000000Z.nc")

        metadata = self._backend().discover()

        self.assertEqual(len(metadata.timestamps), self.n_samples)

    def test_an_odd_sized_file_is_opened(self):
        """Size is what says a file is the same shape as its neighbours.

        A short file is exactly the one whose times cannot be derived, so it has
        to be read -- and here reading it fails, which is the evidence.
        """
        self._scramble("temp_dyamond_atm_1_3_20210603T000000Z.nc", keep_size=False)

        with self.assertRaises(Exception):
            self._backend().discover()

    def test_the_ends_are_always_read(self):
        # truncation happens at the ends of a run, so they are never derived
        self._scramble("temp_dyamond_atm_1_3_20210604T000000Z.nc")

        with self.assertRaises(Exception):
            self._backend().discover()

    def test_reading_agrees_with_the_derived_axis(self):
        # a sample whose times were derived still reads the right data
        backend = self._backend()
        backend.discover()

        # sample 16 is the first of the third day, whose file was described by name
        values = backend.read(2, slice(0, 1), np.array([2]))[0, 0]
        expected = icon_expected_field("temp", 0, ICON_PLEV_HPA.index(500))

        finite = ~np.isnan(expected)
        self.assertTrue(compare_arrays("t500 on a derived day", values[finite], expected[finite], atol=1e-5))


class TestIconFileSurvey(_IconFixture):
    """Which files discovery opens, and which it leaves alone.

    A run reads a dozen variables from a directory that may hold far more, and
    on a parallel filesystem every open is a round trip before the file says
    anything. Names are used to skip work; what a file actually holds is still
    read from inside it.
    """

    def _corrupt(self, name):
        """A file that cannot be opened, so that opening it is detectable."""
        path = os.path.join(self.data_path, name)
        with open(path, "wb") as handle:
            handle.write(b"not an hdf5 file at all")
        self.addCleanup(os.remove, path)
        return path

    def test_a_file_for_an_unwanted_variable_is_never_opened(self):
        """The assertion is the corruption: opening it would raise.

        ``qg`` is a real ICON variable and no channel here asks for it, so
        discovery should recognise the name and pass over the file without
        looking inside.
        """
        self._corrupt("qg_EXCLAIM_coupled_dyamond_atm_1_3_20210601T000000Z.nc")

        metadata = self._backend().discover()

        self.assertEqual(len(metadata.timestamps), self.n_samples)

    def test_a_file_with_an_unrecognised_name_is_still_opened(self):
        """Skipping is a guess; reading is not.

        A dataset whose naming does not follow the convention has to keep
        working, so a name that matches nothing known is opened rather than
        assumed to be irrelevant.
        """
        self._corrupt("mystery_file.nc")

        with self.assertRaises(Exception):
            self._backend().discover()

    def test_the_survey_records_what_it_found(self):
        backend = self._backend()
        backend.discover()

        self.assertIn("temp", backend.survey)
        self.assertIn("u", backend.survey)
        self.assertIn("t_2m", backend.survey)

    def test_reading_is_unaffected_by_the_skipping(self):
        # the same values as without the extra file present
        self._corrupt("qs_EXCLAIM_coupled_dyamond_atm_1_3_20210601T000000Z.nc")
        backend = self._backend()
        backend.discover()

        values = backend.read(0, slice(0, 1), np.array([0]))[0, 0]
        expected = icon_expected_field("u", 0, ICON_PLEV_HPA.index(500))

        finite = ~np.isnan(expected)
        self.assertTrue(compare_arrays("u500 with a skipped file present", values[finite], expected[finite], atol=1e-5))


class TestIconReading(_IconFixture):

    def _read(self, backend, sample, channel):
        """One channel at one sample, through the unit it falls in."""
        offset = 0
        for unit_idx, count in enumerate(backend.metadata.samples_per_file):
            if sample < offset + count:
                return backend.read(unit_idx, slice(sample - offset, sample - offset + 1), np.array([channel]))
            offset += count
        raise AssertionError("sample out of range")

    def setUp(self):
        self.backend = self._backend()
        self.backend.discover()

    def test_a_pressure_level_channel_reads_the_pressure_level_variable(self):
        """Two variables are called ``u``; only one can serve ``u500``.

        The files carry ``u`` on pressure levels and ``u`` on altitude levels.
        Resolving by name alone picks whichever was seen first, which is wrong
        half the time and silently so -- both are plausible wind fields.
        """
        values = self._read(self.backend, sample=0, channel=0)  # u500

        expected = icon_expected_field("u", 0, ICON_PLEV_HPA.index(500))
        self.assertFieldEqual(values[0, 0], expected, "u500")

    def test_levels_are_addressed_by_pressure_not_by_position(self):
        u500 = self._read(self.backend, sample=0, channel=0)
        u850 = self._read(self.backend, sample=0, channel=1)

        for name, values, hpa in (("u500", u500, 500), ("u850", u850, 850)):
            with self.subTest(channel=name):
                self.assertFieldEqual(values[0, 0], icon_expected_field("u", 0, ICON_PLEV_HPA.index(hpa)), name)

    def test_a_surface_channel_follows_its_own_cadence(self):
        """The hourly variable has to be indexed by time, not by sample number.

        Sample 1 is three hours after sample 0, which is index 1 in a three
        hourly file and index 3 in an hourly one. Reusing the sample index for
        both is the obvious mistake and it misdates the surface field.
        """
        values = self._read(self.backend, sample=1, channel=4)  # t2m

        self.assertFieldEqual(values[0, 0], icon_expected_field("t_2m", 3, 0), "t2m at sample 1")

    def test_reading_crosses_a_unit_boundary_correctly(self):
        # sample 8 is the first of the second day, so a different file for the
        # level variables and a later index in the same file for the surface one
        values = self._read(self.backend, sample=8, channel=2)  # t500

        expected = icon_expected_field("temp", 0, ICON_PLEV_HPA.index(500))
        self.assertFieldEqual(values[0, 0], expected, "t500 on the second day")

    def test_fill_values_become_nan(self):
        # cell 7 is written as _FillValue throughout
        values = self._read(self.backend, sample=0, channel=0)

        self.assertTrue(np.isnan(values[0, 0, list(ICON_FILL_CELLS)]).all())
        self.assertFalse(np.isnan(np.delete(values[0, 0], list(ICON_FILL_CELLS))).any())

    def test_the_raw_fill_sentinel_does_not_survive(self):
        values = self._read(self.backend, sample=0, channel=0)
        self.assertFalse(np.any(values == ICON_FILL))

    def test_several_channels_at_once_match_one_at_a_time(self):
        together = self.backend.read(0, slice(0, 2), np.array([0, 2, 4]))

        for position, channel in enumerate([0, 2, 4]):
            with self.subTest(channel=self.channel_names[channel]):
                alone = self.backend.read(0, slice(0, 2), np.array([channel]))
                self.assertFieldEqual(together[:, position], alone[:, 0], "batched vs single channel")

    def test_reads_are_repeatable(self):
        # the run buffer is reused between reads, so a second read has to
        # overwrite it rather than return what the first one left behind
        first = self.backend.read(0, slice(0, 1), np.array([0])).copy()
        self.backend.read(0, slice(1, 2), np.array([2]))
        again = self.backend.read(0, slice(0, 1), np.array([0]))

        self.assertFieldEqual(again, first, "repeated read")


class TestIconDecomposition(_IconFixture):
    """Which cells a rank reads, and the guarantee that makes it safe."""

    def _target_grid(self, nlat=32, nlon=64):
        return GridSpec(
            "equiangular",
            (nlat, nlon),
            np.linspace(90, -90, nlat),
            np.linspace(0, 360, nlon, endpoint=False),
        )

    def test_one_rank_gets_the_whole_mesh(self):
        backend = self._backend()
        backend.discover()
        self.assertEqual(backend.chunk.shape, (ICON_N_CELLS,))

    def test_ranks_cover_the_mesh_between_them(self):
        """Every cell reaches some rank, and the overlap is only the halo.

        A cell that no rank reads is a hole in the field that nothing else
        would notice, since each rank's own chunk looks self consistent.
        """
        target = self._target_grid()
        seen = []
        for rank in range(2):
            backend = self._backend(target_grid=target, io_grid=[2, 1], io_rank=[rank, 0])
            backend.discover()
            seen.append(set(backend.cell_index.tolist()))

        self.assertEqual(set().union(*seen), set(range(ICON_N_CELLS)))
        self.assertTrue(seen[0] & seen[1], "the halo should make the blocks overlap")

    def test_a_rank_reads_its_own_latitudes(self):
        target = self._target_grid()
        backend = self._backend(target_grid=target, io_grid=[2, 1], io_rank=[0, 0])
        backend.discover()

        # the northern half, give or take the halo
        self.assertGreater(backend.chunk.lat.min(), -backend.halo_degrees - 1e-6)

    def test_the_chunk_coordinates_are_the_cells_selected(self):
        target = self._target_grid()
        backend = self._backend(target_grid=target, io_grid=[2, 1], io_rank=[1, 0])
        backend.discover()

        self.assertTrue(compare_arrays("chunk lat", backend.chunk.lat, backend.metadata.grid.lat[backend.cell_index]))
        self.assertTrue(compare_arrays("chunk lon", backend.chunk.lon, backend.metadata.grid.lon[backend.cell_index]))

    def test_a_decomposed_read_matches_the_whole_mesh(self):
        """A rank's values are the global field restricted to its cells.

        This is what ties the selection to the read: the runs are read whole and
        the bridged cells dropped, so an off by one in the compaction shows up
        here as a shifted field and nowhere else.
        """
        target = self._target_grid()
        whole = self._backend()
        whole.discover()
        reference = whole.read(0, slice(0, 1), np.array([0]))

        backend = self._backend(target_grid=target, io_grid=[2, 1], io_rank=[1, 0])
        backend.discover()
        local = backend.read(0, slice(0, 1), np.array([0]))

        self.assertFieldEqual(local[0, 0], reference[0, 0][backend.cell_index], "decomposed read")

    def test_values_and_coordinates_describe_the_same_cells(self):
        """What a rank emits, and where it says those values are, agree.

        Checked against the fixture's formula rather than against another read,
        so it does not rely on the selection being right on both sides: the
        value at position k identifies the cell it came from, and the coordinate
        at position k has to be that cell's.
        """
        target = self._target_grid()
        backend = self._backend(target_grid=target, io_grid=[2, 1], io_rank=[1, 0])
        backend.discover()

        values = backend.read(0, slice(0, 1), np.array([0]))[0, 0]
        expected = icon_expected_field("u", 0, ICON_PLEV_HPA.index(500))[backend.cell_index]

        finite = ~np.isnan(expected)
        self.assertTrue(compare_arrays("decomposed u500", values[finite], expected[finite], atol=1e-5, rtol=1e-5))

        grid = backend.metadata.grid
        self.assertTrue(compare_arrays("chunk latitudes", backend.chunk.lat, grid.lat[backend.cell_index]))
        self.assertTrue(compare_arrays("chunk longitudes", backend.chunk.lon, grid.lon[backend.cell_index]))

    def test_every_target_point_has_its_neighbours_locally(self):
        """The guarantee the halo exists for.

        Resampling onto a target point reads the cells around it, so those cells
        have to be on the same rank -- otherwise the regrid needs a neighbour
        exchange, which is exactly what selecting a margin avoids. Asserted as
        the property rather than as a cell count, so it stays meaningful if the
        default margin changes.
        """
        target = self._target_grid()
        backend = self._backend(target_grid=target, io_grid=[2, 1], io_rank=[0, 0])
        backend.discover()

        grid = backend.metadata.grid
        lat_low, lat_high = block_of_rank(target.lat, [2, 1], [0, 0], 0)
        lat_low, lat_high = min(lat_low, lat_high), max(lat_low, lat_high)
        rows = target.lat[(target.lat >= lat_low) & (target.lat <= lat_high)]

        local = set(backend.cell_index.tolist())
        points = [(lat, lon) for lat in rows[::4] for lon in target.lon[::8]]

        for lat, lon in points:
            # great circle distance from this target point to every cell
            cosine = np.sin(np.radians(lat)) * np.sin(np.radians(grid.lat)) + np.cos(np.radians(lat)) * np.cos(
                np.radians(grid.lat)
            ) * np.cos(np.radians(lon - grid.lon))
            nearest = np.argsort(-cosine)[:3]
            with self.subTest(lat=round(float(lat), 1), lon=round(float(lon), 1)):
                self.assertTrue(set(nearest.tolist()) <= local, "a neighbour of this point is on another rank")

    def test_the_halo_widens_the_selection(self):
        # without it a rank holds only its own block, and the points at the edge
        # have nothing beyond them to interpolate from
        target = self._target_grid()

        bare = self._backend(target_grid=target, io_grid=[2, 1], io_rank=[0, 0], halo_degrees=0.0)
        bare.discover()
        padded = self._backend(target_grid=target, io_grid=[2, 1], io_rank=[0, 0])
        padded.discover()

        self.assertLess(len(bare.cell_index), len(padded.cell_index))
        self.assertTrue(set(bare.cell_index.tolist()) <= set(padded.cell_index.tolist()))

    def test_a_fragmented_selection_still_reads_correctly(self):
        """Several runs rather than one, which the default gap hides.

        With the default ``merge_gap`` a selection this small collapses into a
        single run, so the compaction is never asked to interleave pieces. A gap
        of zero forces one run per stretch, which is what a rank of a real mesh
        would see.
        """
        target = self._target_grid()
        backend = self._backend(target_grid=target, io_grid=[2, 1], io_rank=[1, 0])
        backend.merge_gap = 0
        backend.discover()

        self.assertGreater(len(backend.cell_runs), 1, "the selection should not be one contiguous stretch")

        values = backend.read(0, slice(0, 1), np.array([0]))[0, 0]
        expected = icon_expected_field("u", 0, ICON_PLEV_HPA.index(500))[backend.cell_index]

        finite = ~np.isnan(expected)
        self.assertTrue(compare_arrays("fragmented read", values[finite], expected[finite], atol=1e-5, rtol=1e-5))

    def test_decomposition_without_a_target_grid_is_refused(self):
        # a mesh has no rows and columns to split, so there is nothing to
        # decompose against unless the run says what grid it is heading for
        with self.assertRaises(ValueError) as ctx:
            self._backend(io_grid=[2, 1], io_rank=[0, 0]).discover()
        self.assertIn("target_grid", str(ctx.exception))


class TestIconRejections(_IconFixture):
    """Options and datasets the backend refuses, and how it says so."""

    def test_subsampling_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._backend(subsampling_factor=2).discover()
        self.assertIn("target_grid", str(ctx.exception))

    def test_cropping_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._backend(crop_size=[10, 10]).discover()
        self.assertIn("target_grid", str(ctx.exception))

    def test_channel_names_are_required(self):
        with self.assertRaises(ValueError) as ctx:
            get_backend(self.data_path, grid_file=self.grid_path)
        self.assertIn("channel_names", str(ctx.exception))

    def test_the_grid_file_is_required(self):
        # the output references a grid rather than carrying one, so there is
        # nothing to fall back on
        with self.assertRaises(ValueError) as ctx:
            get_backend(self.data_path, channel_names=list(self.channel_names))
        self.assertIn("grid_file", str(ctx.exception))

    def test_a_grid_from_another_run_is_refused(self):
        """A mismatched grid produces a plausible but scrambled field.

        Nothing downstream can detect it: the values are real and the
        coordinates are real, they simply do not belong together. So the UUIDs
        ICON stamps on both sides are compared.
        """
        wrong = os.path.join(self.tmpdir.name, "wrong_grid.nc")
        with h5.File(self.grid_path, "r") as source, h5.File(wrong, "w") as handle:
            for name in ("clon", "clat"):
                handle.create_dataset(name, data=source[name][...])
            handle.attrs["uuidOfHGrid"] = np.bytes_("00000000-0000-0000-0000-000000000000")

        with self.assertRaises(ValueError) as ctx:
            self._backend(grid_file=wrong).discover()
        # the message has to name the grid that was found, since the usual cause
        # is pointing at a grid from a different run
        self.assertIn("00000000-0000-0000-0000-000000000000", str(ctx.exception))

    def test_a_file_that_is_not_a_grid_is_refused(self):
        not_a_grid = os.path.join(self.tmpdir.name, "not_a_grid.nc")
        with h5.File(not_a_grid, "w") as handle:
            handle.create_dataset("something", data=np.zeros(4))

        with self.assertRaises(ValueError) as ctx:
            self._backend(grid_file=not_a_grid).discover()
        self.assertIn("clon", str(ctx.exception))


class TestIconPickling(_IconFixture):

    def test_handles_and_buffers_do_not_travel(self):
        # a worker process gets the plan, not the open files or the scratch
        # space, both of which it rebuilds for itself
        backend = self._backend()
        backend.discover()
        backend.read(0, slice(0, 1), np.array([0]))

        state = pickle.loads(pickle.dumps(backend)).__getstate__()

        self.assertTrue(all(handle is None for handle in state["files"]))
        self.assertIsNone(state["_buffer"])

    def test_reads_survive_a_round_trip(self):
        backend = self._backend()
        backend.discover()
        before = backend.read(0, slice(0, 2), np.array([0, 4]))

        revived = pickle.loads(pickle.dumps(backend))
        after = revived.read(0, slice(0, 2), np.array([0, 4]))

        self.assertFieldEqual(after, before, "read after unpickling")


class TestWb2Backend(unittest.TestCase):

    def test_datetime64_times_survive_the_conversion(self):
        """WB2 stores times as datetime64[ns], not as seconds.

        They go through a nanosecond to second conversion that no other layout
        needs, and getting that wrong by a factor of 1e9 lands the dataset in
        1970 -- ordered, timezone aware and completely wrong.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = init_wb2_zarr_dataset(tmpdir)[0]
            metadata = get_backend(train_path, channel_names=list(CHANNEL_NAMES)).discover()

            self.assertEqual(metadata.timestamps[0], dt.datetime(TRAIN_YEARS[0], 1, 1, tzinfo=dt.timezone.utc))
            self.assertEqual(metadata.timestamps[1] - metadata.timestamps[0], dt.timedelta(hours=DHOURS))

    def test_channel_names_are_required(self):
        # the layout addresses variables by name, so there is nothing to read
        # without them; failing at construction beats failing mid epoch
        with self.assertRaises(ValueError) as ctx:
            ArcoWB2Backend("/nonexistent", channel_names=None)
        self.assertIn("channel_names", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
